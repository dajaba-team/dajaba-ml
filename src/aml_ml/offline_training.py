"""작고 결정적인 NumPy 로지스틱 기준 모델로 validation에서 선택하고 test는 격리합니다.

오프라인 proxy 기준 모델이며 배포 모델이나 범죄 확률이 아닙니다.
이 NumPy 모델의 합성 소규모 검사는 XGBoost 성능을 입증하지 않습니다.
모델 파일에는 실행 가능한 pickle 대신 JSON 숫자 배열을 담습니다.
"""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .offline_pipeline import (
    FEATURE_NAMES,
    AlertPreprocessor,
    validate_dataset_integrity,
    verify_dataset,
)
from .offline_rules import R02Policy, fingerprint
from .storage import SourceInput, read_json, sha256, write_json


def probabilities(x: np.ndarray, weights: np.ndarray, intercept: float) -> np.ndarray:
    z = np.clip(x @ weights + intercept, -35, 35)
    return 1 / (1 + np.exp(-z))


def matrix(table: pa.Table, columns: list[str]) -> np.ndarray:
    if table.column_names != columns:
        raise ValueError("Ordered model feature columns do not match")
    x = np.column_stack([table.column(n).to_numpy(zero_copy_only=False) for n in columns])
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Model input must be a finite numeric matrix")
    return x.astype(np.float64)


def _evaluation_arrays(y: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y, scores = np.asarray(y), np.asarray(scores)
    if y.ndim != 1 or scores.ndim != 1 or len(y) != len(scores) or not len(y):
        raise ValueError("Aligned nonempty one-dimensional evaluation arrays required")
    if y.dtype.kind not in "biuf" or scores.dtype.kind not in "biuf":
        raise ValueError("Evaluation requires real numeric labels and scores")
    if (
        not np.isfinite(y).all()
        or not np.isfinite(scores).all()
        or not np.isin(y, [0, 1]).all()
        or not ((0 <= scores) & (scores <= 1)).all()
    ):
        raise ValueError("Evaluation requires binary proxy truth and bounded scores")
    return y.astype(np.float64, copy=False), scores.astype(np.float64, copy=False)


def _evaluation_order(scores: np.ndarray, training_alert_uids) -> tuple[np.ndarray, str]:
    if training_alert_uids is None:
        return np.argsort(-scores, kind="stable"), "SCORE_DESC_STABLE_INPUT_ORDER"
    uids = np.asarray(training_alert_uids, dtype=object)
    if (
        uids.ndim != 1
        or len(uids) != len(scores)
        or any(not isinstance(uid, str) or not uid.strip() for uid in uids)
        or len(set(uids)) != len(uids)
    ):
        raise ValueError("Aligned unique nonempty training alert UIDs required")
    return (
        np.lexsort((uids.astype(str), -scores)),
        "SCORE_DESC_THEN_TRAINING_ALERT_UID_ASC",
    )


def select_validation_threshold(y: np.ndarray, scores: np.ndarray) -> float:
    """{0, scores, 1}에서 F1을 최대화하며, 정확히 동률이면 더 높은 임계값을 고릅니다.

    한 번의 내림차순 정렬과 누적 TP 순회로 같은 점수의 전체 묶음을 평가합니다.
    0 경계는 최저 점수 묶음과 겹쳐 동률에서 선택되지 않습니다. 양성이 없는
    데이터에서도 가능한 최고 임계값인 1은 유지합니다.
    """
    y, scores = _evaluation_arrays(y, scores)
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(y) - 1]
    cumulative = np.cumsum(y[order], dtype=np.int64)[ends]
    positives = int(y.sum())
    best_threshold, best_tp, best_denominator = 1.0, 0, 1
    for end, tp in zip(ends, cumulative, strict=True):
        threshold, tp = float(sorted_scores[end]), int(tp)
        denominator = positives + int(end) + 1
        # F1 = 2*TP/(positives + selected); 정수 곱으로 반올림에 따른 동률 오판을 막습니다.
        left, right = tp * best_denominator, best_tp * denominator
        if left > right or (left == right and threshold > best_threshold):
            best_threshold, best_tp, best_denominator = threshold, tp, denominator
    return best_threshold


def metrics(
    y: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    k: int,
    *,
    training_alert_uids=None,
) -> dict:
    cutoff = np.asarray(threshold)
    if (
        type(k) is not int
        or k < 1
        or cutoff.ndim != 0
        or cutoff.dtype.kind not in "biuf"
        or not np.isfinite(cutoff)
        or not 0 <= cutoff <= 1
    ):
        raise ValueError("Positive K and finite bounded scalar threshold required")
    y, scores = _evaluation_arrays(y, scores)
    order, tie_policy = _evaluation_order(scores, training_alert_uids)
    predicted = scores >= threshold
    positive = int(y.sum())
    tp = int(y[predicted].sum())
    count = int(predicted.sum())
    precision = tp / count if count else 0.0
    recall = tp / positive if positive else None
    f1 = 2 * tp / (positive + count) if positive + count else 0.0
    sorted_y, sorted_scores = y[order], scores[order]
    # AP는 서로 다른 점수 임계값의 재현율 증가분을 적분합니다. 동점은 함께 처리합니다.
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(y) - 1]
    cumulative = np.cumsum(sorted_y)[ends]
    ap = (
        float(np.sum(np.diff(np.r_[0, cumulative]) / positive * cumulative / (ends + 1)))
        if positive
        else None
    )
    used_k = min(k, len(y))
    top_positive = int(sorted_y[:used_k].sum())
    clipped = np.clip(scores, 1e-15, 1 - 1e-15)
    return {
        "rows": len(y),
        "positives": positive,
        "selected": count,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "average_precision": ap,
        "k": used_k,
        "requested_k": k,
        "top_k_tie_policy": tie_policy,
        "precision_at_k": top_positive / used_k,
        "recall_at_k": top_positive / positive if positive else None,
        "log_loss": float(-np.mean(y * np.log(clipped) + (1 - y) * np.log(1 - clipped))),
    }


def _labelled(folder: Path, split: str, columns: list[str]):
    x = matrix(pq.read_table(folder / f"X_{split}.parquet"), columns)
    targets = pq.read_table(folder / f"y_{split}.parquet")
    labels = targets.column("y_alert_proxy").to_pylist()
    if len(x) != len(labels):
        raise ValueError(f"Unaligned X/y rows in {split}")
    uids = np.asarray(targets.column("training_alert_uid").to_pylist(), dtype=object)
    known = np.array([v is not None for v in labels], dtype=bool)
    if not known.any():
        raise ValueError(f"No complete proxy labels in {split}; null is never zero")
    return (
        x,
        np.array([v for v in labels if v is not None], dtype=np.float64),
        known,
        uids[known],
    )


def _evaluate_split(
    dataset: Path,
    split: str,
    columns: list[str],
    weights: np.ndarray,
    intercept: float,
    threshold: float,
    k: int,
) -> tuple[dict, list[float]]:
    x, y, known, uids = _labelled(dataset, split, columns)
    scores = probabilities(x, weights, intercept)
    evaluation = dict(
        metrics(y, scores[known], threshold, k, training_alert_uids=uids),
        unresolved_labels=int((~known).sum()),
    )
    return evaluation, scores.tolist()


def require_small_training_source(source: SourceInput, max_source_rows: int) -> None:
    if (
        type(max_source_rows) is not int
        or not 1 <= source.metadata["input_rows"] <= max_source_rows
    ):
        raise ValueError(
            "Small baseline source budget exceeded; "
            "streaming preparation does not authorize large training"
        )


def _verify_dataset_snapshot(
    dataset: Path,
    source: SourceInput,
    policy: R02Policy,
    *,
    execution_scope: str,
    max_source_rows: int,
) -> tuple[dict, str]:
    manifest_sha256 = sha256(dataset / "manifest.json")
    verified = verify_dataset(
        dataset, source, policy, execution_scope=execution_scope, max_source_rows=max_source_rows
    )
    if sha256(dataset / "manifest.json") != manifest_sha256:
        raise ValueError("Dataset manifest changed during source-backed verification")
    return verified, manifest_sha256


def train_evaluate(
    dataset: Path,
    source: SourceInput,
    policy: R02Policy,
    output: Path,
    *,
    execution_scope: str,
    model_version: str,
    k: int,
    iterations: int = 500,
    learning_rate: float = 0.1,
    l2: float = 0.01,
    max_source_rows: int = 100_000,
    support_policy: dict | None = None,
) -> dict:
    require_small_training_source(source, max_source_rows)
    if not isinstance(model_version, str) or not model_version.strip():
        raise ValueError("An explicit model version is required")
    if type(k) is not int or k < 1 or type(iterations) is not int or not 1 <= iterations <= 10_000:
        raise ValueError("Positive K and bounded iteration count required")
    if not np.isfinite([learning_rate, l2]).all() or not 0 < learning_rate <= 1 or l2 < 0:
        raise ValueError("Invalid optimization parameters")
    if output.exists():
        raise FileExistsError("Use a new model directory; existing models are preserved")
    from .offline_readiness import require_evaluation_support

    support = require_evaluation_support(dataset, support_policy, "NUMPY_LOGISTIC_PROXY_BASELINE")
    verified, validated_manifest_sha256 = _verify_dataset_snapshot(
        dataset, source, policy, execution_scope=execution_scope, max_source_rows=max_source_rows
    )
    manifest = read_json(dataset / "manifest.json")
    columns = manifest["feature_columns"]
    train_x, train_y, train_known, _ = _labelled(dataset, "train", columns)
    if set(train_y) != {0.0, 1.0}:
        raise ValueError("Baseline training requires both proxy classes in train")
    weights = np.zeros(len(columns), dtype=np.float64)
    intercept = 0.0
    fit_x = train_x[train_known]
    for _ in range(iterations):
        error = probabilities(fit_x, weights, intercept) - train_y
        weights -= learning_rate * (fit_x.T @ error / len(train_y) + l2 * weights)
        intercept -= learning_rate * float(error.mean())
    if not np.isfinite(weights).all() or not np.isfinite(intercept):
        raise ValueError("Training diverged")
    valid_x, valid_y, valid_known, _ = _labelled(dataset, "validation", columns)
    if set(valid_y) != {0.0, 1.0}:
        raise ValueError("Threshold selection requires both proxy classes in validation")
    valid_scores = probabilities(valid_x[valid_known], weights, intercept)
    threshold = select_validation_threshold(valid_y, valid_scores)
    # 가중치와 validation 임계값을 고정한 뒤에만 test를 평가에 사용합니다.
    report = {}
    predictions = {}
    for split in ("train", "validation", "test"):
        report[split], predictions[split] = _evaluate_split(
            dataset, split, columns, weights, intercept, threshold, k
        )
    model = {
        "model_kind": "NUMPY_LOGISTIC_PROXY_BASELINE",
        "evaluation_support": support,
        "model_version": model_version,
        "execution_scope": execution_scope,
        "dataset_fingerprint": verified["dataset_fingerprint"],
        "validated_dataset_manifest_sha256": validated_manifest_sha256,
        "contract": manifest["contract"],
        "feature_columns": columns,
        "preprocessing_sha256": sha256(dataset / "preprocessing.json"),
        "weights": weights.tolist(),
        "intercept": intercept,
        "recommended_threshold": threshold,
        "threshold_selection": "MAX_VALIDATION_F1_TIE_HIGHER_THRESHOLD",
        "fit_scope": "TRAIN_COMPLETE_PROXY_ONLY",
        "k_selection_scope": "EXPLICIT_INPUT",
        "parameters": {"iterations": iterations, "learning_rate": learning_rate, "l2": l2, "k": k},
        "training_code_sha256": sha256(Path(__file__)),
        "service_id_mapping_verified": False,
        "deployed": False,
    }
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "model.json", model)
    write_json(output / "evaluation.json", report)
    write_json(output / "predictions.json", predictions)
    (output / "preprocessing.json").write_bytes((dataset / "preprocessing.json").read_bytes())
    inventory = {p.name: sha256(p) for p in output.iterdir() if p.is_file()}
    write_json(
        output / "manifest.json",
        {
            "status": "COMPLETED",
            "files": inventory,
            "model_fingerprint": fingerprint(inventory),
        },
    )
    saved_model = load_model(
        output,
        dataset,
        source,
        policy,
        execution_scope=execution_scope,
        max_source_rows=max_source_rows,
    )
    return _verify_model_outputs(output, dataset, saved_model, execution_scope=execution_scope)


def load_model(
    folder: Path,
    dataset: Path,
    source: SourceInput,
    policy: R02Policy,
    *,
    execution_scope: str,
    max_source_rows: int = 100_000,
) -> dict:
    require_small_training_source(source, max_source_rows)
    verified = validate_dataset_integrity(dataset, source, policy, execution_scope=execution_scope)
    inventory = read_json(folder / "manifest.json")
    expected_files = {"model.json", "evaluation.json", "predictions.json", "preprocessing.json"}
    if (
        inventory.get("status") != "COMPLETED"
        or set(inventory["files"]) != expected_files
        or inventory["model_fingerprint"] != fingerprint(inventory["files"])
    ):
        raise ValueError("Incomplete model inventory")
    if any(sha256(folder / n) != h for n, h in inventory["files"].items()):
        raise ValueError("Model artifact hash mismatch")
    model = read_json(folder / "model.json")
    from .offline_readiness import require_evaluation_support

    if model.get("evaluation_support") != require_evaluation_support(
        dataset,
        model.get("evaluation_support", {}).get("policy"),
        "NUMPY_LOGISTIC_PROXY_BASELINE",
    ):
        raise ValueError("Evaluation support policy or eligible population changed")
    manifest = read_json(dataset / "manifest.json")
    if (
        model["execution_scope"] != execution_scope
        or model.get("validated_dataset_manifest_sha256") != verified["manifest_sha256"]
        or verified["manifest_sha256"] != sha256(dataset / "manifest.json")
        or model["dataset_fingerprint"] != verified["dataset_fingerprint"]
        or model["contract"] != manifest["contract"]
        or model["feature_columns"] != manifest["feature_columns"]
        or model["preprocessing_sha256"] != sha256(dataset / "preprocessing.json")
        or model["preprocessing_sha256"] != sha256(folder / "preprocessing.json")
        or model["training_code_sha256"] != sha256(Path(__file__))
    ):
        raise ValueError("Incompatible model/data/policy/code; regenerate or retrain")
    weights = np.asarray(model["weights"], dtype=np.float64)
    if (
        weights.shape != (len(model["feature_columns"]),)
        or not np.isfinite(weights).all()
        or not np.isfinite(model["intercept"])
        or not 0 <= model["recommended_threshold"] <= 1
    ):
        raise ValueError("Invalid numeric model state")
    return model


def predict_raw(
    folder: Path,
    raw: pa.Table,
    dataset: Path,
    source: SourceInput,
    policy: R02Policy,
    *,
    execution_scope: str,
    max_source_rows: int = 100_000,
):
    model = load_model(
        folder,
        dataset,
        source,
        policy,
        execution_scope=execution_scope,
        max_source_rows=max_source_rows,
    )
    if tuple(raw.column_names) != FEATURE_NAMES:
        raise ValueError("Raw alert feature allowlist/order mismatch; labels/IDs are forbidden")
    processed = AlertPreprocessor.load(folder / "preprocessing.json").transform(raw)
    return probabilities(
        matrix(processed, model["feature_columns"]),
        np.asarray(model["weights"]),
        model["intercept"],
    )


def verify_model(
    folder: Path,
    dataset: Path,
    source: SourceInput,
    policy: R02Policy,
    *,
    execution_scope: str,
    max_source_rows: int = 100_000,
) -> dict:
    """명시적으로 원천 기반 데이터셋을 재연산한 뒤 저장 모델의 출력을 재현합니다."""
    require_small_training_source(source, max_source_rows)
    _, validated_manifest_sha256 = _verify_dataset_snapshot(
        dataset, source, policy, execution_scope=execution_scope, max_source_rows=max_source_rows
    )
    model = load_model(
        folder,
        dataset,
        source,
        policy,
        execution_scope=execution_scope,
        max_source_rows=max_source_rows,
    )
    if model["validated_dataset_manifest_sha256"] != validated_manifest_sha256:
        raise ValueError("Dataset manifest changed since source-backed verification")
    return _verify_model_outputs(folder, dataset, model, execution_scope=execution_scope)


def _verify_model_outputs(
    folder: Path, dataset: Path, model: dict, *, execution_scope: str
) -> dict:
    stored = read_json(folder / "predictions.json")
    evaluation = read_json(folder / "evaluation.json")
    for split in ("train", "validation", "test"):
        expected, scores = _evaluate_split(
            dataset,
            split,
            model["feature_columns"],
            np.asarray(model["weights"]),
            model["intercept"],
            model["recommended_threshold"],
            model["parameters"]["k"],
        )
        if scores != stored[split] or expected != evaluation[split]:
            raise ValueError("Stored scores/evaluation are not reproducible from saved model")
    return {
        "status": "VERIFIED",
        "execution_scope": execution_scope,
        "model_version": model["model_version"],
        "evaluation": evaluation,
        "recommended_threshold": model["recommended_threshold"],
        "service_id_mapping_verified": False,
        "deployed": False,
    }
