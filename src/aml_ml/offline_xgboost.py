"""validation 선택과 test 격리를 적용하는 소규모 오프라인 네이티브 XGBoost 어댑터입니다.

명시적 자원 제한은 운영 학습이 아닌 소규모 실행 확인을 지원합니다.
점수는 원천 기반 proxy 라벨을 대상으로 하며, STR 정답이나 범죄 확률이 아닙니다.
네이티브 UBJ·JSON 산출물만 저장하고 원천 UID는 오프라인 ID로 유지합니다.
"""

import json
import math
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pyarrow as pa

from .offline_pipeline import FEATURE_NAMES, AlertPreprocessor, validate_dataset_integrity
from .offline_rules import R02Policy, fingerprint
from .offline_training import (
    _labelled,
    _verify_dataset_snapshot,
    matrix,
    metrics,
    require_small_training_source,
    select_validation_threshold,
)
from .storage import SourceInput, read_json, sha256, write_json

MODEL_KIND = "XGBOOST_BINARY_LOGISTIC_PROXY"
SCHEMA_VERSION = "offline-xgboost-v1"
NATIVE_MODEL_FILE = "booster.ubj"
PR_AUC_DEFINITION = "AVERAGE_PRECISION_STEP_INTEGRAL"
MODEL_FILES = frozenset(
    {"model.json", NATIVE_MODEL_FILE, "preprocessing.json", "evaluation.json", "predictions.json"}
)


def _xgboost():
    """네이티브 백엔드를 불러오지 않아도 기존 NumPy 경로를 사용할 수 있게 합니다."""
    try:
        import xgboost
    except ImportError as exc:
        raise ValueError(f"XGBoost backend is unavailable: {exc}") from exc
    return xgboost


def _environment_versions() -> dict:
    return {
        "python": platform.python_version(),
        "xgboost": _xgboost().__version__,
        "numpy": np.__version__,
        "scipy": version("scipy"),
        "pyarrow": pa.__version__,
    }


def _code_hashes() -> dict:
    package = Path(__file__).parent
    return {name: sha256(package / name) for name in ("offline_xgboost.py", "offline_training.py")}


def _bounded_real(value, name: str, maximum: float, *, positive: bool = False) -> float:
    if (
        type(value) not in (int, float)
        or value < 0
        or value > maximum
        or not math.isfinite(value)
        or (positive and value == 0)
    ):
        raise ValueError(f"Invalid finite bounded XGBoost parameter: {name}")
    return float(value)


def _validate_parameters(
    *,
    k: int,
    seed: int,
    num_boost_round: int,
    max_depth: int,
    learning_rate: float,
    min_child_weight: float,
    l2: float,
    early_stopping_rounds: int,
) -> dict:
    for name, value, minimum, maximum in (
        ("k", k, 1, 2**31 - 1),
        ("seed", seed, 0, 2**31 - 1),
        ("num_boost_round", num_boost_round, 1, 10_000),
        ("max_depth", max_depth, 1, 16),
    ):
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError(f"Invalid bounded integer XGBoost parameter: {name}")
    if type(early_stopping_rounds) is not int or not 1 <= early_stopping_rounds <= num_boost_round:
        raise ValueError("early_stopping_rounds must be within the requested boosting rounds")
    return {
        "k": k,
        "seed": seed,
        "num_boost_round": num_boost_round,
        "max_depth": max_depth,
        "learning_rate": _bounded_real(learning_rate, "learning_rate", 1.0, positive=True),
        "min_child_weight": _bounded_real(min_child_weight, "min_child_weight", 1e6),
        "l2": _bounded_real(l2, "l2", 1e6),
        "early_stopping_rounds": early_stopping_rounds,
    }


def _booster_parameters(parameters: dict) -> dict:
    return {
        "booster": "gbtree",
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "device": "cpu",
        "nthread": 1,
        "seed": parameters["seed"],
        "seed_per_iteration": False,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "colsample_bylevel": 1.0,
        "colsample_bynode": 1.0,
        "max_depth": parameters["max_depth"],
        "learning_rate": parameters["learning_rate"],
        "min_child_weight": parameters["min_child_weight"],
        "reg_lambda": parameters["l2"],
        "validate_parameters": True,
        "verbosity": 0,
    }


def _scores(booster, x: np.ndarray, columns: list[str], selected_rounds: int) -> np.ndarray:
    if x.ndim != 2 or x.shape[1] != len(columns) or not np.isfinite(x).all():
        raise ValueError("XGBoost requires an aligned finite feature matrix")
    data = _xgboost().DMatrix(x, feature_names=columns, nthread=1)
    scores = np.asarray(
        booster.predict(
            data,
            iteration_range=(0, selected_rounds),
            validate_features=True,
            training=False,
        ),
        dtype=np.float64,
    )
    if (
        scores.shape != (len(x),)
        or not np.isfinite(scores).all()
        or not ((0 <= scores) & (scores <= 1)).all()
    ):
        raise ValueError("Native XGBoost returned invalid binary logistic scores")
    return scores


def _split_metrics(y, scores, known, uids, threshold: float, k: int) -> dict:
    result = metrics(y, scores[known], threshold, k, training_alert_uids=uids)
    return dict(
        result,
        pr_auc=result["average_precision"],
        pr_auc_definition=PR_AUC_DEFINITION,
        unresolved_labels=int((~known).sum()),
    )


def _evaluate(booster, dataset: Path, model: dict) -> tuple[dict, dict]:
    report, predictions = {}, {}
    for split in ("train", "validation", "test"):
        x, y, known, uids = _labelled(dataset, split, model["feature_columns"])
        scores = _scores(booster, x, model["feature_columns"], model["selected_rounds"])
        report[split] = _split_metrics(
            y, scores, known, uids, model["recommended_threshold"], model["parameters"]["k"]
        )
        predictions[split] = scores.tolist()
    return report, predictions


def _validate_model_metadata(model: dict) -> None:
    if (
        not isinstance(model, dict)
        or model.get("schema_version") != SCHEMA_VERSION
        or model.get("model_kind") != MODEL_KIND
        or model.get("native_model_file") != NATIVE_MODEL_FILE
        or model.get("fit_scope") != "TRAIN_COMPLETE_PROXY_ONLY"
        or model.get("iteration_selection") != "MIN_VALIDATION_LOGLOSS_FIRST_BEST"
        or model.get("threshold_selection") != "MAX_VALIDATION_F1_TIE_HIGHER_THRESHOLD"
        or model.get("k_selection_scope") != "EXPLICIT_INPUT"
        or model.get("pr_auc_definition") != PR_AUC_DEFINITION
        or model.get("service_id_mapping_verified") is not False
        or model.get("deployed") is not False
        or model.get("id_namespace") != "OFFLINE_SOURCE_UID"
        or model.get("target_definition") != "ANY_POSITIVE_IN_FULL_ALERT_EVIDENCE_NOT_STR_TRUTH"
    ):
        raise ValueError("Incompatible XGBoost metadata contract")
    if not isinstance(model.get("model_version"), str) or not model["model_version"].strip():
        raise ValueError("An explicit XGBoost model version is required")
    try:
        parameters = _validate_parameters(**model["parameters"])
    except (KeyError, TypeError) as exc:
        raise ValueError("Invalid saved XGBoost parameter contract") from exc
    if model.get("native_parameters") != _booster_parameters(parameters):
        raise ValueError("Saved native XGBoost parameters differ from the bounded CPU contract")
    selected = model.get("selected_rounds")
    best = model.get("best_iteration")
    trained = model.get("training_rounds")
    if (
        type(selected) is not int
        or not 1 <= selected <= parameters["num_boost_round"]
        or type(best) is not int
        or best != selected - 1
        or type(trained) is not int
        or not selected <= trained <= parameters["num_boost_round"]
        or model.get("prediction_iteration_range") != [0, selected]
    ):
        raise ValueError("Invalid selected XGBoost iteration range")
    history = model.get("training_history")
    if (
        not isinstance(history, dict)
        or set(history) != {"validation"}
        or not isinstance(history["validation"], dict)
        or set(history["validation"]) != {"logloss"}
    ):
        raise ValueError("XGBoost model selection requires validation-only logloss history")
    losses = history["validation"]["logloss"]
    if (
        not isinstance(losses, list)
        or len(losses) != trained
        or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in losses)
        or best != min(range(len(losses)), key=losses.__getitem__)
        or model.get("best_validation_logloss") != losses[best]
    ):
        raise ValueError("Saved best XGBoost iteration disagrees with validation history")
    _bounded_real(model.get("recommended_threshold"), "recommended_threshold", 1.0)


def train_evaluate(
    dataset: Path,
    source: SourceInput,
    policy: R02Policy,
    output: Path,
    *,
    execution_scope: str,
    model_version: str,
    k: int,
    seed: int,
    num_boost_round: int = 100,
    max_depth: int = 3,
    learning_rate: float = 0.1,
    min_child_weight: float = 1.0,
    l2: float = 1.0,
    early_stopping_rounds: int = 10,
    max_source_rows: int = 100_000,
    support_policy: dict | None = None,
) -> dict:
    policy.require(execution_scope)
    parameters = _validate_parameters(
        k=k,
        seed=seed,
        num_boost_round=num_boost_round,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_child_weight=min_child_weight,
        l2=l2,
        early_stopping_rounds=early_stopping_rounds,
    )
    require_small_training_source(source, max_source_rows)
    if not isinstance(model_version, str) or not model_version.strip():
        raise ValueError("An explicit XGBoost model version is required")
    if output.exists() or output.is_symlink():
        raise FileExistsError("Use a new model directory; existing models are preserved")
    from .offline_readiness import require_evaluation_support

    support = require_evaluation_support(dataset, support_policy, MODEL_KIND)
    backend = _xgboost()
    from xgboost.callback import EarlyStopping

    environment = _environment_versions()
    code_hashes = _code_hashes()
    verified, validated_manifest = _verify_dataset_snapshot(
        dataset, source, policy, execution_scope=execution_scope, max_source_rows=max_source_rows
    )
    manifest = read_json(dataset / "manifest.json")
    columns = manifest["feature_columns"]
    train_x, train_y, train_known, _ = _labelled(dataset, "train", columns)
    valid_x, valid_y, valid_known, _ = _labelled(dataset, "validation", columns)
    if set(train_y) != {0.0, 1.0} or set(valid_y) != {0.0, 1.0}:
        raise ValueError("XGBoost training and validation each require both complete proxy classes")
    dtrain = backend.DMatrix(train_x[train_known], label=train_y, feature_names=columns, nthread=1)
    dvalid = backend.DMatrix(valid_x[valid_known], label=valid_y, feature_names=columns, nthread=1)
    native_parameters = _booster_parameters(parameters)
    history = {}
    stopper = EarlyStopping(
        rounds=early_stopping_rounds,
        data_name="validation",
        metric_name="logloss",
        maximize=False,
        save_best=True,
    )
    booster = backend.train(
        native_parameters,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dvalid, "validation")],
        evals_result=history,
        callbacks=[stopper],
        verbose_eval=False,
    )
    best_iteration = int(booster.best_iteration)
    selected_rounds = best_iteration + 1
    if booster.num_boosted_rounds() != selected_rounds:
        raise ValueError("XGBoost early stopping did not preserve only the selected best rounds")
    validation_scores = _scores(booster, valid_x[valid_known], columns, selected_rounds)
    threshold = select_validation_threshold(valid_y, validation_scores)
    model = {
        "schema_version": SCHEMA_VERSION,
        "model_kind": MODEL_KIND,
        "evaluation_support": support,
        "model_version": model_version,
        "native_model_file": NATIVE_MODEL_FILE,
        "execution_scope": execution_scope,
        "dataset_fingerprint": verified["dataset_fingerprint"],
        "validated_dataset_manifest_sha256": validated_manifest,
        "contract": manifest["contract"],
        "feature_columns": columns,
        "preprocessing_sha256": sha256(dataset / "preprocessing.json"),
        "parameters": parameters,
        "native_parameters": native_parameters,
        "selected_rounds": selected_rounds,
        "best_iteration": best_iteration,
        "training_rounds": len(history["validation"]["logloss"]),
        "best_validation_logloss": float(booster.best_score),
        "training_history": history,
        "prediction_iteration_range": [0, selected_rounds],
        "recommended_threshold": threshold,
        "iteration_selection": "MIN_VALIDATION_LOGLOSS_FIRST_BEST",
        "threshold_selection": "MAX_VALIDATION_F1_TIE_HIGHER_THRESHOLD",
        "fit_scope": "TRAIN_COMPLETE_PROXY_ONLY",
        "k_selection_scope": "EXPLICIT_INPUT",
        "pr_auc_definition": PR_AUC_DEFINITION,
        "environment_versions": environment,
        "code_sha256": code_hashes,
        "id_namespace": "OFFLINE_SOURCE_UID",
        "target_definition": "ANY_POSITIVE_IN_FULL_ALERT_EVIDENCE_NOT_STR_TRUTH",
        "service_id_mapping_verified": False,
        "deployed": False,
    }
    _validate_model_metadata(model)
    # 트리, 선택한 반복 수와 임계값을 고정한 뒤에만 test를 평가에 사용합니다.
    report, predictions = _evaluate(booster, dataset, model)
    current = validate_dataset_integrity(dataset, source, policy, execution_scope=execution_scope)
    if (
        current["manifest_sha256"] != validated_manifest
        or _code_hashes() != code_hashes
        or _environment_versions() != environment
    ):
        raise ValueError("Dataset/code/environment changed during XGBoost training")
    output.mkdir(parents=True, exist_ok=False)
    # 메타데이터·해시의 순환 참조를 막도록 네이티브 SHA는 바깥 파일 목록에만 둡니다.
    booster.set_attr(dajaba_metadata_fingerprint=fingerprint(model))
    booster.save_model(output / NATIVE_MODEL_FILE)
    write_json(output / "model.json", model)
    write_json(output / "evaluation.json", report)
    write_json(output / "predictions.json", predictions)
    (output / "preprocessing.json").write_bytes((dataset / "preprocessing.json").read_bytes())
    inventory = {name: sha256(output / name) for name in sorted(MODEL_FILES)}
    write_json(
        output / "manifest.json",
        {"status": "COMPLETED", "files": inventory, "model_fingerprint": fingerprint(inventory)},
    )
    saved, reloaded = _load_artifacts(
        output,
        dataset,
        source,
        policy,
        execution_scope=execution_scope,
        max_source_rows=max_source_rows,
    )
    return _verify_model_outputs(output, dataset, saved, reloaded)


def _load_artifacts(
    folder: Path,
    dataset: Path,
    source: SourceInput,
    policy: R02Policy,
    *,
    execution_scope: str,
    max_source_rows: int,
):
    policy.require(execution_scope)
    require_small_training_source(source, max_source_rows)
    verified = validate_dataset_integrity(dataset, source, policy, execution_scope=execution_scope)
    manifest_digest = sha256(folder / "manifest.json")
    inventory = read_json(folder / "manifest.json")
    entries, root = list(folder.iterdir()), folder.resolve()
    if (
        inventory.get("status") != "COMPLETED"
        or not isinstance(inventory.get("files"), dict)
        or set(inventory["files"]) != MODEL_FILES
        or inventory.get("model_fingerprint") != fingerprint(inventory["files"])
        or {entry.name for entry in entries} != MODEL_FILES | {"manifest.json"}
        or any(
            not entry.is_file() or entry.is_symlink() or entry.resolve().parent != root
            for entry in entries
        )
    ):
        raise ValueError("Incomplete or altered XGBoost model inventory")
    if any(sha256(folder / name) != digest for name, digest in inventory["files"].items()):
        raise ValueError("XGBoost model artifact hash mismatch")
    model = read_json(folder / "model.json")
    _validate_model_metadata(model)
    from .offline_readiness import require_evaluation_support

    if model.get("evaluation_support") != require_evaluation_support(
        dataset,
        model.get("evaluation_support", {}).get("policy"),
        MODEL_KIND,
    ):
        raise ValueError("Evaluation support policy or eligible population changed")
    manifest = read_json(dataset / "manifest.json")
    if (
        model.get("execution_scope") != execution_scope
        or model.get("validated_dataset_manifest_sha256") != verified["manifest_sha256"]
        or verified["manifest_sha256"] != sha256(dataset / "manifest.json")
        or model.get("dataset_fingerprint") != verified["dataset_fingerprint"]
        or model.get("contract") != manifest["contract"]
        or model.get("feature_columns") != manifest["feature_columns"]
        or model.get("preprocessing_sha256") != sha256(dataset / "preprocessing.json")
        or model.get("preprocessing_sha256") != sha256(folder / "preprocessing.json")
        or model.get("code_sha256") != _code_hashes()
        or model.get("environment_versions") != _environment_versions()
    ):
        raise ValueError("Incompatible XGBoost model/data/policy/code/environment; retrain")
    backend = _xgboost()
    booster = backend.Booster(
        params={"device": "cpu", "nthread": 1}, model_file=folder / NATIVE_MODEL_FILE
    )
    config = json.loads(booster.save_config())["learner"]
    best_score = booster.attr("best_score")
    if (
        booster.attr("dajaba_metadata_fingerprint") != fingerprint(model)
        or booster.feature_names != model["feature_columns"]
        or booster.num_features() != len(model["feature_columns"])
        or booster.num_boosted_rounds() != model["selected_rounds"]
        or booster.attr("best_iteration") != str(model["best_iteration"])
        or best_score is None
        or float(best_score) != model["best_validation_logloss"]
        or config["objective"]["name"] != "binary:logistic"
        or config["gradient_booster"]["name"] != "gbtree"
    ):
        raise ValueError("Native XGBoost model differs from the saved feature/iteration contract")
    if (
        sha256(folder / "manifest.json") != manifest_digest
        or any(sha256(folder / name) != digest for name, digest in inventory["files"].items())
        or sha256(dataset / "manifest.json") != verified["manifest_sha256"]
    ):
        raise ValueError("XGBoost model or dataset changed during loading")
    return model, booster


def load_model(
    folder: Path,
    dataset: Path,
    source: SourceInput,
    policy: R02Policy,
    *,
    execution_scope: str,
    max_source_rows: int = 100_000,
) -> dict:
    """원천 재연산 없이 현재 파일 무결성을 검증하고 일반 메타데이터를 반환합니다."""
    model, _ = _load_artifacts(
        folder,
        dataset,
        source,
        policy,
        execution_scope=execution_scope,
        max_source_rows=max_source_rows,
    )
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
) -> np.ndarray:
    model, booster = _load_artifacts(
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
    return _scores(
        booster,
        matrix(processed, model["feature_columns"]),
        model["feature_columns"],
        model["selected_rounds"],
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
    """명시적으로 원천 기반 데이터셋을 재연산한 뒤 저장된 네이티브 모델을 재현합니다."""
    policy.require(execution_scope)
    require_small_training_source(source, max_source_rows)
    _, validated_manifest = _verify_dataset_snapshot(
        dataset, source, policy, execution_scope=execution_scope, max_source_rows=max_source_rows
    )
    model, booster = _load_artifacts(
        folder,
        dataset,
        source,
        policy,
        execution_scope=execution_scope,
        max_source_rows=max_source_rows,
    )
    if model["validated_dataset_manifest_sha256"] != validated_manifest:
        raise ValueError("Dataset manifest changed since source-backed verification")
    return _verify_model_outputs(folder, dataset, model, booster)


def _verify_model_outputs(folder: Path, dataset: Path, model: dict, booster) -> dict:
    evaluation, predictions = _evaluate(booster, dataset, model)
    if evaluation != read_json(folder / "evaluation.json") or predictions != read_json(
        folder / "predictions.json"
    ):
        raise ValueError("Stored XGBoost scores/evaluation are not reproducible from saved model")
    return {
        "status": "VERIFIED",
        "model_kind": MODEL_KIND,
        "execution_scope": model["execution_scope"],
        "model_version": model["model_version"],
        "evaluation": evaluation,
        "recommended_threshold": model["recommended_threshold"],
        "selected_rounds": model["selected_rounds"],
        "pr_auc_definition": PR_AUC_DEFINITION,
        "service_id_mapping_verified": False,
        "deployed": False,
    }
