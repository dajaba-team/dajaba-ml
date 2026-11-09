"""오프라인 데이터셋 산출물, 원천 연결과 무결성 검증을 제공합니다."""

import hashlib
import json
from dataclasses import asdict
from datetime import datetime
from importlib.resources import files
from pathlib import Path

from .alert_dataset import OfflineAlert
from .dataset_quality import SPLITS as SPLITS
from .offline_eligibility import ELIGIBILITY_VERSION
from .offline_features import FEATURE_NAMES, OFFLINE_FEATURE_VERSION
from .offline_rules import R02Policy, fingerprint
from .offline_serialization import FORMAT_FILES
from .storage import SourceInput, read_json, sha256

FIT_BATCH_ROWS = 256
DATASET_FILES = FORMAT_FILES | {"splits.json", "preprocessing.json", "quality.json"}


def validate_source(source: SourceInput) -> None:
    count = source.metadata["input_rows"]
    if (
        source.metadata.get("status") != "VERIFIED"
        or source.metadata.get("ml_output_rows") != count
        or any(
            m.get("status") != "VERIFIED"
            or m.get("scope") != "FULL_SOURCE"
            or m.get("source_release") != source.metadata["source_release"]
            or m.get("rows") != count
            for m in (source.transactions, source.labels)
        )
    ):
        raise ValueError("Verified complete source manifests are required, never service excerpts")
    if count <= 0 or len(source.parts("transactions")) != len(source.parts("labels")):
        raise ValueError("Incomplete or unaligned source parts")


def manifest_snapshot(path: Path) -> tuple[dict, str]:
    """같은 바이트를 파싱하고 해시하여 반환 지문이 실제 읽은 내용을 증명하게 합니다."""
    content = path.read_bytes()
    value = json.loads(content.decode("utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON manifest object")
    return value, hashlib.sha256(content).hexdigest()


def _source_manifest_hashes(source: SourceInput) -> dict[str, str]:
    """오래되거나 변경된 캐시 입력을 거부하고 현재 manifest 전체 바이트를 연결합니다."""
    validate_source(source)
    folder = source.folder.resolve()
    metadata, digest = manifest_snapshot(folder / "source_manifest.json")
    if metadata != source.metadata or metadata.get("source_id") != source.id:
        raise ValueError("Source manifest changed; reload the source and regenerate")
    hashes = {"source_manifest.json": digest}
    for key, cached in (
        ("ml_transactions", source.transactions),
        ("ml_labels", source.labels),
    ):
        table_folder = (folder / metadata["datasets"][key]["path"]).resolve()
        if not table_folder.is_relative_to(folder):
            raise ValueError("Unsafe source dataset path")
        current, digest = manifest_snapshot(table_folder / "manifest.json")
        if current != cached:
            raise ValueError("Source table manifest changed; reload the source and regenerate")
        if (
            current.get("source_id") != source.id
            or (key == "ml_transactions" and current.get("source_kind") != source.kind)
            or current.get("purpose") != "ml"
            or sum(part["rows"] for part in current["parts"]) != metadata["input_rows"]
        ):
            raise ValueError("Source table identity or complete row count changed")
        hashes[key] = digest
    if [part["rows"] for part in source.transactions["parts"]] != [
        part["rows"] for part in source.labels["parts"]
    ]:
        raise ValueError("Source transaction/label parts are not aligned")
    return hashes


def source_fingerprint(source: SourceInput) -> str:
    manifest_hashes = _source_manifest_hashes(source)
    return fingerprint(
        {
            "id": source.id,
            "kind": source.kind,
            "metadata": source.metadata,
            "transactions": source.transactions,
            "labels": source.labels,
            "source_manifest_sha256": manifest_hashes["source_manifest.json"],
            "table_manifest_sha256": {
                key: manifest_hashes[key] for key in ("ml_transactions", "ml_labels")
            },
        }
    )


def contract(
    source: SourceInput,
    policy: R02Policy,
    scope: str,
    train_before: datetime,
    test_from: datetime,
    *,
    horizon_seconds=86400,
) -> dict:
    policy.require(scope)
    package = files("aml_ml")
    return {
        "execution_scope": scope,
        "rule_policy": asdict(policy),
        "rule_config_hash": policy.config_hash,
        "source_fingerprint": source_fingerprint(source),
        "source_id": source.id,
        "source_release": source.metadata["source_release"],
        "feature_version": OFFLINE_FEATURE_VERSION,
        "raw_feature_columns": list(FEATURE_NAMES),
        "train_before": train_before.isoformat(),
        "test_from": test_from.isoformat(),
        "eligibility_version": ELIGIBILITY_VERSION,
        "horizon_seconds": horizon_seconds,
        "code_sha256": {
            n: sha256(Path(str(package.joinpath(n))))
            for n in (
                "offline_rules.py",
                "offline_pipeline.py",
                "offline_streaming.py",
                "offline_account_stream.py",
                "offline_cli.py",
                "offline_checkpoint.py",
                "offline_resume.py",
                "offline_io.py",
                "offline_progress.py",
                "offline_publish.py",
                "offline_serialization.py",
                "offline_settings.py",
                "offline_eligibility.py",
                "offline_measurements.py",
                "offline_readiness.py",
                "offline_contract.py",
                "offline_features.py",
                "dataset_quality.py",
                "sliding_window.py",
                "storage.py",
                "alert_dataset.py",
                "alert_preparation.py",
                "history.py",
                "schema.py",
                "preprocessing.py",
                "currencies.json",
            )
        },
        "id_namespace": "OFFLINE_SOURCE_UID",
        "target_definition": "ANY_POSITIVE_IN_FULL_ALERT_EVIDENCE_NOT_STR_TRUTH",
        "money_policy": "NO_FX_NO_CROSS_CURRENCY_SUM_SAML_DIRECTIONAL_MONEY_NULL",
    }


def regeneration_required(
    folder: Path, source: SourceInput, policy: R02Policy, *, execution_scope: str
) -> dict:
    """하위 산출물을 보수적으로 무효화하며, 바뀐 룰을 암묵적으로 승인하지 않습니다."""
    saved = read_json(folder / "manifest.json")["contract"]
    try:
        current = contract(
            source,
            policy,
            execution_scope,
            datetime.fromisoformat(saved["train_before"]),
            datetime.fromisoformat(saved["test_from"]),
            horizon_seconds=saved["horizon_seconds"],
        )
    except (ValueError, KeyError, OSError) as exc:
        return {
            "status": "BLOCKED",
            "reason": str(exc),
            "changed_fields": [],
            "regenerate": ["alerts", "features", "preprocessing", "model", "evaluation"],
        }
    changed = sorted(
        key for key in saved.keys() | current.keys() if saved.get(key) != current.get(key)
    )
    return {
        "status": "REGENERATE" if changed else "CURRENT",
        "changed_fields": changed,
        "regenerate": ["alerts", "features", "preprocessing", "model", "evaluation"]
        if changed
        else [],
    }


def alert_records(alerts: list[OfflineAlert]) -> list[dict]:
    records = []
    for alert in alerts:
        record = asdict(alert)
        for key in ("reference_ts", "window_start", "window_end"):
            record[key] = record[key].isoformat()
        records.append(record)
    return records


def validate_artifact_snapshot(folder: Path, manifest: dict, manifest_digest: str) -> None:
    """고정된 manifest와 출력 바이트를 재대조하며, 원천 읽기나 재연산은 하지 않습니다."""
    if sha256(folder / "manifest.json") != manifest_digest:
        raise ValueError("Dataset manifest changed during artifact validation")
    entries = list(folder.iterdir())
    root = folder.resolve()
    if (
        set(manifest["files"]) != DATASET_FILES
        or {entry.name for entry in entries} != DATASET_FILES | {"manifest.json"}
        or any(
            not entry.is_file() or entry.is_symlink() or entry.resolve().parent != root
            for entry in entries
        )
    ):
        raise ValueError("Dataset artifact inventory mismatch")
    for name, digest in manifest["files"].items():
        if sha256(folder / name) != digest:
            raise ValueError("Dataset file hash mismatch")
    if sha256(folder / "manifest.json") != manifest_digest:
        raise ValueError("Dataset manifest changed during artifact validation")


def validate_dataset_integrity(
    folder: Path,
    source: SourceInput,
    policy: R02Policy,
    *,
    execution_scope: str,
) -> dict:
    """룰·피처·학습을 다시 실행하지 않고 현재 원천·산출물 바이트를 확인합니다.

    파일 무결성 확인이며 의미적 검증은 아닙니다. 모델 사용자는 verify_dataset
    성공 후 고정한 지문과 manifest_sha256도 대조해야 합니다.
    호출마다 선언된 원천 파트를 읽으며, 수정 시각만으로 유효성을 캐시하지 않습니다.
    """
    manifest, manifest_digest = manifest_snapshot(folder / "manifest.json")
    declared = manifest.pop("dataset_fingerprint", None)
    if fingerprint(manifest) != declared or manifest.get("status") != "COMPLETED":
        raise ValueError("Incomplete or altered dataset manifest")
    if manifest.get("layout") not in (
        None,
        "ACCOUNT_PARTITIONED_STREAM_V1",
        "ACCOUNT_WINDOW_STREAM_V2",
        "ACCOUNT_CHECKPOINT_THREE_FORMATS_V3",
    ):
        raise ValueError("Unsupported dataset layout")
    binding = manifest["contract"]
    train_before = datetime.fromisoformat(binding["train_before"])
    test_from = datetime.fromisoformat(binding["test_from"])
    if binding != contract(
        source,
        policy,
        execution_scope,
        train_before,
        test_from,
        horizon_seconds=binding["horizon_seconds"],
    ):
        raise ValueError(
            "Policy/source/feature/code changed; regenerate alerts, features and model"
        )
    # 계약에서 원천 manifest를 다시 읽은 뒤 참조된 모든 파트의 해시를 계산합니다.
    # 데이터셋 산출물이 아닌, 관련 없는 기존 원천 파일은 보존합니다.
    for which, key in (("transactions", "ml_transactions"), ("labels", "ml_labels")):
        table_folder = (source.folder / source.metadata["datasets"][key]["path"]).resolve()
        for index, part in enumerate(source.parts(which)):
            parquet_path = source.part_path(which, index)
            parquet_digest = next(
                item["sha256"] for item in part["files"] if item["path"].endswith(".parquet")
            )
            for item in part["files"]:
                path = (table_folder / item["path"]).resolve()
                if path.parent != table_folder:
                    raise ValueError("ML input part changed or uses an unsafe path")
                actual_digest = parquet_digest if path == parquet_path else sha256(path)
                if actual_digest != item["sha256"]:
                    raise ValueError("ML input part changed or uses an unsafe path")
    validate_artifact_snapshot(folder, manifest, manifest_digest)
    if source_fingerprint(source) != binding["source_fingerprint"]:
        raise ValueError("Source manifests changed during integrity validation")
    if sha256(folder / "manifest.json") != manifest_digest:
        raise ValueError("Dataset manifest changed during integrity validation")
    return {
        "status": "INTEGRITY_VERIFIED",
        "dataset_fingerprint": declared,
        "manifest_sha256": manifest_digest,
        "execution_scope": execution_scope,
        "rows": manifest["rows"],
        "service_id_mapping_verified": False,
    }
