"""오프라인 경보 전처리와 기준·원천 재연산의 진입점입니다."""

from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .alert_dataset import OfflineAlert, chronological_split, offline_proxy_targets
from .alert_preparation import AlertBatch, bind_evidence
from .dataset_quality import DatasetQuality
from .offline_contract import DATASET_FILES as DATASET_FILES
from .offline_contract import FIT_BATCH_ROWS as FIT_BATCH_ROWS
from .offline_contract import SPLITS as SPLITS
from .offline_contract import alert_records as _alert_records
from .offline_contract import contract as contract
from .offline_contract import manifest_snapshot as _manifest_snapshot
from .offline_contract import regeneration_required as regeneration_required
from .offline_contract import source_fingerprint as source_fingerprint
from .offline_contract import validate_artifact_snapshot
from .offline_contract import validate_dataset_integrity as validate_dataset_integrity
from .offline_contract import validate_source as validate_source
from .offline_eligibility import DatasetNotReady, EligibilityCounts, ledger_row
from .offline_features import FEATURE_NAMES as FEATURE_NAMES
from .offline_features import OFFLINE_FEATURE_VERSION as OFFLINE_FEATURE_VERSION
from .offline_features import AlertPreprocessor as AlertPreprocessor
from .offline_features import partition_features as partition_features
from .offline_rules import R02Policy, fingerprint, generate_r02, validate_transactions
from .offline_serialization import (
    ALERT_SCHEMA,
    LEDGER_SCHEMA,
    TARGET_SCHEMA,
    finish_formats,
    read_record_array,
    record_table,
    verify_formats,
    write_three,
)
from .schema import epoch_us
from .storage import SourceInput, read_json, sha256, write_json


def load_rows(source: SourceInput, *, max_source_rows: int) -> list[dict]:
    count = source.metadata["input_rows"]
    validate_source(source)
    if type(max_source_rows) is not int or not 1 <= count <= max_source_rows:
        raise ValueError("Full source exceeds explicit in-memory row budget; no truncation")
    rows = []
    for i in range(len(source.parts("transactions"))):
        source.part_path("labels", i)
        table = pq.read_table(source.part_path("transactions", i))
        if table.num_rows != source.parts("transactions")[i]["rows"]:
            raise ValueError("Source part count changed")
        rows.extend(table.to_pylist())
    if len(rows) != count:
        raise ValueError("Incomplete full source")
    validate_transactions(rows, source.id, source.metadata["source_release"])
    return rows


def offline_features(source: SourceInput, rows: list[dict], alerts: list[OfflineAlert]) -> pa.Table:
    """전체 원천 이력과 전체 proxy 근거를 경로까지 구분합니다."""
    validate_transactions(rows, source.id, source.metadata["source_release"])
    return partition_features(
        source, rows, alerts, coverage_start=min(epoch_us(r["ts"]) for r in rows)
    )


def _derive(
    source: SourceInput,
    policy: R02Policy,
    scope: str,
    train_before: datetime,
    test_from: datetime,
    max_source_rows: int,
    horizon_seconds: int = 86400,
):
    rows = load_rows(source, max_source_rows=max_source_rows)
    alerts = generate_r02(
        rows,
        source_id=source.id,
        source_release=source.metadata["source_release"],
        policy=policy,
        execution_scope=scope,
    )
    if not alerts:
        raise ValueError("No R02 alerts; cannot fabricate a training dataset")
    evidence, labels = bind_evidence(source, alerts)
    targets = offline_proxy_targets(alerts, labels)
    ledger = [
        ledger_row(
            alert,
            target,
            train_before=train_before,
            test_from=test_from,
            horizon_seconds=horizon_seconds,
        )
        for alert, target in zip(alerts, targets, strict=True)
    ]
    eligible_alerts = [
        alert for alert, item in zip(alerts, ledger, strict=True) if item["supervised_eligible"]
    ]
    # 명시적인 적격 판정 후에 독립적인 공유 근거 검사를 유지합니다.
    splits = chronological_split(eligible_alerts, train_before=train_before, test_from=test_from)
    targets = [
        dict(target, split=item["original_split"])
        for target, item in zip(targets, ledger, strict=True)
    ]
    batch = AlertBatch(_alert_records(alerts), targets, evidence, splits, {})
    raw = offline_features(source, rows, eligible_alerts)
    return alerts, batch, raw, targets, ledger


def _eligible_targets(targets, ledger):
    return [
        target for target, item in zip(targets, ledger, strict=True) if item["supervised_eligible"]
    ]


def _indices(targets: list[dict], split: str) -> pa.Array:
    return pa.array([i for i, row in enumerate(targets) if row["split"] == split], type=pa.int64())


def _quality(source: SourceInput, raw: pa.Table, targets: list[dict], ledger: list[dict]) -> dict:
    quality = DatasetQuality(source.id)
    offset = 0
    for batch in raw.to_batches(max_chunksize=FIT_BATCH_ROWS):
        for feature in batch.to_pylist():
            quality.add(targets[offset], feature)
            offset += 1
    if offset != len(targets):
        raise ValueError("Quality X/target row count mismatch")
    eligibility = EligibilityCounts()
    for row in ledger:
        eligibility.add(row)
    return dict(quality.report(), eligibility=eligibility.report())


def build_reference_dataset(
    source: SourceInput,
    policy: R02Policy,
    output: Path,
    *,
    train_before: datetime,
    test_from: datetime,
    execution_scope: str,
    max_source_rows: int = 100_000,
    horizon_seconds: int = 86400,
) -> dict:
    binding = contract(
        source, policy, execution_scope, train_before, test_from, horizon_seconds=horizon_seconds
    )
    if output.exists():
        raise FileExistsError("Use a new output directory; existing artifacts are preserved")
    alerts, batch, raw, targets, ledger = _derive(
        source, policy, execution_scope, train_before, test_from, max_source_rows, horizon_seconds
    )
    eligible_targets = _eligible_targets(targets, ledger)
    output.mkdir(parents=True, exist_ok=False)
    write_three(record_table(_alert_records(alerts), ALERT_SCHEMA), output / "alerts.parquet")
    write_three(record_table(targets, TARGET_SCHEMA), output / "targets.parquet")
    write_three(record_table(ledger, LEDGER_SCHEMA), output / "ledger.parquet")
    write_json(output / "splits.json", batch.splits)
    write_three(batch.evidence, output / "evidence.parquet")
    write_three(raw, output / "raw.parquet")
    write_json(output / "quality.json", _quality(source, raw, eligible_targets, ledger))
    if any(not batch.splits[split] for split in SPLITS):
        write_json(
            output / "failure.json",
            {
                "status": "NOT_READY",
                "reason": "EMPTY_ELIGIBLE_SPLIT",
                "contract": binding,
                "complete_candidates_preserved": True,
                "training_executed": False,
                "files": {p.name: sha256(p) for p in output.iterdir() if p.is_file()},
            },
        )
        raise DatasetNotReady("NOT_READY: all three eligible splits must be nonempty")
    processor = AlertPreprocessor()
    for fit_batch in (
        raw.take(_indices(eligible_targets, "train"))
        .combine_chunks()
        .to_batches(max_chunksize=FIT_BATCH_ROWS)
    ):
        processor.partial_fit(pa.Table.from_batches([fit_batch]), split="train")
    try:
        processor.freeze()
    except ValueError as exc:
        write_json(
            output / "failure.json",
            {
                "status": "NOT_READY",
                "reason": "ELIGIBLE_TRAIN_CANNOT_FIT",
                "contract": binding,
                "complete_candidates_preserved": True,
                "training_executed": False,
                "files": {p.name: sha256(p) for p in output.iterdir() if p.is_file()},
            },
        )
        raise DatasetNotReady("NOT_READY: eligible train cannot fit preprocessing") from exc
    processed = processor.transform(raw)
    processor.save(output / "preprocessing.json")
    for split in SPLITS:
        indices = _indices(eligible_targets, split)
        write_three(processed.take(indices), output / f"X_{split}.parquet")
        selected = [row for row in eligible_targets if row["split"] == split]
        write_three(
            pa.table(
                {
                    "training_alert_uid": [r["training_alert_uid"] for r in selected],
                    "y_alert_proxy": pa.array(
                        [r["y_alert_proxy"] for r in selected], type=pa.int8()
                    ),
                }
            ),
            output / f"y_{split}.parquet",
        )
    finish_formats(output)
    manifest = {
        "status": "COMPLETED",
        "contract": binding,
        "prediction_unit": "OFFLINE_ALERT",
        "service_id_mapping_verified": False,
        "model_training_executed": False,
        "rows": {split: len(batch.splits[split]) for split in SPLITS},
        "feature_columns": processed.column_names,
        "files": {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()},
    }
    manifest["dataset_fingerprint"] = fingerprint(manifest)
    write_json(output / "manifest.json", manifest)
    return verify_dataset(
        output, source, policy, execution_scope=execution_scope, max_source_rows=max_source_rows
    )


def verify_dataset(
    folder: Path,
    source: SourceInput,
    policy: R02Policy,
    *,
    execution_scope: str,
    max_source_rows: int | None = None,
) -> dict:
    """불변 원천을 다시 읽고 재계산해 비교하며, 쓰기나 재적합 산출물 생성은 하지 않습니다."""
    manifest, original_digest = _manifest_snapshot(folder / "manifest.json")
    if manifest.get("layout") in (
        "ACCOUNT_PARTITIONED_STREAM_V1",
        "ACCOUNT_WINDOW_STREAM_V2",
        "ACCOUNT_CHECKPOINT_THREE_FORMATS_V3",
    ):
        if max_source_rows is not None and (
            type(max_source_rows) is not int
            or not 1 <= source.metadata["input_rows"] <= max_source_rows
        ):
            raise ValueError("Source exceeds explicit verification row budget")
        from .offline_streaming import verify_streaming_dataset

        return verify_streaming_dataset(folder, source, policy, execution_scope=execution_scope)
    integrity = validate_dataset_integrity(folder, source, policy, execution_scope=execution_scope)
    if integrity["manifest_sha256"] != original_digest:
        raise ValueError("Dataset manifest changed before source replay")
    binding = manifest["contract"]
    train_before = datetime.fromisoformat(binding["train_before"])
    test_from = datetime.fromisoformat(binding["test_from"])
    alerts, batch, raw, targets, ledger = _derive(
        source,
        policy,
        execution_scope,
        train_before,
        test_from,
        100_000 if max_source_rows is None else max_source_rows,
        binding["horizon_seconds"],
    )
    eligible_targets = _eligible_targets(targets, ledger)
    # JSON은 tuple 경로를 배열로 바꾸므로 정규화한 지문을 비교합니다.
    if fingerprint(read_record_array(folder / "alerts.json")) != fingerprint(
        _alert_records(alerts)
    ):
        raise ValueError("Stored alert identity/evidence differs from source generation")
    if read_record_array(folder / "targets.json") != targets:
        raise ValueError("Stored proxy targets differ from full source evidence")
    if read_record_array(folder / "ledger.json") != ledger:
        raise ValueError("Stored eligibility ledger differs from full source candidates")
    if read_json(folder / "splits.json") != batch.splits:
        raise ValueError("Stored split ownership differs from chronological split")
    if not pq.read_table(folder / "evidence.parquet").equals(batch.evidence):
        raise ValueError("Stored full evidence changed")
    if not pq.read_table(folder / "raw.parquet").equals(raw):
        raise ValueError("Stored features differ from past-only source computation")
    if read_json(folder / "quality.json") != _quality(source, raw, eligible_targets, ledger):
        raise ValueError("Stored quality report differs from source-derived features and targets")
    processor = AlertPreprocessor.load(folder / "preprocessing.json")
    independent = AlertPreprocessor()
    for fit_batch in (
        raw.take(_indices(eligible_targets, "train"))
        .combine_chunks()
        .to_batches(max_chunksize=FIT_BATCH_ROWS)
    ):
        independent.partial_fit(pa.Table.from_batches([fit_batch]), split="train")
    if independent.freeze() != processor.state:
        raise ValueError("Preprocessing is not reproducible from train only")
    transformed = processor.transform(raw)
    if transformed.column_names != manifest["feature_columns"]:
        raise ValueError("Feature column order changed")
    for split in SPLITS:
        expected = [r for r in eligible_targets if r["split"] == split]
        x = pq.read_table(folder / f"X_{split}.parquet")
        y = pq.read_table(folder / f"y_{split}.parquet")
        if (
            not x.equals(transformed.take(_indices(eligible_targets, split)))
            or y.column("training_alert_uid").to_pylist() != batch.splits[split]
            or y.column("y_alert_proxy").to_pylist() != [r["y_alert_proxy"] for r in expected]
            or len(x) != manifest["rows"][split]
        ):
            raise ValueError("Split X/y alignment, values or count changed")
    if source_fingerprint(source) != binding["source_fingerprint"]:
        raise ValueError("Source manifests changed during source replay")
    if sha256(folder / "manifest.json") != integrity["manifest_sha256"]:
        raise ValueError("Dataset manifest changed during source replay")
    validate_artifact_snapshot(folder, manifest, integrity["manifest_sha256"])
    verify_formats(folder)
    return dict(integrity, status="VERIFIED")


def build_dataset(
    source: SourceInput,
    policy: R02Policy,
    output: Path,
    *,
    train_before: datetime,
    test_from: datetime,
    execution_scope: str,
    max_source_rows: int | None = None,
    partitions: int = 32,
    read_batch_rows: int = 4096,
    max_partition_rows: int = 100_000,
    max_partition_bytes: int = 64 * 1024**2,
    decode_batch_rows: int | None = None,
    sort_run_rows: int = 65536,
    sort_run_bytes: int | None = None,
    write_batch_rows: int = 8192,
    merge_fan_in: int = 8,
    horizon_seconds: int = 86400,
    checkpoint_owner_rows: int = 100_000,
    checkpoint_dir: Path | None = None,
    resume: bool = False,
    checkpoint_context: dict | None = None,
) -> dict:
    """기본은 디스크 분할이며, 소규모 전체 메모리 경로는 테스트 비교 기준입니다."""
    from .offline_streaming import build_streaming_dataset

    return build_streaming_dataset(
        source,
        policy,
        output,
        train_before=train_before,
        test_from=test_from,
        execution_scope=execution_scope,
        max_source_rows=max_source_rows,
        partitions=partitions,
        read_batch_rows=read_batch_rows,
        max_partition_rows=max_partition_rows,
        max_partition_bytes=max_partition_bytes,
        decode_batch_rows=decode_batch_rows,
        sort_run_rows=sort_run_rows,
        sort_run_bytes=sort_run_bytes,
        write_batch_rows=write_batch_rows,
        merge_fan_in=merge_fan_in,
        horizon_seconds=horizon_seconds,
        checkpoint_owner_rows=checkpoint_owner_rows,
        checkpoint_dir=checkpoint_dir,
        resume=resume,
        checkpoint_context=checkpoint_context,
    )
