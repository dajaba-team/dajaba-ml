"""합성 오프라인 테스트 데이터의 명시적 생성 함수와 동일성 검사를 제공합니다."""

from datetime import datetime

import pyarrow as pa
import pyarrow.parquet as pq

from aml_ml.offline_pipeline import build_dataset, build_reference_dataset
from aml_ml.storage import read_json, sha256, write_json


def synthetic_support(model_kind="NUMPY_LOGISTIC_PROXY_BASELINE"):
    """이 작은 fixture에만 명시한 표본 기준이며 팀의 기본값이 아닙니다."""
    return {
        "version": "offline-evaluation-support-v1",
        "model_kind": model_kind,
        "decision_reference": "SYNTHETIC_TEST_TWO_ALERTS_NOT_A_TEAM_MINIMUM",
        "metric_set": "PR_AUC_PRECISION_RECALL_F1_TOP_K_V1",
        "minimums": {
            split: {"rows": 2, "positive": 1, "negative": 1}
            for split in ("train", "validation", "test")
        },
    }


def _build_review_dataset(source, policy, folder):
    build_reference_dataset(
        source,
        policy,
        folder,
        train_before=datetime(2022, 1, 2),
        test_from=datetime(2022, 1, 4),
        execution_scope="FIXTURE_ONLY",
    )


def rewrite_fixture(source, rows, labels, part_rows=7, *, assign_ordinals=True, tx_schema=None):
    """이 테스트에서 새로 만든 합성 SourceInput만 수정합니다."""
    if tx_schema is None:
        tx_schema = pq.read_schema(source.part_path("transactions", 0))
    label_schema = pq.read_schema(source.part_path("labels", 0))
    release = source.metadata["source_release"]
    if assign_ordinals:
        rows = [
            dict(row, source_row_id=i, transaction_uid=f"{release}:{i}")
            for i, row in enumerate(rows)
        ]
    truth = [
        dict(
            transaction_uid=row["transaction_uid"],
            source_row_id=row["source_row_id"],
            source_positive=value,
        )
        for row, value in zip(rows, labels, strict=True)
    ]
    for which, key, table in (
        (
            "transactions",
            "ml_transactions",
            pa.Table.from_pylist(rows, schema=tx_schema),
        ),
        ("labels", "ml_labels", pa.Table.from_pylist(truth, schema=label_schema)),
    ):
        manifest = source.transactions if which == "transactions" else source.labels
        folder = source.folder / source.metadata["datasets"][key]["path"]
        parts = []
        for start in range(0, len(rows), part_rows):
            selected = table.slice(start, part_rows)
            path = folder / f"stream-fixture-{start}.parquet"
            pq.write_table(selected, path, row_group_size=min(7, len(selected)))
            parts.append(
                {
                    "rows": len(selected),
                    "files": [{"path": path.name, "sha256": sha256(path)}],
                }
            )
        manifest["parts"], manifest["rows"] = parts, len(rows)
        source.metadata["datasets"][key]["rows"] = len(rows)
        write_json(folder / "manifest.json", manifest)
    source.metadata["input_rows"] = source.metadata["ml_output_rows"] = len(rows)
    write_json(source.folder / "source_manifest.json", source.metadata)
    return rows


def source_values(source):
    tx, labels = [], []
    for i in range(len(source.parts("transactions"))):
        tx.extend(pq.read_table(source.part_path("transactions", i)).to_pylist())
        labels.extend(
            pq.read_table(source.part_path("labels", i)).column("source_positive").to_pylist()
        )
    return tx, labels


def prepare(source, policy, path, **settings):
    return build_dataset(
        source,
        policy,
        path,
        train_before=datetime(2022, 1, 2),
        test_from=datetime(2022, 1, 4),
        execution_scope="FIXTURE_ONLY",
        **settings,
    )


def assert_same_artifacts(reference, streamed):
    left, right = (
        read_json(reference / "manifest.json"),
        read_json(streamed / "manifest.json"),
    )
    assert left["rows"] == right["rows"]
    assert left["feature_columns"] == right["feature_columns"]
    for name in (
        "alerts.json",
        "targets.json",
        "ledger.json",
        "splits.json",
        "preprocessing.json",
        "quality.json",
    ):
        assert read_json(reference / name) == read_json(streamed / name), name
    for name in ("raw.parquet", "evidence.parquet"):
        assert pq.read_table(reference / name).equals(pq.read_table(streamed / name)), name
    for split in ("train", "validation", "test"):
        for prefix in ("X", "y"):
            name = f"{prefix}_{split}.parquet"
            assert pq.read_table(reference / name).equals(pq.read_table(streamed / name)), name
