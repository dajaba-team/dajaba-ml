"""실제 소규모 Parquet로 정확한 UID 연결과 전체 경로 proxy 라벨을 검사합니다."""

import hashlib
import json
from dataclasses import replace
from datetime import datetime, timedelta
from decimal import Decimal
from importlib.resources import files

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from aml_ml.alert_dataset import OfflineAlert
from aml_ml.alert_preparation import prepare_alert_batch
from aml_ml.input_inspection import inspect_inputs
from aml_ml.offline_pipeline import offline_features
from aml_ml.storage import SourceInput, load_index, sha256


@pytest.fixture
def assembly(tmp_path):
    folder = tmp_path / "run" / "ibm_hi_small" / "attempt-01"
    release = "ibm_hi_small:" + "a" * 64

    def uid(i):
        return release + ":" + str(i)

    schema = pa.schema(
        [
            ("transaction_uid", pa.string()),
            ("source_row_id", pa.int64()),
            ("ts", pa.timestamp("us")),
            ("from_account_key", pa.string()),
            ("to_account_key", pa.string()),
            ("is_self_transfer", pa.bool_()),
            ("payment_format", pa.string()),
            ("currency_paid_bucket", pa.string()),
            ("currency_received_bucket", pa.string()),
            ("amount_paid", pa.decimal128(38, 12)),
            ("amount_received", pa.decimal128(38, 12)),
            ("currency_paid_raw", pa.string()),
            ("currency_received_raw", pa.string()),
        ]
    )
    # 경로 중간 거래만 양성입니다. 두 번째 그룹은 validation에 속합니다.
    rows = [
        dict(
            transaction_uid=uid(i),
            source_row_id=i,
            ts=datetime(2022, 1, 1 if i < 3 else 3) + timedelta(minutes=i),
            from_account_key=["S", "B", "C"][i % 3],
            to_account_key=["B", "C", "S"][i % 3],
            is_self_transfer=False,
            payment_format="Wire",
            currency_paid_bucket="CRYPTO:BTC",
            currency_received_bucket="CRYPTO:BTC",
            amount_paid=Decimal("0.000001"),
            amount_received=Decimal("0.000001"),
            currency_paid_raw="Bitcoin",
            currency_received_raw="Bitcoin",
        )
        for i in range(6)
    ]
    tx = pa.Table.from_pylist(rows, schema=schema)
    labels = pa.table(
        {
            "transaction_uid": [uid(i) for i in range(6)],
            "source_row_id": pa.array(range(6), type=pa.int64()),
            "source_positive": pa.array([0, 1, 0, 0, 0, 0], type=pa.int8()),
        }
    )
    tables = {}
    datasets = {}
    for key, table in [("ml_transactions", tx), ("ml_labels", labels)]:
        path = folder / key
        path.mkdir(parents=True)
        with pa.OSFile(str(path / "schema.arrow"), "wb") as stream:
            stream.write(table.schema.serialize())
        parts = []
        for i in range(3):
            p = path / f"part-{i}.parquet"
            pq.write_table(table.slice(i * 2, 2), p)
            parts.append(dict(rows=2, files=[dict(path=p.name, sha256=sha256(p))]))
        manifest = dict(
            status="VERIFIED",
            source_id="ibm_hi_small",
            source_kind="ibm",
            source_release=release,
            purpose="ml",
            scope="FULL_SOURCE",
            rows=6,
            parts=parts,
            aliases_sha256=hashlib.sha256(
                files("aml_ml").joinpath("currencies.json").read_bytes()
            ).hexdigest(),
        )
        (path / "manifest.json").write_text(json.dumps(manifest))
        tables[key] = manifest
        datasets[key] = dict(path=key, rows=6)
    metadata = dict(
        source_id="ibm_hi_small",
        status="VERIFIED",
        source_release=release,
        ml_output_rows=6,
        input_rows=6,
        datasets=datasets,
    )
    (folder / "source_manifest.json").write_text(json.dumps(metadata))
    run = folder.parents[1]
    (run / "run.json").write_text(
        json.dumps(
            dict(
                contract_version="dajaba-assembly-v2",
                status="COMPLETED",
                sources={"ibm_hi_small": dict(status="VERIFIED", folder="ibm_hi_small/attempt-01")},
            )
        )
    )
    index = tmp_path / "index.json"
    index.write_text(
        json.dumps(
            dict(
                ml_runs={"ibm_hi_small": "run"},
                run_manifest_sha256={"ibm_hi_small": sha256(run / "run.json")},
            )
        )
    )
    source = SourceInput(
        "ibm_hi_small", "ibm", folder, metadata, tables["ml_transactions"], tables["ml_labels"]
    )

    def alert(start=0, **changes):
        return OfflineAlert(
            source_id=source.id,
            rule_id="R04_CYCLE",
            rule_version="fixture-only",
            rule_config_hash="b" * 64,
            account_key="S",
            reference_ts=datetime(2022, 1, 2 if start == 0 else 4),
            window_start=datetime(2022, 1, 1 if start == 0 else 3),
            window_end=datetime(2022, 1, 2 if start == 0 else 4),
            transaction_uids=tuple(uid(i) for i in range(start, start + 3)),
            paths=(tuple(uid(i) for i in range(start, start + 3)),),
            **changes,
        )

    return source, index, alert


def prepare(source, alerts):
    return prepare_alert_batch(
        source, alerts, train_before=datetime(2022, 1, 3), test_from=datetime(2022, 1, 5)
    )


def test_indirect_path_changes_target_evidence_but_not_direct_account_features(assembly):
    source, _, make_alert = assembly
    alert = make_alert()
    rows = []
    for index in range(len(source.parts("transactions"))):
        rows.extend(pq.read_table(source.part_path("transactions", index)).to_pylist())
    batch = prepare(source, [alert])
    assert batch.targets[0]["y_alert_proxy"] == 1  # 양성인 중간 B->C 거래도 전체 근거에 포함됩니다.
    assert len(batch.evidence) == 3
    before = offline_features(source, rows, [alert])
    changed = [dict(row) for row in rows]
    changed[1]["amount_paid"] = changed[1]["amount_received"] = Decimal("999999")
    changed[1]["currency_paid_raw"] = changed[1]["currency_received_raw"] = "Euro"
    assert offline_features(source, changed, [alert]).equals(before)
    # 직접 S->B 거래는 S 계좌에 속하며 해당 계좌의 피처 문맥에 반영해야 합니다.
    changed[0]["amount_paid"] = Decimal("0.000002")
    assert not offline_features(source, changed, [alert]).equals(before)


def test_full_evidence_uid_join_preserves_indirect_precision_and_separates_labels(assembly):
    source, _, candidate = assembly
    result = prepare(source, [candidate(), candidate(3)])
    assert [r["y_alert_proxy"] for r in result.targets] == [1, 0]
    assert result.evidence.column("source_row_id").to_pylist() == list(range(6))
    assert result.evidence.column("amount_paid")[0].as_py() == Decimal("0.000001")
    assert result.evidence.column("currency_paid_raw")[0].as_py() == "Bitcoin"
    assert "source_positive" not in result.evidence.schema.names
    assert all("y_alert_proxy" not in r for r in result.alerts)
    assert {k: len(v) for k, v in result.splits.items()} == dict(train=1, validation=1, test=0)
    assert result.metadata["features_computed"] is False


def test_wrong_release_and_out_of_range_uids_are_not_guessed(assembly):
    source, _, candidate = assembly
    a = candidate()
    altered = tuple(uid.replace("a" * 64, "c" * 64) for uid in a.transaction_uids)
    with pytest.raises(ValueError, match="source release"):
        prepare(source, [replace(a, transaction_uids=altered, paths=(altered,))])
    altered = tuple(source.metadata["source_release"] + ":" + str(i) for i in [6, 7, 8])
    with pytest.raises(ValueError, match="outside"):
        prepare(source, [replace(a, transaction_uids=altered, paths=(altered,))])


def test_broken_path_is_rejected_without_reordering(assembly):
    source, _, candidate = assembly
    a = candidate()
    with pytest.raises(ValueError, match="order or account"):
        prepare(
            source,
            [
                replace(
                    a,
                    paths=((a.transaction_uids[0], a.transaction_uids[2], a.transaction_uids[1]),),
                )
            ],
        )


def test_shared_evidence_across_splits_and_future_labels_are_rejected(assembly):
    source, _, candidate = assembly
    a = candidate()
    with pytest.raises(ValueError, match="overlaps splits"):
        prepare(source, [a, replace(a, reference_ts=datetime(2022, 1, 4))])
    a = replace(candidate(3), reference_ts=datetime(2022, 1, 2))
    with pytest.raises(ValueError, match="later chronological split"):
        prepare(source, [a])


def test_metadata_inspection_pins_run_and_rejects_large(assembly):
    _, index, _ = assembly
    result = inspect_inputs(index)
    assert result["status"] == "INPUT_METADATA_AND_FIRST_SCHEMA_OK"
    assert result["full_data_verified"] is False
    data = json.loads(index.read_text())
    data["run_manifest_sha256"]["ibm_hi_small"] = "0" * 64
    index.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="manifest changed"):
        load_index(index)
    index.write_text(json.dumps(dict(ml_runs={"ibm_hi_large": "run"})))
    with pytest.raises(ValueError, match="no Large"):
        load_index(index)


def test_changed_parquet_blocks_binding(assembly):
    source, _, candidate = assembly
    p = source.folder / "ml_labels/part-0.parquet"
    with p.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="input part changed"):
        prepare(source, [candidate()])


def test_unconfirmed_label_remains_null_even_with_positive_middle_hop(assembly):
    source, _, candidate = assembly
    path = source.folder / "ml_labels/part-0.parquet"
    table = pq.read_table(path)
    table = table.set_column(2, "source_positive", pa.array([None, 1], type=pa.int8()))
    pq.write_table(table, path)
    source.labels["parts"][0]["files"][0]["sha256"] = sha256(path)
    result = prepare(source, [candidate()])
    assert result.targets[0]["y_alert_proxy"] is None
    assert result.targets[0]["missing_label_count"] == 1
