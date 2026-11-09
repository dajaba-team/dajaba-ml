"""HI 원본 대신 합성 원천으로 외부 임시 저장·구간 처리의 회귀를 검사합니다."""

import json
import math
from dataclasses import replace
from datetime import datetime, timedelta
from decimal import Decimal

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from offline_test_data import assert_same_artifacts, rewrite_fixture, source_values

from aml_ml import offline_account_stream as account_stream
from aml_ml import offline_streaming as streaming
from aml_ml.alert_dataset import offline_proxy_targets
from aml_ml.offline_pipeline import build_dataset, build_reference_dataset, offline_features
from aml_ml.offline_rules import generate_r02
from aml_ml.schema import epoch_us
from aml_ml.storage import read_json


def _emissions(source, policy, rows, labels, *, max_rows=256, max_bytes=8 * 1024**2):
    owned = [
        dict(row, _bucket=0, _owner=owner, _offline_label=label)
        for row, label in zip(rows, labels, strict=True)
        if not row["is_self_transfer"] and row["from_account_key"] != row["to_account_key"]
        for owner in sorted({row["from_account_key"], row["to_account_key"]})
    ]
    stats = dict(
        input_rows=len(rows),
        owned_rows_processed=0,
        max_active_window_rows=0,
        max_active_window_workset_bytes=0,
    )
    result = list(
        account_stream.account_alerts(
            iter(sorted(owned, key=streaming._owner_key)),
            source,
            policy,
            coverage=min(epoch_us(row["ts"]) for row in rows),
            train_before=datetime(2022, 2, 1),
            test_from=datetime(2022, 3, 1),
            settings={"max_partition_rows": max_rows, "max_partition_bytes": max_bytes},
            row_size=streaming._row_workset_bytes,
            stats=stats,
        )
    )
    result.sort(key=lambda item: (item[0].reference_ts, item[0].training_alert_uid))
    assert stats["owned_rows_processed"] == len(owned)
    return result, stats


def _compare(source, policy, rows, labels, **limits):
    expected = generate_r02(
        rows,
        source_id=source.id,
        source_release=source.metadata["source_release"],
        policy=policy,
        execution_scope="FIXTURE_ONLY",
    )
    actual, stats = _emissions(source, policy, rows, labels, **limits)
    assert [item[0] for item in actual] == expected
    if expected:
        assert [item[1] for item in actual] == offline_features(source, rows, expected).to_pylist()
        truth = dict(zip((row["transaction_uid"] for row in rows), labels, strict=True))
        assert [{k: v for k, v in item[2].items() if k != "split"} for item in actual] == (
            offline_proxy_targets(expected, truth)
        )
    evidence_uids = [row["transaction_uid"] for item in actual for row in item[3]]
    assert len(evidence_uids) == len(set(evidence_uids))
    assert set(evidence_uids) == {uid for alert in expected for uid in alert.transaction_uids}
    return actual, stats


@pytest.mark.parametrize("peers", [9, 10, 11])
def test_streamed_n10_same_timestamp_full_evidence_and_null_proxy(source, policy, peers):
    original, _ = source_values(source)
    rows = [
        dict(original[0], ts=datetime(2022, 1, 1), from_account_key=f"peer-{i}")
        for i in range(peers)
    ]
    labels = [1, None] + [0] * (peers - 2)
    rows = rewrite_fixture(source, rows, labels)
    actual, _ = _compare(source, replace(policy, min_counterparties=10), rows, labels)
    assert len(actual) == int(peers >= 10)
    if actual:
        assert len(actual[0][0].transaction_uids) == peers
        assert actual[0][2]["y_alert_proxy"] is None


@pytest.mark.parametrize("source_kind", ["ibm", "saml"])
@pytest.mark.parametrize("currency", ["US Dollar", "Euro"])
def test_clock_expiry_retrigger_currency_direction_and_self_exclusion(
    source, policy, source_kind, currency
):
    original, _ = source_values(source)
    events = [
        (datetime(2022, 1, 1, 0, 0, 5), "A"),
        (datetime(2022, 1, 1, 12), "B"),
        (datetime(2022, 1, 2, 0, 1), "C"),
        (datetime(2022, 1, 2, 0, 2), "D"),
    ]
    rows: list[dict[str, datetime | str | Decimal | bool | int]] = [
        dict(original[0], ts=stamp, from_account_key=peer) for stamp, peer in events
    ]
    rows[1]["currency_received_raw"] = currency
    # 직접 출금 이력은 수취 금액이 아닌 지급 금액을 사용해야 합니다.
    rows.append(
        dict(
            original[0],
            ts=datetime(2022, 1, 1, 11),
            from_account_key="P",
            to_account_key="Z",
            amount_paid=Decimal("12"),
            amount_received=Decimal("99"),
        )
    )
    rows.append(
        dict(
            original[0],
            ts=datetime(2022, 1, 1, 10),
            from_account_key="self-flagged",
            is_self_transfer=True,
        )
    )
    rows.append(dict(original[0], ts=datetime(2022, 1, 1, 10), from_account_key="P"))
    labels = [0] * len(rows)
    rows = rewrite_fixture(source, rows, labels, part_rows=2)
    actual, _ = _compare(replace(source, kind=source_kind), policy, rows, labels)
    assert [item[0].reference_ts for item in actual] == [
        datetime(2022, 1, 1, 12, 1),
        datetime(2022, 1, 2, 0, 2),
    ]
    assert set(actual[0][0].transaction_uids) & set(actual[1][0].transaction_uids)
    if source_kind == "saml" or currency == "Euro":
        assert actual[0][1]["account_in_amount_log1p_24h"] is None
        assert actual[0][1]["account_out_amount_log1p_24h"] is None
    else:
        assert actual[0][1]["account_in_amount_log1p_24h"] == math.log1p(20)
        assert actual[0][1]["account_out_amount_log1p_24h"] == math.log1p(12)


@pytest.mark.parametrize("rows_in_minute", [256, 257])
def test_live_window_exact_row_limit_and_overflow_diagnostic(source, policy, rows_in_minute):
    original, _ = source_values(source)
    rows = [
        dict(original[0], ts=datetime(2022, 1, 1), from_account_key=f"peer-{i}")
        for i in range(rows_in_minute)
    ]
    labels = [0] * len(rows)
    rows = rewrite_fixture(source, rows, labels)
    if rows_in_minute == 256:
        actual, stats = _compare(source, policy, rows, labels)
        assert len(actual) == 1 and stats["max_active_window_rows"] == 256
    else:
        with pytest.raises(ValueError, match="active_window_rows") as error:
            _emissions(source, policy, rows, labels)
        details = json.loads(str(error.value).split(": ", 1)[1])
        assert (details["current"], details["projected"], details["limit"]) == (256, 257, 256)
        assert details["bucket"] == 0 and details["input_rows"] == 257
        assert "peer-" not in str(error.value)


def test_live_window_byte_limit_is_independent_of_lifetime_row_limit(source, policy):
    rows, labels = source_values(source)
    with pytest.raises(ValueError, match="active_window_workset_bytes") as error:
        _emissions(source, policy, rows, labels, max_bytes=1)
    details = json.loads(str(error.value).split(": ", 1)[1])
    assert details["current"] == 0 and details["projected"] > details["limit"] == 1


def test_single_alert_payload_rejected_before_alert_allocation(source, policy, monkeypatch):
    rows, labels = source_values(source)
    # 현재 행들이 한도 안에 있어도 이스케이프된 JSON 메타데이터는 경보 예산을 넘을 수 있습니다.
    policy = replace(policy, rule_version="fixture-" + "한" * 6000)
    policy.require("FIXTURE_ONLY")

    def forbidden(*args, **kwargs):
        raise AssertionError("Alert allocated before payload preflight")

    monkeypatch.setattr(account_stream, "OfflineAlert", forbidden)
    with pytest.raises(ValueError, match="single_alert_payload_bytes") as error:
        _emissions(source, policy, rows, labels, max_bytes=32768)
    details = json.loads(str(error.value).split(": ", 1)[1])
    assert details["current"] == 0 and details["projected"] > details["limit"] == 32768
    assert "fixture-" not in str(error.value)


@pytest.mark.parametrize("partitions,batch_rows", [(1, 7), (7, 31)])
def test_long_hot_accounts_and_colliding_owners_spill_without_truncation(
    source, policy, tmp_path, partitions, batch_rows
):
    original, _ = source_values(source)
    rows, labels = [], []
    for episode in range(35):
        stamp = datetime(2022, 1, 1, 12) + timedelta(days=episode * 2)
        for account in ("P", "Q"):
            for peer in range(10):
                amount = Decimal("10.123456") if account == "P" else Decimal("1.000001")
                rows.append(
                    dict(
                        original[0],
                        ts=stamp,
                        from_account_key=f"peer-{peer}",
                        to_account_key=account,
                        amount_paid=amount,
                        amount_received=amount,
                    )
                )
                labels.append(int(account == "P" and peer == 0))
    # 원천 순번은 연속으로 유지하면서 시각은 의도적으로 뒤집습니다.
    # 기본 fixture는 scale=0으로 추론되므로 이 테스트에서 scale=6 스키마를 명시합니다.
    schema = pa.schema(
        [
            pa.field(field.name, pa.decimal128(20, 6))
            if field.name in {"amount_paid", "amount_received"}
            else field
            for field in pq.read_schema(source.part_path("transactions", 0))
        ]
    )
    rows = rewrite_fixture(
        source,
        list(reversed(rows)),
        list(reversed(labels)),
        part_rows=29,
        tx_schema=schema,
    )
    policy = replace(policy, min_counterparties=10)
    train_before = datetime(2022, 1, 21)
    test_from = datetime(2022, 2, 10)
    reference, output = tmp_path / "reference", tmp_path / "stream"
    build_reference_dataset(
        source,
        policy,
        reference,
        train_before=train_before,
        test_from=test_from,
        execution_scope="FIXTURE_ONLY",
    )
    result = build_dataset(
        source,
        policy,
        output,
        partitions=partitions,
        read_batch_rows=batch_rows,
        sort_run_rows=batch_rows,
        max_partition_rows=256,
        max_partition_bytes=1024**2,
        train_before=train_before,
        test_from=test_from,
        execution_scope="FIXTURE_ONLY",
    )
    assert result["status"] == "VERIFIED"
    assert_same_artifacts(reference, output)
    manifest = read_json(output / "manifest.json")
    stats = manifest["work_stats"]
    assert manifest["layout"] == "ACCOUNT_CHECKPOINT_THREE_FORMATS_V3"
    assert stats["input_rows"] == 700
    assert stats["owned_rows"] == stats["owned_rows_processed"] == 1400
    assert stats["max_active_window_rows"] == 10
    assert manifest["execution_settings"]["sort_run_rows"] == batch_rows
    if partitions == 1:
        assert stats["max_bucket_total_rows"] == 1400 > 256
    assert sum(manifest["rows"].values()) == 66
    assert len(read_json(output / "alerts.json")) == 70
    ledger = read_json(output / "ledger.json")
    assert sum(row["supervised_eligible"] for row in ledger) == 66
    assert sum(not row["time_eligible"] for row in ledger) == 4
    stored = pq.read_table(output / "evidence.parquet")
    assert stored.schema == schema
    assert stored.column("amount_paid").to_pylist() == [row["amount_paid"] for row in rows]
    assert stored.column("amount_received").to_pylist() == [row["amount_received"] for row in rows]
    assert not list(tmp_path.glob(".offline-work-*"))


def test_online_spill_compaction_limits_open_readers_and_preserves_all_rows(tmp_path, monkeypatch):
    schema = pa.schema([("value", pa.int64())])
    sorter = streaming.SortedRuns(
        tmp_path,
        schema,
        lambda r: r["value"],
        prefix="test",
        max_rows=1,
        max_bytes=65536,
    )
    original = streaming._rows
    state = {"active": 0, "peak": 0}

    def tracked(*args, **kwargs):
        state["active"] += 1
        state["peak"] = max(state["peak"], state["active"])
        try:
            yield from original(*args, **kwargs)
        finally:
            state["active"] -= 1

    monkeypatch.setattr(streaming, "_rows", tracked)
    for i in reversed(range(130)):
        sorter.add({"value": i}, {"stage": "test"})
    paths = sorter.paths()
    assert all(len(level) < streaming.MERGE_FAN_IN for level in sorter.levels)
    assert sorter.peak_rows == 1 and sorter.peak_bytes <= 65536
    merged = streaming._bounded_merge(
        paths,
        lambda r: r["value"],
        work=tmp_path,
        schema=schema,
        prefix="final",
        max_bytes=65536,
    )
    try:
        assert [row["value"] for row in merged] == list(range(130))
    finally:
        merged.close()
    assert state["active"] == 0 and state["peak"] <= streaming.MERGE_FAN_IN


def test_account_failure_cleans_spill_and_preserves_unrelated_output(
    source, policy, tmp_path, monkeypatch
):
    untouched = tmp_path / "existing-result.txt"
    untouched.write_text("preserve", encoding="utf-8")

    def fail(*args, **kwargs):
        raise ValueError("injected account failure")

    monkeypatch.setattr(account_stream, "_features", fail)
    output = tmp_path / "blocked"
    with pytest.raises(ValueError, match="injected account failure"):
        build_dataset(
            source,
            policy,
            output,
            train_before=datetime(2022, 1, 2),
            test_from=datetime(2022, 1, 4),
            execution_scope="FIXTURE_ONLY",
            read_batch_rows=1,
        )
    assert untouched.read_text(encoding="utf-8") == "preserve"
    assert not output.exists()
    assert not list(tmp_path.glob(".offline-work-*"))
