"""합성 임계값·ID·거래만으로 확정된 R02 의미를 검사합니다."""

from datetime import datetime

import pyarrow.parquet as pq
import pytest
from offline_test_data import assert_same_artifacts, prepare, rewrite_fixture, source_values

from aml_ml.offline_pipeline import build_reference_dataset, load_rows
from aml_ml.offline_rules import generate_r02
from aml_ml.storage import read_json


def _generate(rows, source, policy):
    return generate_r02(
        rows,
        source_id=source.id,
        source_release=source.metadata["source_release"],
        policy=policy,
        execution_scope="FIXTURE_ONLY",
    )


def test_accounts_not_entities_or_currencies_and_all_incoming_evidence(source, policy):
    rows = load_rows(source, max_source_rows=100)
    # 입금 거래 셋이며, 같은 법인의 서로 다른 계좌 두 개입니다.
    # 같은 계좌가 다른 통화로 다시 지급해도 상대방은 동일합니다.
    rows[0]["from_entity_id"] = rows[1]["from_entity_id"] = "fixture-same-entity"
    rows[2]["from_account_key"] = rows[0]["from_account_key"]
    rows[2]["from_entity_id"] = "fixture-same-entity"
    rows[2]["currency_paid_raw"] = rows[2]["currency_received_raw"] = "Euro"
    rows[2]["currency_paid_bucket"] = rows[2]["currency_received_bucket"] = "EUR"
    for same_key, flagged in ((True, False), (False, True)):
        ordinal = len(rows)
        rows.append(
            dict(
                rows[0],
                source_row_id=ordinal,
                transaction_uid=f"{source.metadata['source_release']}:{ordinal}",
                from_account_key="P" if same_key else "flagged-self",
                is_self_transfer=flagged,
            )
        )
    alerts = [
        alert
        for alert in _generate(rows, source, policy)
        if alert.account_key == "P" and alert.reference_ts == datetime(2022, 1, 1, 12, 1)
    ]
    assert len(alerts) == 1
    # 반복 상대방 거래도 포함하며, 자기 이체 두 건은 모두 룰 근거가 아닙니다.
    assert set(alerts[0].transaction_uids) == {row["transaction_uid"] for row in rows[:3]}
    assert len(alerts[0].transaction_uids) == 3
    assert alerts[0].window_start == datetime(2021, 12, 31, 12, 1)
    assert alerts[0].window_end == alerts[0].reference_ts
    # 서로 다른 두 상대방이 한 계좌가 되면 통화가 달라도 임계값 2를 충족하지 못합니다.
    rows[1]["from_account_key"] = rows[0]["from_account_key"]
    assert not [
        alert
        for alert in _generate(rows, source, policy)
        if alert.account_key == "P" and alert.reference_ts == datetime(2022, 1, 1, 12, 1)
    ]


@pytest.mark.parametrize("partitions,batch_rows", [(1, 1), (7, 3)])
def test_confirmed_r02_semantics_survive_both_preparation_entrypoints(
    source, policy, tmp_path, partitions, batch_rows
):
    rows, labels = source_values(source)
    for row in rows:
        if row["to_account_key"] == "P" and row["from_account_key"] == "sender-2":
            row["from_account_key"] = "sender-0"
            row["currency_paid_raw"] = row["currency_received_raw"] = "Euro"
            row["currency_paid_bucket"] = row["currency_received_bucket"] = "EUR"
    for same_key, flagged in ((True, False), (False, True)):
        rows.append(
            dict(
                rows[0],
                from_account_key="P" if same_key else "flagged-self",
                is_self_transfer=flagged,
            )
        )
        labels.append(0)
    rows = rewrite_fixture(source, rows, labels, part_rows=2)
    reference, streamed = tmp_path / "reference", tmp_path / "streamed"
    build_reference_dataset(
        source,
        policy,
        reference,
        train_before=datetime(2022, 1, 2),
        test_from=datetime(2022, 1, 4),
        execution_scope="FIXTURE_ONLY",
    )
    prepare(
        source,
        policy,
        streamed,
        partitions=partitions,
        read_batch_rows=batch_rows,
        max_partition_rows=256,
    )
    assert_same_artifacts(reference, streamed)
    alerts = read_json(streamed / "alerts.json")
    assert len(alerts) == 6  # P/Q는 두 번의 48시간 유휴 구간 후 각각 다시 발생합니다.
    selected = [
        alert
        for alert in alerts
        if alert["account_key"] == "P" and alert["reference_ts"] == "2022-01-01T12:01:00"
    ]
    assert len(selected) == 1
    assert selected[0]["transaction_uids"] == [row["transaction_uid"] for row in rows[:3]]
    evidence = pq.read_table(streamed / "evidence.parquet").to_pylist()
    by_uid = {row["transaction_uid"]: row for row in evidence}
    assert set(by_uid) == {row["transaction_uid"] for row in rows[:15]}
    for original in rows[:15]:
        saved = by_uid[original["transaction_uid"]]
        for key in ("amount_paid", "amount_received", "currency_paid_raw", "currency_received_raw"):
            assert saved[key] == original[key]
