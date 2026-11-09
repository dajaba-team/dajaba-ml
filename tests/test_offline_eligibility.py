"""5단계 정책 회귀 검사이며 합성 입력만 쓰고 운영 최솟값을 정하지 않습니다."""

import math
from dataclasses import replace
from datetime import datetime, timedelta
from decimal import Decimal

import pyarrow.parquet as pq
import pytest
from offline_test_data import (
    assert_same_artifacts,
    prepare,
    rewrite_fixture,
    source_values,
    synthetic_support,
)

from aml_ml import offline_account_stream
from aml_ml.alert_dataset import offline_proxy_targets
from aml_ml.offline_eligibility import DatasetNotReady, EligibilityCounts, ledger_row
from aml_ml.offline_pipeline import build_dataset, build_reference_dataset, load_rows
from aml_ml.offline_readiness import require_evaluation_support
from aml_ml.offline_rules import generate_r02
from aml_ml.storage import read_json, sha256

FIRST = datetime(2022, 1, 2)
SECOND = datetime(2022, 1, 4)


def _candidate(source, policy):
    return generate_r02(
        load_rows(source, max_source_rows=100),
        source_id=source.id,
        source_release=source.metadata["source_release"],
        policy=policy,
        execution_scope="FIXTURE_ONLY",
    )[0]


@pytest.mark.parametrize("horizon", [86400, 3 * 86400])
def test_boundaries_before_at_after_long_horizon_and_same_time_cohort(source, policy, horizon):
    first, second = FIRST, FIRST + timedelta(days=10)
    h = timedelta(seconds=horizon)
    epsilon = timedelta(microseconds=1)
    expected = [
        (first - epsilon, "train", True),
        (first, "validation", False),
        (first + epsilon, "validation", False),
        (first + h - epsilon, "validation", False),
        (first + h, "validation", True),
        (second - epsilon, "validation", True),
        (second, "test", False),
        (second + h - epsilon, "test", False),
        (second + h, "test", True),
        (second + h + epsilon, "test", True),
    ]
    candidate = _candidate(source, policy)
    for reference, split, time_eligible in expected:
        alert = replace(
            candidate,
            reference_ts=reference,
            window_end=reference,
            window_start=reference - timedelta(hours=24),
        )
        for truth in (0, 1, None):
            target = offline_proxy_targets([alert], dict.fromkeys(alert.transaction_uids, truth))[0]
            row = ledger_row(
                alert, target, train_before=first, test_from=second, horizon_seconds=horizon
            )
            assert row["original_split"] == split
            assert row["time_eligible"] is time_eligible
            assert row["target_complete"] is (truth is not None)
            assert row["supervised_eligible"] is (time_eligible and truth is not None)
            assert row["y_alert_proxy"] == truth
            assert row["evidence_uids"] == list(alert.transaction_uids)


def test_ledger_multiple_reasons_balance_and_actual_positive_unknown_counts(source, policy):
    candidate = _candidate(source, policy)
    reference = FIRST + timedelta(hours=1)
    alert = replace(
        candidate,
        reference_ts=reference,
        window_end=reference,
        window_start=reference - timedelta(hours=24),
    )
    counts = EligibilityCounts()
    for truth in (0, 1, None):
        target = offline_proxy_targets([alert], dict.fromkeys(alert.transaction_uids, truth))[0]
        item = ledger_row(alert, target, train_before=FIRST, test_from=SECOND)
        counts.add(item)
        if truth is None:
            assert item["exclusion_reasons"] == [
                "VALIDATION_HORIZON_PURGE",
                "UNRESOLVED_EVIDENCE_LABEL",
            ]
    report = counts.report()
    validation = report["splits"]["validation"]
    assert report["total_alerts"] == report["excluded_alerts"] == 3
    assert report["eligible_alerts"] == 0
    assert (
        validation["excluded"]["positive"]
        == validation["excluded"]["negative"]
        == validation["excluded"]["unresolved"]
        == 1
    )
    assert validation["exclusion_reason_counts"] == {
        "UNRESOLVED_EVIDENCE_LABEL": 1,
        "VALIDATION_HORIZON_PURGE": 3,
    }
    assert validation["all"]["positive_fraction_known"] == 0.5
    assert validation["all"]["positive_fraction_all"] == 1 / 3
    assert report["model_evaluation_readiness"] == "NOT_READY"


@pytest.mark.parametrize("builder", [build_dataset, build_reference_dataset])
def test_empty_eligible_split_preserves_all_candidates_targets_evidence_and_ledger(
    source, policy, tmp_path, builder
):
    output = tmp_path / "not-ready"
    original_rows, original_labels = source_values(source)
    expected_alerts = generate_r02(
        original_rows,
        source_id=source.id,
        source_release=source.metadata["source_release"],
        policy=policy,
        execution_scope="FIXTURE_ONLY",
    )
    expected_targets = offline_proxy_targets(
        expected_alerts,
        dict(zip((row["transaction_uid"] for row in original_rows), original_labels, strict=True)),
    )
    original_hashes = {str(p): sha256(p) for p in source.folder.rglob("*") if p.is_file()}
    with pytest.raises(DatasetNotReady, match="NOT_READY"):
        builder(
            source,
            policy,
            output,
            train_before=FIRST,
            test_from=FIRST + timedelta(hours=12),
            execution_scope="FIXTURE_ONLY",
        )
    assert read_json(output / "failure.json")["status"] == "NOT_READY"
    assert not (output / "manifest.json").exists()
    assert (
        len(read_json(output / "alerts.json"))
        == len(read_json(output / "targets.json"))
        == len(read_json(output / "ledger.json"))
        == 6
    )
    assert pq.read_table(output / "evidence.parquet").num_rows == 15
    assert pq.read_table(output / "evidence.parquet").to_pylist() == original_rows
    stored_targets = read_json(output / "targets.json")
    assert [
        {key: value for key, value in row.items() if key != "split"} for row in stored_targets
    ] == expected_targets
    assert [item["training_alert_uid"] for item in read_json(output / "ledger.json")] == [
        alert.training_alert_uid for alert in expected_alerts
    ]
    assert [item["transaction_uids"] for item in read_json(output / "alerts.json")] == [
        list(alert.transaction_uids) for alert in expected_alerts
    ]
    counts = read_json(output / "quality.json")["eligibility"]
    assert counts["total_alerts"] == counts["eligible_alerts"] + counts["excluded_alerts"]
    assert counts["splits"]["validation"]["eligible"]["rows"] == 0
    assert {str(p): sha256(p) for p in source.folder.rglob("*") if p.is_file()} == original_hashes


@pytest.mark.parametrize("decode,sort,write,fan", [(1, 1, 2, 2), (7, 3, 5, 8), (31, 17, 11, 3)])
def test_full_b7_state_ledger_uid_projection_and_train_only_parity(
    source, policy, tmp_path, decode, sort, write, fan
):
    original, labels = source_values(source)
    extra = [
        dict(original[0], ts=stamp, from_account_key=peer, to_account_key="shared")
        for stamp, peer in (
            (datetime(2022, 1, 1), "a"),
            (datetime(2022, 1, 1, 12), "b"),
            (datetime(2022, 1, 2, 0, 1), "c"),
        )
    ]
    rows = rewrite_fixture(source, extra + original, [0, 1, None] + labels, part_rows=2)
    reference, streamed = tmp_path / "reference", tmp_path / "streamed"
    build_reference_dataset(
        source,
        policy,
        reference,
        train_before=FIRST,
        test_from=SECOND,
        execution_scope="FIXTURE_ONLY",
    )
    prepare(
        source,
        policy,
        streamed,
        partitions=1,
        decode_batch_rows=decode,
        sort_run_rows=sort,
        write_batch_rows=write,
        merge_fan_in=fan,
    )
    assert_same_artifacts(reference, streamed)
    ledger = read_json(streamed / "ledger.json")
    alerts = read_json(streamed / "alerts.json")
    assert len([alert for alert in alerts if alert["account_key"] == "shared"]) == 2
    excluded = [item for item in ledger if not item["supervised_eligible"]]
    assert len(excluded) == 1 and excluded[0]["target_complete"] is False
    assert excluded[0]["exclusion_reasons"] == [
        "VALIDATION_HORIZON_PURGE",
        "UNRESOLVED_EVIDENCE_LABEL",
    ]
    for split in ("train", "validation", "test"):
        expected = [
            item["training_alert_uid"]
            for item in ledger
            if item["supervised_eligible"] and item["original_split"] == split
        ]
        y = pq.read_table(streamed / f"y_{split}.parquet")
        assert y.column("training_alert_uid").to_pylist() == expected
        assert y.column("y_alert_proxy").null_count == 0
        assert pq.read_table(streamed / f"X_{split}.parquet").num_rows == len(expected)
    state = read_json(streamed / "preprocessing.json")
    assert state["fit_rows"] == 3
    assert state["numeric"]["account_in_count_24h"]["mean"] == pytest.approx(7 / 3)
    assert state["numeric"]["account_in_count_24h"]["std"] == pytest.approx(math.sqrt(2 / 9))
    assert state["numeric"]["account_in_amount_log1p_24h"]["mean"] == pytest.approx(
        (math.log1p(60) + math.log1p(3) + math.log1p(20)) / 3
    )
    assert pq.read_table(streamed / "evidence.parquet").num_rows == len(rows)


def test_eligible_cross_split_guard_is_still_enforced(source, policy, tmp_path, monkeypatch):
    original, labels = source_values(source)
    shared = [
        dict(original[0], ts=stamp, from_account_key=peer, to_account_key="shared")
        for stamp, peer in (
            (datetime(2022, 1, 1), "a"),
            (datetime(2022, 1, 1, 12), "b"),
            (datetime(2022, 1, 2, 0, 1), "c"),
        )
    ]
    rewrite_fixture(source, shared + original, [0, 0, 1] + labels)
    real = offline_account_stream.ledger_row

    def wrongly_eligible(*args, **kwargs):
        return dict(real(*args, **kwargs), supervised_eligible=True)

    monkeypatch.setattr(offline_account_stream, "ledger_row", wrongly_eligible)
    with pytest.raises(ValueError, match="overlaps eligible splits"):
        prepare(source, policy, tmp_path / "blocked")


def test_missing_support_policy_blocks_before_fit_without_scope_bypass(
    source, policy, tmp_path, monkeypatch
):
    from aml_ml import offline_training, offline_xgboost

    dataset = tmp_path / "data"
    prepare(source, policy, dataset)

    def forbidden(*args, **kwargs):
        pytest.fail("Fit/native backend must not be reached before support readiness")

    monkeypatch.setattr(offline_training, "probabilities", forbidden)
    monkeypatch.setattr(offline_xgboost, "_xgboost", forbidden)
    with pytest.raises(ValueError, match="NOT_READY"):
        offline_training.train_evaluate(
            dataset,
            source,
            policy,
            tmp_path / "no-model",
            execution_scope="FIXTURE_ONLY",
            model_version="synthetic",
            k=1,
        )
    with pytest.raises(ValueError, match="NOT_READY"):
        offline_xgboost.train_evaluate(
            dataset,
            source,
            policy,
            tmp_path / "no-model",
            execution_scope="FIXTURE_ONLY",
            model_version="synthetic",
            k=1,
            seed=1,
        )
    assert not (tmp_path / "no-model").exists()
    supplied = synthetic_support()
    assert (
        require_evaluation_support(dataset, supplied, supplied["model_kind"])["status"]
        == "READY_FOR_DECLARED_SUPPORT"
    )
    supplied["minimums"]["train"]["rows"] = 100
    with pytest.raises(ValueError, match="NOT_READY"):
        require_evaluation_support(dataset, supplied, supplied["model_kind"])


def test_validation_test_feature_changes_do_not_refit_frozen_train_state(source, policy, tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    prepare(source, policy, first)
    original, labels = source_values(source)
    changed = [
        dict(row, amount_paid=Decimal(99), amount_received=Decimal(99))
        if row["ts"] >= FIRST
        else row
        for row in original
    ]
    rewrite_fixture(source, changed, labels)
    prepare(source, policy, second)
    assert (first / "preprocessing.json").read_bytes() == (
        second / "preprocessing.json"
    ).read_bytes()
    assert pq.read_table(first / "X_train.parquet").equals(
        pq.read_table(second / "X_train.parquet")
    )
    assert not pq.read_table(first / "X_validation.parquet").equals(
        pq.read_table(second / "X_validation.parquet")
    )


def test_continuously_true_account_does_not_reemit_at_split_boundary(source, policy, tmp_path):
    original, labels = source_values(source)
    sticky = [
        dict(original[0], to_account_key="continuous", from_account_key=f"sticky-{i}", ts=stamp)
        for i, stamp in enumerate(
            (
                FIRST - timedelta(minutes=2),
                FIRST - timedelta(minutes=2) + timedelta(seconds=1),
                FIRST + timedelta(minutes=1),
            )
        )
    ]
    rewrite_fixture(source, sticky + original, [0, 1, 0] + labels, part_rows=1)
    output = tmp_path / "continuous"
    prepare(source, policy, output, sort_run_rows=1, decode_batch_rows=1)
    selected = [
        item for item in read_json(output / "alerts.json") if item["account_key"] == "continuous"
    ]
    assert len(selected) == 1
    assert selected[0]["reference_ts"] == (FIRST - timedelta(minutes=1)).isoformat()


def test_distinct_alert_uids_at_same_reference_remain_in_one_time_cohort(source, policy):
    alerts = generate_r02(
        load_rows(source, max_source_rows=100),
        source_id=source.id,
        source_release=source.metadata["source_release"],
        policy=policy,
        execution_scope="FIXTURE_ONLY",
    )[:2]
    assert alerts[0].training_alert_uid != alerts[1].training_alert_uid
    assert alerts[0].reference_ts == alerts[1].reference_ts
    first = alerts[0].reference_ts
    items = []
    for alert, truth in zip(alerts, (1, None), strict=True):
        target = offline_proxy_targets([alert], dict.fromkeys(alert.transaction_uids, truth))[0]
        items.append(
            ledger_row(alert, target, train_before=first, test_from=first + timedelta(days=2))
        )
    assert {item["original_split"] for item in items} == {"validation"}
    assert {item["time_eligible"] for item in items} == {False}


def test_actual_horizon_target_uid_and_completeness_mismatches_are_rejected(source, policy):
    alert = _candidate(source, policy)
    target = offline_proxy_targets([alert], dict.fromkeys(alert.transaction_uids, 1))[0]
    for change, match in (
        ({"training_alert_uid": "different"}, "UID"),
        ({"missing_label_count": 1}, "completeness"),
    ):
        with pytest.raises(ValueError, match=match):
            ledger_row(alert, dict(target, **change), train_before=FIRST, test_from=SECOND)
    longer = replace(alert, window_start=alert.reference_ts - timedelta(hours=25))
    target = offline_proxy_targets([longer], dict.fromkeys(longer.transaction_uids, 1))[0]
    with pytest.raises(ValueError, match="shorter"):
        ledger_row(longer, target, train_before=FIRST, test_from=SECOND)
