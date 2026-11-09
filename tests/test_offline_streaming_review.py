"""리뷰 회귀 검사이며, fixture 정책값으로 팀 임계값을 확정하지 않습니다."""

import json
from datetime import datetime, timedelta
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from offline_test_data import (
    assert_same_artifacts,
    prepare,
    rewrite_fixture,
    source_values,
)

from aml_ml import offline_rules as rules
from aml_ml import offline_streaming as streaming
from aml_ml.offline_pipeline import build_reference_dataset, partition_features
from aml_ml.storage import read_json


def fixture_rows(events):
    release = "ibm_hi_small:" + "a" * 64
    return [
        dict(
            source_row_id=i,
            transaction_uid=f"{release}:{i}",
            ts=stamp,
            from_account_key=peer,
            to_account_key="receiver",
            is_self_transfer=False,
        )
        for i, (stamp, peer) in enumerate(events)
    ]


def candidates(rows, policy, **limits):
    return rules.r02_partition_candidates(
        rows,
        source_id="ibm_hi_small",
        source_release="ibm_hi_small:" + "a" * 64,
        policy=policy,
        execution_scope="FIXTURE_ONLY",
        **limits,
    )


@pytest.mark.parametrize(
    "value",
    [
        'a"\\\n\t\b\f\r\x00\x1f\x7f',
        "\u00e9\ud55c\U0001f600",
        "",
        {"nested": [None, True, False, 123, -1.25, ("x", "\u2028")]},
        [],
        {},
    ],
)
def test_json_preflight_matches_escaped_serialization_size(value):
    assert rules._json_bytes(value) == len(json.dumps(value, sort_keys=True, allow_nan=False))


def test_arrow_workset_rejects_large_row_before_table_allocation(monkeypatch):
    schema = pa.schema([("payload", pa.string())])

    def forbidden(*args, **kwargs):
        pytest.fail("Arrow allocation must follow the row-byte preflight")

    monkeypatch.setattr(
        streaming,
        "pa",
        SimpleNamespace(
            types=pa.types,
            Table=SimpleNamespace(from_pylist=forbidden),
        ),
    )
    with pytest.raises(ValueError, match="preallocation"):
        streaming._bounded_table([{"payload": "x" * 10_000}], schema, 4096)


def test_dictionary_strings_use_bounded_decode_before_expansion(tmp_path, monkeypatch):
    # 사전 크기가 디코딩된 전체 배치보다 작은 소규모 fixture입니다.
    path = tmp_path / "dictionary.parquet"
    pq.write_table(pa.table({"payload": ["x" * 4096] * 32}), path, use_dictionary=True)
    original = streaming.pq.ParquetFile
    observed = []

    class InspectReader:
        def __init__(self, filename):
            self.reader = original(filename)
            self.metadata = self.reader.metadata
            self.schema_arrow = self.reader.schema_arrow

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.reader.close()

        def iter_batches(self, *, batch_size, row_groups):
            observed.append(batch_size)
            assert 1 <= batch_size <= 3
            yield from self.reader.iter_batches(batch_size=batch_size, row_groups=row_groups)

    monkeypatch.setattr(streaming.pq, "ParquetFile", InspectReader)
    rows = list(streaming._rows(path, 32, max_bytes=64 * 1024))
    assert len(observed) == 1 and observed[0] < 32
    assert len(rows) == 32 and all(row["payload"] == "x" * 4096 for row in rows)


@pytest.mark.parametrize(
    "limit,expected",
    [
        ({"max_evidence_references": 1}, "evidence-reference"),
        ({"max_payload_bytes": 4096}, "payload"),
    ],
)
def test_full_evidence_and_escaped_payload_guard_precedes_alert_allocation(
    policy,
    monkeypatch,
    limit,
    expected,
):
    rows = fixture_rows([(datetime(2022, 1, 1), "A"), (datetime(2022, 1, 1), "B")])
    rows = [dict(row, to_account_key="\x00" * 1000) for row in rows]

    def forbidden(*args, **kwargs):
        pytest.fail("The full-evidence tuple/alert must not be allocated above budget")

    monkeypatch.setattr(rules, "OfflineAlert", forbidden)
    with pytest.raises(ValueError, match=expected):
        candidates(rows, policy, **limit)


def test_payload_budget_is_cumulative_across_alerts(policy):
    rows = fixture_rows(
        [
            (datetime(2022, 1, 1) + timedelta(days=2 * day), peer)
            for day in range(12)
            for peer in ("A", "B")
        ]
    )
    with pytest.raises(ValueError, match="payload byte budget"):
        candidates(rows, policy, max_evidence_references=1000, max_payload_bytes=4096)
    assert len(candidates(rows, policy)) == 12


def test_long_valid_source_uid_is_checked_before_alert_identity(policy, monkeypatch):
    source_id = "source" * 1000
    release = source_id + ":" + "a" * 64
    rows = [
        dict(row, transaction_uid=f"{release}:{i}")
        for i, row in enumerate(
            fixture_rows([(datetime(2022, 1, 1), "A"), (datetime(2022, 1, 1), "B")])
        )
    ]

    def forbidden(*args, **kwargs):
        pytest.fail("Long source UID must be rejected before alert identity JSON")

    monkeypatch.setattr(rules, "OfflineAlert", forbidden)
    with pytest.raises(ValueError, match="payload"):
        rules.r02_partition_candidates(
            rows,
            source_id=source_id,
            source_release=release,
            policy=policy,
            execution_scope="FIXTURE_ONLY",
            max_payload_bytes=4096,
        )


def test_b7_exact_start_end_and_all_incoming_evidence_have_independent_expected_values(
    policy,
):
    rows = fixture_rows(
        [
            (datetime(2022, 1, 1, 0, 0, 59, 999999), "A"),
            (datetime(2022, 1, 1, 0, 1), "A"),
            (datetime(2022, 1, 2), "B"),
            (datetime(2022, 1, 2, 0, 0, 59, 999999), "B"),
            (datetime(2022, 1, 2, 0, 1), "C"),
        ]
    )
    alerts = candidates(rows, policy)
    assert len(alerts) == 1
    alert = alerts[0]
    assert alert.reference_ts == alert.window_end == datetime(2022, 1, 2, 0, 1)
    assert alert.window_start == datetime(2022, 1, 1, 0, 1)
    assert alert.transaction_uids == tuple(rows[i]["transaction_uid"] for i in (1, 2, 3))


def test_previous_clock_minute_expiry_and_retrigger_have_independent_evidence(policy):
    rows = fixture_rows(
        [
            (datetime(2022, 1, 1, 0, 0, 5), "A"),
            (datetime(2022, 1, 1, 0, 0, 30), "B"),
            (datetime(2022, 1, 1, 12), "B"),
            (datetime(2022, 1, 2, 0, 1), "C"),
        ]
    )
    alerts = candidates(rows, policy)
    assert [alert.reference_ts for alert in alerts] == [
        datetime(2022, 1, 1, 0, 1),
        datetime(2022, 1, 2, 0, 2),
    ]
    assert [alert.transaction_uids for alert in alerts] == [
        tuple(rows[i]["transaction_uid"] for i in (0, 1)),
        tuple(rows[i]["transaction_uid"] for i in (2, 3)),
    ]


def test_reference_timestamp_transaction_is_excluded_from_features(source, policy):
    original, _ = source_values(source)
    rows = fixture_rows(
        [
            (datetime(2022, 1, 1, 0, 0, 5), "A"),
            (datetime(2022, 1, 1, 0, 0, 30), "B"),
            (datetime(2022, 1, 1, 0, 1), "C"),
        ]
    )
    rows = [dict(original[0], **row) for row in rows]
    alert = candidates(rows, policy)[0]
    raw = partition_features(source, rows, [alert], coverage_start=0).to_pylist()[0]
    assert raw["account_in_count_24h"] == 2
    assert alert.transaction_uids == (
        rows[0]["transaction_uid"],
        rows[1]["transaction_uid"],
    )


def merge_runs(tmp_path, count=65, *, ties=False):
    schema = pa.schema([("key", pa.int64()), ("payload", pa.string())])
    paths = []
    for i in range(count):
        path = tmp_path / f"input-{i}.parquet"
        rows = (
            [{"key": 0, "payload": str(i)}]
            if ties
            else [{"key": i, "payload": str(i)}, {"key": i + count, "payload": str(i)}]
        )
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)
        paths.append(path)
    return paths, schema


def tracked_readers(monkeypatch):
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
    return state


def test_two_stage_merge_sorting_fan_in_and_reader_cleanup(tmp_path, monkeypatch):
    paths, schema = merge_runs(tmp_path)
    state = tracked_readers(monkeypatch)
    result = list(
        streaming._bounded_merge(
            paths,
            lambda row: row["key"],
            work=tmp_path,
            schema=schema,
            prefix="review",
            max_bytes=1024 * 1024,
        )
    )
    assert [row["key"] for row in result] == list(range(130))
    assert state == {"active": 0, "peak": 8}
    assert len(list(tmp_path.glob("review-merge-0-*.parquet"))) == 9
    assert len(list(tmp_path.glob("review-merge-1-*.parquet"))) == 2


def test_staged_merge_preserves_duplicate_keys_and_closes_on_early_exit(tmp_path, monkeypatch):
    paths, schema = merge_runs(tmp_path, 17, ties=True)
    state = tracked_readers(monkeypatch)
    kwargs = dict(work=tmp_path, schema=schema, max_bytes=1024 * 1024)
    result = list(streaming._bounded_merge(paths, lambda row: row["key"], prefix="ties", **kwargs))
    assert [row["payload"] for row in result] == [str(i) for i in range(17)]
    merged = streaming._bounded_merge(paths, lambda row: row["key"], prefix="early", **kwargs)
    assert next(merged)["payload"] == "0"
    merged.close()
    assert state["active"] == 0 and state["peak"] <= 8


def test_direct_merge_over_fan_in_opens_no_readers(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("No reader may open above the merge fan-in limit")

    monkeypatch.setattr(streaming, "_rows", forbidden)
    with pytest.raises(ValueError, match="fan-in"):
        list(
            streaming._merge(
                [tmp_path / str(i) for i in range(9)],
                lambda row: row,
                max_bytes=4096,
            )
        )


def test_merge_exception_closes_all_active_readers(tmp_path, monkeypatch):
    paths, _ = merge_runs(tmp_path, 3)
    state = tracked_readers(monkeypatch)

    def fail_on_third(row):
        if row["key"] == 2:
            raise ValueError("injected key failure")
        return row["key"]

    with pytest.raises(ValueError, match="injected"):
        list(streaming._merge(paths, fail_on_third, max_bytes=1024 * 1024))
    assert state["active"] == 0


@pytest.mark.parametrize("prefix", ["alerts", "evidence"])
def test_final_duplicate_checks_still_reject_after_staged_merge(
    source,
    policy,
    tmp_path,
    monkeypatch,
    prefix,
):
    original = streaming._bounded_merge

    def duplicated(*args, **kwargs):
        merged = original(*args, **kwargs)
        try:
            first = next(merged)
            yield first
            if kwargs["prefix"] == prefix:
                yield first
            yield from merged
        finally:
            merged.close()

    monkeypatch.setattr(streaming, "_bounded_merge", duplicated)
    output = tmp_path / "duplicate"
    with pytest.raises(ValueError, match="Duplicate/unsorted"):
        prepare(source, policy, output, partitions=17, read_batch_rows=3)
    assert not (output / "manifest.json").exists()


@pytest.mark.parametrize("partitions,batch_rows", [(1, 1), (17, 3), (65, 129)])
def test_review_partition_batch_variants_preserve_reference_artifacts(
    source,
    policy,
    tmp_path,
    partitions,
    batch_rows,
):
    rows, labels = source_values(source)
    rows = [dict(row, ts=row["ts"] - timedelta(seconds=i % 3)) for i, row in enumerate(rows)]
    rewrite_fixture(source, rows, labels, part_rows=4)
    reference, output = tmp_path / "reference", tmp_path / "streamed"
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
        output,
        partitions=partitions,
        read_batch_rows=batch_rows,
        max_partition_rows=256,
    )
    assert_same_artifacts(reference, output)


def test_staged_dataset_merge_keeps_fixed_256_train_fit(source, policy, tmp_path):
    original, truth = source_values(source)
    rows, labels = [], []
    for copy in range(130):
        rows.extend(
            dict(
                row,
                from_account_key=f"{row['from_account_key']}-{copy}",
                to_account_key=f"{row['to_account_key']}-{copy}",
            )
            for row in original
        )
        labels.extend(truth)
    rewrite_fixture(source, rows, labels, part_rows=113)
    reference, output = tmp_path / "reference-many", tmp_path / "streamed-many"
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
        output,
        partitions=65,
        read_batch_rows=17,
        max_partition_rows=5000,
    )
    assert_same_artifacts(reference, output)
    assert read_json(output / "preprocessing.json")["fit_rows"] == 260
    assert read_json(output / "manifest.json")["execution_settings"]["merge_fan_in"] == 8
