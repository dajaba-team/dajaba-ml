"""소규모 원천 기반 분할 동등성과 제한된 작업 공간의 회귀를 검사합니다."""

import json
from dataclasses import asdict
from datetime import datetime, timedelta

import pytest
from offline_test_data import (
    assert_same_artifacts,
    prepare,
    rewrite_fixture,
    source_values,
)

from aml_ml.offline_cli import main
from aml_ml.offline_pipeline import (
    build_dataset,
    build_reference_dataset,
    verify_dataset,
)
from aml_ml.offline_rules import fingerprint
from aml_ml.offline_training import train_evaluate
from aml_ml.storage import read_json, sha256, write_json


@pytest.mark.parametrize("partitions,batch_rows", [(1, 1), (7, 3), (32, 128)])
def test_partition_and_batch_sizes_match_reference(
    source, policy, tmp_path, partitions, batch_rows
):
    rows, labels = source_values(source)
    # 원본 행 순번을 보존하면서 파트 경계까지 포함해 시각 순서를 섞습니다.
    rows = [
        dict(row, ts=row["ts"] + timedelta(seconds=(i % 3) * -10)) for i, row in enumerate(rows)
    ]
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
    result = prepare(
        source,
        policy,
        output,
        partitions=partitions,
        read_batch_rows=batch_rows,
        max_partition_rows=256,
    )
    assert result["status"] == "VERIFIED"
    assert_same_artifacts(reference, output)
    assert verify_dataset(output, source, policy, execution_scope="FIXTURE_ONLY") == result
    stats = read_json(output / "manifest.json")["work_stats"]
    assert stats["input_rows"] == 15
    assert stats["max_active_window_rows"] <= 256
    assert (
        read_json(output / "manifest.json")["execution_settings"]["decode_batch_rows"] == batch_rows
    )


@pytest.mark.parametrize("partitions,batch_rows", [(1, 17), (7, 31)])
def test_fixed_train_batches_and_many_alerts_match_exactly(
    source, policy, tmp_path, partitions, batch_rows
):
    original, truth = source_values(source)
    rows, labels = [], []
    for copy in range(150):
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
        max_partition_rows=5000,
    )
    assert_same_artifacts(reference, output)
    state = read_json(output / "preprocessing.json")
    assert state["fit_rows"] == 300  # 고정된 256행 적합 배치 경계를 넘습니다.
    assert read_json(output / "manifest.json")["work_stats"]["input_rows"] == 2250


def test_exact_24h_start_same_minute_and_reference_exclusion(source, policy, tmp_path):
    original, labels = source_values(source)
    template = original[0]
    dates = [
        datetime(2022, 1, 1, 0, 1),
        datetime(2022, 1, 2, 0, 0),
        datetime(2022, 1, 2, 0, 0, 30),
        datetime(2022, 1, 2, 0, 1),
    ]
    boundary = [
        dict(
            template,
            ts=stamp,
            from_account_key=f"boundary-peer-{i}",
            to_account_key="boundary",
        )
        for i, stamp in enumerate(dates)
    ]
    rows = rewrite_fixture(source, boundary + original, [0, 0, 1, 0] + labels, part_rows=2)
    output = tmp_path / "boundary"
    build_dataset(
        source,
        policy,
        output,
        train_before=datetime(2022, 1, 2, 6),
        test_from=datetime(2022, 1, 4),
        execution_scope="FIXTURE_ONLY",
        partitions=7,
        read_batch_rows=1,
        max_partition_rows=256,
    )
    alerts = read_json(output / "alerts.json")
    selected = [a for a in alerts if a["account_key"] == "boundary"]
    assert len(selected) == 1
    assert selected[0]["reference_ts"] == "2022-01-02T00:01:00"
    assert selected[0]["transaction_uids"] == [r["transaction_uid"] for r in rows[:3]]
    # 기준 P/Q는 48시간 유휴 구간을 지날 때마다 다시 경보가 발생합니다.
    # 마지막 거래의 캐시가 아니라 직전 시계상의 분을 기준으로 합니다.
    assert sum(a["account_key"] in {"P", "Q"} for a in alerts) == 6


@pytest.mark.parametrize(
    "settings",
    [
        {"max_partition_bytes": 1},
        {"max_source_rows": 14},
        {"max_partition_rows": 255},
    ],
)
def test_invalid_or_exceeded_budget_never_publishes_manifest(source, policy, tmp_path, settings):
    output = tmp_path / "blocked"
    with pytest.raises(ValueError, match="budget"):
        prepare(source, policy, output, **settings)
    assert not (output / "manifest.json").exists()
    assert not list(tmp_path.glob(".offline-work-*"))


def test_skewed_account_partition_rejects_before_materializing(source, policy, tmp_path):
    rows, labels = source_values(source)
    # 단일 현재 구간은 여전히 안전 한도를 넘을 수 있습니다. 누적량은 디스크에
    # 저장하므로 모든 행을 같은 분에 넣어 실제 상한을 검사합니다.
    many = [
        dict(rows[i % len(rows)], to_account_key="one-skewed-account", ts=datetime(2022, 1, 1))
        for i in range(300)
    ]
    rewrite_fixture(source, many, [labels[i % len(labels)] for i in range(300)], part_rows=19)
    output = tmp_path / "skew"
    with pytest.raises(ValueError, match="workset"):
        prepare(
            source,
            policy,
            output,
            partitions=1,
            max_partition_rows=256,
            read_batch_rows=3,
        )
    assert not output.exists()


def test_bad_original_order_rejected_and_purged_shared_evidence_preserved(source, policy, tmp_path):
    original, labels = source_values(source)
    invalid = [dict(row) for row in original]
    invalid[0], invalid[1] = invalid[1], invalid[0]
    rewrite_fixture(source, invalid, labels, assign_ordinals=False)
    with pytest.raises(ValueError, match="ordinals"):
        prepare(source, policy, tmp_path / "unordered")
    template = original[0]
    shared = [
        dict(template, ts=stamp, from_account_key=peer, to_account_key="shared")
        for stamp, peer in [
            (datetime(2022, 1, 1), "a"),
            (datetime(2022, 1, 1, 12), "b"),
            (datetime(2022, 1, 2, 0, 1), "c"),
        ]
    ]
    rewrite_fixture(source, shared + original, [0, 0, 1] + labels)
    output = tmp_path / "overlap"
    prepare(source, policy, output)
    ledger = read_json(output / "ledger.json")
    excluded = [item for item in ledger if not item["supervised_eligible"]]
    assert len(excluded) == 1
    assert excluded[0]["exclusion_reasons"] == ["VALIDATION_HORIZON_PURGE"]
    alerts = read_json(output / "alerts.json")
    assert len([item for item in alerts if item["account_key"] == "shared"]) == 2


def test_missing_evidence_cannot_be_hidden_by_updated_file_hash(source, policy, tmp_path):
    output = tmp_path / "tampered"
    prepare(source, policy, output)
    payload = read_json(output / "alerts.json")
    payload[0]["transaction_uids"].pop()
    write_json(output / "alerts.json", payload)
    manifest = read_json(output / "manifest.json")
    manifest["files"]["alerts.json"] = sha256(output / "alerts.json")
    manifest.pop("dataset_fingerprint")
    manifest["dataset_fingerprint"] = fingerprint(manifest)
    write_json(output / "manifest.json", manifest)
    with pytest.raises(ValueError, match="alignment"):
        verify_dataset(output, source, policy, execution_scope="FIXTURE_ONLY")


def test_cli_budgets_and_small_training_guard(source, policy, tmp_path, capsys):
    policy_path = tmp_path / "policy.json"
    write_json(policy_path, asdict(policy))
    common = [
        "--assembly-index",
        str(tmp_path / "index.json"),
        "--source",
        source.id,
        "--policy",
        str(policy_path),
        "--scope",
        "FIXTURE_ONLY",
    ]
    output = tmp_path / "cli-streamed"
    assert (
        main(
            [
                "prepare",
                *common,
                "--output",
                str(output),
                "--train-before",
                "2022-01-02",
                "--test-from",
                "2022-01-04",
                "--partitions",
                "7",
                "--read-batch-rows",
                "1",
                "--max-partition-rows",
                "256",
            ]
        )
        == 0
    )
    settings = read_json(output / "manifest.json")["execution_settings"]
    assert settings["partitions"] == 7 and settings["read_batch_rows"] == 1
    assert main(["verify", *common, "--dataset", str(output), "--max-source-rows", "14"]) == 2
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["status"] == "FAILED"
    with pytest.raises(ValueError, match="Small baseline"):
        train_evaluate(
            tmp_path / "not-read",
            source,
            policy,
            tmp_path / "no-model",
            execution_scope="FIXTURE_ONLY",
            model_version="fixture",
            k=1,
            max_source_rows=14,
        )
