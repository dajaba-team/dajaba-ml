"""작은 원천 전체 -> R02 -> X/y -> 기준 모델 저장과 변조 후 다시 읽기를 검사합니다."""

import json
from dataclasses import asdict, replace
from datetime import datetime, timedelta

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from offline_test_data import synthetic_support

from aml_ml.offline_cli import main
from aml_ml.offline_pipeline import (
    FEATURE_NAMES,
    AlertPreprocessor,
    build_dataset,
    load_rows,
    offline_features,
    regeneration_required,
    verify_dataset,
)
from aml_ml.offline_rules import R02Policy, fingerprint, generate_r02
from aml_ml.offline_training import metrics, predict_raw, train_evaluate, verify_model
from aml_ml.storage import read_json, sha256, write_json


def build(source, policy, path):
    return build_dataset(
        source,
        policy,
        path,
        train_before=datetime(2022, 1, 2),
        test_from=datetime(2022, 1, 4),
        execution_scope="FIXTURE_ONLY",
    )


def candidates(source, policy):
    return generate_r02(
        load_rows(source, max_source_rows=100),
        source_id=source.id,
        source_release=source.metadata["source_release"],
        policy=policy,
        execution_scope="FIXTURE_ONLY",
    )


def test_pending_and_fixture_cannot_pass_official_gate(source, policy, tmp_path):
    with pytest.raises(ValueError, match="Pending/fixture"):
        build_dataset(
            source,
            policy,
            tmp_path / "forbidden",
            train_before=datetime(2022, 1, 2),
            test_from=datetime(2022, 1, 4),
            execution_scope="CONFIRMED",
        )
    assert not (tmp_path / "forbidden").exists()
    for key, value in (
        ("min_counterparties", None),
        ("rule_version", None),
        ("evidence_policy", None),
        ("confirmation_reference", ""),
    ):
        with pytest.raises(ValueError):
            replace(policy, **{key: value}).require("FIXTURE_ONLY")
    pending = tmp_path / "pending.json"
    write_json(pending, asdict(R02Policy()))
    assert (
        main(
            [
                "prepare",
                "--assembly-index",
                "does-not-exist.json",
                "--source",
                source.id,
                "--policy",
                str(pending),
                "--scope",
                "CONFIRMED",
            ]
        )
        == 2
    )


def test_r02_full_minute_idle_recurrence_all_evidence_label_blind(source, policy):
    rows = load_rows(source, max_source_rows=100)
    result = candidates(source, policy)
    assert len(result) == 6
    assert {a.reference_ts.second for a in result} == {0}
    assert {a.reference_ts.minute for a in result} == {1}
    assert sorted(len(a.transaction_uids) for a in result) == [2, 2, 2, 3, 3, 3]
    assert all(a.window_end - a.window_start == timedelta(hours=24) for a in result)
    assert all(a.training_alert_uid.startswith("TRAIN-ALERT-") for a in result)
    changed = [
        dict(r, source_positive=1, pattern_type="SECRET", currency_received_raw="Euro")
        for r in rows
    ]
    same = generate_r02(
        changed,
        source_id=source.id,
        source_release=source.metadata["source_release"],
        policy=policy,
        execution_scope="FIXTURE_ONLY",
    )
    assert result == same
    duplicate = [*rows, rows[0]]
    with pytest.raises(ValueError, match="Duplicate"):
        generate_r02(
            duplicate,
            source_id=source.id,
            source_release=source.metadata["source_release"],
            policy=policy,
            execution_scope="FIXTURE_ONLY",
        )


def test_past_only_features_mixed_currency_and_saml_directional_null(source, policy):
    rows = load_rows(source, max_source_rows=100)
    alerts = candidates(source, policy)
    before = offline_features(source, rows, alerts)
    # 기준 시각과 같거나 미래인 거래가 이 경보에 들어가면 안 됩니다.
    first = alerts[0]
    extra = dict(
        rows[0],
        transaction_uid=source.metadata["source_release"] + ":15",
        source_row_id=15,
        ts=first.reference_ts,
        from_account_key="future",
        to_account_key=first.account_key,
    )
    after = offline_features(source, rows + [extra], [first])
    assert before.slice(0, 1).equals(after)
    mixed = [dict(r, currency_received_raw="Euro") if r["source_row_id"] == 1 else r for r in rows]
    raw = offline_features(source, mixed, alerts).to_pylist()
    p = next(i for i, a in enumerate(alerts) if a.account_key == "P")
    assert raw[p]["account_in_count_24h"] == 3
    assert raw[p]["account_in_amount_log1p_24h"] is None
    saml = offline_features(replace(source, kind="saml"), rows, alerts).to_pylist()
    assert all(r["account_in_amount_log1p_24h"] is None for r in saml)


def test_build_independent_reload_and_no_overwrite(source, policy, tmp_path):
    path = tmp_path / "data"
    hashes = {str(p): sha256(p) for p in source.folder.rglob("*") if p.is_file()}
    result = build(source, policy, path)
    assert result["status"] == "VERIFIED"
    assert result["rows"] == {"train": 2, "validation": 2, "test": 2}
    assert result["service_id_mapping_verified"] is False
    assert {str(p): sha256(p) for p in source.folder.rglob("*") if p.is_file()} == hashes
    assert read_json(path / "preprocessing.json")["fit_rows"] == 2
    assert {t["y_alert_proxy"] for t in read_json(path / "targets.json")} == {0, 1}
    with pytest.raises(FileExistsError):
        build(source, policy, path)
    with pytest.raises(ValueError, match="row budget"):
        load_rows(source, max_source_rows=14)
    with pytest.raises(ValueError, match="regenerate"):
        verify_dataset(
            path, source, replace(policy, min_counterparties=3), execution_scope="FIXTURE_ONLY"
        )
    # 파일 해시가 자체적으로 맞아도 X의 의미적 변조를 숨길 수 없어야 합니다.
    xfile = path / "X_test.parquet"
    x = pq.read_table(xfile)
    pq.write_table(x.set_column(0, x.column_names[0], pa.array([999.0] * len(x))), xfile)
    manifest = read_json(path / "manifest.json")
    manifest["files"][xfile.name] = sha256(xfile)
    manifest.pop("dataset_fingerprint")
    manifest["dataset_fingerprint"] = fingerprint(manifest)
    write_json(path / "manifest.json", manifest)
    with pytest.raises(ValueError, match="alignment"):
        verify_dataset(path, source, policy, execution_scope="FIXTURE_ONLY")


def test_split_overlap_fails_and_null_proxy_is_not_negative(source, policy, tmp_path):
    with pytest.raises(ValueError, match="all three"):
        build_dataset(
            source,
            policy,
            tmp_path / "bad",
            train_before=datetime(2023, 1, 1),
            test_from=datetime(2024, 1, 1),
            execution_scope="FIXTURE_ONLY",
        )
    labels_path = source.part_path("labels", 0)
    labels = pq.read_table(labels_path).to_pylist()
    labels[5]["source_positive"] = None
    pq.write_table(pa.Table.from_pylist(labels), labels_path)
    source.labels["parts"][0]["files"][0]["sha256"] = sha256(labels_path)
    write_json(labels_path.parent / "manifest.json", source.labels)
    path = tmp_path / "nullable"
    build(source, policy, path)
    targets = read_json(path / "targets.json")
    row = next(t for t in targets if t["missing_label_count"])
    assert row["y_alert_proxy"] is None
    assert row["target_status"] == "UNRESOLVED_EVIDENCE_LABEL"


def test_train_saved_reload_scope_policy_and_feature_order(source, policy, tmp_path):
    data, model = tmp_path / "data", tmp_path / "model"
    build(source, policy, data)
    result = train_evaluate(
        data,
        source,
        policy,
        model,
        execution_scope="FIXTURE_ONLY",
        model_version="fixture-baseline",
        support_policy=synthetic_support(),
        k=1,
        iterations=100,
    )
    assert result["status"] == "VERIFIED" and result["deployed"] is False
    assert result["execution_scope"] == "FIXTURE_ONLY"
    assert verify_model(model, data, source, policy, execution_scope="FIXTURE_ONLY") == result
    raw = pq.read_table(data / "raw.parquet")
    scores = predict_raw(model, raw, data, source, policy, execution_scope="FIXTURE_ONLY")
    assert len(scores) == 6 and np.isfinite(scores).all()
    with pytest.raises(ValueError, match="allowlist"):
        predict_raw(
            model,
            raw.select(list(reversed(FEATURE_NAMES))),
            data,
            source,
            policy,
            execution_scope="FIXTURE_ONLY",
        )
    with pytest.raises(ValueError):
        verify_model(model, data, source, policy, execution_scope="CONFIRMED")
    with pytest.raises(ValueError, match="regenerate"):
        verify_model(
            model,
            data,
            source,
            replace(policy, policy_version="changed"),
            execution_scope="FIXTURE_ONLY",
        )
    with pytest.raises(FileExistsError):
        train_evaluate(
            data,
            source,
            policy,
            model,
            execution_scope="FIXTURE_ONLY",
            model_version="fixture-new",
            support_policy=synthetic_support(),
            k=1,
        )


def test_metrics_ties_and_preprocessing_fit_guard():
    result = metrics(np.array([1, 0]), np.array([0.5, 0.5]), 0.5, 1)
    assert result["average_precision"] == 0.5
    assert result["recall"] == 1 and result["precision"] == 0.5
    assert metrics(np.array([0, 0]), np.array([0.1, 0.2]), 0.5, 1)["recall"] is None
    with pytest.raises(ValueError, match="train only"):
        AlertPreprocessor().partial_fit(pa.table({}), split="test")


def test_cli_roundtrip(source, policy, tmp_path, capsys):
    path = tmp_path / "policy.json"
    write_json(path, asdict(policy))
    support = tmp_path / "support.json"
    write_json(support, synthetic_support())
    common = [
        "--assembly-index",
        str(tmp_path / "index.json"),
        "--source",
        source.id,
        "--policy",
        str(path),
        "--scope",
        "FIXTURE_ONLY",
    ]
    dataset, model = tmp_path / "cli-data", tmp_path / "cli-model"
    assert (
        main(
            [
                "prepare",
                *common,
                "--output",
                str(dataset),
                "--train-before",
                "2022-01-02",
                "--test-from",
                "2022-01-04",
            ]
        )
        == 0
    )
    assert main(["verify", *common, "--dataset", str(dataset)]) == 0
    assert (
        main(
            [
                "train",
                *common,
                "--dataset",
                str(dataset),
                "--output",
                str(model),
                "--model-version",
                "fixture-cli",
                "--support-policy",
                str(support),
                "--k",
                "1",
            ]
        )
        == 0
    )
    assert main(["verify-model", *common, "--dataset", str(dataset), "--model", str(model)]) == 0
    outputs = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert all(
        o["status"] == "VERIFIED" and o["execution_scope"] == "FIXTURE_ONLY" for o in outputs
    )


def test_policy_change_requires_descendant_rebuild_and_missing_source_fails(
    source, policy, tmp_path
):
    path = tmp_path / "data"
    build(source, policy, path)
    assert (
        regeneration_required(path, source, policy, execution_scope="FIXTURE_ONLY")["status"]
        == "CURRENT"
    )
    changed = regeneration_required(
        path, source, replace(policy, policy_version="fixture-v2"), execution_scope="FIXTURE_ONLY"
    )
    assert changed["status"] == "REGENERATE"
    assert changed["regenerate"] == ["alerts", "features", "preprocessing", "model", "evaluation"]
    assert "rule_config_hash" in changed["changed_fields"]
    assert (
        regeneration_required(path, source, R02Policy(), execution_scope="CONFIRMED")["status"]
        == "BLOCKED"
    )
    tx_path = source.part_path("transactions", 0)
    with tx_path.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="changed"):
        verify_dataset(path, source, policy, execution_scope="FIXTURE_ONLY")


def test_test_labels_cannot_select_weights_threshold_or_preprocessing(source, policy, tmp_path):
    first, first_model = tmp_path / "first", tmp_path / "first-model"
    build(source, policy, first)
    train_evaluate(
        first,
        source,
        policy,
        first_model,
        execution_scope="FIXTURE_ONLY",
        model_version="fixture-a",
        support_policy=synthetic_support(),
        k=1,
        iterations=100,
    )
    original = read_json(first_model / "model.json")
    labels_path = source.part_path("labels", 0)
    labels = pq.read_table(labels_path).to_pylist()
    # 표본 기준과 train/validation 전체를 보존하면서 test 경보 클래스만 바꿉니다.
    for row in labels[10:]:
        row["source_positive"] = int(row["source_row_id"] >= 13)
    pq.write_table(pa.Table.from_pylist(labels), labels_path)
    source.labels["parts"][0]["files"][0]["sha256"] = sha256(labels_path)
    write_json(labels_path.parent / "manifest.json", source.labels)
    second, second_model = tmp_path / "second", tmp_path / "second-model"
    build(source, policy, second)
    train_evaluate(
        second,
        source,
        policy,
        second_model,
        execution_scope="FIXTURE_ONLY",
        model_version="fixture-b",
        support_policy=synthetic_support(),
        k=1,
        iterations=100,
    )
    changed = read_json(second_model / "model.json")
    for key in (
        "weights",
        "intercept",
        "recommended_threshold",
        "feature_columns",
        "preprocessing_sha256",
    ):
        assert original[key] == changed[key]
    assert original["dataset_fingerprint"] != changed["dataset_fingerprint"]


def test_same_counterparty_twice_is_not_two_and_expiry_reactivates(source, policy):
    rows = load_rows(source, max_source_rows=100)
    # 원천 식별 관계 전체를 보존하면서 P의 송신자 셋을 하나의 고유 상대방으로 만듭니다.
    changed = [dict(r, from_account_key="same") if r["to_account_key"] == "P" else r for r in rows]
    result = generate_r02(
        changed,
        source_id=source.id,
        source_release=source.metadata["source_release"],
        policy=policy,
        execution_scope="FIXTURE_ONLY",
    )
    assert len(result) == 3 and all(a.account_key == "Q" for a in result)
    # 조건이 참인 상태에서 한 분을 추가해도 경보가 중복되지 않아야 합니다.
    extra = dict(
        rows[0],
        source_row_id=15,
        transaction_uid=source.metadata["source_release"] + ":15",
        ts=rows[0]["ts"] + timedelta(minutes=1),
    )
    continued = generate_r02(
        rows + [extra],
        source_id=source.id,
        source_release=source.metadata["source_release"],
        policy=policy,
        execution_scope="FIXTURE_ONLY",
    )
    assert len(continued) == 6


def test_model_file_tampering_is_rejected(source, policy, tmp_path):
    dataset, model = tmp_path / "data", tmp_path / "model"
    build(source, policy, dataset)
    train_evaluate(
        dataset,
        source,
        policy,
        model,
        execution_scope="FIXTURE_ONLY",
        model_version="fixture",
        support_policy=synthetic_support(),
        k=1,
        iterations=50,
    )
    model_json = read_json(model / "model.json")
    model_json["weights"][0] += 1
    write_json(model / "model.json", model_json)
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_model(model, dataset, source, policy, execution_scope="FIXTURE_ONLY")


def test_unverified_or_excerpt_source_is_not_an_offline_training_source(source):
    unverified = replace(source, metadata=dict(source.metadata, status="UNVERIFIED"))
    with pytest.raises(ValueError, match="Verified complete"):
        load_rows(unverified, max_source_rows=100)
    excerpt = replace(source, transactions=dict(source.transactions, scope="SERVICE_SELECTION"))
    with pytest.raises(ValueError, match="never service excerpts"):
        load_rows(excerpt, max_source_rows=100)
