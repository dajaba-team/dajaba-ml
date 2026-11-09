"""SCRUM-77 계약 회귀 검사이며, 운영 설정은 합성 행에만 적용합니다."""

from dataclasses import replace
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pytest
from offline_test_data import assert_same_artifacts, rewrite_fixture, source_values

from aml_ml import backend_handoff
from aml_ml.dataset_quality import DatasetQuality
from aml_ml.offline_features import FEATURE_NAMES
from aml_ml.offline_pipeline import build_dataset, build_reference_dataset, verify_dataset
from aml_ml.offline_rules import fingerprint, generate_r02, load_policy
from aml_ml.storage import read_json, sha256, write_json


def spec_policy():
    path = Path(__file__).resolve().parent / "fixtures" / "legacy" / "offline_r02.spec.json"
    return load_policy(path)[0]


def test_confirmed_offline_spec_uses_ten_without_claiming_a_backend_version():
    policy = spec_policy()
    policy.require("CONFIRMED")
    assert policy.min_counterparties == 10
    assert policy.rule_version == "mldb-offline-r02-v1"
    assert policy.rule_version_origin == "OFFLINE_SPEC"
    assert policy.readiness("CONFIRMED")["policy"]["rule_version_origin"] == "OFFLINE_SPEC"
    with pytest.raises(ValueError, match="BACKEND_EXPORT"):
        policy.require_backend()
    with pytest.raises(ValueError, match="internal version"):
        replace(policy, rule_version_origin="BACKEND_EXPORT").require("CONFIRMED")
    with pytest.raises(ValueError, match="origin"):
        replace(policy, rule_version_origin=None).require("CONFIRMED")
    assert replace(policy, min_counterparties=11).config_hash != policy.config_hash


def test_offline_spec_cannot_be_used_for_backend_comparison(source):
    # 백엔드 본문·원천 확인 전에 정책 출처를 거부합니다.
    with pytest.raises(ValueError, match="BACKEND_EXPORT"):
        backend_handoff.validate_handoff({}, source, policy=spec_policy())


def test_ten_peer_threshold_boundary_is_label_blind(source):
    original, _ = source_values(source)
    release = source.metadata["source_release"]
    rows = [
        dict(
            original[0],
            source_row_id=i,
            transaction_uid=f"{release}:{i}",
            ts=datetime(2022, 1, 1, 12, 0, i),
            from_account_key=f"peer-{i}",
        )
        for i in range(10)
    ]

    def generate(selected):
        return generate_r02(
            selected,
            source_id=source.id,
            source_release=release,
            policy=spec_policy(),
            execution_scope="CONFIRMED",
        )

    assert generate(rows[:9]) == []
    alerts = generate(rows)
    assert len(alerts) == 1 and len(alerts[0].transaction_uids) == 10
    assert alerts[0].reference_ts == datetime(2022, 1, 1, 12, 1)


def test_spec_dataset_both_paths_quality_and_no_backend_ids(source, tmp_path):
    original, _ = source_values(source)
    rows, labels = [], []
    for day in (1, 3, 5):
        for account in ("P", "Q"):
            # 서로 다른 원천 이력으로 train 피처의 변동을 유지하고 라벨은 X에서 제외합니다.
            amount = Decimal("10") if account == "P" else Decimal("1")
            for i in range(10):
                rows.append(
                    dict(
                        original[0],
                        ts=datetime(2022, 1, day, 12, 0, i),
                        from_account_key=f"sender-{i}",
                        to_account_key=account,
                        amount_paid=amount,
                        amount_received=amount,
                    )
                )
                labels.append(int(account == "P" and i == 0))
    rewrite_fixture(source, rows, labels, part_rows=7)
    reference, streamed = tmp_path / "reference", tmp_path / "streamed"
    policy = spec_policy()
    build_reference_dataset(
        source,
        policy,
        reference,
        train_before=datetime(2022, 1, 2),
        test_from=datetime(2022, 1, 4),
        execution_scope="CONFIRMED",
    )
    result = build_dataset(
        source,
        policy,
        streamed,
        partitions=7,
        read_batch_rows=3,
        train_before=datetime(2022, 1, 2),
        test_from=datetime(2022, 1, 4),
        execution_scope="CONFIRMED",
    )
    assert result["status"] == "VERIFIED"
    assert_same_artifacts(reference, streamed)
    manifest = read_json(streamed / "manifest.json")
    assert manifest["contract"]["rule_policy"]["rule_version_origin"] == "OFFLINE_SPEC"
    assert manifest["service_id_mapping_verified"] is False
    report = read_json(streamed / "quality.json")
    for split, counts in report["splits"].items():
        assert counts["alerts"] == manifest["rows"][split] == 2
        assert counts["proxy_positive"] == counts["proxy_negative"] == 1
        assert counts["proxy_unresolved"] == 0
        assert counts["evidence_references"] == 20
    assert report["warnings"] == []
    assert report["model_training_executed"] is False


def target(truth: int | None = 1, missing: int = 0) -> dict:
    return {
        "split": "train",
        "source_id": "ibm_hi_small",
        "reference_ts": "2022-01-01T12:01:00",
        "y_alert_proxy": truth,
        "evidence_count": 10,
        "missing_label_count": missing,
    }


def test_quality_preserves_unknown_labels_and_raw_nulls_without_imputation():
    quality = DatasetQuality("ibm_hi_small")
    feature: dict[str, float | None] = {name: 0.0 for name in FEATURE_NAMES}
    feature["account_in_amount_log1p_24h"] = None
    feature["account_money_unit_unknown"] = 1.0
    quality.add(target(None, 1), feature)
    quality.add(target(0), feature)
    report = quality.report()
    train = report["splits"]["train"]
    assert train["proxy_unresolved"] == train["proxy_negative"] == 1
    assert train["proxy_positive"] == 0
    assert train["evidence_references"] == 20 and train["missing_label_references"] == 1
    assert train["raw_feature_null_rates"]["account_in_amount_log1p_24h"] == 1.0
    assert train["flagged_alerts"]["account_money_unit_unknown"] == 2
    assert report["splits"]["test"]["raw_feature_null_rates"][FEATURE_NAMES[0]] is None
    train["alerts"] = 999
    assert quality.report()["splits"]["train"]["alerts"] == 2
    assert feature["account_in_amount_log1p_24h"] is None


@pytest.mark.parametrize("truth,missing", [(None, 0), (1, 1), (True, 0), (2, 0), (0, -1)])
def test_quality_rejects_inconsistent_proxy_counts(truth, missing):
    with pytest.raises(ValueError, match="proxy/evidence"):
        DatasetQuality("ibm_hi_small").add(
            target(truth, missing), dict.fromkeys(FEATURE_NAMES, 0.0)
        )


@pytest.mark.parametrize("streaming", [False, True])
def test_quality_report_tampering_cannot_pass_source_replay(source, policy, tmp_path, streaming):
    folder = tmp_path / "data"
    builder = build_dataset if streaming else build_reference_dataset
    builder(
        source,
        policy,
        folder,
        train_before=datetime(2022, 1, 2),
        test_from=datetime(2022, 1, 4),
        execution_scope="FIXTURE_ONLY",
    )
    report = read_json(folder / "quality.json")
    report["splits"]["train"]["proxy_positive"] += 1
    write_json(folder / "quality.json", report)
    manifest = read_json(folder / "manifest.json")
    manifest["files"]["quality.json"] = sha256(folder / "quality.json")
    manifest.pop("dataset_fingerprint")
    manifest["dataset_fingerprint"] = fingerprint(manifest)
    write_json(folder / "manifest.json", manifest)
    with pytest.raises(ValueError, match="quality report|source replay"):
        verify_dataset(folder, source, policy, execution_scope="FIXTURE_ONLY")
