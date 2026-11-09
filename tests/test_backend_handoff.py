"""합성 ID 내보내기만 검사하며, 실제 백엔드 인계를 입증하지 않습니다."""

import copy
import hashlib
import json
from dataclasses import asdict, replace

import pytest

from aml_ml import backend_handoff
from aml_ml.offline_rules import R02Policy


@pytest.fixture
def handoff():
    return {
        "id_namespace": "BACKEND_OPS",
        "export_reference": "SYNTHETIC_FIXTURE_NOT_A_REAL_BACKEND_EXPORT",
        "source_row_number_base": 1,
        "mapping": [
            {
                "transaction_id": 9007199254740993,
                "source_file_hash": "a" * 64,
                "source_row_number": 1,
            },
            {
                "transaction_id": 2**63 - 1,
                "source_file_hash": "a" * 64,
                "source_row_number": 15,
            },
        ],
        "alerts": [
            {
                "alert_id": "fixture-alert",
                "rule_id": "R02_FAN_IN_24H",
                "rule_version": "fixture-r02",
                "account_key": "fixture-account",
                "window_start": "2022-01-01T00:00:00",
                "window_end": "2022-01-02T00:00:00",
                "reference_ts": "2022-01-02T00:00:00",
                "transaction_ids": [2**63 - 1, 9007199254740993],
            }
        ],
    }


def test_original_first_last_rows_and_bigint_are_preserved(source, handoff):
    before = copy.deepcopy(handoff)
    result = backend_handoff.validate_handoff(handoff, source)
    linked = result["alerts"][0]
    assert linked["transaction_ids"] == [2**63 - 1, 9007199254740993]
    assert linked["transaction_uids"] == [
        source.metadata["source_release"] + ":14",
        source.metadata["source_release"] + ":0",
    ]
    assert "FULL_RULE_EVIDENCE_COMPLETENESS" in result["not_verified"]
    assert linked["evaluation_minute"] == "2022-01-01T23:59:00"
    assert linked["batch_date"] == "2022-01-01"
    assert linked["currency_evidence_status"] == "NOT_PROVIDED"
    assert result["policy_binding"]["status"] == "NOT_PROVIDED"
    assert result["policy_binding"]["config_hash"] is None
    assert handoff == before


@pytest.mark.parametrize("value", [True, 0, -1, 1.0, "101", 2**63, None])
def test_rejects_non_bigint_mapping_ids(source, handoff, value):
    handoff["mapping"][0]["transaction_id"] = value
    with pytest.raises(ValueError, match="BIGINT"):
        backend_handoff.validate_handoff(handoff, source)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_file_hash", "b" * 64, "source_file_hash"),
        ("source_row_number", 0, "row number"),
        ("source_row_number", 16, "row number"),
        ("source_row_number", True, "row number"),
        ("transaction_uid", "ibm_hi_small:" + "a" * 64 + ":1", "learning UID"),
    ],
)
def test_rejects_wrong_source_and_off_by_one_uid(source, handoff, field, value, message):
    handoff["mapping"][0][field] = value
    with pytest.raises(ValueError, match=message):
        backend_handoff.validate_handoff(handoff, source)


@pytest.mark.parametrize("field", ["transaction_id", "source_row_number"])
def test_mapping_is_one_to_one_in_both_directions(source, handoff, field):
    handoff["mapping"][1][field] = handoff["mapping"][0][field]
    with pytest.raises(ValueError, match="one-to-one"):
        backend_handoff.validate_handoff(handoff, source)


@pytest.mark.parametrize("evidence", [[], [1], [9007199254740993] * 2, [True], [1.0]])
def test_missing_duplicate_and_invalid_evidence_fail_closed(source, handoff, evidence):
    handoff["alerts"][0]["transaction_ids"] = evidence
    with pytest.raises(ValueError):
        backend_handoff.validate_handoff(handoff, source)


def test_row_base_is_explicit_and_candidate_namespace_rejected(source, handoff):
    handoff["source_row_number_base"] = 0
    for record in handoff["mapping"]:
        record["source_row_number"] -= 1
    assert backend_handoff.validate_handoff(handoff, source)["mapping_rows"] == 2
    handoff["source_row_number_base"] = None
    with pytest.raises(ValueError, match="number_base"):
        backend_handoff.validate_handoff(handoff, source)
    handoff["id_namespace"] = "SOURCE_CANDIDATE"
    with pytest.raises(ValueError, match="BACKEND_OPS"):
        backend_handoff.validate_handoff(handoff, source)


def test_cli_saves_exact_ids_and_refuses_overwrite_or_invalid_export(
    source, handoff, tmp_path, monkeypatch
):
    monkeypatch.setattr(backend_handoff, "load_index", lambda *_: [source])
    path, output = tmp_path / "handoff.json", tmp_path / "result.json"
    path.write_text(json.dumps(handoff), encoding="utf-8")
    args = [
        "--assembly-index",
        "synthetic-index.json",
        "--source",
        "ibm_hi_small",
        "--handoff",
        str(path),
        "--output",
        str(output),
    ]
    assert backend_handoff.main(args) == 0
    saved = output.read_bytes()
    assert json.loads(saved)["alerts"][0]["transaction_ids"][-1] == 9007199254740993
    with pytest.raises(SystemExit) as error:
        backend_handoff.main(args)
    assert error.value.code == 2 and output.read_bytes() == saved
    invalid_output = tmp_path / "must-not-exist.json"
    args[-1] = str(invalid_output)
    with pytest.raises(SystemExit):
        backend_handoff.main(args + ["--max-input-bytes", "1"])
    assert not invalid_output.exists()
    handoff["mapping"][0]["source_file_hash"] = "b" * 64
    path.write_text(json.dumps(handoff), encoding="utf-8")
    with pytest.raises(SystemExit):
        backend_handoff.main(args)
    assert not invalid_output.exists()


def test_duplicate_alerts_and_stale_source_manifest_are_rejected(source, handoff):
    handoff["alerts"].append(copy.deepcopy(handoff["alerts"][0]))
    with pytest.raises(ValueError, match="Duplicate alert"):
        backend_handoff.validate_handoff(handoff, source)
    handoff["alerts"].pop()
    source.metadata["input_rows"] += 1
    with pytest.raises(ValueError, match="complete source"):
        backend_handoff.validate_handoff(handoff, source)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("rule_version", None),
        ("rule_version", 1),
        ("rule_id", "R01_FAN_OUT_24H"),
        ("alert_id", 1),
        ("window_start", "2022-01-01T00:00:01"),
        ("window_end", "2022-01-02T00:00:01"),
        ("reference_ts", "2022-01-01T23:59:00"),
        ("reference_ts", "2022-01-02T00:00:00Z"),
    ],
)
def test_r02_versions_and_b7_window_contract(source, handoff, key, value):
    handoff["alerts"][0][key] = value
    with pytest.raises(ValueError):
        backend_handoff.validate_handoff(handoff, source)


def test_currency_evidence_union_is_checked_without_money_conversion(source, handoff):
    ids = handoff["alerts"][0]["transaction_ids"]
    # 공유 참조는 그룹 간 합집합으로 처리하며, 최상위 ID를 추가로 세지 않습니다.
    # 거래의 원천 통화가 아니라 ID의 일관성을 검사합니다.
    handoff["alerts"][0]["rule_facts"] = {
        "amounts_by_currency": [
            {"currency": "USD", "transaction_ids": ids[:1], "total_amount": "0.00000001"},
            {"currency": "Unregistered Raw Unit", "transaction_ids": ids, "total_amount": "7.5"},
        ]
    }
    before = copy.deepcopy(handoff)
    result = backend_handoff.validate_handoff(handoff, source)
    assert result["alerts"][0]["currency_evidence_status"] == "MATCHES_TOP_LEVEL_IDS"
    assert result["alerts"][0]["transaction_ids"] == ids
    assert "CURRENCY_AMOUNTS_AND_TRANSACTION_CURRENCY_MEMBERSHIP" in result["not_verified"]
    assert handoff == before


@pytest.mark.parametrize(
    "groups",
    [
        None,
        {},
        [],
        [None],
        [{"currency": "USD", "transaction_ids": [9007199254740993]}],
        [{"currency": "USD", "transaction_ids": [2**63 - 1, 9007199254740993, 999]}],
        [{"currency": "USD", "transaction_ids": [2**63 - 1, 9007199254740993] * 2}],
        [{"currency": "USD", "transaction_ids": [True]}],
        [{"currency": "USD", "transaction_ids": [1.0]}],
        [{"currency": "USD", "transaction_ids": "9007199254740993"}],
        [{"currency": " ", "transaction_ids": [2**63 - 1, 9007199254740993]}],
    ],
)
def test_supplied_currency_evidence_never_silently_drops_bad_references(source, handoff, groups):
    handoff["alerts"][0]["rule_facts"] = {"amounts_by_currency": groups}
    with pytest.raises(ValueError):
        backend_handoff.validate_handoff(handoff, source)


def test_absent_breakdown_is_not_verified_and_malformed_facts_are_rejected(source, handoff):
    handoff["alerts"][0]["rule_facts"] = {"distinct_senders": 2}
    result = backend_handoff.validate_handoff(handoff, source)
    assert result["alerts"][0]["currency_evidence_status"] == "NOT_PROVIDED"
    handoff["alerts"][0]["rule_facts"] = None
    with pytest.raises(ValueError, match="rule_facts"):
        backend_handoff.validate_handoff(handoff, source)


def test_received_policy_binds_version_threshold_hash_and_reference(source, handoff, policy):
    confirmed = replace(policy, decision_status="CONFIRMED", rule_version_origin="BACKEND_EXPORT")
    before = copy.deepcopy(handoff)
    result = backend_handoff.validate_handoff(handoff, source, policy=confirmed)
    assert result["policy_binding"] == {
        "status": "MATCHES_SUPPLIED_POLICY",
        "checks": ["RULE_VERSION_EQUALITY"],
        "rule_version": confirmed.rule_version,
        "rule_version_origin": "BACKEND_EXPORT",
        "policy_version": confirmed.policy_version,
        "min_counterparties": confirmed.min_counterparties,
        "config_hash": confirmed.config_hash,
        "confirmation_reference": confirmed.confirmation_reference,
    }
    assert "RULE_VERSION_AND_CONFIG_AGREEMENT" in result["not_verified"]
    assert "BACKEND_THRESHOLD_ACTUALLY_APPLIED" in result["not_verified"]
    assert "FULL_RULE_EVIDENCE_COMPLETENESS" in result["not_verified"]
    assert handoff == before


def test_received_policy_checks_every_alert_version_without_normalizing(source, handoff, policy):
    confirmed = replace(policy, decision_status="CONFIRMED", rule_version_origin="BACKEND_EXPORT")
    other = copy.deepcopy(handoff["alerts"][0])
    other.update(alert_id="another", rule_version="fixture-r02 ")
    handoff["alerts"].append(other)
    with pytest.raises(ValueError, match=r"alerts\[1\].rule_version"):
        backend_handoff.validate_handoff(handoff, source, policy=confirmed)


def test_pending_or_fixture_policy_cannot_bind_backend_samples(source, handoff, policy):
    with pytest.raises(ValueError, match="Pending/fixture"):
        backend_handoff.validate_handoff(handoff, source, policy=policy)
    incomplete = replace(
        policy,
        decision_status="CONFIRMED",
        rule_version_origin="BACKEND_EXPORT",
        min_counterparties=None,
    )
    with pytest.raises(ValueError, match="min_counterparties"):
        backend_handoff.validate_handoff(handoff, source, policy=incomplete)


def test_cli_pins_received_policy_bytes_and_preserves_report_on_rejection(
    source, handoff, policy: R02Policy, tmp_path, monkeypatch
):
    confirmed = replace(policy, decision_status="CONFIRMED", rule_version_origin="BACKEND_EXPORT")
    path, output = tmp_path / "handoff.json", tmp_path / "report.json"
    policy_path = tmp_path / "received-policy.json"
    policy_path.write_text(json.dumps(asdict(confirmed)), encoding="utf-8")
    policy_bytes = policy_path.read_bytes()
    path.write_text(json.dumps(handoff), encoding="utf-8")
    monkeypatch.setattr(backend_handoff, "load_index", lambda *_: [source])
    args = [
        "--assembly-index",
        "synthetic-index.json",
        "--source",
        "ibm_hi_small",
        "--handoff",
        str(path),
        "--policy",
        str(policy_path),
        "--output",
        str(output),
    ]
    assert backend_handoff.main(args) == 0
    saved = output.read_bytes()
    binding = json.loads(saved)["policy_binding"]
    assert binding["policy_file_sha256"] == hashlib.sha256(policy_bytes).hexdigest()
    assert binding["config_hash"] == confirmed.config_hash
    assert binding["policy_path"] == str(policy_path.resolve())
    assert policy_path.read_bytes() == policy_bytes
    handoff["alerts"][0]["rule_version"] = "wrong-version"
    path.write_text(json.dumps(handoff), encoding="utf-8")
    with pytest.raises(SystemExit):
        backend_handoff.main(args)
    assert output.read_bytes() == saved
    rejected = tmp_path / "rejected.json"
    args[-1] = str(rejected)
    with pytest.raises(SystemExit) as error:
        backend_handoff.main(args)
    assert error.value.code == 2 and not rejected.exists()


def test_handoff_pending_policy_fails_before_export_or_source_read(tmp_path, monkeypatch):
    policy_path = tmp_path / "pending.json"
    policy_path.write_text(json.dumps(asdict(R02Policy())), encoding="utf-8")

    def forbidden(*args, **kwargs):
        raise AssertionError("Pending policy must not read source")

    monkeypatch.setattr(backend_handoff, "load_index", forbidden)
    with pytest.raises(SystemExit) as error:
        backend_handoff.main(
            [
                "--assembly-index",
                "must-not-be-read.json",
                "--source",
                "ibm_hi_small",
                "--handoff",
                str(tmp_path / "nonexistent.json"),
                "--policy",
                str(policy_path),
                "--output",
                str(tmp_path / "report.json"),
            ]
        )
    assert error.value.code == 2
    assert not (tmp_path / "report.json").exists()


def test_internal_rule_version_is_rejected_even_without_policy(source, handoff):
    handoff["alerts"][0]["rule_version"] = "mldb-offline-r02-v1"
    with pytest.raises(ValueError, match="offline spec version"):
        backend_handoff.validate_handoff(handoff, source)
