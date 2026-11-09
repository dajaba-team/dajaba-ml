"""무결성 전용 검사, 원천 최신성 및 유지된 심층 재연산의 회귀를 검사합니다."""

import hashlib
import json
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from offline_test_data import rewrite_fixture

from aml_ml import (
    offline_cli,
    offline_contract,
    offline_features,
    offline_pipeline,
    offline_streaming,
)
from aml_ml.offline_pipeline import (
    AlertPreprocessor,
    build_dataset,
    build_reference_dataset,
    contract,
    regeneration_required,
    validate_dataset_integrity,
    verify_dataset,
)
from aml_ml.offline_rules import R02Policy, fingerprint
from aml_ml.storage import SourceInput, read_json, sha256, write_json

SCOPE = "FIXTURE_ONLY"


def _build(source, policy, folder, *, streaming=False):
    builder = build_dataset if streaming else build_reference_dataset
    return builder(
        source,
        policy,
        folder,
        train_before=datetime(2022, 1, 2),
        test_from=datetime(2022, 1, 4),
        execution_scope=SCOPE,
    )


def _rehash(folder, artifact):
    manifest = read_json(folder / "manifest.json")
    manifest["files"][artifact] = sha256(folder / artifact)
    manifest.pop("dataset_fingerprint")
    manifest["dataset_fingerprint"] = fingerprint(manifest)
    write_json(folder / "manifest.json", manifest)


def _source_manifest_path(source, which):
    if which == "source":
        return source.folder / "source_manifest.json"
    key = "ml_transactions" if which == "transactions" else "ml_labels"
    return source.folder / source.metadata["datasets"][key]["path"] / "manifest.json"


def test_shared_contract_and_feature_exports_remain_compatible():
    for name in (
        "DATASET_FILES",
        "FIT_BATCH_ROWS",
        "SPLITS",
        "contract",
        "regeneration_required",
        "source_fingerprint",
        "validate_dataset_integrity",
        "validate_source",
    ):
        assert getattr(offline_pipeline, name) is getattr(offline_contract, name)
    for name in (
        "FEATURE_NAMES",
        "OFFLINE_FEATURE_VERSION",
        "AlertPreprocessor",
        "partition_features",
    ):
        assert getattr(offline_pipeline, name) is getattr(offline_features, name)
    assert offline_streaming.FILES is offline_contract.DATASET_FILES


@pytest.mark.parametrize(
    "helper",
    [
        "sliding_window.py",
        "storage.py",
        "offline_contract.py",
        "offline_features.py",
        "offline_account_stream.py",
        "offline_publish.py",
    ],
)
def test_helper_only_hash_change_requires_regeneration(
    source, policy, tmp_path, monkeypatch, helper
):
    saved = contract(source, policy, SCOPE, datetime(2022, 1, 2), datetime(2022, 1, 4))
    assert helper in saved["code_sha256"]
    write_json(tmp_path / "manifest.json", {"contract": saved})
    original_sha256 = offline_contract.sha256

    def changed_helper(path):
        if Path(path).name == helper:
            return "0" * 64
        return original_sha256(path)

    # 실제 설치된 도우미는 수정하지 않고 관측되는 지문만 바꿉니다.
    monkeypatch.setattr(offline_contract, "sha256", changed_helper)
    result = regeneration_required(tmp_path, source, policy, execution_scope=SCOPE)
    assert result["status"] == "REGENERATE"
    assert result["changed_fields"] == ["code_sha256"]
    assert result["regenerate"] == ["alerts", "features", "preprocessing", "model", "evaluation"]


def _pending_r02_configuration():
    path = Path(__file__).resolve().parent / "fixtures" / "legacy" / "offline_r02.pending.json"
    return read_json(path)


def test_pending_configuration_records_confirmed_semantics_and_blocks_official_execution():
    pending = _pending_r02_configuration()
    confirmed_semantics = {
        "window_policy": "B7_E_PLUS_ONE_MINUTE",
        "recurrence_policy": "PREVIOUS_CLOCK_MINUTE_FALSE_TO_TRUE",
        "evidence_policy": "ALL_INCOMING_IN_WINDOW",
        "counterparty_policy": "EXACT_ACCOUNT_KEYS_ACROSS_CURRENCIES",
        "self_transfer_policy": "EXCLUDE_FLAGGED_AND_SAME_ACCOUNT",
        "confirmation_reference": "B7_POLICY_REGRESSION_FIXTURE_NOT_OPERATIONAL_APPROVAL",
    }
    for key, expected in confirmed_semantics.items():
        assert pending[key] == expected
    # 백엔드 룰 버전과 별개인 ML 소유 재현성 식별자입니다.
    assert pending["policy_version"] == "mldb-r02-policy-v1"
    assert pending["decision_status"] == "UNCONFIRMED"
    for key in ("rule_version", "min_counterparties"):
        assert pending[key] is None
    pending_policy = R02Policy(**pending)
    report = pending_policy.readiness("CONFIRMED")
    assert {item["field"] for item in report["issues"]} == {
        "decision_status",
        "rule_version",
        "min_counterparties",
        "rule_version_origin",
    }
    with pytest.raises(ValueError, match="Pending/fixture"):
        pending_policy.require("CONFIRMED")
    # 상태만 바꿔도 누락된 운영값이 기본값으로 채워지면 안 됩니다.
    with pytest.raises(ValueError, match="Explicit rule/policy versions"):
        replace(pending_policy, decision_status="CONFIRMED").require("CONFIRMED")


@pytest.mark.parametrize("missing", ["rule_version", "policy_version", "min_counterparties"])
def test_confirmed_status_never_supplies_missing_operational_values(missing, policy: R02Policy):
    # 메모리 안의 합성 값만 사용하며 운영 설정에는 쓰지 않습니다.
    confirmed = replace(policy, decision_status="CONFIRMED", rule_version_origin="BACKEND_EXPORT")
    confirmed.require("CONFIRMED")
    incomplete = {
        "rule_version": replace(confirmed, rule_version=None),
        "policy_version": replace(confirmed, policy_version=None),
        "min_counterparties": replace(confirmed, min_counterparties=None),
    }
    with pytest.raises(ValueError):
        incomplete[missing].require("CONFIRMED")


def test_current_pending_cli_stops_before_source_loading_and_output(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parent / "fixtures" / "legacy" / "offline_r02.pending.json"
    output = tmp_path / "must-not-be-created"

    def forbidden(*args, **kwargs):
        raise AssertionError("Pending policy must stop before loading data or generating output")

    monkeypatch.setattr(offline_cli, "load_index", forbidden)
    monkeypatch.setattr(offline_cli, "build_dataset", forbidden)
    assert (
        offline_cli.main(
            [
                "prepare",
                "--assembly-index",
                "must-not-be-read.json",
                "--source",
                "ibm_hi_small",
                "--policy",
                str(path),
                "--scope",
                "CONFIRMED",
                "--output",
                str(output),
                "--train-before",
                "2022-01-02T00:00:00",
                "--test-from",
                "2022-01-04T00:00:00",
            ]
        )
        == 2
    )
    assert not output.exists()


def test_policy_diagnostics_report_all_blockers_without_changing_values(policy: R02Policy):
    incomplete = replace(
        policy,
        decision_status="UNCONFIRMED",
        rule_version=None,
        min_counterparties=None,
        evidence_policy=None,
    )
    before = asdict(incomplete)
    report = incomplete.readiness("CONFIRMED")
    assert report["status"] == "POLICY_CONFIG_NOT_READY"
    assert {item["field"] for item in report["issues"]} == {
        "decision_status",
        "rule_version",
        "min_counterparties",
        "evidence_policy",
        "rule_version_origin",
    }
    assert report["policy"] is None and report["config_hash"] is None
    assert asdict(incomplete) == before
    with pytest.raises(ValueError, match="min_counterparties"):
        incomplete.require("CONFIRMED")


@pytest.mark.parametrize("value", [None, True, 0, -1, 2.0, "2"])
def test_policy_diagnostics_never_coerce_threshold(policy, value):
    changed = replace(policy, min_counterparties=value)
    assert changed.diagnostics(SCOPE) == [
        {
            "field": "min_counterparties",
            "code": "REQUIRED_POSITIVE_INTEGER",
            "message": "Confirmed minimum counterparty count required; no default",
        }
    ]


def test_policy_ready_scope_and_hash_do_not_claim_backend_readiness(policy: R02Policy):
    report = policy.readiness(SCOPE)
    assert report["status"] == "POLICY_CONFIG_READY"
    assert report["execution_scope"] == "FIXTURE_ONLY"
    assert report["config_hash"] == policy.config_hash
    assert report["policy"] == asdict(policy)
    assert "BACKEND_POLICY_ACTUALLY_APPLIED" in report["not_verified"]
    assert policy.readiness("CONFIRMED")["status"] == "POLICY_CONFIG_NOT_READY"
    assert any(item["field"] == "execution_scope" for item in policy.diagnostics("EXPERIMENT"))


@pytest.mark.parametrize("confirmed", [False, True])
def test_check_policy_reads_only_configuration(
    policy: R02Policy, tmp_path, monkeypatch, capsys, confirmed
):
    selected = (
        replace(policy, decision_status="CONFIRMED", rule_version_origin="BACKEND_EXPORT")
        if confirmed
        else R02Policy()
    )
    path = tmp_path / "received-policy.json"
    write_json(path, asdict(selected))
    original = path.read_bytes()
    output = tmp_path / "no-data-output"

    def forbidden(*args, **kwargs):
        raise AssertionError("Policy readiness must not access source or run the pipeline")

    monkeypatch.setattr(offline_cli, "load_index", forbidden)
    monkeypatch.setattr(offline_cli, "build_dataset", forbidden)
    monkeypatch.setattr(offline_cli, "_training_module", forbidden)
    code = offline_cli.main(
        ["check-policy", "--policy", str(path), "--scope", "CONFIRMED", "--output", str(output)]
    )
    report = json.loads(capsys.readouterr().out)
    assert code == (0 if confirmed else 2)
    assert report["status"] == ("POLICY_CONFIG_READY" if confirmed else "POLICY_CONFIG_NOT_READY")
    assert report["policy_file_sha256"] == hashlib.sha256(original).hexdigest()
    assert path.read_bytes() == original and not output.exists()
    assert report["config_hash"] == (selected.config_hash if confirmed else None)


@pytest.mark.parametrize("content", ["[]", "{", '{"unknown_field": 1}'])
def test_check_policy_rejects_bad_configuration_without_source_io(
    tmp_path, monkeypatch, capsys, content
):
    path = tmp_path / "invalid-policy.json"
    path.write_text(content, encoding="utf-8")

    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid policy must not access source")

    monkeypatch.setattr(offline_cli, "load_index", forbidden)
    assert offline_cli.main(["check-policy", "--policy", str(path), "--scope", "CONFIRMED"]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "FAILED"


def test_execution_still_requires_source_arguments(policy: R02Policy, tmp_path, monkeypatch):
    path = tmp_path / "policy.json"
    write_json(path, asdict(policy))

    def forbidden(*args, **kwargs):
        raise AssertionError("Missing source arguments must stop before source I/O")

    monkeypatch.setattr(offline_cli, "load_index", forbidden)
    assert offline_cli.main(["prepare", "--policy", str(path), "--scope", SCOPE]) == 2


@pytest.mark.parametrize("streaming", [False, True])
def test_integrity_reads_every_source_part_without_replay_or_writes(
    source, policy, tmp_path, monkeypatch, streaming
):
    folder = tmp_path / "data"
    transactions = pq.read_table(source.part_path("transactions", 0)).to_pylist()
    labels = pq.read_table(source.part_path("labels", 0)).column("source_positive").to_pylist()
    rewrite_fixture(source, transactions, labels, part_rows=4)
    deep = _build(source, policy, folder, streaming=streaming)
    before = {p.name: sha256(p) for p in folder.iterdir()}
    observed = []
    original_part_path = SourceInput.part_path

    def tracked_part_path(self, which, index):
        observed.append((which, index))
        return original_part_path(self, which, index)

    def forbidden(*args, **kwargs):
        raise AssertionError("Integrity must not replay, decode Parquet or refit")

    monkeypatch.setattr(SourceInput, "part_path", tracked_part_path)
    monkeypatch.setattr(offline_pipeline, "_derive", forbidden)
    monkeypatch.setattr(offline_streaming, "build_streaming_dataset", forbidden)
    monkeypatch.setattr(offline_streaming, "verify_streaming_dataset", forbidden)
    monkeypatch.setattr(AlertPreprocessor, "partial_fit", forbidden)
    monkeypatch.setattr(pq, "read_table", forbidden)
    result = validate_dataset_integrity(folder, source, policy, execution_scope=SCOPE)
    assert result["status"] == "INTEGRITY_VERIFIED"
    assert result["dataset_fingerprint"] == deep["dataset_fingerprint"]
    assert result["manifest_sha256"] == deep["manifest_sha256"] == before["manifest.json"]
    assert observed == [
        (which, index)
        for which in ("transactions", "labels")
        for index in range(len(source.parts(which)))
    ]
    assert {p.name: sha256(p) for p in folder.iterdir()} == before


@pytest.mark.parametrize("which", ["source", "transactions", "labels"])
def test_source_manifest_changed_during_part_scan_is_rejected(
    source, policy, tmp_path, monkeypatch, which
):
    folder = tmp_path / "data"
    _build(source, policy, folder)
    original_part_path = SourceInput.part_path

    def change_manifest_during_scan(self, table, index):
        path = original_part_path(self, table, index)
        if table == "transactions" and index == 0:
            manifest_path = _source_manifest_path(source, which)
            manifest_path.write_bytes(manifest_path.read_bytes() + b"\n")
        return path

    monkeypatch.setattr(SourceInput, "part_path", change_manifest_during_scan)
    with pytest.raises(ValueError, match="Source manifests changed during integrity validation"):
        validate_dataset_integrity(folder, source, policy, execution_scope=SCOPE)


@pytest.mark.parametrize("which", ["source", "transactions", "labels"])
def test_stale_source_object_cannot_hide_disk_manifest_change(source, policy, tmp_path, which):
    folder = tmp_path / "data"
    _build(source, policy, folder)
    path = _source_manifest_path(source, which)
    changed = read_json(path)
    changed["unexpected_revision"] = "changed-on-disk"
    write_json(path, changed)
    with pytest.raises(ValueError, match="Source.*manifest changed"):
        validate_dataset_integrity(folder, source, policy, execution_scope=SCOPE)


@pytest.mark.parametrize("which", ["source", "transactions", "labels"])
def test_manifest_raw_bytes_are_pinned_even_when_json_values_match(source, policy, tmp_path, which):
    folder = tmp_path / "data"
    _build(source, policy, folder)
    path = _source_manifest_path(source, which)
    unchanged = read_json(path)
    path.write_bytes(path.read_bytes() + b"\n")
    assert read_json(path) == unchanged
    with pytest.raises(ValueError, match="regenerate"):
        validate_dataset_integrity(folder, source, policy, execution_scope=SCOPE)


def test_mutating_only_cached_manifest_does_not_authorize_a_new_contract(source, policy):
    source.labels["unexpected_revision"] = "memory-only"
    with pytest.raises(ValueError, match="Source table manifest changed"):
        contract(source, policy, SCOPE, datetime(2022, 1, 2), datetime(2022, 1, 4))


@pytest.mark.parametrize("which", ["transactions", "labels"])
def test_source_part_tampering_is_checked_without_replay(source, policy, tmp_path, which):
    folder = tmp_path / "data"
    _build(source, policy, folder)
    part = source.part_path(which, 0)
    part.write_bytes(part.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="ML input part changed"):
        validate_dataset_integrity(folder, source, policy, execution_scope=SCOPE)


@pytest.mark.parametrize("which", ["transactions", "labels"])
def test_additional_declared_source_part_file_is_hashed(source, policy, tmp_path, which):
    manifest_path = _source_manifest_path(source, which)
    extra = manifest_path.parent / "part-notes.json"
    extra.write_text("{}", encoding="utf-8")
    manifest = source.transactions if which == "transactions" else source.labels
    manifest["parts"][0]["files"].append({"path": extra.name, "sha256": sha256(extra)})
    write_json(manifest_path, manifest)
    folder = tmp_path / "data"
    _build(source, policy, folder)
    extra.write_text('{"changed": true}', encoding="utf-8")
    with pytest.raises(ValueError, match="ML input part changed"):
        validate_dataset_integrity(folder, source, policy, execution_scope=SCOPE)


def test_same_resolved_source_file_alias_must_have_the_same_declared_hash(
    source, policy, tmp_path, monkeypatch
):
    folder = tmp_path / "data"
    parquet_path = source.part_path("transactions", 0)
    alias = parquet_path.parent / "alias.txt"
    alias.write_text("synthetic-alias", encoding="utf-8")
    source.transactions["parts"][0]["files"].append({"path": alias.name, "sha256": sha256(alias)})
    write_json(_source_manifest_path(source, "transactions"), source.transactions)
    _build(source, policy, folder)
    original_resolve = Path.resolve

    def resolve_alias(path, *args, **kwargs):
        if path == alias:
            return parquet_path
        return original_resolve(path, *args, **kwargs)

    # Windows 심볼릭 링크 권한 없이 내부 파일 별칭을 검사합니다.
    monkeypatch.setattr(Path, "resolve", resolve_alias)
    with pytest.raises(ValueError, match="ML input part changed"):
        validate_dataset_integrity(folder, source, policy, execution_scope=SCOPE)


@pytest.mark.parametrize("which", ["transactions", "labels"])
def test_missing_source_part_is_not_accepted(source, policy, tmp_path, which):
    folder = tmp_path / "data"
    _build(source, policy, folder)
    source.part_path(which, 0).unlink()
    with pytest.raises(OSError):
        validate_dataset_integrity(folder, source, policy, execution_scope=SCOPE)


@pytest.mark.parametrize("change", ["missing", "extra-file", "extra-directory", "unsafe-name"])
def test_exact_dataset_inventory_rejects_unexpected_entries(source, policy, tmp_path, change):
    folder = tmp_path / "data"
    _build(source, policy, folder)
    if change == "missing":
        (folder / "X_test.parquet").unlink()
    elif change == "extra-file":
        (folder / "unattested.json").write_text("{}", encoding="utf-8")
    elif change == "extra-directory":
        (folder / "unattested").mkdir()
    else:
        manifest = read_json(folder / "manifest.json")
        manifest["files"]["../X_test.parquet"] = manifest["files"].pop("X_test.parquet")
        manifest.pop("dataset_fingerprint")
        manifest["dataset_fingerprint"] = fingerprint(manifest)
        write_json(folder / "manifest.json", manifest)
    with pytest.raises(ValueError, match="inventory"):
        validate_dataset_integrity(folder, source, policy, execution_scope=SCOPE)


def test_unrehashed_dataset_artifact_is_rejected(source, policy, tmp_path):
    folder = tmp_path / "data"
    _build(source, policy, folder)
    path = folder / "alerts.json"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="file hash mismatch"):
        validate_dataset_integrity(folder, source, policy, execution_scope=SCOPE)


def test_stale_dataset_fingerprint_is_rejected(source, policy, tmp_path):
    folder = tmp_path / "data"
    _build(source, policy, folder)
    manifest = read_json(folder / "manifest.json")
    manifest["rows"]["train"] += 1
    write_json(folder / "manifest.json", manifest)
    with pytest.raises(ValueError, match="altered dataset manifest"):
        validate_dataset_integrity(folder, source, policy, execution_scope=SCOPE)


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("artifact", ["alerts.json", "evidence.parquet", "X_test.parquet"])
def test_self_rehashed_evidence_or_x_still_requires_deep_source_replay(
    source, policy, tmp_path, streaming, artifact
):
    folder = tmp_path / "data"
    deep = _build(source, policy, folder, streaming=streaming)
    if artifact == "alerts.json":
        payload = read_json(folder / artifact)
        payload[0]["transaction_uids"].pop()
        write_json(folder / artifact, payload)
    elif artifact == "evidence.parquet":
        evidence = pq.read_table(folder / artifact)
        pq.write_table(evidence.slice(1), folder / artifact)
    else:
        x = pq.read_table(folder / artifact)
        changed = x.set_column(0, x.column_names[0], pa.array([999.0] * len(x)))
        pq.write_table(changed, folder / artifact)
    _rehash(folder, artifact)
    integrity = validate_dataset_integrity(folder, source, policy, execution_scope=SCOPE)
    assert integrity["status"] == "INTEGRITY_VERIFIED"
    # 모델을 읽을 때 이 지문과 앞선 심층 검증에서 고정한 지문을 비교해야 합니다.
    assert integrity["manifest_sha256"] != deep["manifest_sha256"]
    with pytest.raises(ValueError, match="evidence|alignment"):
        verify_dataset(folder, source, policy, execution_scope=SCOPE)


@pytest.mark.parametrize("which", ["dataset", "source", "transactions", "labels"])
def test_manifest_changed_during_reference_replay_cannot_receive_attestation(
    source, policy, tmp_path, monkeypatch, which
):
    folder = tmp_path / "data"
    _build(source, policy, folder)
    derive = offline_pipeline._derive

    def change_manifest_after_deriving(*args, **kwargs):
        result = derive(*args, **kwargs)
        path = (
            folder / "manifest.json" if which == "dataset" else _source_manifest_path(source, which)
        )
        path.write_bytes(path.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(offline_pipeline, "_derive", change_manifest_after_deriving)
    with pytest.raises(ValueError, match="manifests? changed during source replay"):
        verify_dataset(folder, source, policy, execution_scope=SCOPE)


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("change", ["modify", "add", "remove"])
def test_artifact_changed_after_replay_cannot_receive_attestation(
    source, policy, tmp_path, monkeypatch, streaming, change
):
    folder = tmp_path / "data"
    _build(source, policy, folder, streaming=streaming)
    manifest_bytes = (folder / "manifest.json").read_bytes()
    untouched_bytes = (folder / "X_train.parquet").read_bytes()
    module = offline_streaming if streaming else offline_pipeline
    original_fingerprint = module.source_fingerprint
    changed = False

    def change_output_after_replay(current_source):
        nonlocal changed
        result = original_fingerprint(current_source)
        # 의미적 재연산 이후, 최종 VERIFIED 반환 전에 호출됩니다.
        if change == "modify":
            path = folder / "alerts.json"
            path.write_bytes(path.read_bytes() + b"\n")
        elif change == "add":
            (folder / "unexpected.log").write_text("external writer", encoding="utf-8")
        else:
            (folder / "quality.json").unlink()
        changed = True
        return result

    monkeypatch.setattr(module, "source_fingerprint", change_output_after_replay)
    with pytest.raises(ValueError, match="file hash mismatch|artifact inventory mismatch"):
        verify_dataset(folder, source, policy, execution_scope=SCOPE)
    assert changed
    # 보존된 출력을 삭제하거나 다시 쓰지 않고 검증에서 거부합니다.
    assert (folder / "manifest.json").read_bytes() == manifest_bytes
    assert (folder / "X_train.parquet").read_bytes() == untouched_bytes
    if change == "add":
        assert (folder / "unexpected.log").is_file()
    assert not list(tmp_path.glob(".offline-work-*"))


@pytest.mark.parametrize("which", ["transactions", "labels"])
@pytest.mark.parametrize("extra_row", [False, True])
def test_source_row_mismatch_rejected_before_decoding(
    source, policy, tmp_path, monkeypatch, which, extra_row
):
    path = source.part_path(which, 0)
    table = pq.read_table(path)
    changed = pa.concat_tables([table, table.slice(0, 1)]) if extra_row else table.slice(1)
    pq.write_table(changed, path)
    table_manifest = source.transactions if which == "transactions" else source.labels
    table_manifest["parts"][0]["files"][0]["sha256"] = sha256(path)
    # 선언된 행 수를 보존하고 캐시·디스크 manifest를 맞춥니다. 해시 검사만으로
    # 선언된 예산보다 많거나 적은 행을 허용하면 안 됩니다.
    write_json(_source_manifest_path(source, which), table_manifest)

    def forbidden_decode(*args, **kwargs):
        raise AssertionError("Row-count mismatch must fail before any source row decoding")

    monkeypatch.setattr(pq.ParquetFile, "iter_batches", forbidden_decode)
    output = tmp_path / "blocked"
    with pytest.raises(ValueError, match="row count differs from manifest before decoding"):
        build_dataset(
            source,
            policy,
            output,
            train_before=datetime(2022, 1, 2),
            test_from=datetime(2022, 1, 4),
            execution_scope=SCOPE,
            max_source_rows=source.metadata["input_rows"],
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".offline-work-*"))
