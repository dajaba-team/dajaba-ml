"""Independent tiny-source integration checks; never run production sources."""

import copy
import json
from dataclasses import replace
from datetime import datetime, timedelta
from decimal import Decimal

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from aml_ml.alert_dataset import OfflineAlert
from aml_ml.offline_features import AlertPreprocessor, partition_features
from aml_ml.offline_multi import build_source, load_tables, restore_candidate_prefix
from aml_ml.offline_multi_eligibility import POLICY_VERSION, select_time_eligibility
from aml_ml.offline_serialization import verify_table
from aml_ml.schema import epoch_us
from aml_ml.storage import read_json, sha256, write_json


def multi_policy():
    return {
        "enabled_rules": ["R01", "R02", "R04", "R06", "R07"],
        "rules": {
            "R01": {"min_distinct_receivers": 5},
            "R02": {"min_distinct_senders": 5},
            "R04": {"max_hops": 3, "window_hours": 72},
            "R06": {"min_distinct_senders": 3, "min_distinct_receivers": 3},
            "R07": {"criteria_by_currency": [{"currency": "US Dollar",
                "basis_amount": 10000, "lower_bound_ratio": 0.8, "min_count": 3}]},
        },
        "rule_versions": {r: f"fixture-{r}-v1" for r in ("R01", "R02", "R04", "R06", "R07")},
        "semantics_version": "FIXTURE_B7_ONLY_NOT_PRODUCTION_POLICY",
        "eligibility_policy": {"version": POLICY_VERSION},
    }


def rewrite(source, tx, labels, *, kind="ibm"):
    """Rewrite only pytest's synthetic temporary source and pin new hashes."""
    metadata = copy.deepcopy(source.metadata)
    metadata.update(input_rows=len(tx), ml_output_rows=len(tx))
    manifests = []
    for which, rows, old in (("ml_transactions", tx, source.transactions),
                             ("ml_labels", labels, source.labels)):
        manifest = copy.deepcopy(old)
        path = source.folder / which / "part-0.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path)
        manifest.update(rows=len(rows), source_kind=kind)
        manifest["parts"] = [{"rows": len(rows), "files": [
            {"path": path.name, "sha256": sha256(path)}]}]
        metadata["datasets"][which]["rows"] = len(rows)
        write_json(path.parent / "manifest.json", manifest)
        manifests.append(manifest)
    write_json(source.folder / "source_manifest.json", metadata)
    return replace(source, kind=kind, metadata=metadata,
                   transactions=manifests[0], labels=manifests[1])


@pytest.fixture
def tiny_multi(source):
    template = pq.read_table(source.part_path("transactions", 0)).to_pylist()[0]
    tx, labels = [], []
    for day in (1, 5, 9):
        def add(sender, receiver, minute, positive=0):
            n = len(tx)
            row = dict(template, source_row_id=n,
                       transaction_uid=f"{source.metadata['source_release']}:{n}",
                       ts=datetime(2022, 1, day, 12) + timedelta(minutes=minute, seconds=30*(n % 2)),
                       from_account_key=sender, to_account_key=receiver,
                       amount_paid=Decimal(8000 + n), amount_received=Decimal(8000 + n))
            tx.append(row)
            labels.append({"source_row_id": n, "transaction_uid": row["transaction_uid"],
                           "source_positive": positive})
        for i in range(6):
            add(f"sender-{i}", "P", i, int(i == 1))
        for i in range(5):
            add("Q", f"receiver-{i}", i)
        for i in range(3):
            add(f"gather-{i}", "Q", i+1)
        add("C", "D", 0)
        add("D", "E", 1)
        add("E", "C", 2)
    return rewrite(source, tx, labels)


def build(source, path):
    return build_source(source, multi_policy(), path,
                        train_before=datetime(2022, 1, 3), test_from=datetime(2022, 1, 7),
                        memory_limit="256MB", threads=1)


def alerts_from_output(path):
    alerts = []
    for row in pq.read_table(path / "alerts.parquet").to_pylist():
        row["transaction_uids"] = tuple(row["transaction_uids"])
        row["paths"] = tuple(tuple(p) for p in row["paths"])
        alerts.append(OfflineAlert(**row))
    return alerts


def test_end_to_end_features_formats_alignment_and_train_only(tiny_multi, tmp_path):
    output = tmp_path / "multi"
    manifest = build(tiny_multi, output)
    assert manifest["status"] == "COMPLETED"
    assert manifest["model_training_executed"] is False
    assert all(manifest["rule_counts"].values())
    alerts = {a.training_alert_uid: a for a in alerts_from_output(output)}
    assert all(a.reference_ts == a.window_end for a in alerts.values())
    source_rows = pq.read_table(tiny_multi.part_path("transactions", 0)).to_pylist()
    splits = read_json(output / "splits.json")
    ordered_uids = []
    for split in ("train", "validation", "test"):
        ordered_uids.extend(splits[split])
        y = pq.read_table(output / f"y_{split}.parquet")
        assert y["training_alert_uid"].to_pylist() == splits[split]
        assert pq.read_metadata(output / f"X_{split}.parquet").num_rows == len(splits[split])
        for name in (f"X_{split}", f"y_{split}"):
            verify_table(output, name)
    oracle = partition_features(tiny_multi, source_rows,
                                [alerts[uid] for uid in ordered_uids],
                                coverage_start=epoch_us(min(r["ts"] for r in source_rows)))
    actual = pq.read_table(output / "raw.parquet")
    for observed, expected in zip(actual.to_pylist(), oracle.to_pylist(), strict=True):
        for key, value in expected.items():
            assert observed[key] == (pytest.approx(value) if value is not None else None)
    expected_processor = AlertPreprocessor()
    expected_processor.partial_fit(oracle.slice(0, len(splits["train"])), split="train")
    expected_processor.freeze()
    assert read_json(output / "preprocessing.json") == expected_processor.state
    for name in ("alerts", "targets", "evidence", "raw", "ledger", "rule_facts"):
        verify_table(output, name)
    cycle = next(a for a in alerts.values() if a.rule_id == "R04_CYCLE")
    assert {u for p in cycle.paths for u in p} == set(cycle.transaction_uids)
    assert all(u.startswith(tiny_multi.metadata["source_release"] + ":")
               for a in alerts.values() for u in a.transaction_uids)
    labels = {r["source_row_id"]: r["source_positive"] for r in
              pq.read_table(tiny_multi.part_path("labels", 0)).to_pylist()}
    for row in pq.read_table(output / "targets.parquet").to_pylist():
        assert row["y_alert_proxy"] == max(labels[int(u.rsplit(":", 1)[1])]
            for u in alerts[row["training_alert_uid"]].transaction_uids)
    with pytest.raises(FileExistsError):
        build(tiny_multi, output)


def test_saml_omits_directional_money_and_r07(tiny_multi, tmp_path):
    tx = pq.read_table(tiny_multi.part_path("transactions", 0)).to_pylist()
    labels = pq.read_table(tiny_multi.part_path("labels", 0)).to_pylist()
    for row in tx:
        row["amount"] = row.pop("amount_received")
        row.pop("amount_paid")
    saml = rewrite(tiny_multi, tx, labels, kind="saml")
    output = tmp_path / "saml"
    manifest = build(saml, output)
    assert "R07_STRUCTURING" not in manifest["rule_counts"]
    assert manifest["unsupported_rules"][0]["rule_id"] == "R07_STRUCTURING"
    facts = pq.read_table(output / "rule_facts.parquet").to_pylist()
    for row in facts:
        for item in json.loads(row["rule_facts_json"]).get("amounts_by_currency", []):
            assert item["total_amount"] is None
            assert item["money_available"] is False
    with duckdb.connect(str(output / "preprocessing.duckdb"), read_only=True) as con:
        assert con.execute("SELECT count(amount_paid)+count(amount_received) FROM tx").fetchone() == (0,)


def test_null_label_uid_rejected(tiny_multi, tmp_path):
    tx = pq.read_table(tiny_multi.part_path("transactions", 0)).to_pylist()
    labels = pq.read_table(tiny_multi.part_path("labels", 0)).to_pylist()
    labels[0]["transaction_uid"] = None
    changed = rewrite(tiny_multi, tx, labels)
    with duckdb.connect() as con, pytest.raises(ValueError, match="identity/count"):
        load_tables(con, changed, tmp_path)


def test_actual_evidence_disjoint_selection_is_label_blind():
    first, second = datetime(2022, 1, 3), datetime(2022, 1, 7)
    with duckdb.connect() as con:
        con.execute("""CREATE TABLE candidates(candidate_id BIGINT,reference_ts TIMESTAMP,
          rule_id VARCHAR,window_start TIMESTAMP);
          CREATE TABLE evidence_links(candidate_id BIGINT,source_row_id BIGINT);
          CREATE TABLE labels(source_row_id BIGINT,source_positive INTEGER)""")
        definitions = [
            (1, first-timedelta(minutes=1), "R02_FAN_IN_24H", [1, 2]),
            (2, first, "R07_STRUCTURING", [2, 3]),
            (3, first, "R01_FAN_OUT_24H", [4]),
            (4, first+timedelta(minutes=1), "R06_GATHER_SCATTER", [4]),
            (5, second, "R02_FAN_IN_24H", [3]),
            (6, second, "R07_STRUCTURING", [4]),
            (7, second, "R02_FAN_IN_24H", [1]),
            (8, first+timedelta(minutes=1), "R04_CYCLE", [9, 10]),
        ]
        for n, ts, rule, ids in definitions:
            con.execute("INSERT INTO candidates VALUES (?,?,?,?)", [
                n, ts, rule, ts-timedelta(hours=72 if rule == "R04_CYCLE" else 24)])
            con.executemany("INSERT INTO evidence_links VALUES (?,?)", [(n, x) for x in ids])
        con.execute("INSERT INTO labels VALUES (1,NULL),(2,1),(3,0),(4,1)")
        select_time_eligibility(con, first, second)
        query = "SELECT candidate_id,original_split,time_eligible FROM candidate_time ORDER BY candidate_id"
        before = con.execute(query).fetchall()
        assert before == [(1, "train", True), (2, "validation", False),
                          (3, "validation", True), (4, "validation", True),
                          (5, "test", True), (6, "test", False),
                          (7, "test", False), (8, "validation", True)]
        # Unresolved train still reserves evidence; label swapping cannot change selection.
        con.execute("UPDATE labels SET source_positive=CASE WHEN source_positive=1 THEN 0 ELSE 1 END")
        con.execute("DROP TABLE candidate_time")
        select_time_eligibility(con, first, second)
        assert con.execute(query).fetchall() == before


def make_failed_prefix(source, folder):
    """Synthetic pre-metadata OOM state; this new recovery fixture is NOT RUN."""
    from aml_ml.offline_rule_sql import generate_candidates

    folder.mkdir()
    policy = multi_policy()
    write_json(folder / "policy.json", policy)
    write_json(folder / "failure.json", {
        "status": "FAILED", "error_type": "OutOfMemoryException", "message": "fixture only"})
    with duckdb.connect(str(folder / "preprocessing.duckdb")) as con:
        load_tables(con, source, folder)
        prefix = copy.deepcopy(policy)
        prefix["enabled_rules"] = ["R01", "R02"]
        counts = generate_candidates(con, prefix, source.id, source.metadata["source_release"])
        con.execute("CHECKPOINT")
    return counts


def test_recovery_copies_original_and_skips_prefix(tiny_multi, tmp_path, monkeypatch):
    """NOT RUN after user's code-only instruction: manual regression to execute later."""
    import aml_ml.offline_rule_sql as generator

    previous, output = tmp_path / "failed", tmp_path / "recovered"
    prefix_counts = make_failed_prefix(tiny_multi, previous)
    before = sha256(previous / "preprocessing.duckdb")
    calls = []
    original = generator.generate_candidates

    def observed(con, policy, source_id, release, **kwargs):
        calls.append(list(policy["enabled_rules"]))
        assert not {"R01", "R02"}.intersection(policy["enabled_rules"])
        return original(con, policy, source_id, release, **kwargs)

    monkeypatch.setattr(generator, "generate_candidates", observed)
    manifest = build_source(tiny_multi, multi_policy(), output,
                            train_before=datetime(2022, 1, 3), test_from=datetime(2022, 1, 7),
                            memory_limit="256MB", threads=1, recover_from=previous)
    assert calls == [["R04", "R06", "R07"]]
    assert manifest["status"] == "COMPLETED"
    assert all(manifest["rule_counts"][rule] == count for rule, count in prefix_counts.items())
    assert sha256(previous / "preprocessing.duckdb") == before
    receipt = read_json(output / "recovery_receipt.json")
    assert receipt["original_preserved"] is True
    assert receipt["prior_database_sha256"] == before
    assert receipt["reused_rules"] == prefix_counts


def test_recovery_refuses_source_count_mismatch(tiny_multi, tmp_path):
    """NOT RUN: mismatched source metadata must fail before a database copy."""
    previous, output = tmp_path / "failed", tmp_path / "rejected"
    make_failed_prefix(tiny_multi, previous)
    before = sha256(previous / "preprocessing.duckdb")
    metadata = copy.deepcopy(tiny_multi.metadata)
    metadata["input_rows"] += 1
    changed = replace(tiny_multi, metadata=metadata)
    output.mkdir()
    with pytest.raises(ValueError, match="source identity mismatch"):
        restore_candidate_prefix(changed, multi_policy(), previous, output)
    assert sha256(previous / "preprocessing.duckdb") == before
    assert not (output / "preprocessing.duckdb").exists()


def test_recovery_refuses_wal_file(tiny_multi, tmp_path):
    """NOT RUN: a possibly active/uncheckpointed source must never be copied."""
    previous, output = tmp_path / "failed", tmp_path / "rejected"
    make_failed_prefix(tiny_multi, previous)
    before = sha256(previous / "preprocessing.duckdb")
    (previous / "preprocessing.duckdb.wal").write_bytes(b"synthetic WAL guard fixture")
    output.mkdir()
    with pytest.raises(ValueError, match="(?i)wal"):
        restore_candidate_prefix(tiny_multi, multi_policy(), previous, output)
    assert sha256(previous / "preprocessing.duckdb") == before
    assert not (output / "preprocessing.duckdb").exists()


def test_recovery_allows_uncomputed_rule_metadata_change(tiny_multi, tmp_path):
    """NOT RUN: confirmed R06 metadata cannot invalidate identical R01/R02 candidates."""
    previous, output = tmp_path / "failed", tmp_path / "copied"
    expected = make_failed_prefix(tiny_multi, previous)
    before = sha256(previous / "preprocessing.duckdb")
    changed = multi_policy()
    changed["rule_versions"]["R06"] = "mldb-offline-r06-v1"
    changed["backend_rule_status"] = {"R06": "v1 initial values may change"}
    output.mkdir()
    assert restore_candidate_prefix(tiny_multi, changed, previous, output) == expected
    assert sha256(previous / "preprocessing.duckdb") == before


@pytest.mark.parametrize("field", ["params", "version", "semantics", "origin"])
def test_recovery_rejects_changed_reused_rule_contract(tiny_multi, tmp_path, field):
    """NOT RUN: status-only recovery permission never expands reused-rule meaning."""
    previous, output = tmp_path / "failed", tmp_path / "rejected"
    make_failed_prefix(tiny_multi, previous)
    changed = multi_policy()
    if field == "params":
        changed["rules"]["R02"]["min_distinct_senders"] = 10
    elif field == "version":
        changed["rule_versions"]["R02"] = "different-version"
    elif field == "semantics":
        changed["semantics_version"] = "different-semantics"
    else:
        changed["rule_version_origin"] = "different-origin"
    output.mkdir()
    with pytest.raises(ValueError, match="params, versions, or semantics"):
        restore_candidate_prefix(tiny_multi, changed, previous, output)
    assert not (output / "preprocessing.duckdb").exists()
