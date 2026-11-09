"""Tiny pattern-file and completed-run diagnostics fixtures, no real source runs."""

import csv
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import duckdb
import pyarrow.parquet as pq
import pytest
from test_offline_multi import build, rewrite
from test_offline_multi import tiny_multi as tiny_multi

from aml_ml.offline_multi_diagnostics import RAW_HEADER, _event_overlap, diagnose
from aml_ml.storage import sha256


@pytest.fixture
def pattern_case(tmp_path):
    fields = ["2022/09/02 00:00", "001", "A", "002", "B", "8000.00", "US Dollar",
              "7000.00", "Euro", "ACH", "1"]
    raw = tmp_path / "HI-Small_Trans.csv"
    with raw.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(RAW_HEADER)
        writer.writerow(fields)
    pattern = tmp_path / "HI-Small_Patterns.txt"
    pattern.write_text("BEGIN LAUNDERING ATTEMPT - FAN-IN: fixture\n" +
                       ",".join(fields) + "\nEND LAUNDERING ATTEMPT - FAN-IN\n",
                       encoding="utf-8")
    source = SimpleNamespace(kind="ibm", metadata={
        "input_fingerprints": {str(raw): sha256(raw), str(pattern): sha256(pattern)},
        "source_release": "ibm_hi_small:" + sha256(raw)})
    con = duckdb.connect()
    con.execute("""ATTACH ':memory:' AS run;
      CREATE TABLE run.tx(source_row_id BIGINT,ts TIMESTAMP,from_bank_raw VARCHAR,
        from_account_raw VARCHAR,to_bank_raw VARCHAR,to_account_raw VARCHAR,
        amount_received_raw VARCHAR,currency_received_raw VARCHAR,amount_paid_raw VARCHAR,
        currency_paid_raw VARCHAR,payment_format VARCHAR);
      CREATE TABLE run.labels(source_row_id BIGINT,source_positive INTEGER);
      CREATE TABLE run.candidates(candidate_id BIGINT,rule_id VARCHAR);
      CREATE TABLE run.evidence_links(candidate_id BIGINT,source_row_id BIGINT);
      INSERT INTO run.candidates VALUES (1,'R02_FAN_IN_24H');
      INSERT INTO run.evidence_links VALUES (1,0);
      INSERT INTO run.labels VALUES (0,1)""")
    con.execute("INSERT INTO run.tx VALUES (0,TIMESTAMP '2022-09-02',?,?,?,?,?,?,?,?,?)", fields[1:10])
    yield con, source, tmp_path, pattern
    con.close()


def test_exact_original_received_then_paid_mapping(pattern_case):
    con, source, output, pattern = pattern_case
    result = _event_overlap(con, source, output)
    assert result["status"] == "COMPLETE"
    assert con.execute("SELECT source_row_id FROM event_links").fetchall() == [(0,)]
    assert con.execute("SELECT total_events,distinct_events_touched FROM rule_pattern_overlap").fetchone() == (1, 1)
    assert con.execute("SELECT event_id FROM event_links").fetchone()[0] == f"PATTERN-{sha256(pattern)}:0"


@pytest.mark.parametrize("change", ["ambiguous", "amount_order", "label", "raw_spelling"])
def test_no_guessed_pattern_mapping(pattern_case, change):
    con, source, output, _ = pattern_case
    if change == "ambiguous":
        con.execute("INSERT INTO run.tx SELECT 1,* EXCLUDE(source_row_id) FROM run.tx")
        con.execute("INSERT INTO run.labels VALUES (1,1)")
    elif change == "amount_order":
        con.execute("UPDATE run.tx SET amount_received_raw='7000.00',amount_paid_raw='8000.00'")
    elif change == "label":
        con.execute("UPDATE run.labels SET source_positive=0")
    else:
        con.execute("UPDATE run.tx SET amount_received_raw='8000.0'")
    result = _event_overlap(con, source, output)
    assert result["status"] == "PARTIAL_MAPPING"
    assert result["completely_mapped_events"] == 0
    assert con.execute("SELECT count(*) FROM event_links").fetchone() == (0,)
    assert con.execute("SELECT total_events,distinct_events_touched FROM rule_pattern_overlap").fetchone() == (1, 0)


def test_pinned_hash_failure_and_saml_no_event_id(pattern_case):
    con, source, output, pattern = pattern_case
    pattern.write_text(pattern.read_text() + "\n", encoding="utf-8")
    assert _event_overlap(con, source, output)["reason"] == "PINNED_INPUT_HASH_MISMATCH"
    source.kind = "saml"
    assert _event_overlap(con, source, output)["reason"] == "SAML_HAS_NO_VERIFIED_EVENT_ID"


def test_complete_run_ratios_read_only_and_new_output_only(tiny_multi, tmp_path):
    folder = tmp_path / "run"
    manifest = build(tiny_multi, folder)
    before = sha256(folder / "preprocessing.duckdb")
    result = diagnose(folder, tiny_multi)
    assert result["backend_comparison"]["status"] == "PENDING_SERVICE_EXPORT"
    assert result["policy"] == manifest["policy"]
    assert result["event_overlap"]["status"] == "UNAVAILABLE"  # no pinned Patterns fixture
    assert sha256(folder / "preprocessing.duckdb") == before
    with (folder / "diagnostics" / "laundering_alert_ratios.csv").open(newline="") as stream:
        ratios = {row["rule_id"]: row for row in csv.DictReader(stream)}
    assert float(ratios["R02_FAN_IN_24H"]["laundering_alert_ratio"]) == 1
    assert float(ratios["R01_FAN_OUT_24H"]["laundering_alert_ratio"]) == 0
    assert "not precision" in result["disclaimer"]
    assert Path(folder / "diagnostics" / "hi_r02_comparison_evidence.csv").exists()
    with pytest.raises(FileExistsError):
        diagnose(folder, tiny_multi)


def test_incomplete_labels_excluded_from_ratio_denominator(tiny_multi, tmp_path):
    tx = pq.read_table(tiny_multi.part_path("transactions", 0)).to_pylist()
    labels = pq.read_table(tiny_multi.part_path("labels", 0)).to_pylist()
    labels[0]["source_positive"] = None
    source = rewrite(tiny_multi, tx, labels)
    folder = tmp_path / "incomplete"
    build(source, folder)
    diagnose(folder, source)
    with (folder / "diagnostics" / "laundering_alert_ratios.csv").open(newline="") as stream:
        ratios = {row["rule_id"]: row for row in csv.DictReader(stream)}
    assert ratios["R02_FAN_IN_24H"]["all_alerts"] == "3"
    assert ratios["R02_FAN_IN_24H"]["complete_alerts"] == "2"
    assert ratios["R02_FAN_IN_24H"]["incomplete_alerts"] == "1"
    assert float(ratios["R02_FAN_IN_24H"]["laundering_alert_ratio"]) == 1


def test_hi_comparison_batch_date_includes_next_midnight():
    """NOT RUN: B7 batch date is E, while reference_ts is E+1 minute."""
    with duckdb.connect() as con:
        assert con.execute("""
            SELECT reference_ts FROM (VALUES
              (TIMESTAMP '2022-09-02 00:00:00'),
              (TIMESTAMP '2022-09-02 00:01:00'),
              (TIMESTAMP '2022-09-03 00:00:00'),
              (TIMESTAMP '2022-09-03 00:01:00')) t(reference_ts)
            WHERE (reference_ts-INTERVAL '1 minute')::DATE=DATE '2022-09-02'
            ORDER BY reference_ts
        """).fetchall() == [
            (datetime(2022, 9, 2, 0, 1),),
            (datetime(2022, 9, 3, 0, 0),)]
