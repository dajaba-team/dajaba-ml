"""Read-only postrun descriptive diagnostics, never model performance estimates.

Writes a new diagnostics folder only. Pattern events use the pinned Patterns.txt
SHA and zero-based BEGIN ordinal, never invented backend transaction/event IDs.
Only events whose every original row has one exact 11-field match are mapped.
"""

import csv
from datetime import datetime
from pathlib import Path

import duckdb

from .offline_contract import source_fingerprint
from .storage import read_json, sha256, write_json

RAW_HEADER = ["Timestamp", "From Bank", "Account", "To Bank", "Account",
              "Amount Received", "Receiving Currency", "Amount Paid", "Payment Currency",
              "Payment Format", "Is Laundering"]
BEGIN = "BEGIN LAUNDERING ATTEMPT - "
END = "END LAUNDERING ATTEMPT - "
DISCLAIMER = (
    "Descriptive source-label and synthetic-event overlap only. These are not "
    "precision, recall, detection recall, model performance, or STR truth."
)


def _literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def _csv(con, query, path):
    con.execute(f"COPY ({query}) TO {_literal(path)} (HEADER, DELIMITER ',')")


def _ingest_patterns(con, path, digest):
    con.execute("""CREATE TABLE pattern_events(event_id VARCHAR, event_ordinal BIGINT,
        pattern_type VARCHAR, begin_line BIGINT, end_line BIGINT, row_count BIGINT);
        CREATE TABLE pattern_rows(event_id VARCHAR, pattern_row_ordinal BIGINT,
        line_number BIGINT, ts TIMESTAMP, from_bank VARCHAR, from_account VARCHAR,
        to_bank VARCHAR,to_account VARCHAR,amount_received VARCHAR,
        currency_received VARCHAR,amount_paid VARCHAR,currency_paid VARCHAR,
        payment_format VARCHAR,source_positive VARCHAR)""")
    active = None
    event_ordinal = -1
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for line_number, text in enumerate(stream, 1):
            line = text.rstrip("\r\n")
            if not line.strip():
                continue
            if line.startswith(BEGIN):
                if active is not None:
                    raise ValueError(f"Nested event at pattern line {line_number}")
                event_ordinal += 1
                kind = line[len(BEGIN):].split(":", 1)[0].strip()
                if not kind:
                    raise ValueError(f"Missing pattern type at line {line_number}")
                active = [f"PATTERN-{digest}:{event_ordinal}", event_ordinal,
                          kind, line_number, None, 0]
            elif line.startswith(END):
                if active is None or line[len(END):].strip() != active[2]:
                    raise ValueError(f"Unmatched event end at pattern line {line_number}")
                if not active[5]:
                    raise ValueError(f"Empty event at pattern line {line_number}")
                active[4] = line_number
                con.execute("INSERT INTO pattern_events VALUES (?,?,?,?,?,?)", active)
                active = None
            else:
                if active is None:
                    raise ValueError(f"Transaction outside event at pattern line {line_number}")
                fields = next(csv.reader([line], strict=True))
                if len(fields) != 11:
                    raise ValueError(f"Expected 11 original fields at pattern line {line_number}")
                timestamp = datetime.strptime(fields[0], "%Y/%m/%d %H:%M")
                if timestamp.strftime("%Y/%m/%d %H:%M") != fields[0]:
                    raise ValueError(f"Noncanonical timestamp at pattern line {line_number}")
                rows.append([active[0], active[5], line_number, timestamp] + fields[1:])
                active[5] += 1
                if len(rows) >= 4096:
                    con.executemany("INSERT INTO pattern_rows VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
                    rows.clear()
    if active is not None:
        raise ValueError("Unclosed pattern event at EOF")
    if rows:
        con.executemany("INSERT INTO pattern_rows VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)


def _event_overlap(con, source, output):
    if source.kind != "ibm":
        return {"status": "UNAVAILABLE", "reason": "SAML_HAS_NO_VERIFIED_EVENT_ID",
                "source_typology": "Transaction type only, not an event identifier"}
    pins = source.metadata.get("input_fingerprints", {})
    patterns = [(Path(p), digest) for p, digest in pins.items()
                if Path(p).name.endswith("_Patterns.txt")]
    transactions = [(Path(p), digest) for p, digest in pins.items()
                    if Path(p).name.endswith("_Trans.csv")]
    if len(patterns) != 1 or len(transactions) != 1:
        return {"status": "UNAVAILABLE", "reason": "ONE_PINNED_PATTERN_AND_TRANSACTION_FILE_REQUIRED"}
    pattern, digest = patterns[0]
    raw, raw_digest = transactions[0]
    if raw_digest != source.metadata["source_release"].rsplit(":", 1)[1]:
        return {"status": "UNAVAILABLE", "reason": "TRANSACTION_SHA_RELEASE_MISMATCH"}
    # Raw CSV order is verified from its pinned header, never inferred from amount equality.
    if sha256(pattern) != digest or sha256(raw) != raw_digest:
        return {"status": "UNAVAILABLE", "reason": "PINNED_INPUT_HASH_MISMATCH"}
    with raw.open(encoding="utf-8-sig", newline="") as stream:
        if next(csv.reader(stream), None) != RAW_HEADER:
            return {"status": "UNAVAILABLE", "reason": "ORIGINAL_11_FIELD_HEADER_MISMATCH"}
    _ingest_patterns(con, pattern, digest)
    required = {"from_bank_raw", "to_bank_raw", "from_account_raw", "to_account_raw",
                "amount_received_raw", "amount_paid_raw", "currency_received_raw",
                "currency_paid_raw", "payment_format", "ts", "source_row_id"}
    existing = {r[0] for r in con.execute("DESCRIBE run.tx").fetchall()}
    if required - existing:
        return {"status": "UNAVAILABLE", "reason": "EXACT_RAW_FIELDS_UNAVAILABLE",
                "missing_columns": sorted(required-existing)}
    con.execute("CREATE TABLE IF NOT EXISTS diagnostic_rules AS "
                "SELECT DISTINCT rule_id FROM run.candidates")
    con.execute("""
        CREATE TABLE pattern_matches AS
        SELECT p.event_id,p.pattern_row_ordinal,p.line_number,count(t.source_row_id) AS match_count,
          CASE WHEN count(t.source_row_id)=1 THEN min(t.source_row_id) END AS source_row_id
        FROM pattern_rows p LEFT JOIN (
          SELECT t.*,l.source_positive FROM run.tx t JOIN run.labels l USING(source_row_id)
        ) t ON p.ts=t.ts AND p.from_bank=t.from_bank_raw
          AND p.from_account=t.from_account_raw AND p.to_bank=t.to_bank_raw
          AND p.to_account=t.to_account_raw AND p.amount_received=t.amount_received_raw
          AND p.currency_received=t.currency_received_raw AND p.amount_paid=t.amount_paid_raw
          AND p.currency_paid=t.currency_paid_raw AND p.payment_format=t.payment_format
          AND p.source_positive=t.source_positive::VARCHAR
        GROUP BY p.event_id,p.pattern_row_ordinal,p.line_number;
        CREATE TABLE event_mapping_status AS
        SELECT e.*, count(*) FILTER(WHERE m.match_count=0) AS unmatched_rows,
          count(*) FILTER(WHERE m.match_count>1) AS ambiguous_rows,
          bool_and(m.match_count=1) AS completely_mapped
        FROM pattern_events e JOIN pattern_matches m USING(event_id)
        GROUP BY ALL;
        CREATE TABLE event_links AS
        SELECT DISTINCT m.event_id,m.source_row_id FROM pattern_matches m
        JOIN event_mapping_status e USING(event_id) WHERE e.completely_mapped;
        CREATE TABLE event_alert_overlap AS
        SELECT DISTINCT e.event_id,c.rule_id,c.candidate_id
        FROM event_links e JOIN run.evidence_links l USING(source_row_id)
        JOIN run.candidates c USING(candidate_id);
        CREATE TABLE rule_pattern_overlap AS
        WITH rules AS (SELECT rule_id FROM diagnostic_rules),
        types AS (SELECT pattern_type,count(*) AS total_events,
          count(*) FILTER(WHERE completely_mapped) AS mapped_events
          FROM event_mapping_status GROUP BY pattern_type),
        touched AS (SELECT o.rule_id,e.pattern_type,count(DISTINCT o.event_id) AS events_touched,
          count(DISTINCT o.candidate_id) AS alerts_touching_events
          FROM event_alert_overlap o JOIN pattern_events e USING(event_id)
          GROUP BY o.rule_id,e.pattern_type)
        SELECT r.rule_id,t.*,coalesce(x.events_touched,0) AS distinct_events_touched,
          coalesce(x.alerts_touching_events,0) AS alerts_touching_events
        FROM rules r CROSS JOIN types t LEFT JOIN touched x USING(rule_id,pattern_type)
    """)
    for name in ("event_mapping_status", "pattern_matches", "event_links", "event_alert_overlap",
                 "rule_pattern_overlap"):
        _csv(con, f"SELECT * FROM {name}", output / f"{name}.csv")
    total, mapped = con.execute("SELECT count(*),count(*) FILTER(WHERE completely_mapped) "
                                "FROM event_mapping_status").fetchone()
    return {"status": "COMPLETE" if total == mapped else "PARTIAL_MAPPING",
            "pattern_sha256": digest, "total_events": total, "completely_mapped_events": mapped,
            "unmapped_events_excluded_from_overlap": total-mapped,
            "event_identity": "PATTERN_FILE_SHA256_AND_ZERO_BASED_BEGIN_ORDINAL",
            "matching": "ALL_11_ORIGINAL_FIELDS_UNIQUE_EXACT_MATCH",
            "interpretation": "Events touched / all parsed events by type, NOT detection recall"}


def diagnose(run_folder: Path, source, output: Path | None = None, *,
             memory_limit="2GB", threads=2):
    """Verify completed run/source; create new diagnostics with READ_ONLY run attachment.

    source is the verified SourceInput used to generate the run. Any existing
    diagnostics destination fails closed. Pattern mapping failure is reported
    without guessed IDs; label ratios and pending backend comparison still emit.
    """
    run_folder = Path(run_folder)
    manifest = read_json(run_folder / "manifest.json")
    if (manifest.get("status") != "COMPLETED" or manifest.get("source_id") != source.id
            or manifest.get("source_release") != source.metadata["source_release"]
            or manifest.get("source_fingerprint") != source_fingerprint(source)):
        raise ValueError("Completed generated run and verified source binding required")
    db = run_folder / "preprocessing.duckdb"
    expected = manifest.get("files", {}).get(db.name)
    if not expected or sha256(db) != expected:
        raise ValueError("Generated read-only database hash mismatch")
    output = Path(output) if output is not None else run_folder / "diagnostics"
    if output.parent.resolve() != run_folder.resolve():
        raise ValueError("Diagnostics must use a new direct subfolder of the generated run")
    if output.exists():
        raise FileExistsError("Use a new diagnostics folder")
    output.mkdir(parents=True)
    con = duckdb.connect(str(output / "diagnostics.duckdb"))
    try:
        con.execute("SET memory_limit=?", [memory_limit])
        con.execute("SET threads=?", [threads])
        con.execute("SET temp_directory=?", [str(output / "sql_spill")])
        con.execute(f"ATTACH {_literal(db)} AS run (READ_ONLY)")
        con.execute("CREATE TABLE diagnostic_rules(rule_id VARCHAR)")
        con.executemany("INSERT INTO diagnostic_rules VALUES (?)",
                        [(rule,) for rule in manifest["rule_counts"]])
        con.execute("""
            CREATE TABLE laundering_alert_ratios AS
            SELECT r.rule_id,count(c.candidate_id) AS all_alerts,
              count(*) FILTER(WHERE t.missing_label_count=0) AS complete_alerts,
              count(*) FILTER(WHERE t.missing_label_count=0 AND t.any_positive=1)
                AS positive_evidence_alerts,
              count(*) FILTER(WHERE t.missing_label_count>0) AS incomplete_alerts,
              count(*) FILTER(WHERE t.missing_label_count=0 AND t.any_positive=1)::DOUBLE /
                nullif(count(*) FILTER(WHERE t.missing_label_count=0),0) AS laundering_alert_ratio
            FROM diagnostic_rules r LEFT JOIN run.candidates c USING(rule_id)
            LEFT JOIN run.candidate_targets t USING(candidate_id)
            GROUP BY r.rule_id ORDER BY r.rule_id
        """)
        _csv(con, "SELECT * FROM laundering_alert_ratios", output / "laundering_alert_ratios.csv")
        comparison = {"status": "PENDING_SERVICE_EXPORT", "scope": "HI R02 2022-09-02",
                      "identifier_namespace": "ZERO_BASED_SOURCE_ROW_ORDINAL_NOT_BACKEND_ID"}
        if source.id == "ibm_hi_small":
            digest = source.metadata["source_release"].rsplit(":", 1)[1]
            con.execute(f"""
                CREATE TABLE hi_r02_comparison_alerts AS
                SELECT m.training_alert_uid,c.account_key,c.reference_ts,c.window_start,c.window_end,
                  {_literal(digest)} AS source_file_sha256,c.evidence_ids AS source_row_ordinals,
                  c.candidate_id
                FROM run.candidates c JOIN run.meta m USING(candidate_id)
                WHERE c.rule_id='R02_FAN_IN_24H'
                  AND (c.reference_ts-INTERVAL '1 minute')::DATE=DATE '2022-09-02';
                CREATE TABLE hi_r02_comparison_evidence AS
                SELECT a.training_alert_uid,a.account_key,a.reference_ts,a.window_start,a.window_end,
                  a.source_file_sha256,t.source_row_id AS zero_based_source_row_id,
                  t.transaction_uid AS full_evidence_uid,t.ts AS transaction_ts
                FROM hi_r02_comparison_alerts a JOIN run.evidence_links e USING(candidate_id)
                JOIN run.tx t USING(source_row_id)
            """)
            for name in ("hi_r02_comparison_alerts", "hi_r02_comparison_evidence"):
                _csv(con, f"SELECT * FROM {name}", output / f"{name}.csv")
        try:
            events = _event_overlap(con, source, output)
        except (OSError, ValueError, csv.Error, duckdb.Error) as exc:
            events = {"status": "UNAVAILABLE", "reason": type(exc).__name__, "detail": str(exc),
                      "mapping_not_used": True}
        report = {"status": "COMPLETED", "disclaimer": DISCLAIMER,
                  "source_id": source.id, "source_release": source.metadata["source_release"],
                  "policy": manifest["policy"], "rule_counts": manifest["rule_counts"],
                  "run_database_sha256": expected, "event_overlap": events,
                  "backend_comparison": comparison, "model_training_executed": False}
        con.close()
        con = None
        report["files"] = {p.name: sha256(p) for p in output.iterdir() if p.is_file()}
        write_json(output / "diagnostics.json", report)
        return report
    finally:
        if con is not None:
            con.close()
