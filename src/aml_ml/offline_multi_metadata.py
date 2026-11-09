"""Build and validate multi-rule alert identities, evidence, proxy labels and splits."""

from collections import Counter
from contextlib import ExitStack
from dataclasses import asdict

import pyarrow as pa

from .alert_dataset import OfflineAlert
from .offline_eligibility import EligibilityCounts
from .offline_multi_eligibility import POLICY_VERSION, ledger_row, select_time_eligibility
from .offline_multi_io import append, query_batches
from .offline_rules import fingerprint
from .offline_serialization import (
    ALERT_SCHEMA,
    LEDGER_SCHEMA,
    TARGET_SCHEMA,
    ThreeWriter,
    record_table,
)

META_SCHEMA = pa.schema(
    [
        ("candidate_id", pa.int64()),
        ("training_alert_uid", pa.string()),
        ("account_key", pa.string()),
        ("reference_ts", pa.timestamp("us")),
        ("rule_id", pa.string()),
        ("split", pa.string()),
        ("eligible", pa.bool_()),
        ("evidence_ids", pa.list_(pa.int64())),
        ("y_alert_proxy", pa.int8()),
    ]
)
FACT_SCHEMA = pa.schema([("training_alert_uid", pa.string()), ("rule_facts_json", pa.string())])



def candidate_metadata(con, source, policy, folder, first, second):
    release = source.metadata["source_release"]
    con.execute("""
        CREATE TABLE candidates AS SELECT row_number() OVER (
          ORDER BY reference_ts, rule_id, account_key) AS candidate_id, *
        FROM rule_candidates
    """)
    con.execute("""
        CREATE TABLE evidence_links AS
        SELECT candidate_id, unnest(evidence_ids) AS source_row_id FROM candidates
    """)
    bad = con.execute("""
        SELECT count(*) FROM (
          SELECT e.candidate_id, e.source_row_id FROM evidence_links e
          LEFT JOIN tx t USING (source_row_id)
          GROUP BY e.candidate_id, e.source_row_id
          HAVING count(*) != 1 OR count(t.source_row_id) != 1)
    """).fetchone()[0]
    if bad:
        raise ValueError("Candidate evidence is duplicated or absent from the source")
    # Every evidence row must belong to its exact B7 window; self transfers never qualify.
    bad = con.execute("""
        SELECT count(*) FROM evidence_links e JOIN candidates c USING(candidate_id)
        JOIN tx t USING(source_row_id)
        WHERE t.is_self_transfer OR t.from_account_key=t.to_account_key
          OR t.ts<c.window_start OR t.ts>=c.window_end
    """).fetchone()[0]
    if bad:
        raise ValueError("Self/future/out-of-window candidate evidence")
    bad_paths = con.execute("""
        SELECT count(*) FROM candidates c, unnest(c.paths) p(path)
        LEFT JOIN tx t1 ON t1.source_row_id=p.path[1]
        LEFT JOIN tx t2 ON t2.source_row_id=p.path[2]
        LEFT JOIN tx t3 ON t3.source_row_id=p.path[3]
        WHERE c.rule_id='R04_CYCLE' AND (
          len(p.path) NOT IN (2,3) OR t1.source_row_id IS NULL
          OR t2.source_row_id IS NULL
          OR t1.from_account_key<>c.account_key
          OR t1.to_account_key<>t2.from_account_key
          OR date_trunc('minute',t1.ts)>=date_trunc('minute',t2.ts)
          OR (len(p.path)=2 AND t2.to_account_key<>c.account_key)
          OR (len(p.path)=3 AND (
            t3.source_row_id IS NULL OR t2.to_account_key<>t3.from_account_key
            OR t3.to_account_key<>c.account_key
            OR t2.to_account_key=c.account_key
            OR t1.to_account_key=t2.to_account_key
            OR date_trunc('minute',t2.ts)>=date_trunc('minute',t3.ts))))
    """).fetchone()[0]
    if bad_paths:
        raise ValueError("R04 path connectivity/strict-minute ordering/target mismatch")
    select_time_eligibility(con, first, second)
    con.execute("""
        CREATE TABLE candidate_targets AS
        SELECT e.candidate_id, count(*) AS evidence_count,
          count(*) FILTER (WHERE l.source_positive IS NULL) AS missing_label_count,
          max(l.source_positive) AS any_positive
        FROM evidence_links e LEFT JOIN labels l USING(source_row_id)
        GROUP BY e.candidate_id
    """)
    append(con, "meta", pa.Table.from_pylist([], schema=META_SCHEMA), create=True)
    counts = EligibilityCounts()
    per_rule = Counter()
    with ExitStack() as stack:
        writers = {
            name: stack.enter_context(ThreeWriter(folder / f"{name}.parquet", schema))
            for name, schema in (
                ("alerts", ALERT_SCHEMA),
                ("targets", TARGET_SCHEMA),
                ("ledger", LEDGER_SCHEMA),
                ("rule_facts", FACT_SCHEMA),
            )
        }
        for batch in query_batches(
            con,
            """
            SELECT c.*, t.evidence_count,t.missing_label_count,t.any_positive,
              et.original_split,et.time_eligible,et.exclusion_reason
            FROM candidates c JOIN candidate_targets t USING(candidate_id)
            JOIN candidate_time et USING(candidate_id)
            ORDER BY reference_ts,rule_id,account_key
        """,
        ):
            records = {name: [] for name in writers}
            meta = []
            for row in batch.to_pylist():
                alert = OfflineAlert(
                    source_id=source.id,
                    rule_id=row["rule_id"],
                    rule_version=policy["rule_versions"][row["rule_id"].split("_")[0]],
                    rule_config_hash=fingerprint(
                        {
                            "rule_id": row["rule_id"],
                            "params": policy["rules"][row["rule_id"].split("_")[0]],
                            "semantics": policy["semantics_version"],
                        }
                    ),
                    account_key=row["account_key"],
                    reference_ts=row["reference_ts"],
                    window_start=row["window_start"],
                    window_end=row["window_end"],
                    transaction_uids=tuple(f"{release}:{n}" for n in row["evidence_ids"]),
                    paths=tuple(tuple(f"{release}:{n}" for n in path) for path in row["paths"]),
                )
                uid = alert.training_alert_uid
                complete = row["missing_label_count"] == 0
                target = dict(
                    training_alert_uid=uid,
                    source_id=source.id,
                    reference_ts=alert.reference_ts.isoformat(),
                    y_alert_proxy=int(row["any_positive"]) if complete else None,
                    target_status="COMPLETE_PROXY_NOT_STR_TRUTH"
                    if complete
                    else "UNRESOLVED_EVIDENCE_LABEL",
                    target_definition="ANY_POSITIVE_IN_FULL_ALERT_EVIDENCE",
                    evidence_count=row["evidence_count"],
                    missing_label_count=row["missing_label_count"],
                    id_namespace="OFFLINE_SOURCE_UID",
                )
                ledger = ledger_row(alert, target, row, train_before=first, test_from=second)
                target["split"] = ledger["original_split"]
                ar = asdict(alert)
                for key in ("reference_ts", "window_start", "window_end"):
                    ar[key] = ar[key].isoformat()
                records["alerts"].append(ar)
                records["targets"].append(target)
                records["ledger"].append(ledger)
                records["rule_facts"].append(
                    dict(training_alert_uid=uid, rule_facts_json=row["rule_facts_json"])
                )
                meta.append(
                    dict(
                        candidate_id=row["candidate_id"],
                        training_alert_uid=uid,
                        account_key=alert.account_key,
                        reference_ts=alert.reference_ts,
                        rule_id=alert.rule_id,
                        split=target["split"],
                        eligible=ledger["supervised_eligible"],
                        evidence_ids=row["evidence_ids"],
                        y_alert_proxy=target["y_alert_proxy"],
                    )
                )
                counts.add(ledger)
                per_rule[(alert.rule_id, target["split"], ledger["supervised_eligible"])] += 1
            for name, rows in records.items():
                schema = {
                    "alerts": ALERT_SCHEMA,
                    "targets": TARGET_SCHEMA,
                    "ledger": LEDGER_SCHEMA,
                    "rule_facts": FACT_SCHEMA,
                }[name]
                writers[name].write_table(record_table(rows, schema))
            append(con, "meta", pa.Table.from_pylist(meta, schema=META_SCHEMA))
    if con.execute("""SELECT count(*)-count(DISTINCT training_alert_uid)
                       FROM meta""").fetchone()[0]:
        raise ValueError("Duplicate training alert identity")
    overlap = con.execute("""
        SELECT count(*) FROM (
          SELECT e.source_row_id FROM evidence_links e JOIN meta m USING(candidate_id)
          WHERE m.eligible GROUP BY e.source_row_id HAVING count(DISTINCT split)>1)
    """).fetchone()[0]
    if overlap:
        raise ValueError("Full evidence overlaps eligible splits")
    report = counts.report()
    report["policy_version"] = POLICY_VERSION
    return report, [
        dict(rule_id=r, split=s, eligible=e, rows=n) for (r, s, e), n in sorted(per_rule.items())
    ]
