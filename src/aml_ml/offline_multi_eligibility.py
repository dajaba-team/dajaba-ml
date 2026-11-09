"""Label-blind temporal eligibility using actual full-evidence overlap.

Chronological boundaries are unchanged. Features may use observed past history.
All candidate evidence is preserved. Earlier splits get priority, independent of
label availability; an overlapping later candidate is excluded in its entirety.
This protects shared evidence, not account/graph-level statistical independence.
"""

from datetime import datetime

POLICY_VERSION = "evidence-disjoint-temporal-v2"


def select_time_eligibility(con, first: datetime, second: datetime):
    if first.tzinfo is not None or second.tzinfo is not None or first >= second:
        raise ValueError("Ordered naive source-time boundaries required")
    con.execute(
        """
        CREATE TABLE candidate_time AS SELECT candidate_id,
          CASE WHEN reference_ts < ? THEN 'train'
               WHEN reference_ts < ? THEN 'validation' ELSE 'test' END AS original_split,
          true AS time_eligible, NULL::VARCHAR AS exclusion_reason
        FROM candidates
    """,
        [first, second],
    )
    # No labels or target completeness enter any of these statements.
    con.execute("""
        CREATE TEMP TABLE prior_evidence AS
        SELECT DISTINCT e.source_row_id FROM evidence_links e
        JOIN candidate_time c USING(candidate_id) WHERE c.original_split='train';
        UPDATE candidate_time SET time_eligible=false,
          exclusion_reason='FULL_EVIDENCE_OVERLAP_TRAIN'
        WHERE original_split='validation' AND candidate_id IN (
          SELECT e.candidate_id FROM evidence_links e
          JOIN prior_evidence p USING(source_row_id));
        INSERT INTO prior_evidence
        SELECT DISTINCT e.source_row_id FROM evidence_links e
        JOIN candidate_time c USING(candidate_id)
        WHERE c.original_split='validation' AND c.time_eligible;
        UPDATE candidate_time SET time_eligible=false,
          exclusion_reason='FULL_EVIDENCE_OVERLAP_PRIOR_SPLIT'
        WHERE original_split='test' AND candidate_id IN (
          SELECT e.candidate_id FROM evidence_links e
          JOIN prior_evidence p USING(source_row_id));
        DROP TABLE prior_evidence;
    """)
    if con.execute("""
        SELECT count(*) FROM (
          SELECT e.source_row_id FROM evidence_links e
          JOIN candidate_time c USING(candidate_id) WHERE c.time_eligible
          GROUP BY e.source_row_id HAVING count(DISTINCT c.original_split)>1)
    """).fetchone()[0]:
        raise ValueError("Label-blind time selection contains shared cross-split evidence")


def ledger_row(alert, target, selection, *, train_before, test_from):
    if target["training_alert_uid"] != alert.training_alert_uid:
        raise ValueError("Alert/target identity mismatch")
    truth = target["y_alert_proxy"]
    complete = target["missing_label_count"] == 0
    if complete != (truth is not None) or (
        truth is not None and (type(truth) is not int or truth not in (0, 1))
    ):
        raise ValueError("Target completeness inconsistent")
    reasons = [] if selection["time_eligible"] else [selection["exclusion_reason"]]
    if not complete:
        reasons.append("UNRESOLVED_EVIDENCE_LABEL")
    # Audit lookback only, never used as a blanket embargo/purge duration.
    horizon = int((alert.reference_ts - alert.window_start).total_seconds())
    return dict(
        training_alert_uid=alert.training_alert_uid,
        source_id=alert.source_id,
        reference_ts=alert.reference_ts.isoformat(),
        original_split=selection["original_split"],
        time_eligible=selection["time_eligible"],
        target_complete=complete,
        supervised_eligible=selection["time_eligible"] and complete,
        exclusion_reasons=reasons,
        y_alert_proxy=truth,
        evidence_uids=list(alert.transaction_uids),
        horizon_seconds=horizon,
        train_before=train_before.isoformat(),
        test_from=test_from.isoformat(),
        eligibility_version=POLICY_VERSION,
    )
