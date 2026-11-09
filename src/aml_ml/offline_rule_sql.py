"""Label-blind, complete-evidence DuckDB candidates for the frozen B7 rules.

The caller owns connection memory/temp-directory/thread limits and a fresh database.
All transaction work stays in SQL (including path enumeration); no evidence cap or
sampling is applied. Dense cycle graphs may still require substantial spill space.
Rule chronology uses floored minutes; the input ``tx.ts`` is never modified.
"""

import json
from decimal import Decimal, InvalidOperation

RULE_IDS = {
    "R01": "R01_FAN_OUT_24H",
    "R02": "R02_FAN_IN_24H",
    "R04": "R04_CYCLE",
    "R06": "R06_GATHER_SCATTER",
    "R07": "R07_STRUCTURING",
}
CYCLE_FIRST_EDGE_DAY_THRESHOLD = 25000


def _validate(policy):
    enabled = policy["enabled_rules"]
    if not isinstance(enabled, list) or len(set(enabled)) != len(enabled):
        raise ValueError("enabled_rules must be a distinct list")
    for rule in enabled:
        if rule not in RULE_IDS:
            raise ValueError(f"Unsupported rule: {rule}")
        p = policy["rules"][rule]
        expected = {}
        if rule in ("R01", "R02"):
            expected["min_distinct_receivers" if rule == "R01" else "min_distinct_senders"] = 5
        elif rule == "R04":
            expected.update(max_hops=3, window_hours=72)
        elif rule == "R06":
            expected.update(min_distinct_senders=3, min_distinct_receivers=3)
        else:
            criteria = p.get("criteria_by_currency")
            if set(p) != {"criteria_by_currency"} or not isinstance(criteria, list) or len(criteria) != 1:
                raise ValueError("R07 v1 requires exactly one US Dollar criterion")
            item = criteria[0]
            required = {"currency", "basis_amount", "lower_bound_ratio", "min_count"}
            if not isinstance(item, dict) or set(item) != required:
                raise ValueError("R07 v1 requires the exact confirmed parameter keys")
            try:
                basis = Decimal(str(item["basis_amount"]))
                ratio = Decimal(str(item["lower_bound_ratio"]))
            except (InvalidOperation, ValueError, TypeError) as exc:
                raise ValueError("Invalid R07 v1 numeric parameters") from exc
            if (
                item["currency"] != "US Dollar"
                or not basis.is_finite() or basis != Decimal("10000")
                or not ratio.is_finite() or ratio != Decimal("0.8")
                or type(item["min_count"]) is not int or item["min_count"] != 3
            ):
                raise ValueError(
                    "R07 v1 is frozen to US Dollar/10000/0.8/3; "
                    "other criteria require a new rule version and implementation"
                )

        if any(p.get(k) != v for k, v in expected.items()):
            raise ValueError(f"Explicit frozen parameters required for {rule}: {expected}")
    return enabled


def generate_candidates(connection, policy: dict, source_id: str, source_release: str,
                        *, progress=None):
    """Create ``rule_candidates``; return canonical-rule-ID -> candidate count.

    Policy requires enabled_rules (short IDs) and rules mapping with explicit
    min_distinct_receivers=5 for R01, min_distinct_senders=5 for R02;
    max_hops=3/window_hours=72 for R04; min_distinct_senders=3 and
    min_distinct_receivers=3 for R06; criteria_by_currency with currency,
    basis_amount, lower_bound_ratio and min_count for R07. R01/R02/R06/R07 use
    fixed 24-hour B7 windows; R04 starts at two hops. Unknown rules fail.
    R07 v1 is immutable: US Dollar/10000/0.8/3. Other currencies require a new version.
    Input source-row/UID identity is checked, but source completeness and source
    capability checks remain the caller's responsibility. Existing output fails
    closed rather than being overwritten. No labels are read.
    """
    enabled = _validate(policy)
    if not source_id or not source_release.startswith(source_id + ":"):
        raise ValueError("Source release must match source identity")
    invalid = connection.execute("""
        SELECT count(*) FROM tx WHERE source_row_id IS NULL OR source_row_id < 0
          OR transaction_uid IS DISTINCT FROM (? || ':' || source_row_id::VARCHAR)
          OR ts IS NULL OR is_self_transfer IS NULL
          OR from_account_key IS NULL OR from_account_key = ''
          OR to_account_key IS NULL OR to_account_key = ''
    """, [source_release]).fetchone()[0]
    duplicate = connection.execute("""
        SELECT count(*) != count(DISTINCT source_row_id) FROM tx
    """).fetchone()[0]
    if invalid or duplicate:
        raise ValueError("Invalid source row/UID, time, account, or self-transfer flag")
    connection.execute("""
        CREATE TABLE rule_candidates (
          rule_id VARCHAR, account_key VARCHAR, reference_ts TIMESTAMP,
          window_start TIMESTAMP, window_end TIMESTAMP,
          evidence_ids BIGINT[], paths BIGINT[][], rule_facts_json VARCHAR)
    """)
    connection.execute("""
        CREATE TEMP VIEW rule_tx AS
        SELECT source_row_id::BIGINT AS id, date_trunc('minute',ts) AS minute,
          from_account_key AS sender, to_account_key AS receiver,
          amount_paid, currency_paid_raw, amount_received, currency_received_raw
        FROM tx WHERE NOT is_self_transfer AND from_account_key != to_account_key
    """)
    counts = {}
    try:
        for rule in enabled:
            if progress:
                progress('rule_started', rule_id=RULE_IDS[rule])
            if rule == "R04":
                _cycles(connection, progress=progress)
            elif rule == "R06":
                _gather_scatter(connection)
            else:
                _directional(connection, rule, policy["rules"][rule])
            counts[RULE_IDS[rule]] = connection.execute(
                "SELECT count(*) FROM rule_candidates WHERE rule_id = ?", [RULE_IDS[rule]]
            ).fetchone()[0]
            if progress:
                progress("rule_completed", rule_id=RULE_IDS[rule], candidates=counts[RULE_IDS[rule]])
    finally:
        connection.execute("DROP VIEW rule_tx")
    return counts


def _directional(con, rule, params):
    outgoing = rule == "R01"
    account, peer = ("sender", "receiver") if outgoing else ("receiver", "sender")
    amount = "amount_paid" if outgoing else "amount_received"
    currency = "currency_paid_raw" if outgoing else "currency_received_raw"
    predicate = ""
    if rule == "R07":
        criteria = params["criteria_by_currency"]
        criterion = criteria[0]
        basis = Decimal(str(criterion["basis_amount"]))
        lower = basis * Decimal(str(criterion["lower_bound_ratio"]))
        escaped = criterion["currency"].replace("'", "''")
        predicate = (f"WHERE amount_received >= {lower} AND amount_received < {basis} "
                     f"AND currency_received_raw = '{escaped}'")
    metric = "count(*)" if rule == "R07" else "count(DISTINCT peer)"
    threshold = params["criteria_by_currency"][0]["min_count"] if rule == "R07" else 5
    count_fact = "distinct_receivers" if outgoing else "distinct_senders"
    currency_facts = ""
    if rule == "R07":
        count_fact = "transaction_count"
        encoded = json.dumps({k: v for k, v in criterion.items() if k != "currency"})
        currency_facts = ", criteria := '" + encoded.replace("'", "''") + "'::JSON"
    # RANGE evaluates all events in a minute together. The previous range is
    # the actual preceding clock minute, even after days of inactivity.
    con.execute(f"""
        CREATE TEMP TABLE rule_events AS SELECT id, minute, {account} AS account_key,
          {peer} AS peer, {amount} AS amount, {currency} AS currency
        FROM rule_tx {predicate};
        CREATE TEMP TABLE rule_triggers AS
        SELECT DISTINCT account_key, minute AS reference_ts FROM (
          SELECT account_key, minute,
            {metric} OVER (PARTITION BY account_key ORDER BY minute
              RANGE BETWEEN INTERVAL '1439 minutes' PRECEDING AND CURRENT ROW) AS now_n,
            {metric} OVER (PARTITION BY account_key ORDER BY minute
              RANGE BETWEEN INTERVAL '1440 minutes' PRECEDING
                AND INTERVAL '1 minute' PRECEDING) AS prior_n
          FROM rule_events
        ) WHERE now_n >= {threshold} AND prior_n < {threshold};
        CREATE TEMP TABLE rule_evidence AS
        SELECT t.account_key, t.reference_ts, e.id, e.minute, e.peer, e.amount, e.currency
        FROM rule_triggers t JOIN rule_events e USING (account_key)
        WHERE e.minute >= t.reference_ts - INTERVAL '1439 minutes'
          AND e.minute <= t.reference_ts;
        INSERT INTO rule_candidates
        WITH money AS (
          SELECT account_key, reference_ts, currency, count(*) AS tx_count,
            sum(amount)::VARCHAR AS amount_sum, count(amount) AS known_amount_count,
            list(id ORDER BY id) AS ids
          FROM rule_evidence GROUP BY account_key, reference_ts, currency
        ), facts AS (
          SELECT account_key, reference_ts,
            list(struct_pack(currency := currency, transaction_count := tx_count,
              total_amount := amount_sum, transaction_ids := ids,
              known_amount_count := known_amount_count,
              money_available := (known_amount_count=tx_count){currency_facts})
              ORDER BY currency NULLS LAST) AS currencies
          FROM money GROUP BY account_key, reference_ts
        )
        SELECT '{RULE_IDS[rule]}', e.account_key, e.reference_ts + INTERVAL '1 minute',
          e.reference_ts - INTERVAL '1439 minutes',
          e.reference_ts + INTERVAL '1 minute',
          list(e.id ORDER BY e.id), []::BIGINT[][],
          to_json(struct_pack({count_fact} := {
              'count(*)' if rule == 'R07' else 'count(DISTINCT e.peer)'},
            amounts_by_currency := any_value(f.currencies)))::VARCHAR
        FROM rule_evidence e JOIN facts f USING(account_key,reference_ts)
        GROUP BY e.account_key,e.reference_ts;
        DROP TABLE rule_evidence;
        DROP TABLE rule_triggers;
        DROP TABLE rule_events;
    """)


def _gather_scatter(con):
    """Three distinct peers on each side, with no imposed flow ordering."""
    con.execute("""
        CREATE TEMP TABLE rule_events AS
        SELECT id,minute,receiver AS account_key,sender AS peer,1 AS direction,
          amount_received AS amount,currency_received_raw AS currency FROM rule_tx
        UNION ALL
        SELECT id,minute,sender,receiver,0,amount_paid,currency_paid_raw FROM rule_tx;
        CREATE TEMP TABLE rule_triggers AS
        SELECT DISTINCT account_key,minute AS reference_ts FROM (
          SELECT account_key,minute,
            count(DISTINCT CASE WHEN direction=1 THEN peer END) OVER now_w AS now_in,
            count(DISTINCT CASE WHEN direction=0 THEN peer END) OVER now_w AS now_out,
            count(DISTINCT CASE WHEN direction=1 THEN peer END) OVER prior_w AS prior_in,
            count(DISTINCT CASE WHEN direction=0 THEN peer END) OVER prior_w AS prior_out
          FROM rule_events WINDOW
            now_w AS (PARTITION BY account_key ORDER BY minute
              RANGE BETWEEN INTERVAL '1439 minutes' PRECEDING AND CURRENT ROW),
            prior_w AS (PARTITION BY account_key ORDER BY minute
              RANGE BETWEEN INTERVAL '1440 minutes' PRECEDING
                AND INTERVAL '1 minute' PRECEDING)
        ) WHERE now_in>=3 AND now_out>=3 AND (prior_in<3 OR prior_out<3);
        CREATE TEMP TABLE rule_evidence AS
        SELECT t.account_key,t.reference_ts,e.id,e.peer,e.direction,e.amount,e.currency
        FROM rule_triggers t JOIN rule_events e USING(account_key)
        WHERE e.minute>=t.reference_ts-INTERVAL '1439 minutes' AND e.minute<=t.reference_ts;
        INSERT INTO rule_candidates
        WITH money AS (
          SELECT account_key,reference_ts,direction,currency,count(*) AS tx_count,
            sum(amount)::VARCHAR AS amount_sum,count(amount) AS known_amount_count,
            list(id ORDER BY id) AS ids
          FROM rule_evidence GROUP BY account_key,reference_ts,direction,currency
        ), facts AS (
          SELECT account_key,reference_ts,
            list(struct_pack(currency := currency,total_amount := amount_sum,
              transaction_count := tx_count,transaction_ids := ids,
              money_available := (known_amount_count=tx_count)) ORDER BY currency NULLS LAST)
              FILTER (WHERE direction=1) AS inbound,
            list(struct_pack(currency := currency,total_amount := amount_sum,
              transaction_count := tx_count,transaction_ids := ids,
              money_available := (known_amount_count=tx_count)) ORDER BY currency NULLS LAST)
              FILTER (WHERE direction=0) AS outbound
          FROM money GROUP BY account_key,reference_ts
        )
        SELECT 'R06_GATHER_SCATTER',e.account_key,e.reference_ts+INTERVAL '1 minute',
          e.reference_ts-INTERVAL '1439 minutes',e.reference_ts+INTERVAL '1 minute',
          list(e.id ORDER BY e.id),[]::BIGINT[][],
          to_json(struct_pack(
            distinct_senders := count(DISTINCT CASE WHEN e.direction=1 THEN e.peer END),
            distinct_receivers := count(DISTINCT CASE WHEN e.direction=0 THEN e.peer END),
            inbound_by_currency := any_value(f.inbound),
            outbound_by_currency := any_value(f.outbound)))::VARCHAR
        FROM rule_evidence e JOIN facts f USING(account_key,reference_ts)
        GROUP BY e.account_key,e.reference_ts;
        DROP TABLE rule_evidence;
        DROP TABLE rule_triggers;
        DROP TABLE rule_events;
    """)


def _enumerate_cycles(con, *, progress=None):
    """Enumerate exact paths after account-topology filtering, in bounded worksets.

    Long source-namespaced account keys are encoded once as local integer keys.
    These keys never escape into source evidence or exported account identities.
    Hash partitions own the first sender, so no cycle is lost or duplicated.
    Dense first-edge partitions are further split by first-edge day; later edges
    still see the full 72-hour range across dates and partition boundaries.
    These reduce intermediate work, not an absolute RAM bound: a dense single
    account/day can still have arbitrarily many real paths. DuckDB memory/spill
    settings remain required, and a resource failure never truncates evidence.
    """
    con.execute("""
        CREATE TEMP TABLE cycle_accounts AS
        SELECT row_number() OVER(ORDER BY account_key)::BIGINT AS account_id,account_key
        FROM (SELECT sender AS account_key FROM rule_tx UNION SELECT receiver FROM rule_tx);
        CREATE TEMP TABLE cycle_tx AS
        SELECT t.id,t.minute,a.account_id AS sender,b.account_id AS receiver
        FROM rule_tx t JOIN cycle_accounts a ON a.account_key=t.sender
          JOIN cycle_accounts b ON b.account_key=t.receiver;
        CREATE TEMP TABLE cycle_edges AS SELECT DISTINCT sender,receiver FROM cycle_tx;
        CREATE TEMP TABLE rule_cycles(account_key BIGINT,first_minute TIMESTAMP,
          closing_minute TIMESTAMP,path BIGINT[]);
    """)
    # Only a few hundred bucket/day identifiers return to Python, never source rows.
    buckets = con.execute("SELECT DISTINCT hash(sender)%128 FROM cycle_edges ORDER BY 1").fetchall()
    for bucket_index, (bucket,) in enumerate(buckets, 1):
        if progress:
            progress('r04_bucket_started', bucket_index=bucket_index, bucket_count=len(buckets))
        con.execute(f"""
            CREATE TEMP TABLE cycle_pairs AS
            SELECT a.sender,a.receiver FROM cycle_edges a JOIN cycle_edges b
              ON a.receiver=b.sender AND b.receiver=a.sender
              WHERE hash(a.sender)%128={int(bucket)};
            CREATE TEMP TABLE cycle_triangles AS
            SELECT a.sender,a.receiver,b.receiver AS third
            FROM cycle_edges a JOIN cycle_edges b ON a.receiver=b.sender
            JOIN cycle_edges c ON c.sender=b.receiver AND c.receiver=a.sender
            WHERE hash(a.sender)%128={int(bucket)} AND b.receiver<>a.sender;
            CREATE TEMP TABLE cycle_first AS
            SELECT t.* FROM cycle_tx t JOIN (
              SELECT sender,receiver FROM cycle_pairs UNION
              SELECT sender,receiver FROM cycle_triangles
            ) edges USING(sender,receiver);
        """)
        count = con.execute("SELECT count(*) FROM cycle_first").fetchone()[0]
        if count > CYCLE_FIRST_EDGE_DAY_THRESHOLD:
            days = [r[0] for r in con.execute(
                "SELECT DISTINCT minute::DATE FROM cycle_first ORDER BY 1").fetchall()]
        else:
            days = [None] if count else []
        for day in days:
            if day is None:
                con.execute("CREATE TEMP VIEW cycle_first_work AS SELECT * FROM cycle_first")
            else:
                con.execute("CREATE TEMP TABLE cycle_first_work AS "
                            "SELECT * FROM cycle_first WHERE minute::DATE=?", [day])
            con.execute("""
                INSERT INTO rule_cycles
                SELECT a.sender,a.minute,b.minute,[a.id,b.id]::BIGINT[]
                FROM cycle_pairs p JOIN cycle_first_work a
                  ON a.sender=p.sender AND a.receiver=p.receiver
                JOIN cycle_tx b ON b.sender=p.receiver AND b.receiver=p.sender
                  AND a.minute<b.minute AND b.minute-a.minute<INTERVAL '72 hours';
                INSERT INTO rule_cycles
                SELECT a.sender,a.minute,c.minute,[a.id,b.id,c.id]::BIGINT[]
                FROM cycle_triangles p JOIN cycle_first_work a
                  ON a.sender=p.sender AND a.receiver=p.receiver
                JOIN cycle_tx b ON b.sender=p.receiver AND b.receiver=p.third
                  AND a.minute<b.minute AND b.minute-a.minute<INTERVAL '72 hours'
                JOIN cycle_tx c ON c.sender=p.third AND c.receiver=p.sender
                  AND b.minute<c.minute AND c.minute-a.minute<INTERVAL '72 hours';
            """)
            con.execute("DROP VIEW cycle_first_work" if day is None
                        else "DROP TABLE cycle_first_work")
        con.execute("DROP TABLE cycle_first; DROP TABLE cycle_pairs; DROP TABLE cycle_triangles")
        if progress:
            progress("r04_bucket_completed", bucket_index=bucket_index,
                     bucket_count=len(buckets), first_edges=count, day_worksets=len(days))
    con.execute("DROP TABLE cycle_edges; DROP TABLE cycle_tx")


def _cycles(con, *, progress=None):
    # Strict minute inequalities reject same-minute edges, including subminute
    # SAML edges. A full retained path must fit within the B7 half-open window.
    _enumerate_cycles(con, progress=progress)
    con.execute("""
        CREATE TEMP TABLE rule_triggers AS
        SELECT DISTINCT c.account_key, c.closing_minute AS reference_ts
        FROM rule_cycles c WHERE NOT EXISTS (
          SELECT 1 FROM rule_cycles old WHERE old.account_key=c.account_key
          AND old.first_minute >= c.closing_minute-INTERVAL '72 hours'
          AND old.closing_minute < c.closing_minute
        );
        CREATE TEMP TABLE rule_paths AS
        SELECT t.account_key,t.reference_ts,c.first_minute,c.path
        FROM rule_triggers t JOIN rule_cycles c USING(account_key)
        WHERE c.first_minute >= t.reference_ts-INTERVAL '4319 minutes'
          AND c.closing_minute <= t.reference_ts;
        INSERT INTO rule_candidates
        WITH evidence AS (
          SELECT account_key, reference_ts, list(DISTINCT id ORDER BY id) AS ids
          FROM (SELECT account_key,reference_ts,unnest(path) AS id FROM rule_paths)
          GROUP BY account_key,reference_ts
        )
        SELECT 'R04_CYCLE',n.account_key,p.reference_ts+INTERVAL '1 minute',
          p.reference_ts-INTERVAL '4319 minutes',p.reference_ts+INTERVAL '1 minute',
          any_value(e.ids),list(p.path ORDER BY p.first_minute,p.path),
          to_json(struct_pack(paths := list(p.path ORDER BY p.first_minute,p.path),
            hops_by_path := list(len(p.path) ORDER BY p.first_minute,p.path)))::VARCHAR
        FROM rule_paths p JOIN evidence e USING(account_key,reference_ts)
        JOIN cycle_accounts n ON n.account_id=p.account_key
        GROUP BY n.account_key,p.reference_ts;
        DROP TABLE rule_paths;
        DROP TABLE rule_triggers;
        DROP TABLE rule_cycles;
        DROP TABLE cycle_accounts;
    """)
