"""Tiny label-blind fixtures for minute windows, recurrence and complete paths."""

import json
from datetime import datetime, timedelta

import duckdb
import pytest

from aml_ml.offline_rule_sql import RULE_IDS, generate_candidates

RELEASE = "fixture:" + "a" * 64
BASE = datetime(2026, 1, 1)
PARAMS = {
    "R01": {"min_distinct_receivers": 5},
    "R02": {"min_distinct_senders": 5},
    "R04": {"window_hours": 72, "max_hops": 3},
    "R06": {"min_distinct_senders": 3, "min_distinct_receivers": 3},
    "R07": {"criteria_by_currency": [{"currency": "US Dollar", "basis_amount": 10000,
                                     "lower_bound_ratio": 0.8, "min_count": 3}]},
}


@pytest.fixture
def con():
    c = duckdb.connect()
    c.execute("""CREATE TABLE tx(source_row_id BIGINT, transaction_uid VARCHAR,
      ts TIMESTAMP, from_account_key VARCHAR, to_account_key VARCHAR,
      is_self_transfer BOOLEAN, amount_paid DECIMAL(38,12), currency_paid_raw VARCHAR,
      amount_received DECIMAL(38,12), currency_received_raw VARCHAR)""")
    yield c
    c.close()


def add(con, minute, sender, receiver, amount="9000", currency="US Dollar", *,
        paid=None, paid_currency=None, flagged=False, seconds=0):
    n = con.execute("SELECT count(*) FROM tx").fetchone()[0]
    con.execute("INSERT INTO tx VALUES (?,?,?,?,?,?,?,?,?,?)", [
        n, f"{RELEASE}:{n}", BASE + timedelta(minutes=minute, seconds=seconds),
        sender, receiver, flagged, amount if paid is None else paid,
        currency if paid_currency is None else paid_currency, amount, currency])
    return n


def run(con, *rules):
    counts = generate_candidates(con, {"enabled_rules": list(rules), "rules": PARAMS},
                                 "fixture", RELEASE)
    rows = con.execute("SELECT * FROM rule_candidates ORDER BY rule_id,reference_ts").fetchall()
    assert sum(counts.values()) == len(rows)
    assert all(row[2] == row[4] for row in rows)
    return rows


@pytest.mark.parametrize("rule", ["R01", "R02"])
def test_fan_distinct_self_currency_and_recurrence(con, rule):
    for minute in (0, 3000):
        for i in range(5):
            a, b = ("a", f"p{i}") if rule == "R01" else (f"p{i}", "a")
            add(con, minute, a, b, currency="Euro" if i < 2 else "US Dollar")
        add(con, minute, a, b)  # duplicate peer is evidence, not a sixth peer
        add(con, minute, "a", "a")
        add(con, minute, "x", "a", flagged=True)
    rows = run(con, rule)
    assert len(rows) == 2
    assert rows[0][5] == list(range(6))
    facts = json.loads(rows[0][7])
    assert facts["distinct_receivers" if rule == "R01" else "distinct_senders"] == 5
    assert [x["transaction_count"] for x in facts["amounts_by_currency"]] == [2, 4]
    assert all(isinstance(x["total_amount"], str) for x in facts["amounts_by_currency"])
    assert sorted(n for x in facts["amounts_by_currency"] for n in x["transaction_ids"]) == list(range(6))


def test_exact_24h_excluded_and_previous_clock_minute(con):
    for i in range(4):
        add(con, 0, str(i), "a")
    add(con, 1440, "four", "a")
    assert run(con, "R02") == []


def test_previous_true_prevents_retrigger_on_boundary(con):
    for i in range(5):
        add(con, 0, str(i), "a")
    for i in range(5):
        add(con, 1440, str(i), "a")
    rows = run(con, "R02")
    assert len(rows) == 1  # prior [E-24h,E) still contains the first batch


def test_r07_received_bounds_and_all_payments(con):
    add(con, 0, "x", "a", "7999.999999999999")
    add(con, 0, "x", "a", "10000")
    add(con, 0, "x", "a", "9000", "Euro")
    ids = [add(con, i, "x", "a", amount, paid="1", paid_currency="Euro")
           for i, amount in enumerate(("8000", "9999.999999999999", "9000"))]
    rows = run(con, "R07")
    assert len(rows) == 1
    assert rows[0][5] == ids
    facts = json.loads(rows[0][7])
    assert facts["amounts_by_currency"][0]["total_amount"] == "26999.999999999999"
    assert facts["transaction_count"] == 3
    assert facts["amounts_by_currency"][0]["criteria"] == {
        k: v for k, v in PARAMS["R07"]["criteria_by_currency"][0].items() if k != "currency"}


def test_saml_missing_money_is_not_zero(con):
    for i in range(5):
        add(con, 0, str(i), "a", amount=None, currency=None)
    facts = json.loads(run(con, "R02")[0][7])
    assert facts["amounts_by_currency"][0]["total_amount"] is None
    assert facts["amounts_by_currency"][0]["money_available"] is False


def test_cycles_overlap_paths_and_deterministic_union(con):
    add(con, 0, "a", "b")
    add(con, 1, "b", "c")
    add(con, 2, "c", "a")
    add(con, 2, "b", "a")
    add(con, 2, "b", "a")  # all parallel paths retained
    rows = run(con, "R04")
    assert len(rows) == 1
    assert rows[0][1] == "a"
    assert rows[0][5] == [0, 1, 2, 3, 4]
    assert rows[0][6] == [[0, 1, 2], [0, 3], [0, 4]]
    assert json.loads(rows[0][7]) == {"paths": rows[0][6], "hops_by_path": [3, 2, 2]}


@pytest.mark.parametrize("end,expected", [(4319, 1), (4320, 0)])
def test_cycle_72h_half_open(con, end, expected):
    add(con, 0, "a", "b")
    add(con, end, "b", "a")
    assert len(run(con, "R04")) == expected


def test_same_minute_subminute_cycle_rejected(con):
    add(con, 0, "a", "b", seconds=1)
    add(con, 0, "b", "a", seconds=50)
    assert run(con, "R04") == []


def test_cycle_recurrence_after_inactivity(con):
    for start in (0, 10000):
        add(con, start, "a", "b")
        add(con, start + 1, "b", "a")
    rows = run(con, "R04")
    assert len(rows) == 2
    assert [r[6] for r in rows] == [[[0, 1]], [[2, 3]]]


def test_cycle_prior_window_boundary_suppresses_new_cycle(con):
    add(con, 0, "a", "b")
    add(con, 1, "b", "a")
    add(con, 4319, "a", "c")
    add(con, 4320, "c", "a")
    # At E=4320 the old path is absent from current, but present in prior.
    rows = [r for r in run(con, "R04") if r[1] == "a"]
    assert len(rows) == 1
    assert rows[0][2] == BASE + timedelta(minutes=2)


@pytest.mark.parametrize("rule", ["R01", "R02"])
def test_duplicate_peers_do_not_reach_threshold(con, rule):
    for i in range(20):
        a, b = ("a", str(i % 4)) if rule == "R01" else (str(i % 4), "a")
        add(con, i, a, b)
    assert run(con, rule) == []


def test_cycle_self_flags_and_repeated_accounts_excluded(con):
    add(con, 0, "a", "b", flagged=True)
    add(con, 1, "b", "a")
    add(con, 2, "a", "a")
    assert run(con, "R04") == []


def test_bad_uid_and_unsupported_rule_fail_closed(con):
    add(con, 0, "a", "b")
    con.execute("UPDATE tx SET transaction_uid='service-1'")
    with pytest.raises(ValueError, match="Invalid source"):
        run(con, "R02")
    with pytest.raises(ValueError, match="Unsupported rule"):
        run(con, "R99")


def test_r06_reverse_order_currency_facts_and_recurrence(con):
    for offset in (0, 3000):
        for i in range(3):
            add(con, offset, "a", f"out-{i}", currency="Euro")
            add(con, offset+1, f"in-{i}", "a", currency="US Dollar")
        add(con, offset+1, "in-0", "a")
        add(con, offset+1, "a", "a")
        add(con, offset+1, "x", "a", flagged=True)
    rows = run(con, "R06")
    assert len(rows) == 2
    assert rows[0][5] == list(range(7))
    facts = json.loads(rows[0][7])
    assert facts["distinct_senders"] == facts["distinct_receivers"] == 3
    assert facts["inbound_by_currency"][0]["transaction_ids"] == [1, 3, 5, 6]
    assert facts["outbound_by_currency"][0]["transaction_ids"] == [0, 2, 4]
    assert facts["inbound_by_currency"][0]["total_amount"] == "36000.000000000000"


def test_r06_distinct_and_24h_boundary(con):
    for i in range(3):
        add(con, 0, "a", f"out-{i}")
    for i in range(3):
        add(con, 1440, f"in-{i}", "a")
    assert run(con, "R06") == []


def test_r06_prior_true_suppresses_repeated_batch(con):
    for offset in (0, 1440):
        for i in range(3):
            add(con, offset, "a", f"out-{i}")
            add(con, offset, f"in-{i}", "a")
    assert len(run(con, "R06")) == 1


def test_cycle_day_partition_keeps_cross_day_paths(con, monkeypatch):
    # Force the dense-workset branch without constructing a large fixture.
    import aml_ml.offline_rule_sql as generator

    monkeypatch.setattr(generator, "CYCLE_FIRST_EDGE_DAY_THRESHOLD", 0)
    add(con, 1439, "a", "b")
    add(con, 1440, "b", "c")
    add(con, 1441, "c", "a")
    rows = run(con, "R04")
    assert len(rows) == 1
    assert rows[0][5] == [0, 1, 2]
    assert rows[0][6] == [[0, 1, 2]]


def test_rule_ids_are_canonical():
    assert RULE_IDS["R04"] == "R04_CYCLE"


@pytest.mark.parametrize("field,value", [
    ("currency", "Euro"), ("currency", "USD"), ("currency", "US dollar"),
    ("basis_amount", 20000), ("lower_bound_ratio", 0.9), ("min_count", 4),
])
def test_r07_v1_rejects_changed_currency_or_threshold_without_execution(field, value):
    """NOT RUN: v1 cannot silently change criterion while retaining its version."""
    from aml_ml.offline_rule_sql import _validate

    rules = json.loads(json.dumps(PARAMS))
    rules["R07"]["criteria_by_currency"][0][field] = value
    policy = {"enabled_rules": ["R07"], "rules": rules,
              "rule_versions": {"R07": "mldb-offline-r07-v1"}}
    with pytest.raises(ValueError, match="R07 v1"):
        _validate(policy)


def test_r07_v1_rejects_extra_currency_criterion():
    """NOT RUN: adding currencies belongs to a future separately implemented version."""
    from aml_ml.offline_rule_sql import _validate

    rules = json.loads(json.dumps(PARAMS))
    rules["R07"]["criteria_by_currency"].append({
        "currency": "Euro", "basis_amount": 10000, "lower_bound_ratio": 0.8, "min_count": 3})
    with pytest.raises(ValueError, match="R07 v1"):
        _validate({"enabled_rules": ["R07"], "rules": rules})
