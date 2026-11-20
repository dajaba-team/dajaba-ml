import json
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aml_ml import FEATURE_NAMES, AlertInput, Coverage, features, score


def build(f):
    a = dict(f["alert"])
    a["transaction_ids"] = tuple(a["transaction_ids"])
    for k in ("reference_ts", "window_start", "window_end"):
        a[k] = datetime.fromisoformat(a[k])
    t = []
    for row in f["transactions"]:
        r = dict(row)
        r["ts"] = datetime.fromisoformat(r["ts"])
        for k in ("amount_paid", "amount_received"):
            r[k] = Decimal(r[k])
        t.append(r)
    a = AlertInput(**a)
    return (
        a,
        pd.DataFrame(t),
        Coverage(a.reference_ts - timedelta(hours=24), a.reference_ts, None, None, "COMPLETE"),
    )


def test_training_feature_parity_and_predictions():
    fs = json.loads((Path(__file__).parent / "parity_fixture.json").read_text(encoding="utf8"))
    for fixture in fs:
        a, tx, c = build(fixture)
        actual = features([a], tx, coverage=[c]).iloc[0]
        assert list(FEATURE_NAMES) == list(fixture["expected_features"])
        np.testing.assert_array_equal(
            actual[list(FEATURE_NAMES)].to_numpy(dtype=float),
            np.array(list(fixture["expected_features"].values())),
        )
        x = score([a], tx, coverage=[c])[0]
        assert x.data_quality_status == "OK" and x.failure_code is None and 0 <= x.score <= 1
        assert x == score([a], tx, coverage=[c])[0]


def test_failure_isolation_and_coverage():
    fs = json.loads((Path(__file__).parent / "parity_fixture.json").read_text(encoding="utf8"))
    a, tx, c = build(fs[1])
    bad = Coverage(c.requested_start, c.requested_end, None, None, "UNKNOWN")
    r = score([a, a], tx, coverage=[c, bad])
    assert (
        r[0].score is not None
        and r[1].score is None
        and r[1].failure_code == "HISTORY_NOT_SUPPORTED"
    )


def test_input_boundaries_isolation_and_explanations(monkeypatch):
    from dataclasses import replace

    fs = json.loads((Path(__file__).parent / "parity_fixture.json").read_text(encoding="utf8"))
    a, tx, c = build(fs[1])
    txlate = tx.copy()
    txlate.loc[txlate.index[0], "ts"] = a.reference_ts
    assert score([a], txlate, coverage=[c])[0].score is None
    txfloat = tx.copy()
    txfloat.loc[txfloat.index[0], "amount_paid"] = 1.0
    assert score([a], txfloat, coverage=[c])[0].failure_code == "REQUIRED_INPUT_MISSING"
    short = replace(c, status="PARTIAL")
    assert score([a], tx, coverage=[short])[0].failure_code == "HISTORY_NOT_SUPPORTED"
    missing = replace(a, transaction_ids=a.transaction_ids + (987654,))
    z = score([a, missing], tx, coverage=[c, c])
    assert z[0].score is not None and z[1].failure_code == "REQUIRED_INPUT_MISSING"
    assert len(z[0].reasons) <= 3 and set(z[0].reasons[0]) == {
        "name",
        "observation",
        "baseline",
        "contribution",
    }
    assert (
        set(z[0].evidence_transaction_ids) <= set(a.transaction_ids)
        and not z[0].context_transaction_ids
    )
    from aml_ml import serving_candidate as aml_ml

    original = aml_ml._MODEL.predict

    def broken(matrix, *args, **kwargs):
        if kwargs.get("pred_contribs"):
            raise ValueError("explanation failed")
        return original(matrix, *args, **kwargs)

    monkeypatch.setattr(aml_ml._MODEL, "predict", broken)
    q = score([a], tx, coverage=[c])[0]
    assert q.score == z[0].score and q.reasons == [] and q.data_quality_status == "OK"


def test_rule_gate_and_key_identity():
    from dataclasses import replace

    fs = json.loads((Path(__file__).parent / "parity_fixture.json").read_text(encoding="utf8"))
    a, tx, c = build(fs[1])
    assert (
        score([replace(a, rule_id="R99_UNKNOWN")], tx, coverage=[c])[0].failure_code
        == "RULE_NOT_SUPPORTED"
    )
    changed = replace(a, account_key="010_" + a.account_key.split("_", 1)[-1])
    assert score([changed], tx, coverage=[c])[0].score is None


def test_fixed_expected_scores_and_reasons():
    fs = json.loads((Path(__file__).parent / "parity_fixture.json").read_text(encoding="utf8"))
    for f in fs:
        a, tx, c = build(f)
        r = score([a], tx, coverage=[c])[0]
        assert (
            r.score == f["expected_score"]
            and [x["name"] for x in r.reasons] == f["expected_reasons"]
        )


def test_future_evidence_is_preserved_but_excluded():
    from dataclasses import replace

    fs = json.loads((Path(__file__).parent / "parity_fixture.json").read_text(encoding="utf8"))
    a, tx, c = build(fs[0])
    base = score([a], tx, coverage=[c])[0]
    for extra_time in (a.reference_ts, a.reference_ts + timedelta(minutes=1)):
        extra = tx.iloc[0].copy()
        extra["transaction_id"] = 999987
        extra["ts"] = extra_time
        updated = pd.concat([tx, pd.DataFrame([extra])], ignore_index=True)
        b = replace(a, transaction_ids=a.transaction_ids + (999987,))
        out = score([b], updated, coverage=[c])[0]
        assert out.score == base.score and 999987 in out.evidence_transaction_ids


@pytest.mark.parametrize("rule", ["R03_FX_RELAY", "R05_SCATTER_GATHER", "R06_GATHER_SCATTER"])
def test_known_rules_reject_incompatible_input(rule):
    from dataclasses import replace

    fs = json.loads((Path(__file__).parent / "parity_fixture.json").read_text(encoding="utf8"))
    alert, transactions, coverage = build(fs[1])
    result = score([replace(alert, rule_id=rule)], transactions, coverage=[coverage])[0]
    assert result.score is None
    assert result.data_quality_status == "FAILED"
    assert result.failure_code == "REQUIRED_INPUT_MISSING"


def test_mixed_batch_preserves_r04_results_and_r05_superset():
    from dataclasses import replace

    fs = json.loads((Path(__file__).parent / "parity_fixture.json").read_text(encoding="utf8"))
    fixture = next(item for item in fs if item["alert"]["rule_id"] == "R04_CYCLE")
    first, tx, coverage = build(fixture)
    second = replace(first, alert_id="cycle-second")
    paths = first.rule_facts["paths"]
    unsupported = replace(
        first,
        alert_id="r05",
        rule_id="R99_UNKNOWN",
        rule_facts={"paths": paths[:1]},
        transaction_ids=first.transaction_ids + (999991, 999992),
    )
    batch = [first, unsupported, second]
    actual = score(batch, tx, coverage=[coverage] * 3)
    assert actual == score(batch, tx, coverage=[coverage] * 3)
    assert actual[0] == score([first], tx, coverage=[coverage])[0]
    assert actual[2] == score([second], tx, coverage=[coverage])[0]
    assert actual[0].alert_id != actual[2].alert_id
    assert actual[1].failure_code == "RULE_NOT_SUPPORTED" and actual[1].reasons == []
