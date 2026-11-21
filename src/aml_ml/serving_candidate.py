"""Serving adapter for the frozen evidence-only model; parity is assessed per alert."""

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from . import training_kernel as kernel

MODEL_VERSION = "xgb-hi-medium-r04-event-c2-31-v1"
FEATURE_VERSION = "hi-medium-evidence31-96h-r04-event-v1"
THRESHOLD = 0.5
CONTEXT_LIMIT = 0
BASE = (
    "evidence_count",
    "evidence_span_seconds",
    "evidence_min_gap_seconds",
    "evidence_max_gap_seconds",
    "focal_in_count",
    "focal_out_count",
    "focal_distinct_senders",
    "focal_distinct_receivers",
    "evidence_account_count",
    "paid_raw_currency_count",
    "received_raw_currency_count",
    "payment_format_count",
    "rule_r01",
    "rule_r02",
    "rule_r03",
    "rule_r04",
    "rule_r05",
    "rule_r06",
)
FEATURE_NAMES = BASE + kernel.FEATURE_NAMES
RULES = (
    "R01_FAN_OUT",
    "R02_FAN_IN",
    "R03_FX_RELAY",
    "R04_CYCLE",
    "R05_SCATTER_GATHER",
    "R06_GATHER_SCATTER",
)
TX_COLUMNS = (
    "transaction_id",
    "ts",
    "from_account_key",
    "to_account_key",
    "amount_paid",
    "currency_paid",
    "amount_received",
    "currency_received",
    "payment_format",
    "is_self_transfer",
)


@dataclass(frozen=True)
class AlertInput:
    alert_id: str
    rule_id: str
    account_key: str
    reference_ts: datetime
    window_start: datetime
    window_end: datetime
    transaction_ids: tuple[int, ...]
    rule_facts: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Coverage:
    requested_start: datetime | str
    requested_end: datetime | str
    db_min_ts: datetime | str | None
    db_max_ts: datetime | str | None
    status: str


@dataclass(frozen=True)
class ScoreResult:
    alert_id: str
    score: float | None
    data_quality_status: str
    failure_code: str | None
    reasons: list[dict[str, Any]]
    evidence_transaction_ids: list[int]
    context_transaction_ids: list[int]


class InputError(ValueError):
    pass


class HistoryError(ValueError):
    pass


class UnsupportedRuleError(ValueError):
    pass


def _check_new_rules(alert, rows):
    facts = alert.rule_facts
    indexed = {int(row['transaction_id']): row for row in rows}
    if len(indexed) != len(alert.transaction_ids) or not isinstance(facts, dict):
        raise InputError('complete pre-reference rule evidence required')

    if alert.rule_id == 'R06_GATHER_SCATTER':
        incoming = {r['from_account_key'] for r in rows if r['to_account_key'] == alert.account_key}
        outgoing = {r['to_account_key'] for r in rows if r['from_account_key'] == alert.account_key}
        if (any(alert.account_key not in (r['from_account_key'], r['to_account_key']) for r in rows)
                or facts.get('distinct_senders') != len(incoming)
                or facts.get('distinct_receivers') != len(outgoing)
                or not (5 <= len(incoming) <= 100 and 5 <= len(outgoing) <= 100)):
            raise InputError('R06 counterparty facts do not match evidence')
        return

    paths = facts.get('paths')
    if not isinstance(paths, list) or not paths:
        raise InputError('R03/R05 paths are required')
    pairs = []
    for path in paths:
        if (not isinstance(path, list) or len(path) != 2
                or any(type(i) is not int or i not in indexed for i in path)):
            raise InputError('R03/R05 invalid path IDs')
        a, b = (indexed[i] for i in path)
        if (a['to_account_key'] != b['from_account_key'] or _time(a['ts']) >= _time(b['ts'])):
            raise InputError('R03/R05 disconnected or unordered path')
        pairs.append((a, b))
    if len({tuple(path) for path in paths}) != len(paths):
        raise InputError('R03/R05 duplicate paths')

    if alert.rule_id == 'R03_FX_RELAY':
        if ({i for path in paths for i in path} != indexed.keys()
                or any(a['to_account_key'] != alert.account_key or a['currency_received'] == b['currency_paid']
                       for a, b in pairs)):
            raise InputError('R03 relay evidence does not match paths')
        return

    source = facts.get('source_account_key')
    target = alert.account_key
    if not isinstance(source, str) or not source or source == target:
        raise InputError('R05 invalid source')
    mids = {a['to_account_key'] for a, _ in pairs}
    if (len(mids) != len(paths) or len(mids) < 3 or facts.get('intermediates') != len(mids)
            or source in mids or target in mids):
        raise InputError('R05 invalid intermediates')
    representatives, participating = [], set()
    for mid in mids:
        first = sorted((_time(r['ts']), i) for i, r in indexed.items()
                       if r['from_account_key'] == source and r['to_account_key'] == mid)
        second = sorted((_time(r['ts']), i) for i, r in indexed.items()
                        if r['from_account_key'] == mid and r['to_account_key'] == target)
        if not first or not second or first[0][0] >= second[-1][0]:
            raise InputError('R05 incomplete legs')
        next_leg = next(leg for leg in second if leg[0] > first[0][0])
        representatives.append((first[0], next_leg[1]))
        participating.update(i for t, i in first if t < second[-1][0])
        participating.update(i for t, i in second if first[0][0] < t)
    expected = [[first[1], second] for first, second in sorted(representatives)]
    if paths != expected or participating != indexed.keys():
        raise InputError('R05 representative paths or full chosen-source evidence differ')


def _time(v):
    try:
        t = pd.Timestamp(v)
    except Exception as e:
        raise InputError("invalid timestamp") from e
    if pd.isna(t) or t.tzinfo is not None:
        raise InputError("naive event timestamp required")
    return t.to_pydatetime()


def load_candidate():
    """Load the exact candidate from the configured local model path."""
    model_path = os.environ.get("AML_ML_MODEL_PATH")
    if not model_path:
        raise RuntimeError("MODEL_UNAVAILABLE: set AML_ML_MODEL_PATH to the approved checkpoint")
    return _load_candidate(Path(model_path).resolve())


@lru_cache(maxsize=1)
def _load_candidate(model_path: Path):
    pkg = files("aml_ml")
    contract = json.loads(pkg.joinpath("serving_contract.json").read_text(encoding="utf-8"))
    if tuple(contract["features"]) != FEATURE_NAMES or contract["dtype"] != "float32":
        raise RuntimeError("MODEL_UNAVAILABLE: contract feature order or dtype mismatch")
    try:
        model_bytes = model_path.read_bytes()
    except OSError as exc:
        raise RuntimeError("MODEL_UNAVAILABLE: cannot read checkpoint") from exc
    if hashlib.sha256(model_bytes).hexdigest() != contract["model_sha256"]:
        raise RuntimeError("MODEL_UNAVAILABLE: checkpoint SHA-256 mismatch")
    m = xgb.Booster(params={"nthread": 1, "device": "cpu"})
    m.load_model(bytearray(model_bytes))
    if tuple(m.feature_names) != FEATURE_NAMES:
        raise RuntimeError("MODEL_UNAVAILABLE: feature order mismatch")
    return m


def _rows(tx):
    if not isinstance(tx, pd.DataFrame) or not set(TX_COLUMNS) <= set(tx.columns):
        raise InputError("missing transaction columns")
    if set(tx.columns) & {
        "is_laundering",
        "source_positive",
        "pattern_id",
        "pattern_type",
        "outcome",
    }:
        raise InputError("labels or outcomes in inputs")
    out = {}
    for r in tx.loc[:, TX_COLUMNS].to_dict("records"):
        uid = r["transaction_id"]
        if (
            isinstance(uid, (int, np.integer))
            and not isinstance(uid, (bool, np.bool_))
            and int(uid) > 0
        ):
            out.setdefault(int(uid), []).append(r)
    return out


def _calculate(a, coverage, index):
    if (
        not isinstance(a, AlertInput)
        or not isinstance(a.alert_id, str)
        or not a.alert_id
        or not isinstance(a.account_key, str)
        or not a.account_key
    ):
        raise InputError("alert identity")
    if a.rule_id not in RULES:
        raise UnsupportedRuleError("unknown rule")
    if not isinstance(coverage, Coverage) or coverage.status not in (
        "COMPLETE",
        "PARTIAL",
        "UNKNOWN",
    ):
        raise InputError("invalid coverage")
    ref, start, end = map(_time, (a.reference_ts, a.window_start, a.window_end))
    if (
        _time(coverage.requested_start) != ref - timedelta(hours=24)
        or _time(coverage.requested_end) != ref
    ):
        raise InputError("coverage bounds")
    if coverage.status != "COMPLETE":
        raise HistoryError("history is not verified complete")
    if start >= end or end != ref:
        raise InputError("invalid alert window")
    if ref - start != timedelta(hours=96):
        raise HistoryError("trained evidence requires 96-hour alert window")
    if ref.second or ref.microsecond:
        raise InputError("training requires minute-aligned reference")
    ids = a.transaction_ids
    if (
        not isinstance(ids, (list, tuple))
        or len(ids) < 2
        or any(type(x) is not int or x <= 0 for x in ids)
        or len(set(ids)) != len(ids)
    ):
        raise InputError("invalid evidence IDs")
    rows = []
    for uid in ids:
        if len(index.get(uid, [])) != 1:
            raise InputError("missing or duplicate evidence transaction")
        r = index[uid][0]
        stamp = _time(r["ts"])
        if stamp >= ref:
            raise InputError("evidence at or after reference time")
        if stamp < start:
            raise HistoryError("evidence before trained alert window")
        if stamp.second or stamp.microsecond:
            raise InputError("training evidence timestamp is minute-aligned")
        for k in ("from_account_key", "to_account_key"):
            if not isinstance(r[k], str) or not r[k]:
                raise InputError("account key")
        if (
            not isinstance(r["is_self_transfer"], (bool, np.bool_))
            or bool(r["is_self_transfer"])
            or r["from_account_key"] == r["to_account_key"]
        ):
            raise InputError("self transfer")
        for k in ("amount_paid", "amount_received"):
            if not isinstance(r[k], Decimal) or not r[k].is_finite() or r[k] < 0:
                raise InputError("money must be finite nonnegative Decimal")
        for k in ("currency_paid", "currency_received", "payment_format"):
            if not isinstance(r[k], str) or r[k].strip().upper() in kernel._MISSING:
                raise InputError("required raw evidence field missing")
        rows.append(r)
    if len(rows) < 2:
        raise InputError("insufficient pre-reference evidence")
    if a.rule_id in ("R03_FX_RELAY", "R05_SCATTER_GATHER", "R06_GATHER_SCATTER"):
        _check_new_rules(a, rows)
    ids = [int(r["transaction_id"]) for r in rows]
    stamps = sorted(_time(r["ts"]) for r in rows)
    if a.rule_id in ("R01_FAN_OUT", "R02_FAN_IN"):
        if ref - timedelta(minutes=1) not in stamps:
            raise InputError("training trigger minute missing")
        if a.rule_id == "R01_FAN_OUT" and (
            any(r["from_account_key"] != a.account_key for r in rows)
            or len({r["to_account_key"] for r in rows}) < 10
        ):
            raise InputError("R01 trained evidence shape mismatch")
    gaps = [(y - x).total_seconds() for x, y in zip(stamps, stamps[1:])]
    incoming = [r for r in rows if r["to_account_key"] == a.account_key]
    outgoing = [r for r in rows if r["from_account_key"] == a.account_key]
    vals = [
        len(rows),
        (stamps[-1] - stamps[0]).total_seconds(),
        min(gaps),
        max(gaps),
        len(incoming),
        len(outgoing),
        len({r["from_account_key"] for r in incoming}),
        len({r["to_account_key"] for r in outgoing}),
        len({r[k] for r in rows for k in ("from_account_key", "to_account_key")}),
        len({r["currency_paid"] for r in rows}),
        len({r["currency_received"] for r in rows}),
        len({r["payment_format"] for r in rows}),
    ] + [int(a.rule_id == r) for r in RULES]
    out = dict(zip(BASE, map(float, vals), strict=True))
    extra = dict.fromkeys(kernel.FEATURE_NAMES, 0.0)
    if a.rule_id in kernel.TARGET_RULES:
        paths = a.rule_facts.get("paths", []) if isinstance(a.rule_facts, dict) else []
        if a.rule_id == "R04_CYCLE":
            if not isinstance(paths, list) or not paths:
                raise InputError("R04 full paths required")
            if not all(isinstance(path, list) and all(type(txid) is int for txid in path) for path in paths):
                raise InputError("invalid R04 paths")
            if len({tuple(path) for path in paths}) != len(paths):
                raise InputError("duplicate R04 paths")
            if "hops_by_path" in a.rule_facts and a.rule_facts["hops_by_path"] != [len(path) for path in paths]:
                raise InputError("R04 path and hop counts differ")
        alert = {
            "training_alert_uid": a.alert_id,
            "rule_id": a.rule_id,
            "account_key": a.account_key,
            "reference_ts": ref,
            "window_start": start,
            "window_end": end,
            "evidence_ids": list(ids),
            "paths": paths,
        }
        er = [
            {
                "training_alert_uid": a.alert_id,
                "rule_id": a.rule_id,
                "source_row_id": int(r["transaction_id"]),
                "ts": _time(r["ts"]),
                "from_account_key": r["from_account_key"],
                "to_account_key": r["to_account_key"],
                "payment_format": r["payment_format"],
                "is_self_transfer": bool(r["is_self_transfer"]),
                "amount_paid": r["amount_paid"],
                "amount_received": r["amount_received"],
                "currency_paid_raw": r["currency_paid"],
                "currency_received_raw": r["currency_received"],
            }
            for r in rows
        ]
        try:
            extra = kernel.feature_row(alert, er)
        except (ValueError, TypeError, KeyError) as e:
            raise InputError(str(e)) from e
    out.update(extra)
    if tuple(out) != FEATURE_NAMES or not all(math.isfinite(v) for v in out.values()):
        raise InputError("feature mismatch")
    return out


def features(
    alerts: list[AlertInput], transactions: pd.DataFrame, *, coverage: list[Coverage]
) -> pd.DataFrame:
    try:
        index = _rows(transactions)
    except InputError:
        index = {}
    out = []
    for i, a in enumerate(alerts):
        try:
            if i >= len(coverage):
                raise InputError("missing coverage")
            values = _calculate(a, coverage[i], index)
            out.append(
                {
                    "alert_id": a.alert_id,
                    **values,
                    "data_quality_status": "OK",
                    "failure_code": None,
                }
            )
        except (InputError, HistoryError, UnsupportedRuleError) as e:
            failure_codes = {
                InputError: "REQUIRED_INPUT_MISSING",
                HistoryError: "HISTORY_NOT_SUPPORTED",
                UnsupportedRuleError: "RULE_NOT_SUPPORTED",
            }
            out.append(
                {
                    "alert_id": getattr(a, "alert_id", ""),
                    **dict.fromkeys(FEATURE_NAMES, None),
                    "data_quality_status": "FAILED",
                    "failure_code": failure_codes[type(e)],
                }
            )
        except Exception:
            out.append(
                {
                    "alert_id": getattr(a, "alert_id", ""),
                    **dict.fromkeys(FEATURE_NAMES, None),
                    "data_quality_status": "FAILED",
                    "failure_code": "COMPUTE_ERROR",
                }
            )
    return pd.DataFrame(
        out, columns=["alert_id", *FEATURE_NAMES, "data_quality_status", "failure_code"]
    )


def score(
    alerts: list[AlertInput], transactions: pd.DataFrame, *, coverage: list[Coverage]
) -> list[ScoreResult]:
    f = features(alerts, transactions, coverage=coverage)
    model = load_candidate() if not f.empty and f["data_quality_status"].eq("OK").any() else None
    results = []
    for i, a in enumerate(alerts):
        row = f.iloc[i]
        code = row["failure_code"]
        value = None
        if model is not None and row["data_quality_status"] == "OK":
            try:
                value = float(
                    model.predict(
                        xgb.DMatrix(
                            np.asarray([row[list(FEATURE_NAMES)].to_numpy(dtype=np.float32)]),
                            feature_names=list(FEATURE_NAMES),
                            nthread=1,
                        )
                    )[0]
                )
            except Exception:
                code = "COMPUTE_ERROR"
        ok = value is not None and math.isfinite(value) and 0 <= value <= 1
        if value is not None and not ok:
            value, code = None, 'COMPUTE_ERROR'
        reasons = []
        if ok:
            try:
                matrix = xgb.DMatrix(
                    np.asarray([row[list(FEATURE_NAMES)].to_numpy(dtype=np.float32)]),
                    feature_names=list(FEATURE_NAMES),
                    nthread=1,
                )
                contributions = model.predict(matrix, pred_contribs=True)[0][:-1]
                order = sorted(
                    range(len(FEATURE_NAMES)),
                    key=lambda j: (-abs(float(contributions[j])), FEATURE_NAMES[j]),
                )[:3]
                reasons = [
                    {
                        "name": FEATURE_NAMES[j],
                        "observation": {
                            "value": float(row[FEATURE_NAMES[j]]),
                            "unit": None,
                            "window_start": None,
                            "window_end": None,
                            "basis": "model_feature",
                        },
                        "baseline": {"value": None, "definition": None},
                        "contribution": {
                            "method": "TreeSHAP",
                            "value": float(contributions[j]),
                            "scale": "log_odds",
                        },
                    }
                    for j in order
                ]
            except Exception:
                reasons = []
        results.append(
            ScoreResult(
                getattr(a, "alert_id", ""),
                value,
                "OK" if ok else "FAILED",
                None if ok else code,
                reasons,
                list(a.transaction_ids) if ok else [],
                [],
            )
        )
    return results
