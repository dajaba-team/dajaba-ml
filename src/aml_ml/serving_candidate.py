"""Serving adapter for the frozen evidence-only model; parity is assessed per alert."""

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from importlib.resources import files
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from . import training_kernel as kernel

MODEL_VERSION = "xgb-medium-c2-31-w1-d4-dev130"
FEATURE_VERSION = "medium-evidence31-96h-candidate-dev130"
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


def _time(v):
    try:
        t = pd.Timestamp(v)
    except Exception as e:
        raise InputError("invalid timestamp") from e
    if pd.isna(t) or t.tzinfo is not None:
        raise InputError("naive event timestamp required")
    return t.to_pydatetime()


def load_candidate():
    m = xgb.Booster(params={"nthread": 1})
    m.load_model(bytearray(files("aml_ml").joinpath("models/candidate.ubj").read_bytes()))
    if tuple(m.feature_names) != FEATURE_NAMES:
        raise RuntimeError("MODEL_UNAVAILABLE: feature order mismatch")
    return m


_MODEL = load_candidate()


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
        raise InputError("unknown rule")
    if a.rule_id in ("R03_FX_RELAY", "R05_SCATTER_GATHER", "R06_GATHER_SCATTER"):
        raise UnsupportedRuleError("rule-specific service evidence parity unresolved")
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
            continue
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
            if not isinstance(r[k], str) or not r[k].strip():
                raise InputError("required raw evidence field missing")
        rows.append(r)
    if len(rows) < 2:
        raise InputError("insufficient pre-reference evidence")
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
        if a.rule_id == "R04_CYCLE" and not paths:
            raise InputError("R04 full paths required")
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
    results = []
    for i, a in enumerate(alerts):
        row = f.iloc[i]
        code = row["failure_code"]
        value = None
        if row["data_quality_status"] == "OK":
            try:
                value = float(
                    _MODEL.predict(
                        xgb.DMatrix(
                            np.asarray([row[list(FEATURE_NAMES)].to_numpy(dtype=np.float32)]),
                            feature_names=list(FEATURE_NAMES),
                            nthread=1,
                        )
                    )[0]
                )
            except Exception:
                code = "COMPUTE_ERROR"
        ok = value is not None
        reasons = []
        if ok:
            try:
                matrix = xgb.DMatrix(
                    np.asarray([row[list(FEATURE_NAMES)].to_numpy(dtype=np.float32)]),
                    feature_names=list(FEATURE_NAMES),
                    nthread=1,
                )
                contributions = _MODEL.predict(matrix, pred_contribs=True)[0][:-1]
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
