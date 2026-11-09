"""버전을 관리하는 피처 뷰이며, ID와 원천 라벨은 모델 입력이 아닙니다."""

import json
import math
from datetime import datetime
from decimal import Decimal
from importlib.resources import files

FEATURE_VERSION = "causal-account-v2-currency-20261030-self-excluded-24h-primary"
WINDOWS = {"1h": 3600, "24h": 86400, "7d": 604800}
METRICS = (
    "out_count",
    "in_count",
    "out_counterparties",
    "in_counterparties",
    "out_amount_log1p",
    "in_amount_log1p",
    "coverage_partial",
)
TAIL = ("previous_gap_log1p", "history_absent", "money_unit_unknown")
ACCOUNT_FEATURES = tuple(f"{name}_{window}" for window in WINDOWS for name in METRICS) + TAIL
NUMERIC = tuple(
    f"{side}_{name}" for side in ("sender", "receiver") for name in ACCOUNT_FEATURES
) + ("observed_paid_log1p", "observed_received_log1p", "observed_single_log1p", "self_transfer")
CATEGORICAL = ("payment_format", "currency_paid_bucket", "currency_received_bucket")
FORBIDDEN = frozenset(
    {
        "source_positive",
        "source_label_raw",
        "source_typology",
        "is_laundering",
        "Laundering_type",
        "pattern_id",
        "pattern_type",
        "case_id",
        "outcome",
        "score",
        "_target",
        "transaction_id",
        "source_row_id",
        "transaction_uid",
        "alert_id",
        "rule_id",
        "ts",
    }
)
MISSING = frozenset(
    {"", "UNKNOWN", "UNCONFIRMED", "미확정", "NA", "N/A", "NULL", "NONE", "NOT APPLICABLE"}
)


def epoch_us(value: datetime) -> int:
    if not isinstance(value, datetime) or value.tzinfo is not None:
        raise ValueError("Expected a timezone-naive original timestamp")
    delta = value - datetime(1970, 1, 1)
    return (delta.days * 86400 + delta.seconds) * 1000000 + delta.microseconds


def log_amount(value: Decimal | None) -> float | None:
    if value is None:
        return None
    if not value.is_finite() or value < 0:
        raise ValueError("Only finite nonnegative amounts are supported")
    result = math.log1p(float(value))
    if not math.isfinite(result):
        raise ValueError("Derived amount feature is not finite")
    return result


def currency_bucket(source: str, raw: str) -> str | None:
    if not isinstance(raw, str) or raw.strip().upper() in MISSING:
        return None
    aliases = _ALIASES
    return aliases.get(raw) or json.dumps([source, raw], separators=(",", ":"))


_ALIASES = json.loads(files("aml_ml").joinpath("currencies.json").read_text(encoding="utf-8-sig"))[
    "confirmed"
]


def model_numeric(primary_windows=("24h",)) -> tuple[str, ...]:
    if not primary_windows or set(primary_windows) - set(WINDOWS):
        raise ValueError("Unknown/empty primary feature windows")
    return tuple(
        name
        for name in NUMERIC
        if not any(name.endswith("_" + w) for w in WINDOWS)
        or any(name.endswith("_" + w) for w in primary_windows)
    )
