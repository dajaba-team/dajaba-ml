"""요청 형식에 맞는 계좌 피처와 경보별 품질을 제공하며, 점수를 임의로 만들지 않습니다."""

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, cast

import pandas as pd

from .history import AccountHistory, Event
from .interface_contract import FailureCode
from .schema import ACCOUNT_FEATURES, currency_bucket, epoch_us

TX_COLUMNS = frozenset(
    {
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
    }
)


@dataclass(frozen=True)
class AlertInput:
    alert_id: str | int
    rule_id: str
    account_key: str
    reference_ts: datetime
    window_start: datetime
    evidence_transaction_ids: tuple[int, ...]
    typology: str
    typology_source: str = "RULE"

    @classmethod
    def from_mapping(cls, row: dict) -> "AlertInput":
        values = dict(row)
        for key in ["reference_ts", "window_start"]:
            if isinstance(values[key], str):
                values[key] = datetime.fromisoformat(values[key])
        values["evidence_transaction_ids"] = tuple(values["evidence_transaction_ids"])
        result = cls(**values)
        if (
            not (
                (isinstance(result.alert_id, str) and bool(result.alert_id.strip()))
                or (type(result.alert_id) is int and result.alert_id > 0)
            )
            or not result.account_key
            or result.typology_source != "RULE"
        ):
            raise ValueError("Invalid alert identity or typology_source")
        if epoch_us(result.window_start) >= epoch_us(result.reference_ts):
            raise ValueError("Alert window_start must precede reference_ts")
        if not result.evidence_transaction_ids or any(
            type(v) is not int or v < 1 for v in result.evidence_transaction_ids
        ):
            raise ValueError("Evidence must use actual positive backend transaction IDs")
        return result


def _amount(value) -> Decimal | None:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float):
        raise ValueError("Money must retain Decimal; SQL loaders must not coerce Decimal to float")
    result = value if isinstance(value, Decimal) else Decimal(str(value))
    if not result.is_finite() or result < 0:
        raise ValueError("Invalid money value")
    return result


def features(
    alerts: list[AlertInput | dict],
    transactions: pd.DataFrame,
    *,
    coverage_start: datetime | None = None,
    source_id: str = "ibm_hi_small",
    account_aliases: dict[str, str] | None = None,
) -> pd.DataFrame:
    """경보마다 한 계좌를 보며, 오프라인 전처리와 같은 과거 이력 계산을 사용합니다.

    두 계좌를 쓰는 보조 거래 모델의 서빙 어댑터는 아닙니다.
    여기에서는 라벨 처리, 점수 산출, 쓰기나 백엔드 ID 대체를 하지 않습니다.
    """
    if not TX_COLUMNS <= set(transactions.columns):
        raise ValueError(
            f"Missing serving columns: {sorted(TX_COLUMNS - set(transactions.columns))}"
        )
    if {
        "is_laundering",
        "source_positive",
        "source_typology",
        "pattern_id",
        "pattern_type",
        "outcome",
    } & set(transactions.columns):
        raise ValueError(
            "Serving features must not receive labels, pattern truth or investigation outcomes"
        )
    rows = cast(list[dict[str, Any]], transactions.to_dict("records"))
    by_id = {}
    events: dict[str, list[Event]] = defaultdict(list)
    identity_limited = set()
    for row in rows:
        tx_id = row["transaction_id"]
        if type(tx_id) is not int or tx_id < 1 or tx_id in by_id:
            raise ValueError("Serving transactions require unique actual positive IDs")
        if any(
            not isinstance(row[k], str) or not row[k]
            for k in ["from_account_key", "to_account_key"]
        ):
            raise ValueError("Serving account keys must be nonempty strings")
        ts = epoch_us(row["ts"])
        row["_ts_us"] = ts
        row["_from"] = (account_aliases or {}).get(row["from_account_key"], row["from_account_key"])
        row["_to"] = (account_aliases or {}).get(row["to_account_key"], row["to_account_key"])
        if account_aliases is not None and any(
            key not in account_aliases for key in [row["from_account_key"], row["to_account_key"]]
        ):
            identity_limited.update([row["_from"], row["_to"]])
        by_id[tx_id] = row
        if type(row["is_self_transfer"]) is not bool:
            raise ValueError("is_self_transfer must be a typed boolean")
        if row["is_self_transfer"] or row["_from"] == row["_to"]:
            continue
        for side, account, other, amount_key, currency_key in [
            (0, row["_from"], row["_to"], "amount_paid", "currency_paid"),
            (1, row["_to"], row["_from"], "amount_received", "currency_received"),
        ]:
            amount = _amount(row[amount_key])
            unit = currency_bucket(source_id, row[currency_key])
            events[cast(str, account)].append(
                Event(ts, side, cast(str, other), amount, unit if amount is not None else None)
            )
    queries: dict[str, list[tuple[int, AlertInput]]] = defaultdict(list)
    output: list[dict[str, Any]] = [{} for _ in alerts]
    for index, value in enumerate(alerts):
        raw = value if isinstance(value, dict) else vars(value)
        failure_code = FailureCode.REQUIRED_INPUT_MISSING
        try:
            alert = AlertInput.from_mapping(raw)
            account = (account_aliases or {}).get(alert.account_key, alert.account_key)
            low, high = epoch_us(alert.window_start), epoch_us(alert.reference_ts)
            for tx_id in alert.evidence_transaction_ids:
                item = by_id.get(tx_id)
                if item is None:
                    failure_code = FailureCode.TRANSACTION_NOT_FOUND
                    raise ValueError("Evidence transaction was not found")
                if not low <= item["_ts_us"] < high or account not in {item["_from"], item["_to"]}:
                    raise ValueError("Evidence is missing or outside the alert account/time scope")
            queries[account].append((index, alert))
        except (ValueError, TypeError, KeyError) as exc:
            output[index] = {
                "alert_id": raw.get("alert_id"),
                **{f"account_{n}": None for n in ACCOUNT_FEATURES},
                "data_quality_status": "INSUFFICIENT",
                "quality_reason": str(exc),
                "failure_code": failure_code.value,
            }
    coverage = epoch_us(coverage_start) if coverage_start is not None else None
    for account, items in queries.items():
        history = AccountHistory()
        account_events = sorted(events[account], key=lambda e: e.ts)
        cursor = 0
        for index, alert in sorted(items, key=lambda pair: epoch_us(pair[1].reference_ts)):
            as_of = epoch_us(alert.reference_ts)
            while cursor < len(account_events) and account_events[cursor].ts < as_of:
                history.add(account_events[cursor])
                cursor += 1
            history.expire(as_of)
            unit = history.single_unit()
            values = history.snapshot(as_of, unit, coverage)
            partial = values[ACCOUNT_FEATURES.index("coverage_partial_24h")] == 1
            empty = sum(history.window_24h.counts) == 0
            quality = (
                "INSUFFICIENT"
                if empty
                else "PARTIAL"
                if partial or unit is None or account in identity_limited
                else "OK"
            )
            output[index] = {
                "alert_id": alert.alert_id,
                **{f"account_{n}": v for n, v in zip(ACCOUNT_FEATURES, values)},
                "data_quality_status": quality,
                "failure_code": None,
                "quality_reason": "NO_PAST_OBSERVATIONS"
                if empty
                else "COVERAGE_UNIT_OR_IDENTITY_LIMITED"
                if quality == "PARTIAL"
                else None,
            }
    result = pd.DataFrame(output)
    # Pandas 3의 문자열 추론은 성공 결과의 None을 NaN으로 바꿀 수 있습니다.
    # 피처 자료형을 바꾸지 않고 명시적인 코드 또는 null 계약을 보존합니다.
    result["failure_code"] = pd.Series([item["failure_code"] for item in output], dtype=object)
    return result
