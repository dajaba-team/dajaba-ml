"""고정 R02/R04의 전체 근거에서 라벨 없이 무차원 관측 피처를 계산한다.

과거 이력·탐지 재실행이 아니라 현재 완료 경보의 [window_start, reference_ts)
근거를 요약한다. 실제 수집/탐지 완료 시각을 새로 입증하지 않는다.
"""

import math
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from decimal import Decimal, localcontext

VERSION = "medium-current-evidence-unitfree-v1"
TARGET_RULES = ("R02_FAN_IN", "R04_CYCLE")
ALERT_COLUMNS = (
    "training_alert_uid", "rule_id", "account_key", "reference_ts",
    "window_start", "window_end", "evidence_ids", "paths",
)
EVIDENCE_COLUMNS = (
    "training_alert_uid", "rule_id", "source_row_id", "ts",
    "from_account_key", "to_account_key", "payment_format", "is_self_transfer",
    "amount_paid", "amount_received", "currency_paid_raw", "currency_received_raw",
)
FEATURE_NAMES = (
    "evidence_ach_fraction", "evidence_cheque_fraction", "evidence_cash_fraction",
    "evidence_paid_max_share", "evidence_received_max_share",
    "evidence_span_fraction_96h", "evidence_last_hour_fraction",
    "evidence_min_path_hops_scaled", "evidence_features_applicable",
    "evidence_payment_known_fraction", "evidence_paid_ratio_coverage",
    "evidence_received_ratio_coverage", "evidence_path_available",
)
_MISSING = {"", "UNKNOWN", "UNCONFIRMED", "미확정", "NA", "N/A", "NULL", "NONE", "NOT APPLICABLE"}
# 기존 DBcheck/checks.py의 IBM payment 허용값. 새 문자열을 알려진 수단으로 추정하지 않는다.
_PAYMENT_FORMATS = {"ach", "wire", "cheque", "credit card", "cash", "reinvestment", "bitcoin"}


def _text(value, *, nullable=False):
    if value is None and nullable:
        return None
    if not isinstance(value, str):
        raise ValueError("관측 문자열의 타입이 올바르지 않습니다.")
    value = value.strip()
    if value.upper() in _MISSING:
        if nullable:
            return None
        raise ValueError("경보·거래 식별자가 비어 있습니다.")
    return value


def _time(value):
    if not isinstance(value, datetime) or value.tzinfo is not None:
        raise ValueError("시간대 없는 원본 관측 시각이 필요합니다.")
    return value


def _projected(row, columns):
    if not isinstance(row, dict) or set(row) != set(columns):
        raise ValueError("명시한 관측 열만 투영해야 합니다. 라벨·사건 열은 허용하지 않습니다.")


def _amount(value):
    if value is None:
        return None
    # Arrow decimal128의 Decimal을 유지한다. float 경유로 원본 정밀도를 잃지 않는다.
    if not isinstance(value, Decimal) or not value.is_finite() or value < 0:
        raise ValueError("금액은 유한한 비음수 Decimal 또는 명시적 결측이어야 합니다.")
    return value


def _validate(alert, evidence):
    _projected(alert, ALERT_COLUMNS)
    uid = _text(alert["training_alert_uid"])
    rule = _text(alert["rule_id"])
    focal = _text(alert["account_key"])
    reference, start, end = (_time(alert[k]) for k in ("reference_ts", "window_start", "window_end"))
    if end != reference or reference - start != timedelta(hours=96):
        raise ValueError("고정 경보의 96시간·기준시각 계약이 다릅니다.")
    ids = alert["evidence_ids"]
    if (not isinstance(ids, list) or not ids
            or any(type(v) is not int or v < 0 for v in ids) or len(set(ids)) != len(ids)):
        raise ValueError("전체 근거의 고유 원본 행 ID가 필요합니다.")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("전체 관측 근거가 필요합니다.")
    rows = {}
    for row in evidence:
        _projected(row, EVIDENCE_COLUMNS)
        row_id = row["source_row_id"]
        if type(row_id) is not int or row_id < 0 or row_id in rows:
            raise ValueError("근거 행 ID가 잘못됐거나 중복입니다.")
        if row["training_alert_uid"] != uid or row["rule_id"] != rule:
            raise ValueError("경보와 근거의 UID·룰 연결이 다릅니다.")
        if not start <= _time(row["ts"]) < reference:
            raise ValueError("미래·기준시각 동일·96시간 밖 근거는 제거하지 않고 거절합니다.")
        sender, receiver = (_text(row[k]) for k in ("from_account_key", "to_account_key"))
        if row["is_self_transfer"] is not False or sender == receiver:
            raise ValueError("자기이체 또는 미확인 자기이체 상태는 허용하지 않습니다.")
        for side in ("paid", "received"):
            _amount(row[f"amount_{side}"])
            _text(row[f"currency_{side}_raw"], nullable=True)
        _text(row["payment_format"], nullable=True)
        rows[row_id] = row
    if set(rows) != set(ids):
        raise ValueError("경보 전체 근거와 관측 거래 집합이 일치하지 않습니다.")
    if rule == "R02_FAN_IN" and (
            any(row["to_account_key"] != focal for row in evidence)
            or len({row["from_account_key"] for row in evidence}) < 5 or alert["paths"] != []):
        raise ValueError("고정 R02의 방향·상대 수·경로 계약이 다릅니다.")
    return rows


def _path_feature(alert, rows):
    paths = alert["paths"]
    if not isinstance(paths, list) or not paths:
        raise ValueError("R04 전체 경로가 필요합니다.")
    used, hops = set(), []
    for path in paths:
        if (not isinstance(path, list) or not 2 <= len(path) <= 4
                or any(type(v) is not int or v not in rows for v in path)
                or len(set(path)) != len(path)):
            raise ValueError("고정 R04의 2~4홉 고유 거래 경로가 필요합니다.")
        legs = [rows[v] for v in path]
        senders = [r["from_account_key"] for r in legs]
        minutes = [r["ts"].replace(second=0, microsecond=0) for r in legs]
        if (senders[0] != alert["account_key"] or legs[-1]["to_account_key"] != alert["account_key"]
                or len(set(senders)) != len(path)
                or minutes[-1] != alert["reference_ts"] - timedelta(minutes=1)
                or any(a["to_account_key"] != b["from_account_key"] for a, b in zip(legs, legs[1:]))
                or any(a >= b for a, b in zip(minutes, minutes[1:]))):
            raise ValueError("R04 경로의 연결·귀환·관측 순서·발화 분이 다릅니다.")
        used.update(path)
        hops.append(len(path))
    if used != set(rows):
        raise ValueError("R04 경로 합집합이 전체 근거와 다릅니다.")
    return min(hops) / 4.0


def _concentration(evidence, side):
    groups = defaultdict(list)
    for row in evidence:
        currency = _text(row[f"currency_{side}_raw"], nullable=True)
        if currency is not None:
            # 결측 판정만 정리한다. 실제 그룹 키의 raw 표기는 합치거나 정규화하지 않는다.
            groups[row[f"currency_{side}_raw"]].append(row[f"amount_{side}"])
    eligible, weighted = 0, Decimal(0)
    # 통화·방향별 합계만 만든다. 서로 다른 통화 금액을 더하거나 환산하지 않는다.
    with localcontext() as context:
        context.prec = 80
        for amounts in groups.values():
            if len(amounts) < 2 or any(v is None for v in amounts):
                continue
            total = sum(amounts, Decimal(0))
            if total == 0:
                continue
            eligible += len(amounts)
            weighted += max(amounts) / total * len(amounts)
        ratio = float(weighted / eligible) if eligible else 0.0
    return ratio, eligible / len(evidence)


def feature_row(alert: dict, evidence: list[dict]) -> dict[str, float]:
    """고정된 열 순서로 8개 값과 5개 적용·가용성/coverage 열을 반환한다."""
    rows = _validate(alert, evidence)
    result = dict.fromkeys(FEATURE_NAMES, 0.0)
    if alert["rule_id"] not in TARGET_RULES:
        return result
    result["evidence_features_applicable"] = 1.0
    formats = [_text(r["payment_format"], nullable=True) for r in evidence]
    known = [value.casefold() for value in formats
             if value is not None and value.casefold() in _PAYMENT_FORMATS]
    counts = Counter(known)
    for name in ("ach", "cheque", "cash"):
        result[f"evidence_{name}_fraction"] = counts[name] / len(known) if known else 0.0
    result["evidence_payment_known_fraction"] = len(known) / len(evidence)
    for side in ("paid", "received"):
        ratio, coverage = _concentration(evidence, side)
        result[f"evidence_{side}_max_share"] = ratio
        result[f"evidence_{side}_ratio_coverage"] = coverage
    stamps = [r["ts"] for r in evidence]
    result["evidence_span_fraction_96h"] = (max(stamps) - min(stamps)).total_seconds() / (96 * 3600)
    cutoff = alert["reference_ts"] - timedelta(hours=1)
    result["evidence_last_hour_fraction"] = sum(ts >= cutoff for ts in stamps) / len(stamps)
    if alert["rule_id"] == "R04_CYCLE":
        result["evidence_min_path_hops_scaled"] = _path_feature(alert, rows)
        result["evidence_path_available"] = 1.0
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in result.values()):
        raise ValueError("생성된 무차원 피처가 유한한 0~1 범위를 벗어났습니다.")
    return result
