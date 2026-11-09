"""공통 과거 기반 계좌 피처와 train에만 적합하는 경보 전처리를 제공합니다."""

from collections import defaultdict
from pathlib import Path

import pyarrow as pa

from .alert_dataset import OfflineAlert
from .history import AccountHistory, Event
from .preprocessing import Preprocessor
from .schema import ACCOUNT_FEATURES, currency_bucket, epoch_us
from .storage import SourceInput, read_json

OFFLINE_FEATURE_VERSION = "offline-alert-account-24h-v1"
FEATURE_NAMES = tuple(
    "account_" + name
    for name in ACCOUNT_FEATURES
    if name.endswith("_24h")
    or name in {"previous_gap_log1p", "history_absent", "money_unit_unknown"}
)


class AlertPreprocessor(Preprocessor):
    """계좌 피처 허용 목록과 함께 train 전용 통계·선택·변환을 재사용합니다."""

    def __init__(self):
        super().__init__(
            standardize=True,
            numeric=FEATURE_NAMES,
            categorical=(),
            feature_version=OFFLINE_FEATURE_VERSION,
        )

    @classmethod
    def load(cls, path: Path) -> "AlertPreprocessor":
        state = read_json(path)
        if (
            state.get("feature_version") != OFFLINE_FEATURE_VERSION
            or state.get("fit_scope") != "TRAIN_ONLY"
            or tuple(state.get("numeric", {})) != FEATURE_NAMES
            or state.get("categorical") != {}
        ):
            raise ValueError("Incompatible offline alert preprocessing state")
        result = cls()
        result.state = state
        return result


def partition_features(
    source: SourceInput, rows: list[dict], alerts: list[OfflineAlert], *, coverage_start: int
) -> pa.Table:
    """전체 기간 범위를 적용해 담당 계좌의 완전한 이력을 공통 과거 기반 계산으로 처리합니다."""
    events: dict[str, list[Event]] = defaultdict(list)
    for row in rows:
        if row["is_self_transfer"] or row["from_account_key"] == row["to_account_key"]:
            continue
        for side, key, peer, amount_key, currency_key in (
            (0, "from_account_key", "to_account_key", "amount_paid", "currency_paid_raw"),
            (1, "to_account_key", "from_account_key", "amount_received", "currency_received_raw"),
        ):
            # SAML의 단일 Amount만으로 방향별 금액이 확정되지는 않습니다.
            amount = row[amount_key] if source.kind == "ibm" else None
            unit = currency_bucket(source.id, row[currency_key]) if amount is not None else None
            events[row[key]].append(Event(epoch_us(row["ts"]), side, row[peer], amount, unit))
    grouped: dict[str, list[tuple[int, OfflineAlert]]] = defaultdict(list)
    for i, alert in enumerate(alerts):
        if alert.source_id != source.id:
            raise ValueError("Cross-source alert feature request")
        grouped[alert.account_key].append((i, alert))
    output: list[dict] = [{} for _ in alerts]
    for account, items in grouped.items():
        history = AccountHistory()
        stream = sorted(events[account], key=lambda e: e.ts)
        cursor = 0
        for i, alert in sorted(items, key=lambda pair: pair[1].reference_ts):
            cutoff = epoch_us(alert.reference_ts)
            while cursor < len(stream) and stream[cursor].ts < cutoff:
                history.add(stream[cursor])
                cursor += 1
            history.expire(cutoff)
            snapshot = dict(
                zip(
                    ACCOUNT_FEATURES,
                    history.snapshot(cutoff, history.single_unit(), coverage_start),
                )
            )
            output[i] = {name: snapshot[name.removeprefix("account_")] for name in FEATURE_NAMES}
    return pa.Table.from_pylist(
        output, schema=pa.schema([(n, pa.float64()) for n in FEATURE_NAMES])
    )
