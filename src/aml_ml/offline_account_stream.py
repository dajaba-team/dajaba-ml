"""외부 정렬된 소유자 행에 R02를 적용하며 현재 B7 구간만 유지합니다.

계좌의 전 기간 이력이나 경보·근거 전체를 누적하지 않습니다.
제한은 원천·버킷의 누적량이 아니라 현재 상태와 완결된 경보 하나에 적용합니다.
"""

import json
import math
from collections import deque
from dataclasses import dataclass
from decimal import Decimal
from itertools import groupby

from .alert_dataset import OfflineAlert, offline_proxy_targets
from .history import Event, Window
from .offline_eligibility import ledger_row
from .offline_features import FEATURE_NAMES
from .offline_measurements import milestone
from .offline_progress import progress
from .offline_rules import RULE_ID, _json_bytes
from .schema import currency_bucket, epoch_us, log_amount
from .sliding_window import evaluate_minute, minute_window


def check_budget(metric, current, addition, limit, context):
    projected = current + addition
    if projected > limit:
        details = dict(context, metric=metric, current=current, projected=projected, limit=limit)
        raise ValueError("Offline workset budget exceeded: " + json.dumps(details, sort_keys=True))


@dataclass(slots=True)
class ActiveRow:
    row: dict
    size: int
    evidence_split: str | None = None
    evidence_emitted: bool = False


def _features(window, cutoff, last_ts, coverage):
    units = set(window.money_counts[0]) | set(window.money_counts[1])
    unit = next(iter(units)) if len(units) == 1 else None
    present = sum(window.counts) > 0
    values = {
        "account_out_count_24h": float(window.counts[0]),
        "account_in_count_24h": float(window.counts[1]),
        "account_out_counterparties_24h": float(len(window.parties[0])),
        "account_in_counterparties_24h": float(len(window.parties[1])),
        "account_out_amount_log1p_24h": None
        if unit is None
        else log_amount(window.sums[0].get(unit, Decimal(0))),
        "account_in_amount_log1p_24h": None
        if unit is None
        else log_amount(window.sums[1].get(unit, Decimal(0))),
        "account_coverage_partial_24h": float(cutoff - window.duration < coverage),
        "account_previous_gap_log1p": math.log1p((cutoff - last_ts) / 1_000_000)
        if present and last_ts is not None
        else None,
        "account_history_absent": float(not present),
        "account_money_unit_unknown": float(unit is None),
    }
    if set(values) != set(FEATURE_NAMES):
        raise ValueError("Streaming account feature allowlist drift")
    return {name: values[name] for name in FEATURE_NAMES}


def account_alerts(
    ordered_rows,
    source,
    policy,
    *,
    coverage,
    train_before,
    test_from,
    settings,
    row_size,
    stats,
    acknowledge=None,
    owner_done=None,
):
    """정확한 (bucket, owner, ts, ordinal) 순서로 읽고 완결 경보를 하나씩 반환합니다."""
    max_rows, max_bytes = settings["max_partition_rows"], settings["max_partition_bytes"]
    threshold = policy.min_counterparties
    if type(threshold) is not int or threshold < 1:
        raise ValueError("A positive confirmed counterparty threshold is required")
    config_hash = policy.config_hash
    previous_key = None
    for (bucket, account), owner_rows in groupby(
        ordered_rows, key=lambda r: (r["_bucket"], r["_owner"])
    ):
        active: deque[ActiveRow] = deque()
        window = Window(86400)
        state_bytes = incoming_uid_bytes = 0
        last_ts = None
        context = {"stage": "account_window", "bucket": bucket, "input_rows": stats["input_rows"]}

        def expire(cutoff):
            nonlocal state_bytes, incoming_uid_bytes
            lower = cutoff - window.duration
            while active and epoch_us(active[0].row["ts"]) < lower:
                expired = active.popleft()
                state_bytes -= expired.size
                if expired.row["to_account_key"] == account:
                    incoming_uid_bytes -= _json_bytes(expired.row["transaction_uid"])
            window.expire(cutoff)

        for minute, minute_rows in groupby(
            owner_rows, key=lambda r: r["ts"].replace(second=0, microsecond=0)
        ):
            # 이번 분을 읽기 전에 유휴 만료를 포함해 직전 시계상의 분과 비교합니다.
            # 같은 시각의 행이 많아도 groupby는 필요한 만큼만 읽습니다.
            expire(epoch_us(minute))
            previous_matches = len(window.parties[1]) >= threshold
            current_window = minute_window(minute)
            cutoff = epoch_us(current_window.end)
            expire(cutoff)
            for row in minute_rows:
                milestone("first_final_owner_sort_row")
                key = (bucket, account, row["ts"], row["source_row_id"])
                if previous_key is not None and key <= previous_key:
                    raise ValueError("Duplicate/unsorted owner transaction")
                previous_key = key
                if account not in (row["from_account_key"], row["to_account_key"]):
                    raise ValueError("Owner does not match the original transaction")
                context["source_row_id"] = row["source_row_id"]
                context["owned_rows_processed"] = stats["owned_rows_processed"]
                # Event, deque 참조와 정확한 상대방 집계에 필요한 여유 공간입니다.
                size = row_size(row) + 512
                check_budget("active_window_rows", len(active), 1, max_rows, context)
                check_budget("active_window_workset_bytes", state_bytes, size, max_bytes, context)
                side = int(row["to_account_key"] == account)
                amount_key, currency_key, peer_key = (
                    ("amount_received", "currency_received_raw", "from_account_key")
                    if side
                    else ("amount_paid", "currency_paid_raw", "to_account_key")
                )
                amount = row[amount_key] if source.kind == "ibm" else None
                unit = currency_bucket(source.id, row[currency_key]) if amount is not None else None
                last_ts = epoch_us(row["ts"])
                window.add(Event(last_ts, side, row[peer_key], amount, unit))
                active.append(ActiveRow(row, size))
                state_bytes += size
                if side:
                    incoming_uid_bytes += _json_bytes(row["transaction_uid"])
                stats["owned_rows_processed"] += 1
                stats["max_active_window_rows"] = max(stats["max_active_window_rows"], len(active))
                stats["max_active_window_workset_bytes"] = max(
                    stats["max_active_window_workset_bytes"], state_bytes
                )
                if acknowledge is not None:
                    acknowledge(row)
                if stats["owned_rows_processed"] % 4096 == 0:
                    progress(
                        "account_windows",
                        processed=stats["owned_rows_processed"],
                        total=stats["owned_rows"],
                        unit="owned_rows",
                    )
            current_matches = len(window.parties[1]) >= threshold
            decision = evaluate_minute(
                minute,
                has_account_transactions=True,
                condition=lambda w: previous_matches if w.end == minute else current_matches,
            )
            if not decision.emit:
                continue
            count = window.counts[1]
            check_budget("alert_evidence_references", 0, count, max(1, max_bytes // 512), context)
            metadata = dict(
                source_id=source.id,
                rule_id=RULE_ID,
                rule_version=str(policy.rule_version),
                rule_config_hash=config_hash,
                account_key=account,
                reference_ts=decision.reference_ts.isoformat(timespec="microseconds"),
                window_start=decision.window.start.isoformat(timespec="microseconds"),
                window_end=decision.window.end.isoformat(timespec="microseconds"),
                transaction_uids=(),
                paths=(),
            )
            # 가장 긴 식별자·레코드 이름과 생성 UID가 이 고정 여유 공간에 들어갑니다.
            # UID의 tuple/list/set을 할당하기 전에 확인합니다.
            payload_size = _json_bytes(metadata) + 1024 + incoming_uid_bytes + max(0, count - 1) * 2
            check_budget("single_alert_payload_bytes", 0, payload_size, max_bytes, context)
            incoming = [entry for entry in active if entry.row["to_account_key"] == account]
            alert = OfflineAlert(
                source_id=source.id,
                rule_id=RULE_ID,
                rule_version=str(policy.rule_version),
                rule_config_hash=config_hash,
                account_key=account,
                reference_ts=decision.reference_ts,
                window_start=decision.window.start,
                window_end=decision.window.end,
                transaction_uids=tuple(entry.row["transaction_uid"] for entry in incoming),
            )
            labels = {
                entry.row["transaction_uid"]: entry.row["_offline_label"] for entry in incoming
            }
            target = offline_proxy_targets([alert], labels)[0]
            ledger = ledger_row(
                alert,
                target,
                train_before=train_before,
                test_from=test_from,
                horizon_seconds=settings.get("horizon_seconds", 86400),
            )
            split = ledger["original_split"]
            target["split"] = split
            new_evidence = []
            for entry in incoming:
                if ledger["supervised_eligible"]:
                    if entry.evidence_split is not None and entry.evidence_split != split:
                        raise ValueError("Full evidence overlaps eligible splits")
                    entry.evidence_split = split
                if not entry.evidence_emitted:
                    new_evidence.append(entry.row)
                    entry.evidence_emitted = True
            # 후보·근거 보존은 학습 적격 여부와 독립적입니다. B7 상태는
            # 양쪽 경계와 라벨 누락 행을 지나서도 연속적으로 유지됩니다.
            yield alert, _features(window, cutoff, last_ts, coverage), target, new_evidence, ledger
            del incoming, labels, new_evidence, alert, target, ledger
        # groupby가 다음 소유자의 첫 행을 미리 보관할 수 있습니다. 위에서 처리한
        # 행만 확인 완료했으므로, 저장 커서에는 아직 처리하지 않은 선행 읽기를 넣지 않습니다.
        if owner_done is not None:
            owner_done(bucket, account)
