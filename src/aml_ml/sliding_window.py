"""룰 임계값과 독립적인 B7 분 경계와 재발 조건을 제공합니다.

호출자는 완전한 원천 이력에 룰을 두 번 평가합니다. 거래가 없는 동안 만료된
상태를 처리하려면 직전 거래만 보는 것으로는 부족합니다. 여기에는 탐지기,
금액 합산, 경보 영속 저장이나 스케줄러를 구현하지 않습니다.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta


@dataclass(frozen=True)
class MinuteWindow:
    start: datetime
    end: datetime

    def contains(self, timestamp: datetime) -> bool:
        if timestamp.tzinfo is not None:
            raise ValueError("Use timezone-naive source timestamps")
        return self.start <= timestamp < self.end


@dataclass(frozen=True)
class MinuteDecision:
    window: MinuteWindow
    previous_window: MinuteWindow
    batch_date: date
    previous_matches: bool
    current_matches: bool
    emit: bool

    @property
    def reference_ts(self) -> datetime:
        return self.window.end


def minute_window(evaluation_minute: datetime) -> MinuteWindow:
    """E 시점에는 해당 분 전체를 포함하고 다음 분의 거래는 제외합니다."""
    if evaluation_minute.tzinfo is not None:
        raise ValueError("Use timezone-naive source timestamps")
    if evaluation_minute.second or evaluation_minute.microsecond:
        raise ValueError("E must be an exact minute; do not silently truncate time")
    end = evaluation_minute + timedelta(minutes=1)
    return MinuteWindow(end - timedelta(hours=24), end)


def evaluate_minute(
    evaluation_minute: datetime,
    *,
    has_account_transactions: bool,
    condition: Callable[[MinuteWindow], bool],
) -> MinuteDecision:
    """마지막 거래가 오래전이어도 시계상 연속한 두 분을 비교합니다.

    판정 함수는 마지막 관측 거래의 캐시된 참·거짓 대신 제공된 각 구간과
    동일한 고정 룰 매개변수를 사용해야 합니다. 예외는 그대로 전파하며,
    이력 누락이나 조회 실패를 조건 미충족으로 바꾸면 안 됩니다.
    """
    if type(has_account_transactions) is not bool:
        raise TypeError("has_account_transactions must be a boolean")
    current = minute_window(evaluation_minute)
    previous = minute_window(evaluation_minute - timedelta(minutes=1))
    old_match = condition(previous)
    new_match = condition(current)
    if type(old_match) is not bool or type(new_match) is not bool:
        raise TypeError("Rule conditions must return a boolean, not null or a count")
    return MinuteDecision(
        window=current,
        previous_window=previous,
        batch_date=evaluation_minute.date(),
        previous_matches=old_match,
        current_matches=new_match,
        emit=has_account_transactions and not old_match and new_match,
    )
