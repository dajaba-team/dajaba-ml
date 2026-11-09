"""한 계좌의 과거 정보만 유지하며, 오프라인과 요청 피처가 공유하는 상태입니다."""

import math
from collections import Counter, deque
from dataclasses import dataclass
from decimal import Decimal, localcontext

from .schema import ACCOUNT_FEATURES, WINDOWS, log_amount


@dataclass(frozen=True, slots=True)
class Event:
    ts: int
    direction: int  # 0은 출금, 1은 입금
    other: str
    amount: Decimal | None
    unit: str | None


class Window:
    def __init__(self, seconds: int):
        self.duration = seconds * 1000000
        self.events: deque[Event] = deque()
        self.counts = [0, 0]
        self.parties = [Counter(), Counter()]
        self.sums: list[dict[str, Decimal]] = [{}, {}]
        self.money_counts = [Counter(), Counter()]

    def _change(self, event: Event, sign: int):
        side = event.direction
        self.counts[side] += sign
        parties = self.parties[side]
        parties[event.other] += sign
        if parties[event.other] == 0:
            del parties[event.other]
        if event.unit is not None and event.amount is not None:
            with localcontext() as context:
                context.prec = 60
                self.sums[side][event.unit] = (
                    self.sums[side].get(event.unit, Decimal(0)) + sign * event.amount
                )
            self.money_counts[side][event.unit] += sign
            if self.money_counts[side][event.unit] == 0:
                del self.money_counts[side][event.unit]
                del self.sums[side][event.unit]

    def expire(self, as_of: int):
        while self.events and self.events[0].ts < as_of - self.duration:
            self._change(self.events.popleft(), -1)

    def add(self, event: Event):
        self.events.append(event)
        self._change(event, 1)


class AccountHistory:
    def __init__(self):
        windows = {name: Window(seconds) for name, seconds in WINDOWS.items()}
        self.windows = list(windows.values())
        self.window_24h = windows["24h"]
        self.last_ts: int | None = None
        self.total = 0

    def expire(self, as_of: int):
        for window in self.windows:
            window.expire(as_of)

    def add(self, event: Event):
        if self.last_ts is not None and event.ts < self.last_ts:
            raise ValueError("History events are not chronological")
        for window in self.windows:
            window.add(event)
        self.last_ts = event.ts
        self.total += 1

    def single_unit(self) -> str | None:
        units = set(self.window_24h.money_counts[0]) | set(self.window_24h.money_counts[1])
        return next(iter(units)) if len(units) == 1 else None

    def snapshot(
        self, as_of: int, unit: str | None, coverage_start: int | None
    ) -> tuple[float | None, ...]:
        if self.last_ts is not None and self.last_ts >= as_of:
            raise ValueError("Current/future timestamp reached a past-only snapshot")
        self.expire(as_of)
        values: list[float | None] = []
        for window in self.windows:
            sums = (
                [None, None]
                if unit is None
                else [log_amount(s.get(unit, Decimal(0))) for s in window.sums]
            )
            values.extend(
                [
                    float(window.counts[0]),
                    float(window.counts[1]),
                    float(len(window.parties[0])),
                    float(len(window.parties[1])),
                    sums[0],
                    sums[1],
                    float(coverage_start is None or as_of - window.duration < coverage_start),
                ]
            )
        primary_has_history = sum(self.window_24h.counts) > 0
        values.extend(
            [
                None
                if not primary_has_history or self.last_ts is None
                else math.log1p((as_of - self.last_ts) / 1000000),
                float(not primary_has_history),
                float(unit is None),
            ]
        )
        if len(values) != len(ACCOUNT_FEATURES):
            raise RuntimeError("Feature schema drift")
        return tuple(values)
