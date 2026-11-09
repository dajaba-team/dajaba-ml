from datetime import datetime, timedelta, timezone

import pytest

from aml_ml.sliding_window import evaluate_minute, minute_window


def test_midnight_entire_minute_and_half_open_boundaries():
    e = datetime(2022, 9, 2, 23, 59)
    window = minute_window(e)
    assert window.start == datetime(2022, 9, 2)
    assert window.end == datetime(2022, 9, 3)
    assert window.contains(window.start)
    assert window.contains(e + timedelta(seconds=59, microseconds=999999))
    assert not window.contains(window.start - timedelta(microseconds=1))
    assert not window.contains(window.end)
    result = evaluate_minute(e, has_account_transactions=True, condition=lambda _: True)
    assert result.reference_ts == window.end
    assert result.batch_date == e.date()  # 자정을 넘긴 reference_ts의 날짜를 쓰지 않습니다.
    assert not result.emit  # 참 상태가 유지되며, 같은 분의 모든 거래는 한 번의 평가를 공유합니다.


def test_idle_expiry_then_new_trade_reactivates_without_last_trade_state():
    # 어제의 마지막 관측 거래에서 상대방 둘이 예시 fixture 임계값을 충족했습니다.
    # 유휴 중 만료된 뒤 다음 분에는 새로운 상대방 둘이 있습니다.
    e = datetime(2022, 9, 3, 12, 0)
    trades = [
        (datetime(2022, 9, 2, 11, 0), "old-A"),
        (datetime(2022, 9, 2, 11, 1), "old-B"),
        (e, "new-A"),
        (e, "new-B"),
    ]

    def fixture_condition(window):
        return len({peer for ts, peer in trades if window.contains(ts)}) >= 2

    result = evaluate_minute(e, has_account_transactions=True, condition=fixture_condition)
    assert not result.previous_matches
    assert result.current_matches
    assert result.emit
    last_trade_minute = datetime(2022, 9, 2, 11, 1)
    assert fixture_condition(minute_window(last_trade_minute))
    assert not evaluate_minute(
        e + timedelta(minutes=1),
        has_account_transactions=False,
        condition=fixture_condition,
    ).emit


def test_continuous_true_is_not_repeated_and_missing_condition_is_not_false():
    e = datetime(2022, 9, 3, 12)
    assert not evaluate_minute(e, has_account_transactions=True, condition=lambda _: True).emit
    assert not evaluate_minute(e, has_account_transactions=True, condition=lambda _: False).emit
    with pytest.raises(TypeError, match="boolean"):
        evaluate_minute(e, has_account_transactions=True, condition=lambda _: None)  # type: ignore

    def unavailable(_):
        raise ValueError("history unavailable")

    with pytest.raises(ValueError, match="history unavailable"):
        evaluate_minute(e, has_account_transactions=True, condition=unavailable)


@pytest.mark.parametrize(
    "timestamp",
    [datetime(2022, 9, 3, 12, 0, 1), datetime(2022, 9, 3, tzinfo=timezone.utc)],
)
def test_never_reinterprets_source_timezone_or_silently_truncates(timestamp):
    with pytest.raises(ValueError):
        minute_window(timestamp)
