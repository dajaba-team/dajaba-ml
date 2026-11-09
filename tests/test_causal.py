import math
from datetime import datetime
from decimal import Decimal

import pandas as pd
import pyarrow as pa
import pytest

from aml_ml import features
from aml_ml import history as history_module
from aml_ml.history import AccountHistory, Event
from aml_ml.preprocessing import Preprocessor
from aml_ml.schema import ACCOUNT_FEATURES, CATEGORICAL, NUMERIC, epoch_us
from aml_ml.targets import alert_proxy_targets


def test_strict_history_boundary_currency_and_no_current_timestamp():
    history = AccountHistory()
    origin = epoch_us(datetime(2022, 1, 1))
    history.add(Event(origin, 0, "B", Decimal("10.123456789012"), "USD"))
    history.add(Event(origin, 0, "B", Decimal("900"), "EUR"))
    history.add(Event(origin, 1, "C", Decimal("5"), "USD"))
    query = origin + 3600 * 1000000
    data = dict(zip(ACCOUNT_FEATURES, history.snapshot(query, "USD", origin - 7 * 86400 * 1000000)))
    assert data["out_count_1h"] == 2 and data["out_counterparties_1h"] == 1
    assert data["out_amount_log1p_1h"] == pytest.approx(math.log1p(10.123456789012))
    later = dict(zip(ACCOUNT_FEATURES, history.snapshot(query + 1, "USD", origin)))
    assert later["out_count_1h"] == 0
    with pytest.raises(ValueError, match="Current/future"):
        history.snapshot(origin, "USD", origin)


def test_primary_tail_does_not_use_older_than_24h_history():
    h = AccountHistory()
    h.add(Event(0, 0, "B", Decimal(2), "USD"))
    data = dict(zip(ACCOUNT_FEATURES, h.snapshot(2 * 86400 * 1000000, "USD", 0)))
    assert data["out_count_7d"] == 1 and data["out_count_24h"] == 0
    assert data["history_absent"] == 1 and data["previous_gap_log1p"] is None


@pytest.mark.parametrize("order", [("1h", "24h", "7d"), ("24h", "1h", "7d"), ("1h", "7d", "24h")])
def test_24h_currency_and_tail_do_not_depend_on_window_definition_order(monkeypatch, order):
    windows = {name: history_module.WINDOWS[name] for name in order}
    monkeypatch.setattr(history_module, "WINDOWS", windows)
    history = AccountHistory()
    hour = 3600 * 1000000
    assert isinstance(history.windows, list)
    assert [window.duration for window in history.windows] == [
        seconds * 1000000 for seconds in windows.values()
    ]
    assert history.window_24h is history.windows[order.index("24h")]
    history.add(Event(0, 0, "B", Decimal(10), "USD"))
    history.add(Event(46 * hour, 1, "C", Decimal(20), "EUR"))
    history.expire(48 * hour)
    assert history.single_unit() == "EUR"
    snapshot = history.snapshot(48 * hour, "EUR", -7 * 24 * hour)
    assert snapshot[-3] == pytest.approx(math.log1p(2 * 3600))
    assert snapshot[-2:] == (0.0, 0.0)
    history.expire(73 * hour)
    assert history.single_unit() is None
    assert history.snapshot(73 * hour, None, -7 * 24 * hour)[-3:] == (None, 1.0, 1.0)


def serving_rows():
    return pd.DataFrame(
        [
            dict(
                transaction_id=1,
                ts=datetime(2022, 1, 2, 0),
                from_account_key="001_A",
                to_account_key="2_B",
                amount_paid=Decimal(10),
                currency_paid="US Dollar",
                amount_received=Decimal(10),
                currency_received="US Dollar",
                payment_format="Wire",
                is_self_transfer=False,
            ),
            dict(
                transaction_id=2,
                ts=datetime(2022, 1, 2, 0, 30),
                from_account_key="01_A",
                to_account_key="2_B",
                amount_paid=Decimal(999),
                currency_paid="US Dollar",
                amount_received=Decimal(999),
                currency_received="US Dollar",
                payment_format="Wire",
                is_self_transfer=False,
            ),
            dict(
                transaction_id=3,
                ts=datetime(2022, 1, 2, 0, 10),
                from_account_key="001_A",
                to_account_key="001_A",
                amount_paid=Decimal(9999),
                currency_paid="US Dollar",
                amount_received=Decimal(9999),
                currency_received="US Dollar",
                payment_format="Wire",
                is_self_transfer=True,
            ),
        ]
    )


def test_alert_feature_adapter_is_shared_causal_and_isolates_missing_evidence():
    rows = serving_rows()
    alert = dict(
        alert_id=7,
        rule_id="R02_FAN_IN_24H",
        account_key="001_A",
        reference_ts="2022-01-02T00:30:00",
        window_start="2022-01-01T00:30:00",
        evidence_transaction_ids=[1],
        typology="FAN_IN",
        typology_source="RULE",
    )
    bad = {**alert, "alert_id": 8, "evidence_transaction_ids": [999]}
    aliases = {"001_A": "A", "01_A": "A", "2_B": "B"}
    result = features(
        [alert, bad], rows, coverage_start=datetime(2022, 1, 1), account_aliases=aliases
    )
    assert result.loc[0, "account_out_count_24h"] == 1
    assert result.loc[0, "account_out_amount_log1p_24h"] == pytest.approx(math.log1p(10))
    assert result.loc[0, "data_quality_status"] == "OK"
    assert result.loc[1, "data_quality_status"] == "INSUFFICIENT"
    assert result.loc[1, "failure_code"] == "TRANSACTION_NOT_FOUND"
    assert result.loc[0, "failure_code"] is None
    # 같은 시각이나 미래의 이벤트 변경이 이전 경보 피처를 바꾸면 안 됩니다.
    changed = rows.copy()
    future_amount = Decimal("999999999999")
    amounts = [
        future_amount if index == 1 else value for index, value in rows["amount_paid"].items()
    ]
    changed["amount_paid"] = pd.Series(amounts, index=rows.index, dtype=object)
    assert isinstance(changed.at[1, "amount_paid"], Decimal)
    assert changed.at[1, "amount_paid"] == future_amount
    pd.testing.assert_series_equal(
        result.iloc[0],
        features(
            [alert, bad], changed, coverage_start=datetime(2022, 1, 1), account_aliases=aliases
        ).iloc[0],
    )
    limited = features([alert], rows)
    assert limited.loc[0, "data_quality_status"] == "PARTIAL"
    # 확정된 정규화 서비스 키가 이미 공통 키 공간이므로,
    # 완전한 이력 조회에 추가 별칭 테이블은 필요하지 않습니다.
    direct = features([alert], rows, coverage_start=datetime(2022, 1, 1))
    assert direct.loc[0, "data_quality_status"] == "OK"
    assert direct.loc[0, "account_out_count_24h"] == 1
    labelled = rows.assign(is_laundering=0)
    with pytest.raises(ValueError, match="labels"):
        features([alert], labelled)


@pytest.mark.parametrize(
    "reference, expected_count, quality",
    [("2022-01-02T02:00:00", 1, "OK"), ("2022-01-04T00:00:00", 0, "INSUFFICIENT")],
)
def test_serving_quality_uses_24h_instead_of_1h_or_7d(reference, expected_count, quality):
    alert = dict(
        alert_id=7,
        rule_id="FIXTURE_HISTORY_QUALITY",
        account_key="001_A",
        reference_ts=reference,
        window_start="2022-01-01T00:00:00",
        evidence_transaction_ids=[1],
        typology="FAN_IN",
    )
    result = features([alert], serving_rows(), coverage_start=datetime(2021, 12, 25))
    assert result.loc[0, "account_out_count_1h"] == 0
    assert result.loc[0, "account_out_count_7d"] == 1
    assert result.loc[0, "account_out_count_24h"] == expected_count
    assert result.loc[0, "data_quality_status"] == quality


def raw_table(values, categories):
    return pa.table(
        {
            **{name: pa.array(values, type=pa.float64()) for name in NUMERIC},
            **{name: pa.array(categories, type=pa.string()) for name in CATEGORICAL},
        }
    )


def test_preprocessing_is_train_only_unknown_safe_and_serializable(tmp_path):
    processor = Preprocessor()
    train = raw_table([1.0, 3.0, None], ["a", "b", None])
    processor.partial_fit(train, split="train")
    state = processor.freeze()
    assert state["numeric"]["sender_out_count_24h"]["mean"] == 2.0
    assert not any("_1h" in c or "_7d" in c for c in state["feature_columns"])
    future = raw_table([100000.0, None], ["unseen", "a"])
    encoded = processor.transform(future)
    assert encoded.column("payment_format__category").to_pylist()[0] == 0
    assert encoded.column("sender_out_count_24h").to_pylist()[1] == 2.0
    path = tmp_path / "state.json"
    processor.save(path)
    assert Preprocessor.load(path).transform(future).equals(encoded)
    with pytest.raises(ValueError, match="train only"):
        Preprocessor().partial_fit(future, split="test")
    assert processor.state == state


def test_alert_proxy_labels_require_explicit_ops_mapping_and_do_not_guess_missing():
    alerts = [
        {"alert_id": 1, "evidence_transaction_ids": [101, 102]},
        {"alert_id": 2, "evidence_transaction_ids": [999]},
    ]
    mapping = pd.DataFrame({"transaction_id": [101, 102], "transaction_uid": ["a", "b"]})
    labels = pd.DataFrame({"transaction_uid": ["a", "b"], "source_positive": [0, 1]})
    target = alert_proxy_targets(alerts, mapping, labels, id_namespace="BACKEND_OPS")
    assert target.loc[0, "y_alert_proxy"] == 1 and pd.isna(target.loc[1, "y_alert_proxy"])
    with pytest.raises(ValueError, match="actual-backend-ID"):
        alert_proxy_targets(alerts, mapping, labels, id_namespace="SOURCE_CANDIDATE")
