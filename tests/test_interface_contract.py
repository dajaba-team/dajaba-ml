from datetime import datetime, timedelta, timezone

import pytest

from aml_ml.api import AlertInput
from aml_ml.interface_contract import (
    FailureCode,
    format_system_timestamp,
    is_score_retryable,
    parse_system_timestamp,
)
from aml_ml.schema import epoch_us


@pytest.mark.parametrize("code", [None, *FailureCode])
def test_confirmed_retry_filter_preserves_success_and_permanent_failures(code):
    assert is_score_retryable("FAILED", code) is (code in (None, FailureCode.COMPUTE_ERROR))
    assert not is_score_retryable("SUCCEEDED", code)
    assert not is_score_retryable("PENDING", code)


def test_unknown_failure_code_is_not_silently_retried():
    with pytest.raises(ValueError):
        is_score_retryable("FAILED", "UNKNOWN_NEW_CODE")


def test_system_utc_and_source_time_are_separate():
    instant = datetime(2026, 11, 2, 9, 0, tzinfo=timezone(timedelta(hours=9)))
    wire = format_system_timestamp(instant)
    assert wire == "2026-11-02T00:00:00Z"
    assert parse_system_timestamp(wire) == instant
    source = datetime(2022, 9, 2, 11, 0)
    assert source.isoformat() == "2022-09-02T11:00:00"
    with pytest.raises(ValueError):
        format_system_timestamp(source)
    with pytest.raises(ValueError):
        parse_system_timestamp("2026-11-02T00:00:00+00:00")
    with pytest.raises(ValueError):
        epoch_us(parse_system_timestamp(wire))


def test_string_alert_id_is_preserved_separately_from_transaction_ids():
    alert = AlertInput.from_mapping(
        dict(
            alert_id="ALT-EXAMPLE-001",
            rule_id="R02_FAN_IN_24H",
            account_key="10_000ABC",
            reference_ts="2022-09-02T11:01:00",
            window_start="2022-09-01T11:01:00",
            evidence_transaction_ids=[9007199254740993],
            typology="FAN_IN",
        )
    )
    assert alert.alert_id == "ALT-EXAMPLE-001"
    assert alert.evidence_transaction_ids == (9007199254740993,)
