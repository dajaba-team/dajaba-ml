from dataclasses import replace
from datetime import datetime, timezone

import pytest

from aml_ml.alert_dataset import OfflineAlert, chronological_split, offline_proxy_targets


def uid(row):
    return "saml_d:" + "a" * 64 + ":" + str(row)


def candidate(day=2, rows=(0, 1), **changes):
    base = OfflineAlert(
        source_id="saml_d",
        rule_id="R04_CYCLE",
        rule_version="test-fixture-v1",
        rule_config_hash="b" * 64,
        account_key="001_A",
        reference_ts=datetime(2022, 1, day),
        window_start=datetime(2022, 1, 1),
        window_end=datetime(2022, 1, day),
        transaction_uids=tuple(uid(row) for row in rows),
        paths=(tuple(uid(row) for row in rows),),
    )
    return replace(base, **changes)


def test_full_path_proxy_missing_label_stays_unresolved():
    alert = candidate(rows=(0, 1, 2))
    result = offline_proxy_targets([alert], {uid(0): 0, uid(1): 0, uid(2): 1})[0]
    assert result["y_alert_proxy"] == 1
    assert result["evidence_count"] == 3
    result = offline_proxy_targets([alert], {uid(0): 1, uid(1): 0})[0]
    assert result["y_alert_proxy"] is None
    assert result["missing_label_count"] == 1
    assert offline_proxy_targets([alert], {uid(i): 0 for i in range(3)})[0]["y_alert_proxy"] == 0


def test_identity_stable_but_changes_with_contract_or_exact_account():
    alert = candidate()
    assert (
        alert.training_alert_uid
        == replace(
            alert, transaction_uids=tuple(reversed(alert.transaction_uids))
        ).training_alert_uid
    )
    for changes in (
        {"account_key": "01_A"},
        {"rule_config_hash": "c" * 64},
        {"window_end": datetime(2022, 1, 3)},
    ):
        assert alert.training_alert_uid != replace(alert, **changes).training_alert_uid


@pytest.mark.parametrize("version", [None, 1, 1.0, True, "", " \t\n"])
def test_direct_candidate_rejects_non_string_or_blank_rule_version(version):
    with pytest.raises(ValueError, match="rule_version must be a nonempty string"):
        candidate(rule_version=version)


def test_direct_candidate_preserves_explicit_version_and_identity():
    alert = candidate(rule_version="fixture-r02-v2")
    assert alert.rule_version == "fixture-r02-v2"
    assert alert.training_alert_uid != candidate(rule_version="fixture-r02-v1").training_alert_uid


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"transaction_uids": ("10001",)}, "source-namespaced"),
        ({"transaction_uids": ("saml_d:1",)}, "SHA-256"),
        ({"transaction_uids": (uid(0), uid(0))}, "unique"),
        ({"paths": ()}, "complete paths"),
        ({"paths": ((uid(0),),)}, "Path union"),
        ({"reference_ts": datetime(2022, 1, 2, tzinfo=timezone.utc)}, "naive"),
        ({"rule_config_hash": "pending"}, "fingerprint"),
    ],
)
def test_rejects_ambiguous_candidate(changes, match):
    with pytest.raises(ValueError, match=match):
        candidate(**changes)


def test_split_exact_boundaries_atomic_timestamp_and_label_blind():
    alerts = [candidate(4, (4, 5)), candidate(2), candidate(3, (2, 3)), candidate(3, (6, 7))]
    splits = chronological_split(
        alerts, train_before=datetime(2022, 1, 3), test_from=datetime(2022, 1, 4)
    )
    assert {key: len(value) for key, value in splits.items()} == {
        "train": 1,
        "validation": 2,
        "test": 1,
    }
    assert splits == chronological_split(
        list(reversed(alerts)), train_before=datetime(2022, 1, 3), test_from=datetime(2022, 1, 4)
    )


def test_split_rejects_shared_evidence_and_mixed_sources():
    with pytest.raises(ValueError, match="overlaps splits"):
        chronological_split(
            [candidate(2), candidate(3, (1, 2))],
            train_before=datetime(2022, 1, 3),
            test_from=datetime(2022, 1, 4),
        )
    other = candidate(
        source_id="other",
        transaction_uids=("other:" + "a" * 64 + ":0",),
        paths=(("other:" + "a" * 64 + ":0",),),
    )
    with pytest.raises(ValueError, match="independently"):
        chronological_split(
            [candidate(), other], train_before=datetime(2022, 1, 3), test_from=datetime(2022, 1, 4)
        )


def test_duplicate_candidates_and_invalid_labels_fail():
    with pytest.raises(ValueError, match="Duplicate"):
        offline_proxy_targets([candidate(), candidate()], {})
    with pytest.raises(ValueError, match="integer"):
        offline_proxy_targets([candidate()], {uid(0): True})


def test_r04_rejects_four_hops_but_r05_limit_is_not_invented():
    with pytest.raises(ValueError, match="at most 3 hops"):
        candidate(rows=(0, 1, 2, 3))
    path = tuple(uid(i) for i in range(4))
    alert = replace(candidate(), rule_id="R05_STACK", transaction_uids=path, paths=(path,))
    assert len(alert.paths[0]) == 4
