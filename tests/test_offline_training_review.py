"""UID 동률, 정확한 validation 선택과 재연산 없는 모델 다시 읽기의 회귀 검사입니다."""

from datetime import datetime
from fractions import Fraction
from itertools import permutations, product

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from offline_test_data import _build_review_dataset, synthetic_support

from aml_ml import offline_pipeline, offline_training
from aml_ml.offline_rules import fingerprint
from aml_ml.offline_training import (
    load_model,
    metrics,
    predict_raw,
    select_validation_threshold,
    train_evaluate,
    verify_model,
)
from aml_ml.storage import read_json, sha256, write_json


@pytest.fixture
def trained_review_model(source, policy, tmp_path):
    dataset, model = tmp_path / "review-data", tmp_path / "review-model"
    _build_review_dataset(source, policy, dataset)
    train_evaluate(
        dataset,
        source,
        policy,
        model,
        execution_scope="FIXTURE_ONLY",
        model_version="synthetic-review-baseline",
        support_policy=synthetic_support(),
        k=1,
        iterations=3,
    )
    return dataset, model


def test_top_k_ties_use_uid_over_reference_time_and_input_order():
    rows = [
        (datetime(2022, 1, 1), "TRAIN-ALERT-z", 0),
        (datetime(2022, 1, 2), "TRAIN-ALERT-a", 1),
        (datetime(2022, 1, 3), "TRAIN-ALERT-m", 0),
    ]
    expected = None
    for ordered in permutations(rows):
        result = metrics(
            np.array([row[2] for row in ordered]),
            np.full(3, 0.5),
            0.5,
            1,
            training_alert_uids=[row[1] for row in ordered],
        )
        assert result["top_k_tie_policy"] == "SCORE_DESC_THEN_TRAINING_ALERT_UID_ASC"
        assert result["precision_at_k"] == result["recall_at_k"] == 1.0
        assert result["average_precision"] == pytest.approx(1 / 3)
        if expected is None:
            expected = result
        assert result == expected


def test_metrics_without_uids_reports_input_order_fallback():
    first = metrics(np.array([1, 0]), np.array([0.5, 0.5]), 0.5, 1)
    reversed_rows = metrics(np.array([0, 1]), np.array([0.5, 0.5]), 0.5, 1)
    assert first["top_k_tie_policy"] == "SCORE_DESC_STABLE_INPUT_ORDER"
    assert reversed_rows["top_k_tie_policy"] == first["top_k_tie_policy"]
    assert first["precision_at_k"] == 1.0
    assert reversed_rows["precision_at_k"] == 0.0
    assert first["average_precision"] == reversed_rows["average_precision"] == 0.5


def test_score_order_precedes_uid_and_unsigned_endpoints_are_valid():
    result = metrics(
        np.array([0, 1], dtype=np.uint8),
        np.array([0, 1], dtype=np.uint8),
        1.0,
        1,
        training_alert_uids=["a", "z"],
    )
    assert result["precision_at_k"] == result["f1"] == 1.0
    assert np.isfinite(result["log_loss"])


@pytest.mark.parametrize(
    "uids",
    [["same", "same"], ["a"], ["a", ""], ["a", None], ["a", 1], [["a"], ["b"]], "ab"],
)
def test_metrics_rejects_invalid_uid_alignment(uids):
    with pytest.raises(ValueError, match="UID"):
        metrics(np.array([1, 0]), np.array([0.5, 0.5]), 0.5, 1, training_alert_uids=uids)


def _brute_threshold(y, scores):
    candidates = {0.0, 1.0, *scores}
    positives = sum(y)

    def objective(threshold):
        chosen = [label for label, score in zip(y, scores, strict=True) if score >= threshold]
        denominator = positives + len(chosen)
        return (
            Fraction(2 * sum(chosen), denominator) if denominator else Fraction(0),
            threshold,
        )

    return max(candidates, key=objective)


def test_threshold_matches_independent_brute_force_for_small_score_grids():
    for labels in product((0, 1), repeat=3):
        for scores in product((0.0, 0.5, 1.0), repeat=3):
            assert select_validation_threshold(np.array(labels), np.array(scores)) == (
                _brute_threshold(labels, scores)
            )


@pytest.mark.parametrize(
    "labels,scores,expected",
    [
        ([1, 0, 0, 1], [0.9, 0.8, 0.8, 0.7], 0.9),
        ([1, 0], [0.0, 0.0], 0.0),
        ([1, 0], [1.0, 1.0], 1.0),
        ([0, 0], [0.0, 0.8], 1.0),
        ([1, 1], [0.3, 0.7], 0.3),
    ],
)
def test_threshold_boundaries_and_higher_threshold_ties(labels, scores, expected):
    assert select_validation_threshold(np.array(labels), np.array(scores)) == expected


@pytest.mark.parametrize(
    "labels,scores",
    [
        ([], []),
        ([0, 1], [0.5]),
        ([[0], [1]], [0.1, 0.9]),
        ([0, 1], [[0.1], [0.9]]),
        (1, 0.5),
        ([0, 2], [0.1, 0.9]),
        ([0, np.nan], [0.1, 0.9]),
        ([0, 1], [np.nan, 0.9]),
        ([0, 1], [np.inf, 0.9]),
        ([0, 1], [-0.1, 0.9]),
        ([0, 1], [0.1, 1.1]),
        (["0", "1"], [0.1, 0.9]),
        ([0, 1], [0.1 + 0j, 0.9 + 0j]),
    ],
)
def test_metrics_and_threshold_reject_invalid_evaluation_arrays(labels, scores):
    with pytest.raises(ValueError):
        metrics(np.asarray(labels), np.asarray(scores), 0.5, 1)
    with pytest.raises(ValueError):
        select_validation_threshold(np.asarray(labels), np.asarray(scores))


@pytest.mark.parametrize(
    "threshold,k",
    [
        (np.nan, 1),
        (np.inf, 1),
        (-0.1, 1),
        (1.1, 1),
        ([0.5], 1),
        ("0.5", 1),
        (0.5, 0),
        (0.5, True),
        (0.5, 1.0),
    ],
)
def test_metrics_rejects_invalid_threshold_and_k(threshold, k):
    with pytest.raises(ValueError):
        metrics(np.array([0, 1]), np.array([0.1, 0.9]), threshold, k)


def test_labelled_keeps_uid_mask_aligned_with_nullable_truth(tmp_path):
    pq.write_table(pa.table({"feature": [1.0, 2.0, 3.0]}), tmp_path / "X_test.parquet")
    pq.write_table(
        pa.table({"training_alert_uid": ["z", "unknown", "a"], "y_alert_proxy": [0, None, 1]}),
        tmp_path / "y_test.parquet",
    )
    _, labels, known, uids = offline_training._labelled(tmp_path, "test", ["feature"])
    assert labels.tolist() == [0.0, 1.0]
    assert known.tolist() == [True, False, True]
    assert uids.tolist() == ["z", "a"]


def test_train_and_explicit_verify_share_uid_metrics_and_each_replay_once(
    source, policy, tmp_path, monkeypatch
):
    dataset, model = tmp_path / "data", tmp_path / "model"
    _build_review_dataset(source, policy, dataset)
    original_verify = offline_training.verify_dataset
    original_labelled = offline_training._labelled
    original_selector = offline_training.select_validation_threshold
    calls, events = [], []

    def track_verify(*args, **kwargs):
        calls.append("replay")
        return original_verify(*args, **kwargs)

    def track_labelled(folder, split, columns):
        events.append(split)
        return original_labelled(folder, split, columns)

    def track_selector(y, scores):
        events.append("threshold_frozen")
        return original_selector(y, scores)

    monkeypatch.setattr(offline_training, "verify_dataset", track_verify)
    monkeypatch.setattr(offline_training, "_labelled", track_labelled)
    monkeypatch.setattr(offline_training, "select_validation_threshold", track_selector)
    monkeypatch.setattr(offline_training, "probabilities", lambda x, w, b: np.full(len(x), 0.5))
    result = train_evaluate(
        dataset,
        source,
        policy,
        model,
        execution_scope="FIXTURE_ONLY",
        model_version="synthetic-tied-baseline",
        support_policy=synthetic_support(),
        k=1,
        iterations=2,
    )
    assert calls == ["replay"]
    assert events.index("test") > events.index("threshold_frozen")
    state = read_json(model / "model.json")
    assert state["model_kind"] == "NUMPY_LOGISTIC_PROXY_BASELINE"
    assert state["fit_scope"] == "TRAIN_COMPLETE_PROXY_ONLY"
    assert state["validated_dataset_manifest_sha256"] == sha256(dataset / "manifest.json")
    for split in ("train", "validation", "test"):
        truth = pq.read_table(dataset / f"y_{split}.parquet").to_pylist()
        first = min(truth, key=lambda row: row["training_alert_uid"])
        evaluation = result["evaluation"][split]
        assert evaluation["top_k_tie_policy"] == "SCORE_DESC_THEN_TRAINING_ALERT_UID_ASC"
        assert evaluation["precision_at_k"] == first["y_alert_proxy"]
    assert verify_model(model, dataset, source, policy, execution_scope="FIXTURE_ONLY") == result
    assert calls == ["replay", "replay"]


def test_load_and_predict_do_not_replay_or_refit(trained_review_model, source, policy, monkeypatch):
    dataset, model = trained_review_model

    def replay_forbidden(*args, **kwargs):
        pytest.fail("Model loading and prediction must not replay data or fit preprocessing")

    monkeypatch.setattr(offline_training, "verify_dataset", replay_forbidden)
    monkeypatch.setattr(offline_pipeline, "_derive", replay_forbidden)
    monkeypatch.setattr(offline_pipeline.AlertPreprocessor, "partial_fit", replay_forbidden)
    state = load_model(model, dataset, source, policy, execution_scope="FIXTURE_ONLY")
    assert state["validated_dataset_manifest_sha256"] == sha256(dataset / "manifest.json")
    raw = pq.read_table(dataset / "raw.parquet")
    scores = predict_raw(model, raw, dataset, source, policy, execution_scope="FIXTURE_ONLY")
    assert len(scores) == len(raw)
    assert np.isfinite(scores).all()


@pytest.mark.parametrize("pin", [None, "0" * 64])
def test_load_rejects_missing_or_changed_validation_pin(trained_review_model, source, policy, pin):
    dataset, model = trained_review_model
    state = read_json(model / "model.json")
    if pin is None:
        state.pop("validated_dataset_manifest_sha256")
    else:
        state["validated_dataset_manifest_sha256"] = pin
    write_json(model / "model.json", state)
    inventory = read_json(model / "manifest.json")
    inventory["files"]["model.json"] = sha256(model / "model.json")
    inventory["model_fingerprint"] = fingerprint(inventory["files"])
    write_json(model / "manifest.json", inventory)
    with pytest.raises(ValueError, match="Incompatible"):
        load_model(model, dataset, source, policy, execution_scope="FIXTURE_ONLY")


def test_load_rejects_changed_manifest_bytes_with_same_dataset_fingerprint(
    trained_review_model, source, policy
):
    dataset, model = trained_review_model
    manifest = dataset / "manifest.json"
    before = read_json(manifest)
    manifest.write_bytes(manifest.read_bytes() + b"\n")
    assert read_json(manifest) == before
    with pytest.raises(ValueError, match="Incompatible"):
        load_model(model, dataset, source, policy, execution_scope="FIXTURE_ONLY")


def test_training_rejects_manifest_change_during_deep_verification(
    source, policy, tmp_path, monkeypatch
):
    dataset, model = tmp_path / "data", tmp_path / "model"
    _build_review_dataset(source, policy, dataset)
    original_verify = offline_training.verify_dataset

    def changed_after_verify(*args, **kwargs):
        result = original_verify(*args, **kwargs)
        manifest = dataset / "manifest.json"
        manifest.write_bytes(manifest.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(offline_training, "verify_dataset", changed_after_verify)
    with pytest.raises(ValueError, match="changed during"):
        train_evaluate(
            dataset,
            source,
            policy,
            model,
            execution_scope="FIXTURE_ONLY",
            model_version="synthetic-changed-input",
            support_policy=synthetic_support(),
            k=1,
            iterations=2,
        )
    assert not model.exists()
