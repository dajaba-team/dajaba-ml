"""NumPy 분할 평가가 null 가능한 정답, UID 동률과 저장 출력을 보존하는지 검사합니다."""

from math import log

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from aml_ml import offline_training
from aml_ml.storage import write_json

SPLITS = {
    "train": ([0, None, 1], ["TRAIN-z", "TRAIN-unresolved", "TRAIN-a"]),
    "validation": ([0, 0], ["VALIDATION-z", "VALIDATION-a"]),
    "test": ([1, 1], ["TEST-z", "TEST-a"]),
}


def _write_split(folder, split, labels, uids):
    pq.write_table(pa.table({"feature": [0.0] * len(labels)}), folder / f"X_{split}.parquet")
    pq.write_table(
        pa.table(
            {
                "y_alert_proxy": pa.array(labels, type=pa.int8()),
                "training_alert_uid": uids,
            }
        ),
        folder / f"y_{split}.parquet",
    )


def _expected_evaluation(split, k):
    expected = {
        "rows": 2,
        "selected": 2,
        "k": 1 if k == 1 else 2,
        "requested_k": k,
        "top_k_tie_policy": "SCORE_DESC_THEN_TRAINING_ALERT_UID_ASC",
        "log_loss": log(2),
    }
    expected.update(
        {
            "train": {
                "positives": 1,
                "precision": 0.5,
                "recall": 1.0,
                "f1": 2 / 3,
                "average_precision": 0.5,
                "precision_at_k": 1.0 if k == 1 else 0.5,
                "recall_at_k": 1.0,
                "unresolved_labels": 1,
            },
            "validation": {
                "positives": 0,
                "precision": 0.0,
                "recall": None,
                "f1": 0.0,
                "average_precision": None,
                "precision_at_k": 0.0,
                "recall_at_k": None,
                "unresolved_labels": 0,
            },
            "test": {
                "positives": 2,
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "average_precision": 1.0,
                "precision_at_k": 1.0,
                "recall_at_k": 0.5 if k == 1 else 1.0,
                "unresolved_labels": 0,
            },
        }[split]
    )
    return expected


@pytest.mark.parametrize("split", ["train", "validation", "test"])
@pytest.mark.parametrize("k", [1, 10])
def test_evaluate_split_preserves_complete_metrics_and_all_row_scores(tmp_path, split, k):
    labels, uids = SPLITS[split]
    _write_split(tmp_path, split, labels, uids)

    evaluation, scores = offline_training._evaluate_split(
        tmp_path, split, ["feature"], np.zeros(1), 0.0, 0.5, k
    )

    expected = _expected_evaluation(split, k)
    expected["log_loss"] = pytest.approx(log(2))
    assert evaluation == expected
    assert scores == [0.5] * len(labels)


def test_evaluate_split_rejects_all_null_truth(tmp_path):
    _write_split(tmp_path, "train", [None, None], ["TRAIN-z", "TRAIN-a"])

    with pytest.raises(ValueError, match="No complete proxy labels in train; null is never zero"):
        offline_training._evaluate_split(tmp_path, "train", ["feature"], np.zeros(1), 0.0, 0.5, 1)


@pytest.mark.parametrize("corruption", [None, "predictions", "evaluation"])
def test_saved_outputs_preserve_extra_keys_and_fail_at_first_mismatch(
    tmp_path, monkeypatch, corruption
):
    for split, (labels, uids) in SPLITS.items():
        _write_split(tmp_path, split, labels, uids)
    evaluation = {split: _expected_evaluation(split, 1) for split in SPLITS}
    predictions = {split: [0.5] * len(labels) for split, (labels, _) in SPLITS.items()}
    evaluation["extra"] = {"preserved": True}
    predictions["extra"] = [0.25]
    if corruption == "predictions":
        predictions["train"][0] = 0.25
    elif corruption == "evaluation":
        evaluation["train"]["unresolved_labels"] = 0
    write_json(tmp_path / "evaluation.json", evaluation)
    write_json(tmp_path / "predictions.json", predictions)
    model = {
        "feature_columns": ["feature"],
        "weights": [0.0],
        "intercept": 0.0,
        "recommended_threshold": 0.5,
        "parameters": {"k": 1},
        "model_version": "synthetic-evaluation-contract",
    }
    labelled = offline_training._labelled
    seen = []

    def record_split(folder, split, columns):
        seen.append(split)
        return labelled(folder, split, columns)

    monkeypatch.setattr(offline_training, "_labelled", record_split)
    if corruption is not None:
        with pytest.raises(
            ValueError, match="Stored scores/evaluation are not reproducible from saved model"
        ):
            offline_training._verify_model_outputs(
                tmp_path, tmp_path, model, execution_scope="FIXTURE_ONLY"
            )
        assert seen == ["train"]
    else:
        result = offline_training._verify_model_outputs(
            tmp_path, tmp_path, model, execution_scope="FIXTURE_ONLY"
        )
        assert seen == ["train", "validation", "test"]
        assert result == {
            "status": "VERIFIED",
            "execution_scope": "FIXTURE_ONLY",
            "model_version": "synthetic-evaluation-contract",
            "evaluation": evaluation,
            "recommended_threshold": 0.5,
            "service_id_mapping_verified": False,
            "deployed": False,
        }
