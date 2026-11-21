import json
from datetime import datetime, timedelta
from decimal import Decimal
from importlib.resources import files

import numpy as np
import pandas as pd
import pytest

from aml_ml import FEATURE_NAMES, FEATURE_VERSION, MODEL_VERSION, AlertInput, Coverage, features, score
from aml_ml import serving_candidate


def test_public_synthetic_score_regression(monkeypatch):
    class FixedBooster:
        def predict(self, matrix, *, pred_contribs=False):
            assert matrix.feature_names == list(FEATURE_NAMES)
            assert matrix.num_col() == 31
            return np.zeros((1, 32)) if pred_contribs else np.array([0.625])

    monkeypatch.setattr(serving_candidate, "load_candidate", lambda: FixedBooster())
    ref = datetime(2022, 9, 10, 12)
    ids = tuple(range(1, 6))
    alert = AlertInput("synthetic", "R02_FAN_IN", "target", ref, ref - timedelta(hours=96), ref, ids)
    tx = pd.DataFrame([
        dict(transaction_id=i, ts=ref - timedelta(minutes=i), from_account_key=f"sender-{i}",
             to_account_key="target", amount_paid=Decimal("10"), currency_paid="US Dollar",
             amount_received=Decimal("10"), currency_received="US Dollar", payment_format="Wire",
             is_self_transfer=False)
        for i in ids
    ])
    cover = Coverage(ref - timedelta(hours=24), ref, None, None, "COMPLETE")
    contract = json.loads(files("aml_ml").joinpath("serving_contract.json").read_text(encoding="utf-8"))
    assert (MODEL_VERSION, FEATURE_VERSION) == (
        "xgb-hi-medium-r04-event-c2-31-v1", "hi-medium-evidence31-96h-r04-event-v1")
    assert tuple(contract["features"]) == FEATURE_NAMES and contract["dtype"] == "float32"
    observed = features([alert], tx, coverage=[cover]).iloc[0]
    assert observed["evidence_count"] == 5 and observed["focal_in_count"] == 5
    [result] = score([alert], tx, coverage=[cover])
    assert (result.score, result.failure_code) == (0.625, None)
    assert result.evidence_transaction_ids == list(ids)


def test_private_checkpoint_guard(tmp_path, monkeypatch):
    monkeypatch.delenv("AML_ML_MODEL_PATH", raising=False)
    with pytest.raises(RuntimeError, match="set AML_ML_MODEL_PATH"):
        serving_candidate.load_candidate()
    wrong = tmp_path / "wrong.ubj"
    wrong.write_bytes(b"synthetic invalid checkpoint")
    monkeypatch.setenv("AML_ML_MODEL_PATH", str(wrong))
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        serving_candidate.load_candidate()
