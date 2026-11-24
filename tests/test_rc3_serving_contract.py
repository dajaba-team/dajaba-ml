import json
from datetime import datetime, timedelta
from decimal import Decimal
from hashlib import sha256
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aml_ml import (
    FEATURE_NAMES,
    FEATURE_VERSION,
    MODEL_VERSION,
    AlertInput,
    CheckpointUnavailableError,
    Coverage,
    features,
    score,
    serving_candidate,
)


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


@pytest.fixture
def approved_checkpoint(tmp_path, monkeypatch):
    approved = b"synthetic approved checkpoint"
    checkpoint = tmp_path / "checkpoint.ubj"
    checkpoint.write_bytes(approved)
    contract = json.loads(files("aml_ml").joinpath("serving_contract.json").read_text(encoding="utf-8"))
    contract["model_sha256"] = sha256(approved).hexdigest()
    (tmp_path / "serving_contract.json").write_text(json.dumps(contract), encoding="utf-8")
    loads = []

    class Booster:
        feature_names = FEATURE_NAMES

        def __init__(self, *, params):
            pass

        def load_model(self, model_bytes):
            loads.append(bytes(model_bytes))

    monkeypatch.setattr(serving_candidate, "files", lambda _: tmp_path)
    monkeypatch.setattr(serving_candidate.xgb, "Booster", Booster)
    monkeypatch.setenv("AML_ML_MODEL_PATH", str(checkpoint))
    serving_candidate._load_candidate.cache_clear()
    yield checkpoint, approved, loads
    serving_candidate._load_candidate.cache_clear()


@pytest.mark.parametrize("warm", [True, False], ids=["warm-worker", "fresh-worker"])
@pytest.mark.parametrize("fault", ["missing", "wrong-hash", "read-error"])
def test_checkpoint_failure_and_recovery(approved_checkpoint, monkeypatch, warm, fault):
    checkpoint, approved, loads = approved_checkpoint
    model = serving_candidate.load_candidate() if warm else None
    if warm:
        assert serving_candidate.load_candidate() is model
        assert loads == [approved]

    read_bytes = Path.read_bytes

    def denied(path):
        if path == checkpoint:
            raise PermissionError("synthetic read failure")
        return read_bytes(path)

    with monkeypatch.context() as mock:
        if fault == "missing":
            checkpoint.unlink()
        elif fault == "wrong-hash":
            checkpoint.write_bytes(b"changed")
        else:
            mock.setattr(Path, "read_bytes", denied)
        message = "SHA-256 mismatch" if fault == "wrong-hash" else "cannot read checkpoint"
        with pytest.raises(CheckpointUnavailableError, match=message):
            serving_candidate.load_candidate()
        assert loads == ([approved] if warm else [])

    checkpoint.write_bytes(approved)
    restored = serving_candidate.load_candidate()
    assert loads == [approved]
    if warm:
        assert restored is model


def test_first_load_uses_verified_bytes_after_path_changes(approved_checkpoint, monkeypatch):
    checkpoint, approved, loads = approved_checkpoint
    load_model = serving_candidate.xgb.Booster.load_model

    def replace_before_load(model, model_bytes):
        checkpoint.write_bytes(b"replaced after verification")
        load_model(model, model_bytes)

    monkeypatch.setattr(serving_candidate.xgb.Booster, "load_model", replace_before_load)
    serving_candidate.load_candidate()
    assert loads == [approved]
    with pytest.raises(CheckpointUnavailableError, match="SHA-256 mismatch"):
        serving_candidate.load_candidate()
