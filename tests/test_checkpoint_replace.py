"""checkpoint만 명시적으로 제한 재시도하며 실패 시 generation을 전진시키지 않습니다."""

import json

import pytest

from aml_ml import offline_progress as progress
from aml_ml.offline_checkpoint import Workspace


def denied():
    error = PermissionError("injected checkpoint replacement")
    error.winerror = 5
    return error


def test_checkpoint_initial_and_commit_explicitly_retry_identical_bytes(tmp_path, monkeypatch):
    original, calls, waits = progress.os.replace, [], []
    monkeypatch.setattr(progress, "_WINDOWS", True)
    monkeypatch.setattr(progress, "sleep", waits.append)

    def replace(temporary, destination):
        calls.append((temporary, destination))
        if len(calls) in {1, 3}:
            raise denied()
        return original(temporary, destination)

    monkeypatch.setattr(progress.os, "replace", replace)
    with Workspace(tmp_path / "work", {"fixture": True}) as workspace:
        workspace.commit("ACCOUNTS", {"cursor": 2}, [])
        assert workspace.state["generation"] == 1
        document = json.loads((workspace.folder / "checkpoint.json").read_text(encoding="utf-8"))
        assert document["generation"] == 1 and document["data"] == {"cursor": 2}
    assert len(calls) == 4 and calls[0] == calls[1] and calls[2] == calls[3]
    assert waits == [0.05, 0.05]


def test_checkpoint_exhaustion_does_not_advance_commit(tmp_path, monkeypatch):
    with Workspace(tmp_path / "work", {}) as workspace:
        before = (workspace.folder / "checkpoint.json").read_bytes()
        error, calls, waits = denied(), [], []
        monkeypatch.setattr(progress, "_WINDOWS", True)
        monkeypatch.setattr(progress, "sleep", waits.append)

        def replace(temporary, destination):
            calls.append(temporary)
            raise error

        monkeypatch.setattr(progress.os, "replace", replace)
        with pytest.raises(PermissionError) as caught:
            workspace.commit("ACCOUNTS", {"cursor": 2}, [])
        assert caught.value is error and len(calls) == 4 and len(set(calls)) == 1
        assert waits == [0.05, 0.1, 0.2]
        assert (
            workspace.state["generation"] == 0
            and (workspace.folder / "checkpoint.json").read_bytes() == before
        )
        assert json.loads(calls[0].read_text(encoding="utf-8"))["generation"] == 1
