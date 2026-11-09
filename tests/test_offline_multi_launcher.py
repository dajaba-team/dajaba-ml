"""Launcher safety checks using mocked subprocesses, never real production data."""

import importlib.util
import io
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/run_offline_rules_v1.py"
SPEC = importlib.util.spec_from_file_location("manual_rules_launcher", PATH)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


def test_launcher_rejects_noncompleted_or_missing_sources():
    complete = {"status": "COMPLETED", "sources": {
        s: {"status": "COMPLETED"} for s in launcher.SOURCES}}
    launcher.require_completed(complete)
    for status in ("FAILED", "RUNNING", "COMPLETED_WITH_NOT_READY"):
        with pytest.raises(RuntimeError, match="All three"):
            launcher.require_completed(dict(complete, status=status))
    with pytest.raises(RuntimeError, match="All three"):
        launcher.require_completed({"status": "COMPLETED", "sources": {}})


def test_nonzero_child_exit_is_a_hard_stop(tmp_path, monkeypatch):
    class FailedChild:
        stdout = io.StringIO("synthetic error\n")

        def wait(self):
            return 7

    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *a, **kw: FailedChild())
    logs = []
    with pytest.raises(RuntimeError, match="exit code 7"):
        launcher.run_checked(["never-executed"], cwd=tmp_path, env={}, emit=logs.append,
                             step="synthetic_tests")
    assert any("synthetic error" in line for line in logs)
    assert any("exit_code=7" in line for line in logs)