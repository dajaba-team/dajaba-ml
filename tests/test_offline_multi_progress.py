"""Small manual regressions for live phase logs. Latest changes are NOT RUN."""

import json
import threading

import pytest

from aml_ml import offline_multi_progress as progress


def records(folder):
    return [json.loads(line) for line in (folder / "progress.jsonl").read_text().splitlines()]


def test_phase_records_start_and_end(tmp_path):
    with progress.phase(tmp_path, "tiny", interval=60):
        progress.log(tmp_path, "real_count", rows=3)
    rows = records(tmp_path)
    assert rows[0]["event"] == "started"
    assert rows[1]["rows"] == 3
    assert rows[-1]["event"] == "finished"
    assert json.loads((tmp_path / "progress.json").read_text())["event"] == "finished"


def test_phase_failure_is_not_completion(tmp_path):
    with pytest.raises(ValueError, match="fixture"):
        with progress.phase(tmp_path, "tiny", interval=60):
            raise ValueError("fixture")
    assert records(tmp_path)[-1]["event"] == "failed"


def test_heartbeat_is_honest_and_stops(tmp_path, monkeypatch):
    observed = threading.Event()
    original = progress.log

    def logged(folder, stage, **details):
        original(folder, stage, **details)
        if details.get("event") == "heartbeat":
            observed.set()

    monkeypatch.setattr(progress, "log", logged)
    with progress.phase(tmp_path, "blocking_fixture", interval=0.01):
        assert observed.wait(2), "heartbeat thread did not report"
    beats = [r for r in records(tmp_path) if r.get("event") == "heartbeat"]
    assert beats
    assert all(r["eta_seconds"] is None for r in beats)
    assert all(r["meaning"] == "LOGGER_ALIVE_NOT_QUERY_PROGRESS" for r in beats)
    assert records(tmp_path)[-1]["event"] == "finished"
    assert not any(t.name == "multi-rule-log-heartbeat" for t in threading.enumerate())


def test_heartbeat_retains_last_real_work_event(tmp_path):
    progress.log(tmp_path, "r04_bucket_started", bucket_index=4, bucket_count=128)
    progress.log(tmp_path, "generate_candidates", event="heartbeat", eta_seconds=None)
    current = json.loads((tmp_path / "progress.json").read_text())
    assert current["last_work_event"]["stage"] == "r04_bucket_started"
    assert current["last_work_event"]["bucket_index"] == 4


def test_end_logging_does_not_mask_original_failure(tmp_path, monkeypatch):
    original = progress.log

    def broken_end(folder, stage, **details):
        if details.get("event") == "failed":
            raise OSError("synthetic log failure")
        original(folder, stage, **details)

    monkeypatch.setattr(progress, "log", broken_end)
    with pytest.raises(ValueError, match="original SQL failure") as caught:
        with progress.phase(tmp_path, "tiny", interval=60):
            raise ValueError("original SQL failure")
    assert any("logging also failed" in note for note in caught.value.__notes__)


def test_progress_snapshot_does_not_rename(tmp_path, monkeypatch):
    from pathlib import Path

    def forbidden_replace(self, target):
        raise AssertionError("Telemetry must not use Windows replace/rename")

    monkeypatch.setattr(Path, "replace", forbidden_replace)
    progress.log(tmp_path, "no_rename", rows=7)
    assert json.loads((tmp_path / "progress.json").read_text())["rows"] == 7
    assert records(tmp_path)[-1]["stage"] == "no_rename"


def test_progress_snapshot_retries_permission_error(tmp_path, monkeypatch):
    from pathlib import Path

    original = Path.open
    attempts = []

    def sometimes_locked(self, mode="r", *args, **kwargs):
        if self.name == "progress.json" and mode == "w":
            attempts.append(1)
            if len(attempts) < 3:
                raise PermissionError("synthetic transient sharing violation")
        return original(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", sometimes_locked)
    monkeypatch.setattr(progress.time, "sleep", lambda seconds: None)
    progress.log(tmp_path, "retried", rows=7)
    assert len(attempts) == 3
    assert json.loads((tmp_path / "progress.json").read_text())["rows"] == 7
    assert records(tmp_path)[-1]["stage"] == "progress_snapshot_recovered"
    assert records(tmp_path)[-1]["retries"] == 2


def test_persistent_snapshot_failure_is_logged_and_raised(tmp_path, monkeypatch):
    from pathlib import Path

    original = Path.open
    attempts = []

    def always_locked(self, mode="r", *args, **kwargs):
        if self.name == "progress.json" and mode == "w":
            attempts.append(1)
            raise PermissionError("synthetic persistent lock")
        return original(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", always_locked)
    monkeypatch.setattr(progress.time, "sleep", lambda seconds: None)
    with pytest.raises(PermissionError, match="persistent lock"):
        progress.log(tmp_path, "preserved_before_failure", rows=7)
    assert len(attempts) == 5
    saved = records(tmp_path)
    assert saved[0]["stage"] == "preserved_before_failure"
    assert saved[0]["rows"] == 7
    assert saved[-1]["stage"] == "progress_snapshot_failed"