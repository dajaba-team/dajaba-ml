"""진행 알림, stdout 분리, 최종 상태 저장과 취소 처리의 회귀입니다."""

import json
from dataclasses import asdict

import pytest

from aml_ml import offline_cli
from aml_ml import offline_progress as progress_module
from aml_ml.offline_progress import Progress, check_cancelled, progress, reporting, stage_scope
from aml_ml.storage import read_json, write_json


def test_heartbeat_does_not_advance_work_or_count_resumed_rows_as_current_rate(tmp_path, capsys):
    clock = [100.0]
    with Progress(tmp_path / "logs", interval=300, clock=lambda: clock[0]) as reporter:
        with reporting(reporter):
            progress("source", processed=500, total=1000, unit="rows")
            clock[0] += 30
            reporter._emit("heartbeat")
            first = read_json(tmp_path / "logs" / "progress.json")
            assert first["processed"] == 500 and first["heartbeat_only"] is True
            assert first["observed_units_per_second"] is None
            clock[0] += 1
            progress("source", processed=562, total=1000, unit="rows")
            second = read_json(tmp_path / "logs" / "progress.json")
            assert second["observed_units_per_second"] == 2
            assert second["eta_seconds"] == 219
            with stage_scope("independent_replay/"):
                progress("source", processed=0, total=1000, unit="rows")
            reporter.finish("VERIFIED")
    status = read_json(tmp_path / "logs" / "status.json")
    assert status["status"] == "VERIFIED"
    assert status["stage"] == "independent_replay/source"
    assert capsys.readouterr().out == ""
    assert "heartbeat" in (tmp_path / "logs" / "progress.log").read_text()
    with pytest.raises(ValueError, match="300"):
        Progress(tmp_path / "bad", interval=301)
    assert not (tmp_path / "bad").exists()


def test_cooperative_cancellation_preserves_primary_and_writes_terminal(tmp_path):
    flag = tmp_path / "cancel.request"
    with pytest.raises(KeyboardInterrupt, match="Cooperative"):
        with Progress(tmp_path / "logs", cancel_file=flag) as reporter:
            with reporting(reporter):
                flag.write_text("cancel")
                check_cancelled()
    assert read_json(tmp_path / "logs" / "status.json")["status"] == "CANCELLED"


def test_hot_loop_cancellation_poll_is_throttled_without_losing_request(tmp_path, monkeypatch):
    clock, probes = [0.0], []
    reporter = Progress(tmp_path / "logs", clock=lambda: clock[0])

    class Flag:
        def exists(self):
            probes.append(clock[0])
            return clock[0] >= 1

    monkeypatch.setattr(reporter, "cancel_file", Flag())
    for _ in range(1000):
        reporter.check_cancelled()
    assert probes == [0.0]
    clock[0] = 1.0
    with pytest.raises(KeyboardInterrupt):
        reporter.check_cancelled()
    assert probes == [0.0, 1.0]


def test_post_terminal_display_failure_cannot_revert_committed_status(tmp_path, monkeypatch):
    with Progress(tmp_path / "logs") as reporter:

        def broken_display(kind):
            raise OSError("display log failed")

        monkeypatch.setattr(reporter, "_emit", broken_display)
        reporter.finish("VERIFIED")
        assert reporter.state["status"] == "VERIFIED"
    assert read_json(tmp_path / "logs" / "status.json")["status"] == "VERIFIED"


def test_cli_relative_paths_from_other_cwd_cancelled_stdout_is_single_json(
    source,
    policy,
    tmp_path,
    monkeypatch,
    capsys,
):
    write_json(tmp_path / "policy.json", asdict(policy))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    def cancel(*args, **kwargs):
        assert kwargs["checkpoint_context"]["assembly_index"] == str(tmp_path / "index.json")
        raise KeyboardInterrupt("fixture cancellation")

    monkeypatch.setattr(offline_cli, "build_dataset", cancel)
    code = offline_cli.main(
        [
            "prepare",
            "--source",
            source.id,
            "--scope",
            "FIXTURE_ONLY",
            "--assembly-index",
            "../index.json",
            "--policy",
            "../policy.json",
            "--output",
            "../output",
            "--checkpoint-dir",
            "../work",
            "--progress-dir",
            "../logs",
            "--train-before",
            "2022-01-02",
            "--test-from",
            "2022-01-04",
        ]
    )
    assert code == 130
    assert json.loads(capsys.readouterr().out) == {"status": "CANCELLED"}
    assert read_json(tmp_path / "logs" / "status.json")["status"] == "CANCELLED"


def test_heartbeat_persistence_error_blocks_success_and_records_failure(tmp_path, monkeypatch):
    reporter = Progress(tmp_path / "logs")
    atomic = progress_module.atomic_json

    class SingleHeartbeat:
        calls = 0

        def wait(self, interval):
            self.calls += 1
            return self.calls > 1

        def set(self):
            pass

    def fail_heartbeat(path, value, **kwargs):
        if value.get("event") == "heartbeat":
            raise OSError("heartbeat disk failure")
        return atomic(path, value, **kwargs)

    monkeypatch.setattr(reporter, "stop", SingleHeartbeat())
    monkeypatch.setattr(progress_module, "atomic_json", fail_heartbeat)
    # 실행 속도에 의존하지 않도록 알림 작업의 실패 경로를 동기 호출합니다.
    reporter._heartbeat()
    with pytest.raises(OSError, match="persistence failed"):
        reporter.finish("VERIFIED")
    assert not (tmp_path / "logs" / "status.json").exists()
    reporter.finish("FAILED")
    assert read_json(tmp_path / "logs" / "status.json")["status"] == "FAILED"


def test_terminal_write_failure_preserves_original_body_exception(tmp_path, monkeypatch):
    atomic = progress_module.atomic_json

    def fail_terminal(path, value, **kwargs):
        if path.name == "status.json":
            raise OSError("terminal disk failure")
        return atomic(path, value, **kwargs)

    with pytest.raises(ValueError, match="original processing error"):
        with Progress(tmp_path / "logs") as reporter:
            monkeypatch.setattr(progress_module, "atomic_json", fail_terminal)
            raise ValueError("original processing error")
    assert isinstance(reporter.write_error, OSError)
    assert reporter.thread is not None and not reporter.thread.is_alive()
    assert not (tmp_path / "logs" / "status.json").exists()


def test_unstopped_heartbeat_cannot_commit_terminal_success(tmp_path, monkeypatch):
    reporter = Progress(tmp_path / "logs")

    class Unstopped:
        def join(self, timeout):
            assert timeout == 5

        def is_alive(self):
            return True

    monkeypatch.setattr(reporter, "thread", Unstopped())
    with pytest.raises(OSError, match="did not stop"):
        reporter.finish("VERIFIED")
    assert reporter.state["status"] == "RUNNING"
    assert not (tmp_path / "logs" / "status.json").exists()


class ReplaceClock:
    def __init__(self):
        self.now = 0.0
        self.waits = []

    def sleep(self, seconds):
        self.waits.append(seconds)
        self.now += seconds


def access_denied(code=5):
    error = PermissionError("fixture progress replace denied")
    if code is not None:
        error.winerror = code
    return error


@pytest.fixture
def replace_fixture(tmp_path, monkeypatch):
    path = tmp_path / "progress.json"
    clock = ReplaceClock()
    monkeypatch.setattr(progress_module, "_WINDOWS", True)
    monkeypatch.setattr(progress_module, "monotonic", lambda: clock.now)
    monkeypatch.setattr(progress_module, "sleep", clock.sleep)
    return path, clock


@pytest.mark.parametrize("existing", [False, True])
def test_progress_replace_retries_same_completed_partial(
    replace_fixture, monkeypatch, caplog, existing
):
    path, clock = replace_fixture
    if existing:
        progress_module.atomic_json(path, {"count": 1})
    original, calls = progress_module.os.replace, []

    def replace(source, target):
        calls.append(source)
        # 매 시도에서 같은 임시 파일의 완결된 JSON을 사용합니다.
        with source.open("r", encoding="utf-8") as stream:
            assert json.load(stream) == {"count": 2}
        if len(calls) == 1:
            raise access_denied()
        return original(source, target)

    monkeypatch.setattr(progress_module.os, "replace", replace)
    progress_module.atomic_json(path, {"count": 2}, retry_progress_replace=True)
    assert len(calls) == 2 and calls[0] == calls[1]
    assert clock.waits == [0.05] and len(caplog.records) == 1
    assert read_json(path) == {"count": 2} and not calls[0].exists()


def test_progress_replace_exhaustion_preserves_original_and_partial(replace_fixture, monkeypatch):
    path, clock = replace_fixture
    progress_module.atomic_json(path, {"count": 1})
    before = path.read_bytes()
    failure, calls = access_denied(), []

    def replace(source, target):
        calls.append(source)
        raise failure

    monkeypatch.setattr(progress_module.os, "replace", replace)
    with pytest.raises(PermissionError) as caught:
        progress_module.atomic_json(path, {"count": 2}, retry_progress_replace=True)
    assert caught.value is failure
    assert len(calls) == 4 and len(set(calls)) == 1
    assert clock.waits == [0.05, 0.1, 0.2] and clock.now < 1
    assert path.read_bytes() == before and read_json(calls[0]) == {"count": 2}


@pytest.mark.parametrize("windows,code", [(False, 5), (True, 32), (True, 2), (True, None)])
def test_progress_replace_other_platform_or_error_is_not_retried(
    replace_fixture, monkeypatch, windows, code
):
    path, clock = replace_fixture
    failure, calls = access_denied(code), []
    monkeypatch.setattr(progress_module, "_WINDOWS", windows)

    def replace(source, target):
        calls.append(source)
        raise failure

    monkeypatch.setattr(progress_module.os, "replace", replace)
    with pytest.raises(PermissionError) as caught:
        progress_module.atomic_json(path, {"count": 2}, retry_progress_replace=True)
    assert caught.value is failure and len(calls) == 1 and clock.waits == []
    assert not path.exists() and calls[0].is_file()


def test_checkpoint_atomic_json_keeps_single_attempt_default(replace_fixture, monkeypatch):
    path, clock = replace_fixture
    failure, calls = access_denied(), []

    def replace(source, target):
        calls.append(source)
        raise failure

    monkeypatch.setattr(progress_module.os, "replace", replace)
    with pytest.raises(PermissionError) as caught:
        progress_module.atomic_json(path.with_name("checkpoint.json"), {"generation": 1})
    assert caught.value is failure and len(calls) == 1 and clock.waits == []
    assert read_json(calls[0]) == {"generation": 1}


@pytest.mark.parametrize("when", ["replace", "sleep", "state"])
def test_progress_replace_deadline_prevents_another_attempt(replace_fixture, monkeypatch, when):
    path, clock = replace_fixture
    failure, calls, states = access_denied(), [], []
    original_state = progress_module._replace_state

    def replace(source, target):
        calls.append(source)
        if when == "replace":
            clock.now = 0.96
        raise failure

    def sleep(seconds):
        clock.sleep(seconds)
        if when == "sleep":
            clock.now = 1.1

    def state(*args):
        states.append(None)
        result = original_state(*args)
        if when == "state" and len(states) == 3:
            clock.now = 1.1
        return result

    monkeypatch.setattr(progress_module.os, "replace", replace)
    monkeypatch.setattr(progress_module, "sleep", sleep)
    monkeypatch.setattr(progress_module, "_replace_state", state)
    with pytest.raises(PermissionError) as caught:
        progress_module.atomic_json(path, {"count": 2}, retry_progress_replace=True)
    assert caught.value is failure and len(calls) == 1
    assert not path.exists() and read_json(calls[0]) == {"count": 2}


@pytest.mark.parametrize("change", ["destination", "partial", "missing_partial"])
def test_progress_replace_state_change_stops_before_retry(replace_fixture, monkeypatch, change):
    path, clock = replace_fixture
    failure, calls = access_denied(), []
    preserved = path.with_name("preserved.partial")

    def replace(source, target):
        calls.append(source)
        raise failure

    def sleep(seconds):
        clock.sleep(seconds)
        if change == "destination":
            path.write_text('{"other": true}', encoding="utf-8")
        elif change == "partial":
            calls[0].write_text('{"other": "different"}', encoding="utf-8")
        else:
            calls[0].rename(preserved)

    monkeypatch.setattr(progress_module.os, "replace", replace)
    monkeypatch.setattr(progress_module, "sleep", sleep)
    with pytest.raises(PermissionError) as caught:
        progress_module.atomic_json(path, {"count": 2}, retry_progress_replace=True)
    assert caught.value is failure and len(calls) == 1 and clock.waits == [0.05]
    if change == "destination":
        assert read_json(path) == {"other": True}
    elif change == "partial":
        assert read_json(calls[0]) == {"other": "different"} and not path.exists()
    else:
        assert read_json(preserved) == {"count": 2} and not path.exists()


def test_progress_replace_then_error_is_neither_success_nor_retried(replace_fixture, monkeypatch):
    path, clock = replace_fixture
    original, failure, calls = progress_module.os.replace, access_denied(), []

    def replace(source, target):
        calls.append(source)
        original(source, target)
        raise failure

    monkeypatch.setattr(progress_module.os, "replace", replace)
    with pytest.raises(PermissionError) as caught:
        progress_module.atomic_json(path, {"count": 2}, retry_progress_replace=True)
    assert caught.value is failure and len(calls) == 1 and clock.waits == []
    assert read_json(path) == {"count": 2} and not calls[0].exists()


@pytest.mark.parametrize("status", ["VERIFIED", "CANCELLED"])
def test_progress_and_terminal_opt_in_to_bounded_replace(replace_fixture, monkeypatch, status):
    path, clock = replace_fixture
    reporter = Progress(path.parent / "logs")
    original, calls = progress_module.os.replace, {}

    def replace(source, target):
        calls[target.name] = calls.get(target.name, 0) + 1
        if calls[target.name] == 1:
            raise access_denied()
        return original(source, target)

    monkeypatch.setattr(progress_module.os, "replace", replace)
    reporter._emit("started")
    reporter.finish(status)
    assert calls == {"progress.json": 3, "status.json": 2}
    assert clock.waits == [0.05, 0.05]
    assert read_json(reporter.folder / "status.json")["status"] == status
    assert reporter.state["status"] == status


def test_terminal_replace_exhaustion_does_not_commit_verified(replace_fixture, monkeypatch):
    path, clock = replace_fixture
    reporter = Progress(path.parent / "logs")
    failure, calls = access_denied(), []

    def replace(source, target):
        calls.append(source)
        raise failure

    monkeypatch.setattr(progress_module.os, "replace", replace)
    with pytest.raises(PermissionError) as caught:
        reporter.finish("VERIFIED")
    assert caught.value is failure and len(calls) == 4
    assert clock.waits == [0.05, 0.1, 0.2]
    assert reporter.state["status"] == "RUNNING"
    assert not (reporter.folder / "status.json").exists()
    assert calls[0].is_file()


@pytest.mark.parametrize("failure", [OSError("fixture IO failure"), KeyboardInterrupt()])
def test_progress_replace_other_exceptions_propagate_once(replace_fixture, monkeypatch, failure):
    path, clock = replace_fixture
    calls = []

    def replace(source, target):
        calls.append(source)
        raise failure

    monkeypatch.setattr(progress_module.os, "replace", replace)
    with pytest.raises(type(failure)) as caught:
        progress_module.atomic_json(path, {"count": 2}, retry_progress_replace=True)
    assert caught.value is failure and len(calls) == 1 and clock.waits == []
