"""폴더 발행의 제한 재시도와 실패 시 데이터 보존을 작은 fixture로 확인합니다."""

import os
from pathlib import Path

import pytest

from aml_ml import offline_publish as publishing


class Clock:
    def __init__(self):
        self.now = 0.0
        self.waits = []

    def sleep(self, seconds):
        self.waits.append(seconds)
        self.now += seconds


def denied(code=5):
    error = PermissionError("fixture publication denied")
    if code is not None:
        error.winerror = code
    return error


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    stage, output = tmp_path / "stage", tmp_path / "result"
    stage.mkdir()
    (stage / "payload").write_bytes(b"preserve me")
    clock = Clock()
    monkeypatch.setattr(publishing, "_WINDOWS", True)
    monkeypatch.setattr(publishing, "monotonic", lambda: clock.now)
    monkeypatch.setattr(publishing, "sleep", clock.sleep)
    return stage, output, clock


def test_temporary_denial_then_atomic_publish(fixture, monkeypatch, caplog):
    stage, output, clock = fixture
    original, calls = Path.rename, []

    def rename(path, target):
        calls.append((path, target))
        if len(calls) == 1:
            raise denied()
        return original(path, target)

    monkeypatch.setattr(Path, "rename", rename)
    publishing.publish_directory(stage, output)
    assert len(calls) == 2 and clock.waits == [0.05]
    assert not stage.exists() and (output / "payload").read_bytes() == b"preserve me"
    assert len(caplog.records) == 1


def test_permanent_denial_is_bounded_and_preserves_stage(fixture, monkeypatch):
    stage, output, clock = fixture
    failure, calls = denied(), []

    def rename(path, target):
        calls.append(path)
        raise failure

    monkeypatch.setattr(Path, "rename", rename)
    with pytest.raises(PermissionError) as caught:
        publishing.publish_directory(stage, output)
    assert caught.value is failure
    assert len(calls) == 4 and clock.waits == [0.05, 0.1, 0.2]
    assert clock.now < 1 and (stage / "payload").read_bytes() == b"preserve me"
    assert not output.exists()


@pytest.mark.parametrize("windows,code", [(False, 5), (True, 32), (True, 2), (True, None)])
def test_other_os_or_error_is_not_retried(fixture, monkeypatch, windows, code):
    stage, output, clock = fixture
    failure, calls = denied(code), []
    monkeypatch.setattr(publishing, "_WINDOWS", windows)

    def rename(path, target):
        calls.append(path)
        raise failure

    monkeypatch.setattr(Path, "rename", rename)
    with pytest.raises(PermissionError) as caught:
        publishing.publish_directory(stage, output)
    assert caught.value is failure and len(calls) == 1 and clock.waits == []


@pytest.mark.parametrize(
    "error", [OSError("injected"), RuntimeError("unexpected"), KeyboardInterrupt()]
)
def test_other_exceptions_keep_original_identity(fixture, monkeypatch, error):
    stage, output, clock = fixture
    calls = []

    def rename(path, target):
        calls.append(path)
        raise error

    monkeypatch.setattr(Path, "rename", rename)
    with pytest.raises(type(error)) as caught:
        publishing.publish_directory(stage, output)
    assert caught.value is error and len(calls) == 1 and clock.waits == []


@pytest.mark.parametrize("when", ["rename", "sleep", "next_snapshot"])
def test_elapsed_budget_prevents_another_attempt(fixture, monkeypatch, when):
    stage, output, clock = fixture
    failure, calls, snapshots = denied(), [], []
    original_snapshot = publishing._snapshot

    def rename(path, target):
        calls.append(path)
        if when == "rename":
            clock.now = 0.96
        raise failure

    def sleep(seconds):
        clock.sleep(seconds)
        if when == "sleep":
            clock.now = 1.1

    def snapshot(*args):
        snapshots.append(None)
        result = original_snapshot(*args)
        if when == "next_snapshot" and len(snapshots) == 3:
            clock.now = 1.1
        return result

    monkeypatch.setattr(Path, "rename", rename)
    monkeypatch.setattr(publishing, "sleep", sleep)
    monkeypatch.setattr(publishing, "_snapshot", snapshot)
    with pytest.raises(PermissionError) as caught:
        publishing.publish_directory(stage, output)
    assert caught.value is failure and len(calls) == 1
    assert (stage / "payload").read_bytes() == b"preserve me" and not output.exists()


@pytest.mark.parametrize(
    "change", ["destination", "missing_source", "replace_source", "content", "add_file", "parent"]
)
def test_state_change_during_wait_aborts_without_second_rename(fixture, monkeypatch, change):
    stage, output, clock = fixture
    original, calls = Path.rename, []
    preserved = stage.with_name("preserved")
    failure = denied()

    def rename(path, target):
        calls.append(path)
        raise failure

    def sleep(seconds):
        clock.sleep(seconds)
        if change == "destination":
            output.mkdir()
            (output / "unrelated").write_bytes(b"do not overwrite")
        elif change in {"missing_source", "replace_source"}:
            original(stage, preserved)
            if change == "replace_source":
                stage.mkdir()
                (stage / "payload").write_bytes(b"different source")
        elif change == "content":
            (stage / "payload").write_bytes(b"changed content")
        elif change == "add_file":
            (stage / "new").write_bytes(b"new")
        else:
            original(stage.parent, stage.parent.with_name(stage.parent.name + "-preserved"))
            stage.parent.mkdir()
            stage.mkdir()
            (stage / "payload").write_bytes(b"replacement parent")

    monkeypatch.setattr(Path, "rename", rename)
    monkeypatch.setattr(publishing, "sleep", sleep)
    with pytest.raises((ValueError, FileNotFoundError, FileExistsError)) as caught:
        publishing.publish_directory(stage, output)
    assert caught.value.__cause__ is failure
    assert len(calls) == 1
    if change == "destination":
        assert (output / "unrelated").read_bytes() == b"do not overwrite"
    else:
        assert not output.exists()


def test_move_then_raise_is_not_assumed_success_or_retried(fixture, monkeypatch):
    stage, output, clock = fixture
    original, calls = Path.rename, []

    def rename(path, target):
        calls.append(path)
        original(path, target)
        raise denied()

    monkeypatch.setattr(Path, "rename", rename)
    with pytest.raises(FileExistsError):
        publishing.publish_directory(stage, output)
    assert len(calls) == 1 and not clock.waits
    assert not stage.exists() and (output / "payload").read_bytes() == b"preserve me"


@pytest.mark.parametrize("cancel_at", [1, 3, 4])
def test_cancel_before_attempt_or_wait_or_after_wait(fixture, monkeypatch, cancel_at):
    stage, output, clock = fixture
    checks, calls = [], []

    def cancel():
        checks.append(None)
        if len(checks) == cancel_at:
            raise KeyboardInterrupt("fixture cancellation")

    def rename(path, target):
        calls.append(path)
        raise denied()

    monkeypatch.setattr(publishing, "check_cancelled", cancel)
    monkeypatch.setattr(Path, "rename", rename)
    with pytest.raises(KeyboardInterrupt, match="fixture cancellation"):
        publishing.publish_directory(stage, output)
    assert len(calls) <= 1 and not output.exists()
    assert (stage / "payload").read_bytes() == b"preserve me"


@pytest.mark.parametrize("link", ["source", "child", "destination", "parent"])
def test_links_are_rejected_without_rename(fixture, monkeypatch, link, tmp_path):
    stage, output, clock = fixture
    original, calls = Path.rename, []
    real = tmp_path / "real"
    real.mkdir()
    if link == "source":
        original(stage, real / "old")
        stage.symlink_to(real, target_is_directory=True)
    elif link == "child":
        (stage / "link").symlink_to(real, target_is_directory=True)
    elif link == "destination":
        output.symlink_to(tmp_path / "absent", target_is_directory=True)
    else:
        alias = tmp_path / "alias"
        alias.symlink_to(stage.parent, target_is_directory=True)
        stage = alias / stage.name

    def rename(path, target):
        calls.append(path)
        pytest.fail("Unsafe publication must not call rename")

    monkeypatch.setattr(Path, "rename", rename)
    with pytest.raises((ValueError, FileExistsError)):
        publishing.publish_directory(stage, output)
    assert not calls and not clock.waits


def test_existing_destination_is_preserved(fixture):
    stage, output, clock = fixture
    output.mkdir()
    (output / "existing").write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        publishing.publish_directory(stage, output)
    assert (output / "existing").read_bytes() == b"keep" and stage.exists()
    assert not clock.waits


@pytest.mark.parametrize("snapshot_number", [2, 3])
def test_snapshot_io_failure_is_not_retried(fixture, monkeypatch, snapshot_number):
    stage, output, clock = fixture
    original, checks, calls = publishing._snapshot, [], []
    failure = OSError("fixture metadata failure")
    rename_failure = denied()

    def snapshot(*args):
        checks.append(None)
        if len(checks) == snapshot_number:
            raise failure
        return original(*args)

    def rename(path, target):
        calls.append(None)
        raise rename_failure

    monkeypatch.setattr(publishing, "_snapshot", snapshot)
    monkeypatch.setattr(Path, "rename", rename)
    with pytest.raises(OSError) as caught:
        publishing.publish_directory(stage, output)
    assert caught.value is failure and len(calls) == 1
    assert (caught.value.__cause__ or caught.value.__context__) is rename_failure
    assert clock.waits == ([] if snapshot_number == 2 else [0.05])


@pytest.mark.skipif(os.name != "nt", reason="Windows의 자기 소유 핸들 동작을 확인합니다")
def test_owned_windows_handle_release_allows_publish(fixture, monkeypatch):
    stage, output, clock = fixture
    owned = (stage / "payload").open("rb")

    def release(seconds):
        clock.sleep(seconds)
        owned.close()

    monkeypatch.setattr(publishing, "sleep", release)
    try:
        publishing.publish_directory(stage, output)
    finally:
        owned.close()
    assert clock.waits and (output / "payload").read_bytes() == b"preserve me"
