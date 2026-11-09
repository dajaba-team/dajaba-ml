"""진행 기록은 데이터셋 밖에 저장하고 stdout은 최종 결과 JSON에만 사용합니다."""

import json
import logging
import os
import sys
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic, sleep
from uuid import uuid4

_CURRENT: ContextVar["Progress | None"] = ContextVar("offline_progress", default=None)
_PREFIX: ContextVar[str] = ContextVar("offline_progress_prefix", default="")
TERMINAL = frozenset({"VERIFIED", "FAILED", "CANCELLED", "BUILT", "NOT_READY"})
_LOG = logging.getLogger(__name__)
_WINDOWS = os.name == "nt"
_REPLACE_DELAYS = (0.05, 0.1, 0.2)
_REPLACE_MAX_SECONDS = 1.0


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _replace_state(temporary: Path, destination: Path):
    def stamp(path):
        info = path.lstat()
        return info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns

    parent = destination.parent.lstat()
    try:
        target = stamp(destination)
    except FileNotFoundError:
        target = None
    return (parent.st_dev, parent.st_ino), stamp(temporary), target


def _require_replace_state(temporary, destination, expected, failure):
    try:
        actual = _replace_state(temporary, destination)
    except OSError as state_error:
        failure.add_note("Progress replace state could not be confirmed; no further retry")
        raise failure from state_error
    if actual != expected:
        failure.add_note("Progress replace state changed; no further retry")
        raise failure


def _replace_control_file(temporary: Path, destination: Path):
    """닫힌 제어 JSON만 제한 재시도합니다. 원인이나 전체 호출 시간을 보장하지 않습니다."""
    expected = _replace_state(temporary, destination)
    started = monotonic()
    for attempt in range(len(_REPLACE_DELAYS) + 1):
        try:
            os.replace(temporary, destination)
            return
        except PermissionError as exc:
            if not _WINDOWS or getattr(exc, "winerror", None) != 5:
                raise
            # 교체가 일부 진행됐거나 다른 작성자가 개입한 경우에는 덮어쓰지 않습니다.
            _require_replace_state(temporary, destination, expected, exc)
            if attempt == len(_REPLACE_DELAYS):
                exc.add_note("Progress replace retry count exhausted; partial file preserved")
                raise
            delay = _REPLACE_DELAYS[attempt]
            if monotonic() - started + delay >= _REPLACE_MAX_SECONDS:
                exc.add_note("Progress replace retry deadline reached; partial file preserved")
                raise
            _LOG.warning(
                "Progress replace denied (winerror=5); retry %d/%d for %s after %.2fs",
                attempt + 1,
                len(_REPLACE_DELAYS),
                destination.name,
                delay,
            )
            sleep(delay)
            _require_replace_state(temporary, destination, expected, exc)
            # 1초는 다음 시도의 시작 제한이며 OS 호출 자체의 제한 시간은 아닙니다.
            if monotonic() - started >= _REPLACE_MAX_SECONDS:
                exc.add_note("Progress replace retry deadline reached; partial file preserved")
                raise


def atomic_json(path: Path, value, *, retry_progress_replace=False, retry_control_replace=False):
    """같은 폴더에서 파일을 fsync한 뒤 교체합니다. 단전 내구성은 파일시스템에 달려 있습니다."""
    temporary = path.with_name(path.name + "." + uuid4().hex + ".partial")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    # 기본 호출은 단회입니다. 진행 기록과 checkpoint만 명시적으로 제한 재시도합니다.
    if retry_progress_replace or retry_control_replace:
        _replace_control_file(temporary, path)
    else:
        os.replace(temporary, path)


class Progress:
    """주기 알림은 저장된 진행 수치만 읽으며 실제 작업을 진행시키지 않습니다."""

    def __init__(
        self,
        folder: Path,
        *,
        interval=30.0,
        console=False,
        clock=time.monotonic,
        cancel_file: Path | None = None,
    ):
        if not 0 < interval <= 300:
            raise ValueError("Heartbeat interval must be positive and at most 300 seconds")
        self.folder, self.interval, self.clock = folder, interval, clock
        self.folder.mkdir(parents=True, exist_ok=False)
        self.started = self.clock()
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.state: dict = dict(
            status="RUNNING", stage="starting", processed=0, total=None, unit=None
        )
        self.last_work_at = utc_now()
        self.last_emit = self.started
        self.stage_started = self.started
        self.stage_start_count = 0
        self.cancel_file = cancel_file
        self.cancel_check_after = self.started
        self.thread = None
        self.console = None
        self.console_requested = console
        self.write_error = None

    def __enter__(self):
        if self.console_requested:
            try:
                self.console = open(
                    "CONOUT$" if os.name == "nt" else "/dev/tty", "w", encoding="utf-8"
                )
            except OSError:
                self.console = None  # 직접 표시가 안 되면 실행기가 progress.json을 읽어 표시합니다.
        try:
            self._emit("started")
            self.thread = threading.Thread(
                target=self._heartbeat, daemon=True, name="aml-heartbeat"
            )
            self.thread.start()
        except BaseException:
            self._close_console()
            raise
        return self

    def _close_console(self):
        handle, self.console = self.console, None
        if handle is not None:
            try:
                handle.close()
            except OSError:
                pass  # 화면 표시 실패는 무시합니다. progress/status 파일이 실제 기록입니다.

    def check_cancelled(self):
        if self.write_error is not None:
            raise OSError("Progress persistence failed") from self.write_error
        # 행마다 호출되어도 취소 파일 확인은 1초에 한 번으로 제한합니다.
        if self.cancel_file is not None:
            now = self.clock()
            if now >= self.cancel_check_after:
                self.cancel_check_after = now + 1
                if self.cancel_file.exists():
                    raise KeyboardInterrupt("Cooperative cancellation requested")

    def update(self, stage, *, processed=0, total=None, unit=None):
        self.check_cancelled()
        if type(processed) is not int or processed < 0:
            raise ValueError("Progress count must be a nonnegative integer")
        if total is not None and (type(total) is not int or total < processed):
            raise ValueError("Progress denominator must be a known total, never an estimate")
        with self.lock:
            if self.write_error is not None:
                raise OSError("Progress persistence failed") from self.write_error
            if self.state["status"] in TERMINAL:
                return
            changed = stage != self.state["stage"]
            if changed or processed < self.state["processed"] or unit != self.state["unit"]:
                self.stage_started = self.clock()
                self.stage_start_count = processed
            self.state.update(stage=stage, processed=processed, total=total, unit=unit)
            self.last_work_at = utc_now()
            if changed or self.clock() - self.last_emit >= 1:
                self._emit("progress")

    def _snapshot(self, kind):
        elapsed = max(0, self.clock() - self.stage_started)
        delta = self.state["processed"] - self.stage_start_count
        rate = delta / elapsed if elapsed > 0 and delta > 0 else None
        total = self.state["total"]
        return dict(
            self.state,
            event=kind,
            observed_at=utc_now(),
            last_work_at=self.last_work_at,
            elapsed_seconds=round(self.clock() - self.started, 3),
            heartbeat_only=kind == "heartbeat",
            stage_elapsed_seconds=round(elapsed, 3),
            observed_units_per_second=rate,
            eta_seconds=(total - self.state["processed"]) / rate
            if total is not None and rate is not None
            else None,
            eta_basis="current_stage_observed_delta_only",
        )

    def _emit(self, kind):
        with self.lock:
            snapshot = self._snapshot(kind)
            # 최신 상태는 원자적으로 교체하고 이전 관측은 텍스트 로그에 남깁니다.
            atomic_json(self.folder / "progress.json", snapshot, retry_progress_replace=True)
            denominator = "?" if snapshot["total"] is None else str(snapshot["total"])
            message = (
                f"{snapshot['observed_at']} {kind} status={snapshot['status']} "
                f"stage={snapshot['stage']} count={snapshot['processed']}/{denominator} "
                f"unit={snapshot['unit']} elapsed={snapshot['elapsed_seconds']}s "
                f"rate={snapshot['observed_units_per_second']} eta_s={snapshot['eta_seconds']} "
                f"last_work={snapshot['last_work_at']}"
            )
            with (self.folder / "progress.log").open("a", encoding="utf-8") as stream:
                stream.write(message + "\n")
                stream.flush()
            if self.console is not None:
                try:
                    self.console.write(message + "\n")
                    self.console.flush()
                except OSError:
                    self._close_console()
            self.last_emit = self.clock()

    def _heartbeat(self):
        while not self.stop.wait(self.interval):
            try:
                with self.lock:
                    if self.state["status"] not in TERMINAL:
                        self._emit("heartbeat")
            except OSError as exc:
                self.write_error = exc
                return

    def finish(self, status, *, error_type=None):
        if status not in TERMINAL:
            raise ValueError("Unknown terminal run status")
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=5)
            if self.thread.is_alive():
                raise OSError("Heartbeat sink did not stop; no terminal state was committed")
        with self.lock:
            if self.state["status"] in TERMINAL:
                return
            if self.write_error is not None and status in {"VERIFIED", "BUILT"}:
                raise OSError("Progress persistence failed before completion") from self.write_error
            terminal = dict(self._snapshot("terminal"), status=status, error_type=error_type)
            # 이 파일이 최종 상태의 기준입니다. 이후 화면 표시 실패로 상태를 되돌리지 않습니다.
            atomic_json(self.folder / "status.json", terminal, retry_progress_replace=True)
            self.state.update(status=status, error_type=error_type)
            try:
                self._emit("terminal")
            except OSError as exc:
                self.write_error = exc  # status.json에는 이미 최종 상태가 저장됐습니다.

    def __exit__(self, kind, value, traceback):
        self.stop.set()
        try:
            if self.state["status"] not in TERMINAL:
                try:
                    self.finish(
                        "CANCELLED" if kind is KeyboardInterrupt else "FAILED",
                        error_type=None if kind is None else kind.__name__,
                    )
                except OSError as exc:
                    self.write_error = exc
                    if kind is None:
                        raise
                    # 진단 기록 실패로 원래 작업 오류나 취소 예외를 덮지 않습니다.
        finally:
            self._close_console()


@contextmanager
def reporting(reporter):
    token = _CURRENT.set(reporter)
    try:
        yield reporter
    finally:
        _CURRENT.reset(token)


def progress(stage, *, processed=0, total=None, unit=None):
    reporter = _CURRENT.get()
    if reporter is not None:
        reporter.update(_PREFIX.get() + stage, processed=processed, total=total, unit=unit)


def check_cancelled():
    reporter = _CURRENT.get()
    if reporter is not None:
        reporter.check_cancelled()


@contextmanager
def stage_scope(prefix):
    token = _PREFIX.set(_PREFIX.get() + prefix)
    try:
        yield
    finally:
        _PREFIX.reset(token)


def cancelled_result():
    # stdout에는 최종 JSON 한 건만 씁니다. 진행 기록은 별도 경로를 사용합니다.
    sys.stdout.write(json.dumps({"status": "CANCELLED"}) + "\n")
    sys.stdout.flush()
    return 130
