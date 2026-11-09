"""Observable phases only: heartbeats are not completion percentages or SQL progress."""

import json
import os
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone

_LOG_LOCK = threading.RLock()
_LAST_EVENTS = {}


def _persist_event(folder, record):
    """Append the authoritative event before touching the replaceable snapshot."""
    line = json.dumps(record, ensure_ascii=False)
    with (folder / "progress.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(line, flush=True)


def _write_snapshot(path, record):
    """Avoid Windows rename collisions for this telemetry-only latest snapshot.

    This file is non-atomic and may briefly be partial while written. Consumers
    should retry a JSON read or use progress.jsonl. No data/manifest writer changes.
    """
    text = json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False)
    for attempt in range(5):
        try:
            with path.open("w", encoding="utf-8") as stream:
                stream.write(text)
                stream.flush()
            return attempt
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.05 * (2 ** attempt))


def log(folder, stage, **details):
    record = dict(stage=stage, utc=datetime.now(timezone.utc).isoformat(), **details)
    with _LOG_LOCK:
        folder.mkdir(parents=True, exist_ok=True)
        key = str(folder.resolve())
        if details.get("event") == "heartbeat":
            record["last_work_event"] = _LAST_EVENTS.get(key)
        else:
            _LAST_EVENTS[key] = record.copy()
        _persist_event(folder, record)
        try:
            retries = _write_snapshot(folder / "progress.json", record)
        except OSError as exc:
            try:
                _persist_event(folder, dict(
                    stage="progress_snapshot_failed", event="failed",
                    utc=datetime.now(timezone.utc).isoformat(), failed_stage=stage,
                    error_type=type(exc).__name__, message=str(exc),
                    authoritative_log="progress.jsonl",
                ))
            except Exception as diagnostic_error:
                exc.add_note("Diagnostic logging also failed: " + str(diagnostic_error))
            raise
        if retries:
            _persist_event(folder, dict(
                stage="progress_snapshot_recovered", event="warning",
                utc=datetime.now(timezone.utc).isoformat(), retries=retries,
                previous_stage=stage, authoritative_log="progress.jsonl",
            ))

@contextmanager
def phase(folder, stage, *, interval=30.0, **details):
    """Report start/end and a responsive logging thread during blocking work.

    The thread never queries DuckDB or reads transaction data. A heartbeat does
    not establish that SQL is advancing. No ETA or fabricated row count is used.
    """
    if interval <= 0:
        raise ValueError("Heartbeat interval must be positive")
    started = time.monotonic()
    stopped = threading.Event()
    errors = []
    log(folder, stage, event="started", **details)

    def beat():
        try:
            while not stopped.wait(interval):
                log(folder, stage, event="heartbeat",
                    phase_elapsed_seconds=round(time.monotonic() - started, 1),
                    eta_seconds=None, meaning="LOGGER_ALIVE_NOT_QUERY_PROGRESS", **details)
        except Exception as exc:
            errors.append(exc)

    worker = threading.Thread(target=beat, name="multi-rule-log-heartbeat", daemon=True)
    worker.start()
    failure = None
    try:
        yield
    except BaseException as exc:
        failure = exc
        raise
    finally:
        stopped.set()
        worker.join(timeout=5)
        if worker.is_alive():
            errors.append(RuntimeError("Logging thread did not stop within five seconds"))
        else:
            try:
                log(folder, stage, event="failed" if failure or errors else "finished",
                    phase_elapsed_seconds=round(time.monotonic() - started, 1), **details)
            except Exception as exc:
                errors.append(exc)
        if errors:
            if failure is not None:
                failure.add_note("Progress logging also failed: " + str(errors[0]))
            else:
                raise RuntimeError("Progress logging failed; inspect console output") from errors[0]