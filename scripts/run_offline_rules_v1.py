"""User-run, fail-stop checks then HI/LI/SAML generation, with a live saved log.

This file is only written by the assistant. Running it starts real preprocessing
AFTER the small checks pass. It never deletes old outputs or trains a model.
"""

import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

SOURCES = ("ibm_hi_small", "ibm_li_small", "saml_d")


def run_checked(command, *, cwd, env, emit, step):
    """Merge native streams in Python, avoiding PowerShell stderr/exit pitfalls."""
    emit(f"STEP_START {step}: {subprocess.list2cmdline(command)}\n")
    started = time.monotonic()
    child = subprocess.Popen(
        command, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    try:
        for line in child.stdout:
            emit(line)
        code = child.wait()
    except BaseException:
        # Stop only the child created by this manual session. Keep partial files.
        if child.poll() is None:
            child.terminate()
            child.wait()
        raise
    finally:
        child.stdout.close()
    emit(f"STEP_END {step}: exit_code={code} elapsed_seconds={time.monotonic()-started:.1f}\n")
    if code != 0:
        raise RuntimeError(f"{step} failed with exit code {code}; later steps were NOT started")


def require_completed(summary):
    sources = summary.get("sources", {})
    if (
        summary.get("status") != "COMPLETED"
        or set(sources) != set(SOURCES)
        or any(sources[s].get("status") != "COMPLETED" for s in SOURCES)
    ):
        raise RuntimeError("All three sources must be COMPLETED; inspect run.json and manifests")


def main():
    if len(sys.argv) != 1:
        raise ValueError("This frozen one-shot launcher takes no arguments")
    root = Path(__file__).resolve().parents[2]
    python = root / ".venv/Scripts/python.exe"
    if Path(sys.executable).resolve() != python.resolve():
        raise ValueError(f"Run with {python}")
    config_path = root / "ml_preparation/configs/offline_rules_v1_20261108.json"
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    if [s["id"] for s in config["sources"]] != list(SOURCES):
        raise ValueError("Expected exactly HI, LI, SAML, in that order")
    if config["memory_limit"] != "4GB" or config["threads"] != 2:
        raise ValueError("Resource settings changed; review before this manual run")
    if shutil.disk_usage(root).free < 40 * 1024**3:
        raise RuntimeError("At least 40 GiB free space is required; this is not a sufficiency guarantee")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    session = root / "artifacts" / ("multi-rules-manual-" + stamp)
    output = root / "ml_preparation/runs" / ("rules-v1-n5-manual-" + stamp)
    if session.exists() or output.exists():
        raise FileExistsError("Refusing to reuse an existing output or log folder")
    session.mkdir(parents=True)
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
    # Pin imports to these reviewed source files, not a different installed checkout.
    env["PYTHONPATH"] = str(root / "ml_preparation/src")
    status = "FAILED"
    with (session / "session.log").open("x", encoding="utf-8", buffering=1) as stream:
        def emit(line):
            stream.write(line)
            stream.flush()
            print(line, end="", flush=True)

        emit(f"LOG_FOLDER={session}\nOUTPUT_FOLDER={output}\n")
        emit("Checks first. Fresh HI/LI/SAML from source; no failed-database reuse.\n")
        emit("On failure: stop, preserve files, do not retry automatically.\n")
        emit("30-second phase heartbeats mean logger alive, NOT SQL progress or an ETA.\n")
        try:
            code_files = [
                "ml_preparation/src/aml_ml/" + name for name in (
                    "offline_rule_sql.py", "offline_multi.py", "offline_multi_eligibility.py",
                    "offline_multi_diagnostics.py", "offline_multi_progress.py", "offline_multi_io.py",
                    "offline_multi_metadata.py", "offline_multi_dataset.py", "offline_multi_recovery.py",
                )
            ] + ["ml_preparation/scripts/run_offline_rules_v1.py"]
            tests = [
                "ml_preparation/tests/" + name for name in (
                    "test_offline_rule_sql.py", "test_offline_multi.py",
                    "test_offline_multi_diagnostics.py", "test_offline_multi_progress.py",
                    "test_offline_multi_launcher.py", "test_offline_multi_io.py",
                    "test_offline_multi_modules.py",
                )
            ]
            run_checked([str(python), "-u", "-m", "ruff", "check", "--no-cache", *code_files],
                        cwd=root, env=env, emit=emit, step="ruff")
            run_checked([str(python), "-u", "-m", "pytest", *tests, "-v", "-p", "no:cacheprovider",
                         "--basetemp=" + str(session / "synthetic-test-temp")],
                        cwd=root, env=env, emit=emit, step="synthetic_tests")
            if shutil.disk_usage(root).free < 40 * 1024**3:
                raise RuntimeError("Free space fell below 40 GiB; generation NOT started")
            command = [str(python), "-u", "-m", "aml_ml.offline_multi",
                       "--config", str(config_path), "--output", str(output)]
            for source in SOURCES:
                command.extend(["--source", source])
            run_checked(command, cwd=root, env=env, emit=emit, step="HI_LI_SAML_generation")
            summary = json.loads((output / "run.json").read_text(encoding="utf-8-sig"))
            require_completed(summary)
            emit(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
            status = "COMPLETED"
            emit("ALL_THREE_COMPLETED: preprocessing only; no model training or old-output deletion.\n")
        except BaseException:
            emit(traceback.format_exc())
            raise
        finally:
            record = dict(status=status, output=str(output), log=str(session / "session.log"),
                          model_training_executed=False, automatic_retry=False)
            (session / "session.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
            emit(f"SESSION_STATUS={status}\nLOG={session / 'session.log'}\n")


if __name__ == "__main__":
    main()