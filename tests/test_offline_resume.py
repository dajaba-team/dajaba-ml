"""작은 합성 원천으로 중단 위치, 처리 ACK, 독립 재검증을 확인하도록 작성한 회귀입니다."""

import copy
import json
from dataclasses import asdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from offline_test_data import prepare, rewrite_fixture, source_values

from aml_ml import offline_checkpoint as checkpoint
from aml_ml import offline_progress as progress_module
from aml_ml import offline_resume as resume_module
from aml_ml import offline_streaming as streaming
from aml_ml.offline_checkpoint import CheckpointCleanupError, MergeCursor, Workspace
from aml_ml.offline_cli import main
from aml_ml.offline_contract import DATASET_FILES
from aml_ml.offline_io import rows
from aml_ml.storage import read_json, sha256, write_json


class _InjectedOperationError(OSError):
    """테스트가 의도한 중단만 구분하고 실제 파일 오류는 숨기지 않습니다."""


def _seal(ws, name, values):
    path = ws.attempt / name
    table = pa.table({"n": values}, schema=pa.schema([("n", pa.int64())]))
    pq.write_table(table, path)
    return ws.seal(path, rows=len(values), schema=table.schema)


@pytest.mark.parametrize(
    "cut",
    [
        "SOURCE",
        "OWNERS",
        "ACCOUNTS",
        "OUTPUT",
        "PUBLISH_INTENT",
        "PUBLISHED",
        "REPLAY_SOURCE",
        "VERIFIED",
    ],
)
def test_committed_crash_boundaries_resume_exactly_without_duplicate_rows(
    source,
    policy,
    tmp_path,
    monkeypatch,
    cut,
):
    values, labels = source_values(source)
    rewrite_fixture(source, values, labels, part_rows=5)
    settings = dict(partitions=2, sort_run_rows=2, merge_fan_in=2, checkpoint_owner_rows=1)
    baseline, output, work = (tmp_path / name for name in ("baseline", "output", "work"))
    expected = prepare(source, policy, baseline, **settings)
    original = Workspace.commit
    tripped = []

    def crash(ws, phase, data, files):
        result = original(ws, phase, data, files)
        selected = (
            "REPLAY_SOURCE" if ws.folder.name == "replay-work" and phase == "SOURCE" else phase
        )
        account_done = phase != "ACCOUNTS" or data["stats"]["owned_rows_processed"] > 0
        if not tripped and selected == cut and account_done:
            tripped.append(copy.deepcopy(result))
            raise KeyboardInterrupt("injected after atomic commit")
        return result

    monkeypatch.setattr(Workspace, "commit", crash)
    with pytest.raises(KeyboardInterrupt):
        prepare(source, policy, output, checkpoint_dir=work, **settings)
    assert tripped
    orphan = work / "uncommitted.parquet.partial"
    orphan.write_bytes(b"do not adopt or delete")
    monkeypatch.setattr(Workspace, "commit", original)
    actual = prepare(source, policy, output, checkpoint_dir=work, resume=True, **settings)
    assert actual == expected
    for name in DATASET_FILES | {"manifest.json"}:
        assert (baseline / name).read_bytes() == (output / name).read_bytes(), name
    assert orphan.read_bytes() == b"do not adopt or delete"
    assert read_json(work / "checkpoint.json")["phase"] == "VERIFIED"
    assert read_json(work / "replay-work" / "checkpoint.json")["phase"] == "BUILT"
    assert (work / "replay-dataset" / "manifest.json").is_file()


def test_partial_fit_output_restarts_in_new_attempt(source, policy, tmp_path, monkeypatch):
    output, work = tmp_path / "data", tmp_path / "work"
    original = streaming._write_feature_splits

    def interrupt(*args, **kwargs):
        original(*args, **kwargs)
        raise KeyboardInterrupt("after frozen transform before final receipts")

    monkeypatch.setattr(streaming, "_write_feature_splits", interrupt)
    with pytest.raises(KeyboardInterrupt):
        prepare(source, policy, output, checkpoint_dir=work)
    assert read_json(work / "checkpoint.json")["phase"] == "OUTPUT"
    partials = list(work.glob("attempt-*/dataset/preprocessing.json"))
    assert len(partials) == 1
    old_hash = sha256(partials[0])
    monkeypatch.setattr(streaming, "_write_feature_splits", original)
    assert prepare(source, policy, output, checkpoint_dir=work, resume=True)["status"] == "VERIFIED"
    assert partials[0].exists() and sha256(partials[0]) == old_hash
    assert sha256(output / "preprocessing.json") == old_hash


@pytest.mark.parametrize("tamper", ["size", "same_length_hash"])
def test_old_scratch_binding_receipts_lock_and_no_inferred_completion(tmp_path, tamper):
    old = tmp_path / ".offline-work-old"
    old.mkdir()
    (old / "owners.parquet.partial").write_bytes(b"old")
    with pytest.raises(ValueError, match="No committed checkpoint"):
        with Workspace(old, {}, resume=True):
            pytest.fail("old scratch adopted")
    assert sorted(p.name for p in old.iterdir()) == ["owners.parquet.partial"]
    folder = tmp_path / "work"
    with Workspace(folder, {"code": "pinned"}) as ws:
        name = _seal(ws, "a.parquet", [1, 2])
        ws.commit("SOURCE", {}, [name])
        with pytest.raises(ValueError, match="locked"):
            with Workspace(folder, {"code": "pinned"}, resume=True):
                pytest.fail("second writer entered")
    with pytest.raises(ValueError, match="mismatch"):
        with Workspace(folder, {"code": "changed"}, resume=True):
            pytest.fail("changed binding accepted")
    content = (folder / name).read_bytes()
    (folder / name).write_bytes(
        b"corrupt" if tamper == "size" else bytes([content[0] ^ 1]) + content[1:]
    )
    with pytest.raises(ValueError, match="size changed" if tamper == "size" else "hash changed"):
        with Workspace(folder, {"code": "pinned"}, resume=True):
            pytest.fail("corrupt file accepted")


def test_commit_failure_keeps_old_frontier_and_gc_failure_keeps_new_generation(
    tmp_path, monkeypatch
):
    with Workspace(tmp_path / "work", {}) as ws:
        old = _seal(ws, "old.parquet", [1])
        ws.commit("SOURCE", {"cursor": 1}, [old])
        old_generation = ws.state["generation"]
        new = _seal(ws, "new.parquet", [1, 2])
        atomic = checkpoint.atomic_json

        def fail_replace(*args, **kwargs):
            assert kwargs == {"retry_control_replace": True}
            raise OSError("replace failed")

        monkeypatch.setattr(checkpoint, "atomic_json", fail_replace)
        with pytest.raises(OSError, match="replace failed"):
            ws.commit("SOURCE", {"cursor": 2}, [new])
        assert ws.path(old).exists() and ws.state["generation"] == old_generation
        monkeypatch.setattr(checkpoint, "atomic_json", atomic)
        unlink = Path.unlink

        def fail_retirement(path, *args, **kwargs):
            if path == ws.path(old):
                raise OSError("retirement failed")
            return unlink(path, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", fail_retirement)
        with pytest.raises(CheckpointCleanupError) as error:
            ws.commit("OUTPUT", {"cursor": 2}, [new])
        saved = read_json(ws.folder / "checkpoint.json")
        assert saved["generation"] == ws.state["generation"] == error.value.committed_generation
        assert saved["data"] == {"cursor": 2} and set(ws.receipts) == {new}
        assert ws.path(old).exists() and ws.path(new).exists()


def test_cursor_excludes_lookahead_and_requires_ack_even_before_eof(tmp_path):
    path = tmp_path / "rows.parquet"
    pq.write_table(pa.table({"n": [1, 2]}), path)

    def cursor(offset=0):
        return MergeCursor(
            [path],
            lambda row: row["n"],
            reader=rows,
            offsets=[offset],
            row_counts=[2],
            fan_in=2,
            max_bytes=1024**2,
            batch_rows=2,
        )

    first = cursor()
    row = next(first)
    first.acknowledge(row)
    following = next(first)
    assert first.offsets == [1] and following["n"] == 2
    first.close()
    resumed = cursor(1)
    assert next(resumed)["n"] == 2
    with pytest.raises(ValueError, match="acknowledged"):
        next(resumed)
    complete = cursor(1)
    row = next(complete)
    complete.acknowledge(row)
    assert list(complete) == []
    complete.require_complete()


def test_normal_close_failure_propagates_and_primary_error_is_preserved(tmp_path, monkeypatch):
    class BrokenClose:
        def __init__(self, handle):
            self.handle = handle

        def seek(self, *args):
            return self.handle.seek(*args)

        def fileno(self):
            return self.handle.fileno()

        def close(self):
            self.handle.close()
            raise OSError("close failed")

    for with_primary in (False, True):
        primary = KeyboardInterrupt("original cancellation") if with_primary else None
        ws = Workspace(tmp_path / str(with_primary), {})
        ws.__enter__()
        monkeypatch.setattr(ws, "lock_file", BrokenClose(ws.lock_file))
        if primary is None:
            with pytest.raises(OSError, match="close failed"):
                ws.__exit__(None, None, None)
        else:
            ws.__exit__(KeyboardInterrupt, primary, None)
            assert any("cleanup" in note for note in primary.__notes__)


@pytest.mark.parametrize(
    "overlap",
    [
        "source",
        "output",
        "workspace",
        "ancestor",
        "policy",
        "completed_dataset",
    ],
)
def test_cli_progress_path_rejected_before_mutation(source, policy, tmp_path, capsys, overlap):
    policy_path = tmp_path / "policy.json"
    write_json(policy_path, asdict(policy))
    output, work = tmp_path / "new-data", tmp_path / "new-work"
    previous = tmp_path / "previous-dataset"
    if overlap == "completed_dataset":
        write_json(previous / "manifest.json", {"status": "COMPLETED"})
    forbidden = {
        "source": source.folder / "diagnostics",
        "output": output / "diagnostics",
        "workspace": work / "diagnostics",
        "ancestor": tmp_path,
        "policy": policy_path / "diagnostics",
        "completed_dataset": previous / "diagnostics",
    }[overlap]
    before = set(tmp_path.rglob("*"))
    code = main(
        [
            "prepare",
            "--policy",
            str(policy_path),
            "--scope",
            "FIXTURE_ONLY",
            "--source",
            source.id,
            "--assembly-index",
            str(tmp_path / "index.json"),
            "--output",
            str(output),
            "--checkpoint-dir",
            str(work),
            "--progress-dir",
            str(forbidden),
            "--train-before",
            "2022-01-02",
            "--test-from",
            "2022-01-04",
        ]
    )
    result = json.loads(capsys.readouterr().out)
    assert code == 2 and result["status"] == "FAILED"
    assert result["error"] == (
        "New control/output directory is inside an existing artifact"
        if overlap == "completed_dataset"
        else "Progress directory must be outside output/workspace"
    )
    assert set(tmp_path.rglob("*")) == before


def test_cross_filesystem_publication_rejected_before_any_output(source, tmp_path, monkeypatch):
    output, work = tmp_path / "data", tmp_path / "work"
    monkeypatch.setattr(resume_module, "_existing_device", lambda path: 1 if path == output else 2)
    before = set(tmp_path.rglob("*"))
    with pytest.raises(ValueError, match="same filesystem"):
        resume_module._preflight(source, output, work, False)
    assert set(tmp_path.rglob("*")) == before


def test_atomic_replace_failure_leaves_fsynced_partial_and_prior_generation(tmp_path, monkeypatch):
    with Workspace(tmp_path / "work", {}) as ws:
        old = _seal(ws, "old.parquet", [1])
        ws.commit("SOURCE", {"cursor": 1}, [old])
        before = (ws.folder / "checkpoint.json").read_bytes()
        new = _seal(ws, "new.parquet", [1, 2])
        fsync = progress_module.os.fsync
        calls = []

        def tracked_fsync(descriptor):
            calls.append("fsync")
            return fsync(descriptor)

        def replace_failure(source, destination):
            calls.append("replace")
            raise OSError("atomic replace failure")

        monkeypatch.setattr(progress_module.os, "fsync", tracked_fsync)
        monkeypatch.setattr(progress_module.os, "replace", replace_failure)
        with pytest.raises(OSError, match="atomic replace failure"):
            ws.commit("SOURCE", {"cursor": 2}, [new])
        assert (ws.folder / "checkpoint.json").read_bytes() == before
        pending = list(ws.folder.glob("checkpoint.json.*.partial"))
        assert calls == ["fsync", "replace"]
        assert len(pending) == 1 and read_json(pending[0])["data"] == {"cursor": 2}
        assert ws.path(old).exists() and ws.path(new).exists()


@pytest.mark.parametrize(
    "cut",
    [
        "source_second_owner",
        "source_next_part",
        "owner_merge_closed_before_commit",
        "account_partial_evidence",
        "before_publication_rename",
        "after_publication_rename",
        "retired_file_cleanup",
    ],
)
def test_mid_operation_failures_reopen_exact_committed_frontier(
    source,
    policy,
    tmp_path,
    monkeypatch,
    cut,
):
    values, labels = source_values(source)
    rewrite_fixture(source, values, labels, part_rows=5)
    settings = dict(partitions=2, sort_run_rows=1, merge_fan_in=2, checkpoint_owner_rows=1)
    baseline, output, work = (tmp_path / name for name in ("baseline", "result", "work"))
    expected = prepare(source, policy, baseline, **settings)
    original_add, original_write = streaming.SortedRuns.add, streaming.SortedRuns._write
    original_rename, original_unlink = Path.rename, Path.unlink
    seen, failed, retained = {"owners": 0, "evidence-rows": 0}, [], []

    def interrupt_add(sorter, row, context):
        prefix = sorter.prefix
        if prefix in seen:
            seen[prefix] += 1
        target = 2 if cut == "source_second_owner" else 12
        if (
            not failed
            and cut in {"source_second_owner", "source_next_part"}
            and prefix == "owners"
            and seen[prefix] == target
        ):
            failed.append(cut)
            raise _InjectedOperationError("injected between derived owner rows")
        original_add(sorter, row, context)
        if (
            not failed
            and cut == "account_partial_evidence"
            and prefix == "evidence-rows"
            and seen[prefix] == 2
        ):
            failed.append(cut)
            raise _InjectedOperationError("injected after part of an alert evidence was added")

    def interrupt_write(sorter, iterator):
        path = original_write(sorter, iterator)
        if (
            not failed
            and cut == "owner_merge_closed_before_commit"
            and sorter.prefix == "owner-merge"
        ):
            failed.append(cut)
            retained.append((path, path.read_bytes()))
            raise _InjectedOperationError("injected after closed owner merge before checkpoint")
        return path

    def interrupt_rename(path, destination):
        selected = path.name == "dataset" and Path(destination) == output and not failed
        if selected and cut == "before_publication_rename":
            failed.append(cut)
            raise _InjectedOperationError("injected before publication rename")
        result = original_rename(path, destination)
        if selected and cut == "after_publication_rename":
            failed.append(cut)
            raise _InjectedOperationError("injected after publication rename before checkpoint")
        return result

    def interrupt_unlink(path, *args, **kwargs):
        if (
            not failed
            and cut == "retired_file_cleanup"
            and path.is_relative_to(work)
            and path.suffix == ".parquet"
        ):
            failed.append(cut)
            retained.append((path, path.read_bytes()))
            raise _InjectedOperationError("injected after checkpoint replace during retirement")
        return original_unlink(path, *args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(streaming.SortedRuns, "add", interrupt_add)
        context.setattr(streaming.SortedRuns, "_write", interrupt_write)
        context.setattr(Path, "rename", interrupt_rename)
        context.setattr(Path, "unlink", interrupt_unlink)
        expected_error = (
            CheckpointCleanupError if cut == "retired_file_cleanup" else _InjectedOperationError
        )
        with pytest.raises(expected_error) as caught:
            prepare(source, policy, output, checkpoint_dir=work, **settings)
        if cut == "retired_file_cleanup" and not isinstance(
            caught.value.__cause__, _InjectedOperationError
        ):
            raise caught.value
    assert failed == [cut]
    saved = read_json(work / "checkpoint.json")
    if cut == "source_second_owner":
        assert saved["phase"] == "SOURCE" and saved["data"] == {}
    elif cut == "source_next_part":
        assert saved["phase"] == "SOURCE" and saved["data"]["next_part"] == 1
    elif cut == "owner_merge_closed_before_commit":
        assert saved["phase"] == "OWNERS"
    elif cut == "account_partial_evidence":
        assert saved["phase"] == "ACCOUNTS"
    elif "publication_rename" in cut:
        assert saved["phase"] == "PUBLISH_INTENT"
    elif cut == "retired_file_cleanup":
        assert saved["generation"] > 0
    actual = prepare(source, policy, output, checkpoint_dir=work, resume=True, **settings)
    assert actual == expected
    for name in DATASET_FILES | {"manifest.json"}:
        assert (baseline / name).read_bytes() == (output / name).read_bytes(), name
    for path, original_bytes in retained:
        assert path.read_bytes() == original_bytes
