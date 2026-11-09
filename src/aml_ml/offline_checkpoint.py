"""완결 Parquet 파일에 대해 잠금과 버전이 있는 체크포인트를 저장합니다.

파일을 fsync한 뒤 같은 폴더에서 원자적으로 교체합니다.
단전 시 디렉터리 항목의 내구성은 파일시스템에 달려 있습니다.
커밋되지 않은 파일은 완료로 추정하거나 제자리에서 수리하지 않습니다.
"""

import copy
import hashlib
import os
import sys
from pathlib import Path
from uuid import uuid4

import pyarrow.parquet as pq

from .offline_io import close_all
from .offline_measurements import temporary_closed, temporary_retired, temporary_root
from .offline_progress import atomic_json, progress
from .offline_rules import fingerprint
from .storage import read_json, sha256

CHECKPOINT_VERSION = "aml-offline-checkpoint-v1"


class CheckpointCleanupError(OSError):
    """새 generation 저장은 끝났고 이전 파일 정리만 실패한 경우입니다."""

    def __init__(self, generation, count):
        self.committed_generation = generation
        super().__init__(
            f"Checkpoint generation {generation} committed; {count} retired files remain"
        )


def schema_hash(schema):
    return hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()


class Workspace:
    def __init__(self, folder: Path, binding: dict, *, resume=False):
        if folder.is_symlink() or folder.is_junction():
            raise ValueError("Workspace root must not be a link/junction")
        self.folder = folder.resolve()
        self.binding = binding
        self.resume = resume
        self.lock_file = None
        self.state: dict = {}
        self.receipts = {}
        self.attempt = self.folder / "unallocated-attempt"

    @staticmethod
    def require_checkpoint(folder):
        if folder.is_symlink() or folder.is_junction():
            raise ValueError("Workspace root must not be a link/junction")
        folder = folder.resolve()
        if not folder.is_dir() or not (folder / "checkpoint.json").is_file():
            raise ValueError("No committed checkpoint; preserve old/partial work and use a new run")
        if (folder / "checkpoint.json").is_symlink():
            raise ValueError("Checkpoint control file must not be a link")

    def __enter__(self):
        if self.resume:
            self.require_checkpoint(self.folder)
        else:
            self.folder.mkdir(parents=True, exist_ok=False)
        if (self.folder / "workspace.lock").is_symlink():
            raise ValueError("Workspace lock must not be a link")
        self.lock_file = (self.folder / "workspace.lock").open("a+b")
        try:
            self.lock_file.seek(0, os.SEEK_END)
            if self.lock_file.tell() == 0:
                self.lock_file.write(b"\0")
                self.lock_file.flush()
            self.lock_file.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.lock_file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle, self.lock_file = self.lock_file, None
            try:
                handle.close()
            except OSError as close_error:
                exc.add_note("Lock handle close also failed: " + type(close_error).__name__)
            raise ValueError("Workspace already locked; concurrent execution is forbidden") from exc
        try:
            if self.resume:
                saved = read_json(self.folder / "checkpoint.json")
                declared = saved.pop("checkpoint_sha256", None)
                if declared != fingerprint(saved):
                    raise ValueError("Checkpoint manifest digest mismatch")
                if (
                    saved.get("version") != CHECKPOINT_VERSION
                    or saved.get("binding") != self.binding
                ):
                    raise ValueError("Checkpoint source/policy/schema/code/settings/mode mismatch")
                if type(saved.get("generation")) is not int or saved["generation"] < 0:
                    raise ValueError("Invalid checkpoint generation")
                self.state = saved
                self.receipts = copy.deepcopy(saved["files"])
                for index, (name, receipt) in enumerate(self.receipts.items()):
                    self.verify_receipt(name, receipt)
                    progress(
                        "checkpoint_readback",
                        processed=index + 1,
                        total=len(self.receipts),
                        unit="files",
                    )
            else:
                self.state = {
                    "version": CHECKPOINT_VERSION,
                    "binding": self.binding,
                    "generation": 0,
                    "phase": "SOURCE",
                    "data": {},
                    "files": {},
                }
                atomic_json(
                    self.folder / "checkpoint.json",
                    dict(self.state, checkpoint_sha256=fingerprint(self.state)),
                    retry_control_replace=True,
                )
            self.attempt = self.folder / ("attempt-" + uuid4().hex)
            self.attempt.mkdir(exist_ok=False)
            temporary_root(self.folder)
            for name in self.receipts:
                temporary_closed(self.path(name))
            return self
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise

    def __exit__(self, kind, value, traceback):
        if self.lock_file is not None:
            handle, self.lock_file = self.lock_file, None
            cleanup_error = None
            try:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError as exc:
                cleanup_error = exc
            finally:
                try:
                    handle.close()
                except OSError as exc:
                    if cleanup_error is None:
                        cleanup_error = exc
                    else:
                        cleanup_error.add_note("Workspace handle close also failed")
            if cleanup_error is not None:
                if value is None:
                    raise cleanup_error
                value.add_note("Workspace cleanup also failed: " + type(cleanup_error).__name__)

    def path(self, name):
        path = self.folder / name
        resolved = path.resolve()
        if (
            Path(name).is_absolute()
            or resolved == self.folder
            or not resolved.is_relative_to(self.folder)
            or path.is_symlink()
            or ".partial" in path.name
            or resolved.relative_to(self.folder).as_posix() != name
        ):
            raise ValueError("Unsafe or partial checkpoint file")
        return resolved

    def seal(self, path, *, rows, schema):
        """닫힌 완결 파일만 재사용 가능한 해시·행 수 기록에 등록합니다."""
        name = path.relative_to(self.folder).as_posix()
        path = self.path(name)
        metadata = pq.read_metadata(path)
        if metadata.num_rows != rows or pq.read_schema(path) != schema:
            raise ValueError("Incomplete spill footer/schema/row count")
        with path.open("r+b") as stream:
            stream.flush()
            os.fsync(stream.fileno())
        receipt = {
            "rows": rows,
            "schema_sha256": schema_hash(schema),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        self.receipts[name] = receipt
        return name

    def verify_receipt(self, name, receipt):
        path = self.path(name)
        if not path.is_file() or path.stat().st_size != receipt["bytes"]:
            raise ValueError("Checkpoint file missing or size changed")
        if sha256(path) != receipt["sha256"]:
            raise ValueError("Checkpoint file hash changed")
        if (
            pq.read_metadata(path).num_rows != receipt["rows"]
            or schema_hash(pq.read_schema(path)) != receipt["schema_sha256"]
        ):
            raise ValueError("Checkpoint file rows/schema changed")

    def commit(self, phase, data, files):
        if not self.state or self.lock_file is None:
            raise RuntimeError("Checkpoint requires the workspace lock")
        names = list(files)
        if len(names) != len(set(names)) or any(name not in self.receipts for name in names):
            raise ValueError("Duplicate or unsealed file in checkpoint frontier")
        saved = {
            "version": CHECKPOINT_VERSION,
            "binding": self.binding,
            "generation": self.state["generation"] + 1,
            "phase": phase,
            "data": copy.deepcopy(data),
            "files": {name: self.receipts[name] for name in sorted(names)},
        }
        retired = set(self.receipts) - set(names)
        atomic_json(
            self.folder / "checkpoint.json",
            dict(saved, checkpoint_sha256=fingerprint(saved)),
            retry_control_replace=True,
        )
        # 교체 전 실패하면 입력을 삭제하지 않습니다. 교체 후 실패하면 새 generation을 읽어 재개합니다.
        self.state = saved
        old_receipts = self.receipts
        self.receipts = {name: old_receipts[name] for name in names}
        failures = []
        for name in sorted(retired):
            try:
                self.path(name).unlink()
                temporary_retired(self.path(name))
            except OSError as exc:
                failures.append(exc)
        if failures:
            # 재개 시 반드시 디스크의 새 phase를 읽습니다. 저장된 ACK 이전으로 되돌리지 않습니다.
            raise CheckpointCleanupError(saved["generation"], len(failures)) from failures[0]
        return copy.deepcopy(saved)


def frontier_files(sort_state):
    return [name for level in sort_state["levels"] for name in level]


def verify_frontier(workspace, state, expected_schema):
    names = frontier_files(state)
    if len(names) != len(set(names)):
        raise ValueError("Duplicate file in sort frontier")
    total = 0
    for name in names:
        item = workspace.receipts.get(name)
        if item is None or item["schema_sha256"] != schema_hash(expected_schema):
            raise ValueError("Sort frontier schema/receipt mismatch")
        total += item["rows"]
    if total != state["total_rows"]:
        raise ValueError("Sort frontier has missing/duplicated rows")
    return names


class MergeCursor:
    """실제 처리 ACK만 저장하고 groupby·heap의 미리 읽은 행은 제외합니다."""

    def __init__(self, paths, key, *, reader, offsets, row_counts, fan_in, max_bytes, batch_rows):
        if len(paths) > fan_in or len(paths) != len(offsets) or len(paths) != len(row_counts):
            raise ValueError("Invalid bounded merge frontier")
        if any(type(n) is not int or not 0 <= n <= total for n, total in zip(offsets, row_counts)):
            raise ValueError("Merge cursor outside committed rows")
        self.paths, self.key, self.reader = paths, key, reader
        self.offsets, self.row_counts = list(offsets), list(row_counts)
        self.max_bytes, self.batch_rows = max_bytes, batch_rows
        self.pending = {}
        self.iteration = self._iterate()

    def _iterate(self):
        import heapq

        streams = [
            self.reader(path, self.batch_rows, self.max_bytes, start_row=offset)
            for path, offset in zip(self.paths, self.offsets)
        ]
        positions = list(self.offsets)
        heap = []
        try:
            for i, stream in enumerate(streams):
                row = next(stream, None)
                if row is not None:
                    heapq.heappush(heap, (self.key(row), i, row))
            while heap:
                _, i, row = heapq.heappop(heap)
                if self.pending:
                    raise ValueError("Logical row must be acknowledged before consuming another")
                positions[i] += 1
                self.pending[id(row)] = (row, i, positions[i])
                yield row
                if self.pending:
                    raise ValueError("Logical row must be acknowledged before advancing/EOF")
                following = next(streams[i], None)
                if following is not None:
                    heapq.heappush(heap, (self.key(following), i, following))
        finally:
            close_all(streams)

    def acknowledge(self, row):
        pending = self.pending.pop(id(row), None)
        if pending is None:
            raise ValueError("Unknown or duplicate logical owner row acknowledgement")
        kept_row, index, position = pending
        if kept_row is not row:
            raise ValueError("Logical merge row identity changed")
        if position != self.offsets[index] + 1:
            raise ValueError("Skipped logical owner row")
        self.offsets[index] = position

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.iteration)

    def close(self):
        self.iteration.close()

    def require_complete(self):
        if self.pending or self.offsets != self.row_counts:
            raise ValueError("Merge frontier has unacknowledged or missing rows")
