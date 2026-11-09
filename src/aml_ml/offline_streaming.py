"""소유자별 외부 정렬, 크기를 제한한 계좌 구간과 결정적 원천 재연산을 제공합니다.

입력 행은 원본 순번을 유지하며, 정렬되지 않았을 수 있는 시각은 외부 정렬합니다.
버킷·계좌의 누적 데이터는 자르지 않고 디스크로 내보냅니다. 현재 구간이나 단일
경보의 크기 초과는 거부합니다. 학습에는 별도의 자원 제한을 적용합니다.

max_partition_bytes는 논리 Arrow 데이터, Python 행 작업 공간 추정치와
직렬화 본문을 각각 제한합니다. 디코더·할당자·인덱스 및 여러 표현의 동시 보관
비용이 존재하므로 프로세스 RSS 제한은 아닙니다.
"""

import hashlib
import heapq
import json
from contextlib import ExitStack, contextmanager
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq

from .dataset_quality import DatasetQuality
from .offline_account_stream import account_alerts, check_budget
from .offline_contract import DATASET_FILES as FILES
from .offline_contract import (
    FIT_BATCH_ROWS,
    SPLITS,
    alert_records,
    contract,
    source_fingerprint,
    validate_artifact_snapshot,
    validate_dataset_integrity,
    validate_source,
)
from .offline_eligibility import DatasetNotReady, EligibilityCounts
from .offline_features import FEATURE_NAMES, AlertPreprocessor
from .offline_io import DEFAULT_DECODE_ROWS, DEFAULT_WRITE_ROWS, close_all
from .offline_io import row_workset_bytes as _row_workset_bytes
from .offline_io import rows as _rows
from .offline_measurements import (
    event,
    milestone,
    temporary_closed,
    temporary_retired,
    temporary_root,
)
from .offline_progress import check_cancelled, progress
from .offline_publish import publish_directory
from .offline_rules import R02Policy, _json_bytes, fingerprint, validate_transactions
from .offline_serialization import (
    ALERT_SCHEMA,
    LEDGER_SCHEMA,
    TARGET_SCHEMA,
    TableWriterFactory,
    ThreeWriter,
    finish_formats,
    record_table,
)
from .offline_settings import execution_settings, reader_budget, validate_boundaries
from .schema import epoch_us
from .storage import SourceInput, read_json, sha256, write_json

LAYOUT = "ACCOUNT_CHECKPOINT_THREE_FORMATS_V3"
MERGE_FAN_IN = 8
RAW_SCHEMA = pa.schema([(n, pa.float64()) for n in FEATURE_NAMES])
Y_SCHEMA = pa.schema([("training_alert_uid", pa.string()), ("y_alert_proxy", pa.int8())])
RECORD_SCHEMA = pa.schema(
    [
        ("reference_ts", pa.timestamp("us")),
        ("training_alert_uid", pa.string()),
        ("split", pa.string()),
        ("alert_json", pa.string()),
        ("target_json", pa.string()),
        ("ledger_json", pa.string()),
    ]
    + list(RAW_SCHEMA)
)


@contextmanager
def _scratch(parent: Path):
    parent = parent.resolve()
    parent.mkdir(parents=True, exist_ok=True)
    # 정리는 이 부모 경로 아래에서 작업용으로 새로 만든 임시 폴더로 제한합니다.
    with TemporaryDirectory(prefix=".offline-work-", dir=parent) as name:
        resolved = Path(name).resolve()
        if resolved.parent != parent or not resolved.name.startswith(".offline-work-"):
            raise ValueError("Unsafe scratch cleanup target")
        temporary_root(resolved)
        try:
            yield resolved
        finally:
            temporary_retired(resolved)


def _bucket(account: str, count: int) -> int:
    return int.from_bytes(hashlib.blake2b(account.encode(), digest_size=8).digest(), "big") % count


def _arrow_row_bytes(row: dict, schema: pa.Schema) -> int:
    """Arrow 할당 전에 null 비트를 포함한 평면 행 크기를 보수적으로 구합니다."""
    total = 0
    for field in schema:
        value, kind = row.get(field.name), field.type
        total += 1  # 행별 유효성 표시에 필요한 공간을 보수적으로 반영합니다.
        if pa.types.is_string(kind) or pa.types.is_large_string(kind):
            total += 8 if pa.types.is_large_string(kind) else 4
            if value is not None:
                total += sum(
                    1 if ord(c) < 128 else 2 if ord(c) < 2048 else 3 if ord(c) < 65536 else 4
                    for c in value
                )
        elif pa.types.is_binary(kind) or pa.types.is_large_binary(kind):
            total += (8 if pa.types.is_large_binary(kind) else 4) + (
                0 if value is None else len(value)
            )
        elif pa.types.is_null(kind):
            continue
        elif pa.types.is_fixed_size_binary(kind):
            total += kind.byte_width
        else:
            # 원천 계약은 평면 기본형·decimal·timestamp 데이터입니다.
            try:
                total += max(1, (kind.bit_width + 7) // 8)
            except (AttributeError, ValueError) as exc:
                raise ValueError("Unsupported type for bounded offline row allocation") from exc
    return total


def _bounded_table(rows, schema, max_bytes):
    arrow_bytes = workset_bytes = 0
    for row in rows:
        arrow_size, workset_size = _arrow_row_bytes(row, schema), _row_workset_bytes(row)
        context = {"stage": "preallocation"}
        check_budget("arrow_bytes", arrow_bytes, arrow_size, max_bytes, context)
        check_budget("workset_bytes", workset_bytes, workset_size, max_bytes, context)
        arrow_bytes += arrow_size
        workset_bytes += workset_size
    table = pa.Table.from_pylist(rows, schema=schema)
    check_budget("allocated_arrow_bytes", 0, table.nbytes, max_bytes, {"stage": "allocation"})
    return table


def _merge(
    paths: list[Path],
    key,
    *,
    max_bytes: int,
    fan_in=MERGE_FAN_IN,
    decode_rows=DEFAULT_DECODE_ROWS,
):
    if len(paths) > fan_in:
        raise ValueError("Merge fan-in exceeds bounded reader limit")
    iterators = [_rows(p, decode_rows, reader_budget(max_bytes, fan_in)) for p in paths]
    heap = []
    try:
        for i, iterator in enumerate(iterators):
            row = next(iterator, None)
            if row is not None:
                heapq.heappush(heap, (key(row), i, row))
        while heap:
            check_cancelled()
            _, i, row = heapq.heappop(heap)
            yield row
            following = next(iterators[i], None)
            if following is not None:
                heapq.heappush(heap, (key(following), i, following))
    finally:
        close_all(iterators)


def _bounded_merge(
    paths,
    key,
    *,
    work,
    schema,
    prefix,
    max_bytes,
    fan_in=MERGE_FAN_IN,
    decode_rows=DEFAULT_DECODE_ROWS,
    write_rows=DEFAULT_WRITE_ROWS,
):
    """안정적인 다단계 병합이며, 병합마다 최대 MERGE_FAN_IN개 리더를 엽니다."""
    paths = list(paths)
    stage = 0
    while len(paths) > fan_in:
        following_paths = []
        for start in range(0, len(paths), fan_in):
            event("merge_operations")
            event(f"merge_pass_{stage}")
            path = work / f"{prefix}-merge-{stage}-{start}.parquet"
            partial = path.with_suffix(".parquet.partial")
            if path.exists() or partial.exists():
                raise FileExistsError("Merge stage already exists")
            merged = _merge(
                paths[start : start + fan_in],
                key,
                max_bytes=max_bytes,
                fan_in=fan_in,
                decode_rows=decode_rows,
            )
            write_budget = reader_budget(max_bytes, fan_in)
            with partial.open("xb") as stream, pq.ParquetWriter(stream, schema) as writer:
                buffer, size = [], 0
                try:
                    for row in merged:
                        row_bytes = max(_arrow_row_bytes(row, schema), _row_workset_bytes(row))
                        check_budget(
                            "merge_row_bytes",
                            0,
                            row_bytes,
                            write_budget,
                            {"stage": "merge_preallocation", "merge_pass": stage},
                        )
                        if buffer and (
                            len(buffer) == write_rows or size + row_bytes > write_budget
                        ):
                            writer.write_table(_bounded_table(buffer, schema, write_budget))
                            buffer.clear()
                            size = 0
                        buffer.append(row)
                        size += row_bytes
                    if buffer:
                        writer.write_table(_bounded_table(buffer, schema, write_budget))
                finally:
                    close_all([merged])
            partial.rename(path)
            event("parquet_closed_write_bytes", path.stat().st_size)
            temporary_closed(path)
            following_paths.append(path)
        paths = following_paths
        stage += 1
    yield from _merge(paths, key, max_bytes=max_bytes, fan_in=fan_in, decode_rows=decode_rows)


class SortedRuns:
    """크기를 제한한 정렬 조각을 즉시 8방향 병합하며 O(log N)개 경로만 유지합니다."""

    def __init__(
        self,
        work,
        schema,
        key,
        *,
        prefix,
        max_rows,
        max_bytes,
        workspace=None,
        fan_in=MERGE_FAN_IN,
        decode_rows=DEFAULT_DECODE_ROWS,
        write_rows=DEFAULT_WRITE_ROWS,
        sort_buffer_bytes=None,
    ):
        self.work, self.schema, self.key, self.prefix = work, schema, key, prefix
        self.max_rows, self.max_bytes = max_rows, max_bytes
        self.sort_buffer_bytes = max_bytes if sort_buffer_bytes is None else sort_buffer_bytes
        self.buffer, self.levels = [], []
        self.size = self.sequence = self.total_rows = 0
        self.peak_rows = self.peak_bytes = 0
        self.workspace = workspace
        self.fan_in, self.decode_rows, self.write_rows = fan_in, decode_rows, write_rows

    def add(self, row, context):
        size = max(_arrow_row_bytes(row, self.schema), _row_workset_bytes(row))
        check_budget("spill_row_bytes", 0, size, self.sort_buffer_bytes, context)
        if self.buffer and (
            len(self.buffer) >= self.max_rows or self.size + size > self.sort_buffer_bytes
        ):
            self.flush()
        self.buffer.append(row)
        self.size += size
        self.total_rows += 1
        self.peak_rows = max(self.peak_rows, len(self.buffer))
        self.peak_bytes = max(self.peak_bytes, self.size)

    def _write(self, rows):
        suffix = "" if self.workspace is None else "-" + uuid4().hex
        path = self.work / f"{self.prefix}-run-{self.sequence}{suffix}.parquet"
        partial = path.with_suffix(".parquet.partial")
        self.sequence += 1
        count, previous = 0, None
        write_budget = reader_budget(self.max_bytes, self.fan_in)
        with partial.open("xb") as stream, pq.ParquetWriter(stream, self.schema) as writer:
            buffer, size = [], 0
            for row in rows:
                current = self.key(row)
                if previous is not None and current < previous:
                    raise ValueError("Spill output is not ordered")
                previous = current
                count += 1
                row_bytes = max(_arrow_row_bytes(row, self.schema), _row_workset_bytes(row))
                check_budget("merge_row_bytes", 0, row_bytes, write_budget, {"stage": self.prefix})
                if buffer and (len(buffer) == self.write_rows or size + row_bytes > write_budget):
                    writer.write_table(_bounded_table(buffer, self.schema, write_budget))
                    buffer.clear()
                    size = 0
                buffer.append(row)
                size += row_bytes
            if buffer:
                writer.write_table(_bounded_table(buffer, self.schema, write_budget))
        if path.exists():
            raise FileExistsError("Spill output already exists")
        partial.rename(path)
        event("parquet_closed_write_bytes", path.stat().st_size)
        temporary_closed(path)
        milestone("first_spill")
        if self.workspace is not None:
            self.workspace.seal(path, rows=count, schema=self.schema)
        return path

    def flush(self):
        if not self.buffer:
            return
        self.buffer.sort(key=self.key)
        path = self._write(self.buffer)
        self.buffer.clear()
        self.size = 0
        level = 0
        while True:
            if level == len(self.levels):
                self.levels.append([])
            self.levels[level].append(path)
            if len(self.levels[level]) < self.fan_in:
                return
            inputs = self.levels[level]
            event("merge_operations")
            event(f"sort_compaction_level_{level}")
            merged = _merge(
                inputs,
                self.key,
                max_bytes=self.max_bytes,
                fan_in=self.fan_in,
                decode_rows=self.decode_rows,
            )
            try:
                path = self._write(merged)
            finally:
                close_all([merged])
            # 새 임시 폴더 안에서 내부적으로 만들고 닫은 정렬 조각만 처리합니다.
            if self.workspace is None:
                for item in inputs:
                    if item.parent != self.work or not item.name.startswith(self.prefix + "-run-"):
                        raise ValueError("Unsafe spill cleanup target")
                    item.unlink()
                    temporary_retired(item)
            self.levels[level] = []
            level += 1

    def paths(self):
        self.flush()
        return [path for level in self.levels for path in level]

    def snapshot(self):
        if self.workspace is None:
            raise ValueError("Only durable sorters have checkpoint snapshots")
        self.flush()
        return {
            "levels": [
                [path.relative_to(self.workspace.folder).as_posix() for path in level]
                for level in self.levels
            ],
            "sequence": self.sequence,
            "total_rows": self.total_rows,
            "peak_rows": self.peak_rows,
            "peak_bytes": self.peak_bytes,
        }

    def restore(self, state):
        from .offline_checkpoint import verify_frontier

        workspace = self.workspace
        if workspace is None:
            raise ValueError("Only durable sorters can restore checkpoint snapshots")
        verify_frontier(workspace, state, self.schema)
        if any(len(level) >= self.fan_in for level in state["levels"]):
            raise ValueError("Invalid sort compaction frontier")
        self.levels = [[workspace.path(name) for name in level] for level in state["levels"]]
        self.sequence = state["sequence"]
        self.total_rows = state["total_rows"]
        self.peak_rows, self.peak_bytes = state["peak_rows"], state["peak_bytes"]


def _sorter(work, schema, key, prefix, settings, workspace=None):
    return SortedRuns(
        work,
        schema,
        key,
        prefix=prefix,
        max_rows=settings["sort_run_rows"],
        max_bytes=settings["max_partition_bytes"],
        workspace=workspace,
        sort_buffer_bytes=settings["sort_run_bytes"],
        fan_in=settings["merge_fan_in"],
        decode_rows=settings["decode_batch_rows"],
        write_rows=settings["write_batch_rows"],
    )


def _partition_source(source: SourceInput, work: Path, settings: dict, *, workspace=None):
    validate_source(source)
    count = source.metadata["input_rows"]
    cap = settings["max_source_rows"]
    if cap is not None:
        if type(cap) is not int or cap < 1 or count < 1:
            raise ValueError("Invalid full-source row budget; no truncation")
        check_budget("full_source_rows", 0, count, cap, {"stage": "source_preflight"})
    parts = settings["partitions"]
    max_bytes = settings["max_partition_bytes"]
    batch_rows = settings["decode_batch_rows"]
    tx_schema = pq.read_schema(source.part_path("transactions", 0))
    if {"_owner", "_bucket", "_offline_label"} & set(tx_schema.names):
        raise ValueError("Input uses reserved offline work columns")
    schema = pa.schema(
        list(tx_schema)
        + [("_owner", pa.string()), ("_bucket", pa.int32()), ("_offline_label", pa.int8())]
    )
    sorter = _sorter(work, schema, _owner_key, "owners", settings, workspace)
    processed, coverage = 0, None
    bucket_counts = [0] * parts
    next_part = 0
    if workspace is not None and workspace.state["data"]:
        saved = workspace.state["data"]
        next_part, processed, coverage = saved["next_part"], saved["processed"], saved["coverage"]
        bucket_counts = list(saved["bucket_counts"])
        if (
            type(next_part) is not int
            or not 0 <= next_part <= len(source.parts("transactions"))
            or processed != sum(part["rows"] for part in source.parts("transactions")[:next_part])
            or len(bucket_counts) != parts
        ):
            raise ValueError("Checkpoint source cursor/count mismatch")
        sorter.restore(saved["sorter"])
        if sorter.total_rows != sum(bucket_counts):
            raise ValueError("Checkpoint owner row totals differ")
        for i in range(next_part):
            source.part_path("transactions", i)
            source.part_path("labels", i)
            progress("source_prefix_hash", processed=i + 1, total=next_part, unit="parts")
    progress("source_partition", processed=processed, total=count, unit="source_rows")
    for i in range(next_part, len(source.parts("transactions"))):
        declared = source.parts("transactions")[i]
        tx = source.part_path("transactions", i)
        labels = source.part_path("labels", i)
        if pq.read_schema(tx) != tx_schema:
            raise ValueError("Source part schema drift")
        if (
            pq.read_metadata(tx).num_rows != declared["rows"]
            or pq.read_metadata(labels).num_rows != source.parts("labels")[i]["rows"]
        ):
            raise ValueError("Source Parquet row count differs from manifest before decoding")
        seen = 0
        tx_rows, label_rows = _rows(tx, batch_rows, max_bytes), _rows(labels, batch_rows, max_bytes)
        try:
            for row, truth in zip(tx_rows, label_rows, strict=True):
                if row["source_row_id"] != processed:
                    raise ValueError(
                        "Source row ordinals must be contiguous and ordered; no implicit repair"
                    )
                check_budget("source_rows", processed, 1, count, {"stage": "source", "part": i})
                check_budget("part_rows", seen, 1, declared["rows"], {"stage": "source", "part": i})
                validate_transactions(
                    [row], source.id, source.metadata["source_release"], complete_source=False
                )
                if (
                    truth["source_row_id"] != processed
                    or truth["transaction_uid"] != row["transaction_uid"]
                    or truth["source_positive"] is not None
                    and (
                        type(truth["source_positive"]) is not int
                        or truth["source_positive"] not in (0, 1)
                    )
                ):
                    raise ValueError(
                        "Transaction/label ordinal, UID or binary proxy source mismatch"
                    )
                stamp = epoch_us(row["ts"])
                coverage = stamp if coverage is None else min(coverage, stamp)
                processed += 1
                seen += 1
                if row["is_self_transfer"] or row["from_account_key"] == row["to_account_key"]:
                    continue
                for account in sorted({row["from_account_key"], row["to_account_key"]}):
                    index = _bucket(account, parts)
                    sorter.add(
                        dict(
                            row,
                            _owner=account,
                            _bucket=index,
                            _offline_label=truth["source_positive"],
                        ),
                        {
                            "stage": "source_spill",
                            "bucket": index,
                            "processed_source_rows": processed,
                        },
                    )
                    bucket_counts[index] += 1
                if processed % 4096 == 0:
                    progress(
                        "source_partition", processed=processed, total=count, unit="source_rows"
                    )
        finally:
            close_all([tx_rows, label_rows])
        if seen != declared["rows"] or seen != source.parts("labels")[i]["rows"]:
            raise ValueError("Part row count differs from source manifest")
        # 거래·라벨 검사와 파생 소유자 양쪽 처리가 모두 끝나야 이 커서를
        # 재사용할 수 있습니다. 행·파트 중간 실패 시 파트 전체를 재처리합니다.
        sorter.flush()
        if workspace is not None:
            from .offline_checkpoint import frontier_files

            saved_sorter = sorter.snapshot()
            workspace.commit(
                "SOURCE",
                {
                    "next_part": i + 1,
                    "processed": processed,
                    "coverage": coverage,
                    "bucket_counts": bucket_counts,
                    "sorter": saved_sorter,
                },
                frontier_files(saved_sorter),
            )
        progress("source_partition", processed=processed, total=count, unit="source_rows")
    if processed != count or coverage is None:
        raise ValueError("Incomplete full source")
    sorter.flush()
    return (
        sorter,
        tx_schema,
        coverage,
        {
            "input_rows": processed,
            "owned_rows": sorter.total_rows,
            "owned_rows_processed": 0,
            "max_bucket_total_rows": max(bucket_counts),
            "max_source_sort_buffer_rows": sorter.peak_rows,
            "max_source_sort_buffer_bytes": sorter.peak_bytes,
            "max_active_window_rows": 0,
            "max_active_window_workset_bytes": 0,
            "configured_decode_batch_rows": batch_rows,
            "source_flush_batch_rows": settings["sort_run_rows"],
            "output_batch_rows": FIT_BATCH_ROWS,
            "merge_fan_in": settings["merge_fan_in"],
        },
    )


def _owner_key(row):
    return row["_bucket"], row["_owner"], row["ts"], row["source_row_id"]


def _merge_options(settings):
    return dict(
        fan_in=settings["merge_fan_in"],
        decode_rows=settings["decode_batch_rows"],
        write_rows=settings["write_batch_rows"],
    )


def _consume_emissions(emissions, records, evidence, tx_schema, stats, max_bytes):
    for alert, feature, target, new_evidence, ledger in emissions:
        context = {"stage": "alert_spill", "input_rows": stats["input_rows"]}
        payload = alert_records([alert])[0]
        row = dict(
            feature,
            reference_ts=alert.reference_ts,
            training_alert_uid=target["training_alert_uid"],
            split=target["split"],
            alert_json="",
            target_json="",
            ledger_json="",
        )
        size = _json_bytes(payload) + _json_bytes(target) + _json_bytes(ledger)
        check_budget("serialized_alert_bytes", 0, size, max_bytes, context)
        row["alert_json"] = json.dumps(payload, sort_keys=True, allow_nan=False)
        row["target_json"] = json.dumps(target, sort_keys=True, allow_nan=False)
        row["ledger_json"] = json.dumps(ledger, sort_keys=True, allow_nan=False)
        records.add(row, context)
        for owned in new_evidence:
            evidence.add({name: owned[name] for name in tx_schema.names}, context)
        del alert, feature, target, new_evidence, ledger, payload, row


def _shards(
    source, policy, scope, work, sorter, tx_schema, coverage, first, second, settings, stats
):
    policy.require(scope)
    max_bytes = settings["max_partition_bytes"]
    records = _sorter(
        work,
        RECORD_SCHEMA,
        lambda r: (r["reference_ts"], r["training_alert_uid"]),
        "records",
        settings,
    )
    evidence = _sorter(
        work,
        tx_schema,
        lambda r: r["source_row_id"],
        "evidence-rows",
        settings,
    )
    ordered = _bounded_merge(
        sorter.paths(),
        _owner_key,
        work=work,
        schema=sorter.schema,
        prefix="ordered-owners",
        max_bytes=max_bytes,
        **_merge_options(settings),
    )
    emissions = account_alerts(
        ordered,
        source,
        policy,
        coverage=coverage,
        train_before=first,
        test_from=second,
        settings=settings,
        row_size=_row_workset_bytes,
        stats=stats,
        owner_done=lambda bucket, owner: progress(
            "account_windows",
            processed=stats["owned_rows_processed"],
            total=stats["owned_rows"],
            unit="owned_rows",
        ),
    )
    try:
        _consume_emissions(emissions, records, evidence, tx_schema, stats, max_bytes)
    finally:
        close_all([emissions, ordered])
    if stats["owned_rows_processed"] != stats["owned_rows"]:
        raise ValueError("Incomplete externally ordered owner history")
    stats["max_record_sort_buffer_bytes"] = records.peak_bytes
    stats["max_evidence_sort_buffer_bytes"] = evidence.peak_bytes
    return records.paths(), evidence.paths()


def _write_raw_outputs(
    output,
    work,
    record_paths,
    max_bytes,
    source_id,
    *,
    merge_options=None,
    writer_type: TableWriterFactory = ThreeWriter,
):
    processor = AlertPreprocessor()
    quality = DatasetQuality(source_id)
    eligibility = EligibilityCounts()
    counts = dict.fromkeys(SPLITS, 0)
    raw_buffer, train_buffer = [], []
    y_buffer = {s: [] for s in SPLITS}
    raw_bytes = train_bytes = 0
    y_bytes = dict.fromkeys(SPLITS, 0)
    with ExitStack() as stack:
        raw_writer = stack.enter_context(
            writer_type(output / "raw.parquet", RAW_SCHEMA, max_bytes=max_bytes)
        )
        y_writers = {
            s: stack.enter_context(
                writer_type(output / f"y_{s}.parquet", Y_SCHEMA, max_bytes=max_bytes)
            )
            for s in SPLITS
        }
        record_writers = {
            "alerts": stack.enter_context(
                writer_type(output / "alerts.parquet", ALERT_SCHEMA, max_bytes=max_bytes)
            ),
            "targets": stack.enter_context(
                writer_type(output / "targets.parquet", TARGET_SCHEMA, max_bytes=max_bytes)
            ),
            "ledger": stack.enter_context(
                writer_type(output / "ledger.parquet", LEDGER_SCHEMA, max_bytes=max_bytes)
            ),
        }
        record_schemas = {"alerts": ALERT_SCHEMA, "targets": TARGET_SCHEMA, "ledger": LEDGER_SCHEMA}
        record_buffers = {name: [] for name in record_schemas}
        record_bytes = dict.fromkeys(record_schemas, 0)

        def write_record(name, payload, size):
            buffer = record_buffers[name]
            if buffer and (len(buffer) == FIT_BATCH_ROWS or record_bytes[name] + size > max_bytes):
                record_writers[name].write_table(record_table(buffer, record_schemas[name]))
                buffer.clear()
                record_bytes[name] = 0
            buffer.append(payload)
            record_bytes[name] += size

        split_files = {
            s: stack.enter_context((work / f"split-{s}.txt").open("w", encoding="utf-8"))
            for s in SPLITS
        }
        previous, n = None, 0
        merged = _bounded_merge(
            record_paths,
            lambda r: (r["reference_ts"], r["training_alert_uid"]),
            work=work,
            schema=RECORD_SCHEMA,
            prefix="alerts",
            max_bytes=max_bytes,
            **(merge_options or {}),
        )
        stack.callback(merged.close)
        for row in merged:
            milestone("first_final_row")
            key = (row["reference_ts"], row["training_alert_uid"])
            if previous is not None and key <= previous:
                raise ValueError("Duplicate/unsorted merged alert identity")
            previous = key
            split = row["split"]
            feature = {name: row[name] for name in FEATURE_NAMES}
            feature_bytes = _row_workset_bytes(feature)
            context = {"stage": "fixed_feature_buffer", "output_rows": n, "split": split}
            full_target = json.loads(row["target_json"])
            alert_size = (
                len(row["alert_json"].encode()) * 4 + 128 * full_target["evidence_count"] + 1024
            )
            target_size = len(row["target_json"].encode()) * 4 + 1024
            ledger_size = (
                len(row["ledger_json"].encode()) * 4 + 128 * full_target["evidence_count"] + 2048
            )
            check_budget("alert_record_decode_bytes", 0, alert_size, max_bytes, context)
            check_budget("target_record_decode_bytes", 0, target_size, max_bytes, context)
            check_budget("ledger_record_decode_bytes", 0, ledger_size, max_bytes, context)
            ledger = json.loads(row["ledger_json"])
            eligibility.add(ledger)
            write_record("alerts", json.loads(row["alert_json"]), alert_size)
            write_record("targets", full_target, target_size)
            write_record("ledger", ledger, ledger_size)
            n += 1
            if n % 4096 == 0:
                progress("final_features", processed=n, unit="alerts")
            if not ledger["supervised_eligible"]:
                continue
            check_budget("raw_feature_bytes", raw_bytes, feature_bytes, max_bytes, context)
            if split == "train":
                check_budget("train_feature_bytes", train_bytes, feature_bytes, max_bytes, context)
            split_files[split].write(
                ("," if counts[split] else "") + json.dumps(row["training_alert_uid"])
            )
            counts[split] += 1
            raw_buffer.append(feature)
            raw_bytes += feature_bytes
            quality.add(full_target, feature)
            truth = full_target["y_alert_proxy"]
            target = {"training_alert_uid": row["training_alert_uid"], "y_alert_proxy": truth}
            target_bytes = _row_workset_bytes(target)
            check_budget("target_buffer_bytes", y_bytes[split], target_bytes, max_bytes, context)
            y_buffer[split].append(target)
            y_bytes[split] += target_bytes
            if split == "train":
                train_buffer.append(feature)
                train_bytes += feature_bytes
                if len(train_buffer) == FIT_BATCH_ROWS:
                    processor.partial_fit(
                        pa.Table.from_pylist(train_buffer, schema=RAW_SCHEMA), split="train"
                    )
                    train_buffer.clear()
                    train_bytes = 0
            if len(raw_buffer) == FIT_BATCH_ROWS:
                raw_writer.write_table(pa.Table.from_pylist(raw_buffer, schema=RAW_SCHEMA))
                raw_buffer.clear()
                raw_bytes = 0
            if len(y_buffer[split]) == FIT_BATCH_ROWS:
                y_writers[split].write_table(pa.Table.from_pylist(y_buffer[split], schema=Y_SCHEMA))
                y_buffer[split].clear()
                y_bytes[split] = 0
        if raw_buffer:
            raw_writer.write_table(pa.Table.from_pylist(raw_buffer, schema=RAW_SCHEMA))
        if train_buffer:
            processor.partial_fit(
                pa.Table.from_pylist(train_buffer, schema=RAW_SCHEMA), split="train"
            )
        for split in SPLITS:
            if y_buffer[split]:
                y_writers[split].write_table(pa.Table.from_pylist(y_buffer[split], schema=Y_SCHEMA))
        for name, buffer in record_buffers.items():
            if buffer:
                record_writers[name].write_table(record_table(buffer, record_schemas[name]))
    write_json(output / "quality.json", dict(quality.report(), eligibility=eligibility.report()))
    return processor, counts


def _write_split_index(output, work):
    with (output / "splits.json").open("w", encoding="utf-8") as stream:
        stream.write("{")
        for i, split in enumerate(SPLITS):
            stream.write(("," if i else "") + json.dumps(split) + ":[")
            with (work / f"split-{split}.txt").open(encoding="utf-8") as src:
                for block in iter(lambda: src.read(65536), ""):
                    stream.write(block)
            stream.write("]")
        stream.write("}")


def _write_evidence(
    output,
    work,
    evidence_paths,
    tx_schema,
    max_bytes,
    *,
    merge_options=None,
    writer_type: TableWriterFactory = ThreeWriter,
):
    with writer_type(output / "evidence.parquet", tx_schema, max_bytes=max_bytes) as writer:
        buffer, previous_id, buffered_bytes = [], -1, 0
        merged = _bounded_merge(
            evidence_paths,
            lambda r: r["source_row_id"],
            work=work,
            schema=tx_schema,
            prefix="evidence",
            max_bytes=max_bytes,
            **(merge_options or {}),
        )
        try:
            for row in merged:
                if row["source_row_id"] <= previous_id:
                    raise ValueError("Duplicate/unsorted evidence across account partitions")
                previous_id = row["source_row_id"]
                size = max(_arrow_row_bytes(row, tx_schema), _row_workset_bytes(row))
                check_budget(
                    "evidence_row_bytes",
                    0,
                    size,
                    max_bytes,
                    {"stage": "evidence_preallocation", "source_row_id": previous_id},
                )
                if buffer and (len(buffer) == FIT_BATCH_ROWS or buffered_bytes + size > max_bytes):
                    writer.write_table(_bounded_table(buffer, tx_schema, max_bytes))
                    buffer.clear()
                    buffered_bytes = 0
                buffer.append(row)
                buffered_bytes += size
            if buffer:
                writer.write_table(_bounded_table(buffer, tx_schema, max_bytes))
        finally:
            close_all([merged])


def _write_feature_splits(
    output, processor, counts, max_bytes, decode_rows, writer_type: TableWriterFactory = ThreeWriter
):
    # 원시 행이 전체 시간순이므로 각 분할은 연속 구간을 차지합니다.
    with ExitStack() as stack:
        writers = {}
        raw_iter = _rows(output / "raw.parquet", decode_rows, max_bytes)
        try:
            for split in SPLITS:
                remaining = counts[split]
                while remaining:
                    size = min(FIT_BATCH_ROWS, remaining)
                    table = pa.Table.from_pylist(
                        [next(raw_iter) for _ in range(size)], schema=RAW_SCHEMA
                    )
                    x = processor.transform(table)
                    if split not in writers:
                        writers[split] = stack.enter_context(
                            writer_type(
                                output / f"X_{split}.parquet", x.schema, max_bytes=max_bytes
                            )
                        )
                    writers[split].write_table(x)
                    remaining -= size
            if next(raw_iter, None) is not None:
                raise ValueError("Unassigned raw feature rows")
        finally:
            raw_iter.close()


def _write_outputs(
    output,
    work,
    record_paths,
    evidence_paths,
    tx_schema,
    max_bytes,
    source_id,
    *,
    settings=None,
    writer_type: TableWriterFactory = ThreeWriter,
):
    merge_options = {} if settings is None else _merge_options(settings)
    progress("final_features", unit="alerts")
    processor, counts = _write_raw_outputs(
        output,
        work,
        record_paths,
        max_bytes,
        source_id,
        merge_options=merge_options,
        writer_type=writer_type,
    )
    # 학습 준비 조건을 확인하기 전에 후보 전체와 근거를 보존합니다.
    _write_split_index(output, work)
    progress("final_evidence", unit="rows")
    _write_evidence(
        output,
        work,
        evidence_paths,
        tx_schema,
        max_bytes,
        merge_options=merge_options,
        writer_type=writer_type,
    )
    if any(not counts[s] for s in SPLITS):
        raise DatasetNotReady("NOT_READY: all three eligible splits must be nonempty")
    try:
        processor.freeze()
    except ValueError as exc:
        raise DatasetNotReady("NOT_READY: eligible train cannot fit preprocessing") from exc
    processor.save(output / "preprocessing.json")
    _write_feature_splits(
        output,
        processor,
        counts,
        max_bytes,
        DEFAULT_DECODE_ROWS if settings is None else settings["decode_batch_rows"],
        writer_type,
    )
    if writer_type is ThreeWriter:
        finish_formats(output, max_bytes=max_bytes)
    if processor.state is None:
        raise RuntimeError("Preprocessing was not frozen")
    return counts, processor.state["feature_columns"]


def build_streaming_dataset(
    source: SourceInput,
    policy: R02Policy,
    output: Path,
    *,
    train_before: datetime,
    test_from: datetime,
    execution_scope: str,
    max_source_rows: int | None = None,
    partitions: int = 32,
    read_batch_rows: int = 4096,
    max_partition_rows: int = 100_000,
    max_partition_bytes: int = 64 * 1024**2,
    decode_batch_rows: int | None = None,
    sort_run_rows: int = 65536,
    sort_run_bytes: int | None = None,
    write_batch_rows: int = 8192,
    merge_fan_in: int = 8,
    horizon_seconds: int = 86400,
    checkpoint_owner_rows: int = 100_000,
    checkpoint_dir: Path | None = None,
    resume: bool = False,
    checkpoint_context: dict | None = None,
    _verify: bool = True,
) -> dict:
    settings = execution_settings(
        max_source_rows=max_source_rows,
        partitions=partitions,
        read_batch_rows=read_batch_rows,
        max_partition_rows=max_partition_rows,
        max_partition_bytes=max_partition_bytes,
        decode_batch_rows=decode_batch_rows,
        sort_run_rows=sort_run_rows,
        sort_run_bytes=sort_run_bytes,
        write_batch_rows=write_batch_rows,
        merge_fan_in=merge_fan_in,
        horizon_seconds=horizon_seconds,
        checkpoint_owner_rows=checkpoint_owner_rows,
    )
    validate_boundaries(train_before, test_from, horizon_seconds)
    binding = contract(
        source, policy, execution_scope, train_before, test_from, horizon_seconds=horizon_seconds
    )
    if checkpoint_dir is not None:
        from .offline_resume import build_durable_dataset

        return build_durable_dataset(
            source,
            policy,
            output,
            train_before=train_before,
            test_from=test_from,
            execution_scope=execution_scope,
            settings=settings,
            binding=binding,
            checkpoint_dir=checkpoint_dir,
            resume=resume,
            verify=_verify,
            checkpoint_context=checkpoint_context,
        )
    if resume or checkpoint_context is not None:
        raise ValueError("Resume/context requires an explicit checkpoint directory")
    if output.exists() or output.is_symlink():
        raise FileExistsError("Use a new output directory; existing artifacts are preserved")
    with _scratch(output.parent) as work:
        sink, tx_schema, coverage, stats = _partition_source(source, work, settings)
        records, evidence = _shards(
            source,
            policy,
            execution_scope,
            work,
            sink,
            tx_schema,
            coverage,
            train_before,
            test_from,
            settings,
            stats,
        )
        if not records:
            raise ValueError("No R02 alerts; cannot fabricate a training dataset")
        stage = work / "dataset"
        stage.mkdir(exist_ok=False)
        try:
            counts, columns = _write_outputs(
                stage,
                work,
                records,
                evidence,
                tx_schema,
                max_partition_bytes,
                source.id,
                settings=settings,
            )
        except DatasetNotReady as exc:
            write_json(
                stage / "failure.json",
                {
                    "status": "NOT_READY",
                    "reason": str(exc),
                    "contract": binding,
                    "complete_candidates_preserved": True,
                    "training_executed": False,
                    "files": {p.name: sha256(p) for p in stage.iterdir() if p.is_file()},
                },
            )
            publish_directory(stage, output)
            raise
        manifest = _manifest(stage, binding, settings, stats, counts, columns)
        write_json(stage / "manifest.json", manifest)
        publish_directory(stage, output)
        milestone("build_complete")
    if _verify:
        return verify_streaming_dataset(output, source, policy, execution_scope=execution_scope)
    return manifest


def _manifest(output, binding, settings, stats, counts, columns):
    # 물리적 임시 저장·재시도·디코딩 통계는 I/O 설정에 따라 달라 시도별 진단에 둡니다.
    # 여기의 논리 집계는 정확한 원천 재연산 후에도 동일하게 유지됩니다.
    logical = {
        key: stats[key]
        for key in (
            "input_rows",
            "owned_rows",
            "owned_rows_processed",
            "max_bucket_total_rows",
            "max_active_window_rows",
            "max_active_window_workset_bytes",
        )
    }
    manifest = {
        "status": "COMPLETED",
        "layout": LAYOUT,
        "contract": binding,
        "execution_settings": settings,
        "work_stats": logical,
        "prediction_unit": "OFFLINE_ALERT",
        "service_id_mapping_verified": False,
        "model_training_executed": False,
        "rows": counts,
        "feature_columns": columns,
        "files": {n: sha256(output / n) for n in sorted(FILES)},
    }
    manifest["dataset_fingerprint"] = fingerprint(manifest)
    return manifest


def verify_streaming_dataset(
    folder: Path,
    source: SourceInput,
    policy: R02Policy,
    *,
    execution_scope: str,
    replay_workspace: Path | None = None,
    replay_output: Path | None = None,
) -> dict:
    integrity = validate_dataset_integrity(folder, source, policy, execution_scope=execution_scope)
    saved = read_json(folder / "manifest.json")
    declared = saved["dataset_fingerprint"]
    if (
        declared != integrity["dataset_fingerprint"]
        or sha256(folder / "manifest.json") != integrity["manifest_sha256"]
    ):
        raise ValueError("Dataset manifest changed before source replay")
    if saved.get("layout") != LAYOUT:
        raise ValueError("Incomplete or altered streaming dataset manifest")
    binding = saved["contract"]
    first, second = (
        datetime.fromisoformat(binding["train_before"]),
        datetime.fromisoformat(binding["test_from"]),
    )
    if (replay_workspace is None) != (replay_output is None):
        raise ValueError("Independent replay requires both output and workspace")

    def rebuild(destination, checkpoint=None):
        return build_streaming_dataset(
            source,
            policy,
            destination,
            train_before=first,
            test_from=second,
            execution_scope=execution_scope,
            _verify=False,
            checkpoint_dir=checkpoint,
            resume=checkpoint is not None and checkpoint.exists(),
            **saved["execution_settings"],
        )

    if replay_workspace is None:
        with _scratch(folder.parent) as work:
            replay = rebuild(work / "replay")
    else:
        assert replay_output is not None
        replay = rebuild(replay_output, replay_workspace)
    if (
        saved["files"] != replay["files"]
        or saved["rows"] != replay["rows"]
        or saved["feature_columns"] != replay["feature_columns"]
        or saved["work_stats"] != replay["work_stats"]
    ):
        raise ValueError(
            "Split X/y alignment, evidence, features or preprocessing differs from source replay"
        )
    if sha256(folder / "manifest.json") != integrity["manifest_sha256"]:
        raise ValueError("Dataset manifest changed during deep verification")
    if source_fingerprint(source) != binding["source_fingerprint"]:
        raise ValueError("Source manifests changed during deep verification")
    validate_artifact_snapshot(folder, saved, integrity["manifest_sha256"])
    return {
        "status": "VERIFIED",
        "dataset_fingerprint": declared,
        "manifest_sha256": integrity["manifest_sha256"],
        "execution_scope": execution_scope,
        "rows": saved["rows"],
        "service_id_mapping_verified": False,
    }
