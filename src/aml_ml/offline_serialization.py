"""조립 코드와 독립적으로 CSV·JSON 배열·Parquet의 타입과 크기 제한을 맞춥니다."""

import csv
import hashlib
import json
import math
from contextlib import AbstractContextManager, ExitStack
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Protocol

import pyarrow as pa
import pyarrow.parquet as pq

from .offline_io import close_all, python_decode_bytes
from .offline_io import rows as parquet_rows
from .offline_measurements import event, temporary_closed
from .offline_progress import progress
from .storage import read_json, write_json

CODEC = "aml-typed-three-formats-v1"
TABLES = ("alerts", "targets", "evidence", "raw", "ledger") + tuple(
    f"{prefix}_{split}" for split in ("train", "validation", "test") for prefix in ("X", "y")
)
FORMAT_FILES = {f"{name}.{suffix}" for name in TABLES for suffix in ("csv", "json", "parquet")}
FORMAT_FILES |= {f"{name}.schema.arrow" for name in TABLES} | {"formats.json"}
ALERT_SCHEMA = pa.schema(
    [
        (name, pa.string())
        for name in ("source_id", "rule_id", "rule_version", "rule_config_hash", "account_key")
    ]
    + [(name, pa.timestamp("us")) for name in ("reference_ts", "window_start", "window_end")]
    + [("transaction_uids", pa.list_(pa.string())), ("paths", pa.list_(pa.list_(pa.string())))]
)
TARGET_SCHEMA = pa.schema(
    [
        ("training_alert_uid", pa.string()),
        ("source_id", pa.string()),
        ("reference_ts", pa.timestamp("us")),
        ("y_alert_proxy", pa.int8()),
        ("target_status", pa.string()),
        ("target_definition", pa.string()),
        ("evidence_count", pa.int64()),
        ("missing_label_count", pa.int64()),
        ("id_namespace", pa.string()),
        ("split", pa.string()),
    ]
)
LEDGER_SCHEMA = pa.schema(
    [
        ("training_alert_uid", pa.string()),
        ("source_id", pa.string()),
        ("reference_ts", pa.timestamp("us")),
        ("original_split", pa.string()),
        ("time_eligible", pa.bool_()),
        ("target_complete", pa.bool_()),
        ("supervised_eligible", pa.bool_()),
        ("exclusion_reasons", pa.list_(pa.string())),
        ("y_alert_proxy", pa.int8()),
        ("evidence_uids", pa.list_(pa.string())),
        ("horizon_seconds", pa.int64()),
        ("train_before", pa.timestamp("us")),
        ("test_from", pa.timestamp("us")),
        ("eligibility_version", pa.string()),
    ]
)


def _json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def wire_value(value, field):
    if value is None:
        if not field.nullable:
            raise ValueError("Null in a non-nullable serialized field")
        return None
    kind = field.type
    if pa.types.is_list(kind) or pa.types.is_large_list(kind):
        return [wire_value(item, kind.value_field) for item in value]
    if pa.types.is_decimal(kind):
        if not isinstance(value, Decimal) or not value.is_finite():
            raise ValueError("Decimal serialization requires an exact finite Decimal")
        return format(value, "f")
    if pa.types.is_timestamp(kind):
        if not isinstance(value, datetime) or value.tzinfo is not None or kind.tz is not None:
            raise ValueError("Preserve naive source timestamps")
        if kind.unit not in ("s", "ms", "us") or value.microsecond % {
            "s": 1_000_000,
            "ms": 1000,
            "us": 1,
        }.get(kind.unit, 1):
            raise ValueError("Timestamp cannot be represented losslessly at declared unit")
        return value.isoformat()
    if pa.types.is_boolean(kind):
        if type(value) is not bool:
            raise ValueError("Boolean serialization type mismatch")
    elif pa.types.is_integer(kind):
        if type(value) is not int:
            raise ValueError("Integer serialization type mismatch")
    elif pa.types.is_floating(kind):
        if type(value) is not float or not math.isfinite(value):
            raise ValueError("Only finite floating point values may be serialized")
    elif pa.types.is_string(kind) or pa.types.is_large_string(kind):
        if type(value) is not str:
            raise ValueError("String serialization type mismatch")
    elif not pa.types.is_null(kind):
        raise ValueError("Unsupported offline serialization type")
    return value


def typed_value(value, field):
    if value is None:
        return wire_value(value, field)
    kind = field.type
    if pa.types.is_list(kind) or pa.types.is_large_list(kind):
        if type(value) is not list:
            raise ValueError("Nested arrays must remain JSON arrays")
        return [typed_value(item, kind.value_field) for item in value]
    if pa.types.is_decimal(kind):
        if type(value) is not str:
            raise ValueError("Decimal must use an exact string, not JSON float")
        result = Decimal(value)
    elif pa.types.is_timestamp(kind):
        if type(value) is not str:
            raise ValueError("Timestamp must use an ISO string")
        result = datetime.fromisoformat(value)
        if result.isoformat() != value:
            raise ValueError("Noncanonical or over-precision timestamp; no truncation")
    elif pa.types.is_floating(kind):
        if type(value) not in (int, float):
            raise ValueError("Floating point wire type mismatch")
        result = float(value)
    else:
        result = value
    wire_value(result, field)  # 타입, null 허용, 유한값, timezone 없는 시각을 확인합니다.
    return result


def typed_row(row, schema):
    if list(row) != schema.names:
        raise ValueError("Serialized column names/order differ from schema")
    return {field.name: typed_value(row[field.name], field) for field in schema}


def record_table(rows, schema):
    """R02 기록의 ISO 시각 문자열을 타입이 있는 Arrow 행으로 옮깁니다."""

    def native_value(value, field):
        if value is not None and (
            pa.types.is_list(field.type) or pa.types.is_large_list(field.type)
        ):
            if not isinstance(value, (list, tuple)):
                raise ValueError("Native record evidence must be an array/tuple")
            return [native_value(item, field.type.value_field) for item in value]
        return value

    normalized = [
        {field.name: native_value(row[field.name], field) for field in schema} for row in rows
    ]
    return pa.Table.from_pylist([typed_row(row, schema) for row in normalized], schema=schema)


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field")
        result[key] = value
    return result


def _nonstandard_constant(value):
    raise ValueError("Nonstandard JSON numeric constant: " + value)


def iter_json_array(path: Path, *, max_record_bytes=256 * 1024**2):
    """객체 배열을 나누어 읽고 뒷부분 잔여 내용·잘못된 쉼표·미완료 기록을 거부합니다."""
    decoder = json.JSONDecoder(object_pairs_hook=_object, parse_constant=_nonstandard_constant)
    with path.open(encoding="utf-8", newline="") as stream:
        buffer = ""
        eof = False

        def fill():
            nonlocal buffer, eof
            chunk = stream.read(65536)
            eof = not chunk
            buffer += chunk
            if len(buffer.encode("utf-8")) > max_record_bytes + 65536 * 4:
                raise ValueError("JSON record exceeds bounded readback budget")

        def whitespace():
            nonlocal buffer
            buffer = buffer.lstrip()
            while not buffer and not eof:
                fill()
                buffer = buffer.lstrip()

        whitespace()
        if not buffer.startswith("["):
            raise ValueError("Expected a complete JSON array, never an object or JSONL")
        buffer = buffer[1:]
        first = True
        while True:
            whitespace()
            if buffer.startswith("]"):
                buffer = buffer[1:]
                break
            if not first:
                if not buffer.startswith(","):
                    raise ValueError("Expected JSON array separator")
                buffer = buffer[1:]
                whitespace()
            while True:
                try:
                    row, end = decoder.raw_decode(buffer)
                    break
                except json.JSONDecodeError as exc:
                    if eof:
                        raise ValueError("Incomplete or invalid JSON array") from exc
                    fill()
            if not isinstance(row, dict) or len(buffer[:end].encode("utf-8")) > max_record_bytes:
                raise ValueError("Expected one bounded JSON object per array row")
            buffer = buffer[end:]
            first = False
            yield row
        whitespace()
        if buffer or not eof:
            raise ValueError("Unexpected content after JSON array")


def read_record_array(path: Path):
    """작은 기준·테스트용입니다. 큰 비교는 배열을 순회하며 읽습니다."""
    return list(iter_json_array(path))


def _csv_cell(value):
    if value is None:
        return "\\N"
    if type(value) is bool:
        return "true" if value else "false"
    if isinstance(value, list):
        return _json(value)
    text = str(value)
    return "\\" + text if text.startswith("\\") else text


def _csv_value(cell, field):
    if cell == "\\N":
        return typed_value(None, field)
    if cell.startswith("\\\\"):
        cell = cell[1:]
    elif cell.startswith("\\"):
        raise ValueError("Unescaped leading backslash in CSV")
    kind = field.type
    if pa.types.is_list(kind) or pa.types.is_large_list(kind):
        value = json.loads(cell, object_pairs_hook=_object, parse_constant=_nonstandard_constant)
    elif pa.types.is_boolean(kind):
        if cell not in ("true", "false"):
            raise ValueError("Invalid CSV boolean")
        value = cell == "true"
    elif pa.types.is_integer(kind):
        value = int(cell)
        if str(value) != cell:
            raise ValueError("Noncanonical CSV integer")
    elif pa.types.is_floating(kind):
        value = float(cell)
    else:
        value = cell
    return typed_value(value, field)


def iter_csv(path, schema, *, max_record_bytes):
    previous = csv.field_size_limit()
    csv.field_size_limit(max_record_bytes)
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream, strict=True)
            if next(reader, None) != schema.names:
                raise ValueError("CSV header/order differs from schema")
            for cells in reader:
                if len(cells) != len(schema):
                    raise ValueError("CSV column count mismatch")
                if sum(len(cell.encode("utf-8")) for cell in cells) > max_record_bytes:
                    raise ValueError("CSV record exceeds readback budget")
                yield {field.name: _csv_value(cell, field) for field, cell in zip(schema, cells)}
    finally:
        csv.field_size_limit(previous)


class TableWriter(Protocol):
    """최종 출력에서 공통으로 사용하는 테이블 쓰기 동작입니다."""

    def write_table(self, table: pa.Table) -> None: ...


class TableWriterFactory(Protocol):
    """경로·스키마·예산을 받아 종료 처리가 있는 writer를 만듭니다."""

    def __call__(
        self, path: Path, schema: pa.Schema, /, *, max_bytes: int
    ) -> AbstractContextManager[TableWriter]: ...


class ThreeWriter:
    """세 형식의 완결 파일을 함께 쓰고 다시 읽어 확인한 뒤 게시합니다."""

    def __init__(self, parquet_path, schema, *, max_bytes=256 * 1024**2):
        self.path, self.schema, self.max_bytes = parquet_path, schema, max_bytes
        self.stack = ExitStack()
        self.rows = 0

    def __enter__(self):
        outputs = [
            self.path.with_suffix(suffix)
            for suffix in (".parquet", ".json", ".csv", ".schema.arrow")
        ]
        if any(path.exists() or path.is_symlink() for path in outputs):
            raise FileExistsError("Three-format output exists; preserve it and use a new stage")
        try:
            parquet_stream = self.stack.enter_context(self.path.open("xb"))
            self.parquet = self.stack.enter_context(pq.ParquetWriter(parquet_stream, self.schema))
            self.json = self.stack.enter_context(
                self.path.with_suffix(".json").open("x", encoding="utf-8", newline="\n")
            )
            self.csv = csv.writer(
                self.stack.enter_context(
                    self.path.with_suffix(".csv").open("x", encoding="utf-8", newline="")
                ),
                lineterminator="\n",
            )
            with self.path.with_suffix(".schema.arrow").open("xb") as schema_stream:
                schema_stream.write(self.schema.serialize().to_pybytes())
            self.json.write("[")
            self.csv.writerow(self.schema.names)
            return self
        except BaseException:
            close_all([self.stack])
            raise

    def write_table(self, table):
        if table.schema != self.schema or table.nbytes > self.max_bytes:
            raise ValueError("Three-format table schema/byte budget mismatch")
        for batch in table.to_batches(max_chunksize=256):
            start = 0
            while start < len(batch):
                chunk = batch.slice(start)
                estimate = python_decode_bytes(chunk)
                while estimate > self.max_bytes and len(chunk) > 1:
                    chunk = chunk.slice(0, max(1, len(chunk) // 2))
                    estimate = python_decode_bytes(chunk)
                if estimate > self.max_bytes:
                    raise ValueError("Three-format Python row workset estimate exceeds budget")
                decoded = chunk.to_pylist()
                for row in decoded:
                    wire = {field.name: wire_value(row[field.name], field) for field in self.schema}
                    encoded = _json(wire)
                    if len(encoded.encode("utf-8")) > self.max_bytes:
                        raise ValueError("Three-format record exceeds serialized byte budget")
                    self.json.write(("," if self.rows else "") + encoded)
                    self.csv.writerow([_csv_cell(wire[name]) for name in self.schema.names])
                    self.rows += 1
                start += len(chunk)
                del row, wire, encoded, decoded, chunk
            self.parquet.write_table(pa.Table.from_batches([batch], schema=self.schema))

    def __exit__(self, kind, value, traceback):
        error = value
        try:
            if kind is None:
                self.json.write("]")
        except BaseException as exc:
            error = exc
        try:
            self.stack.close()
        except BaseException as exc:
            if error is None:
                error = exc
            else:
                error.add_note("Three-format writer close also failed: " + type(exc).__name__)
        if error is not None and value is None:
            raise error
        if error is None:
            for suffix in (".parquet", ".json", ".csv", ".schema.arrow"):
                path = self.path.with_suffix(suffix)
                event(f"{suffix[1:]}_closed_write_bytes", path.stat().st_size)
                temporary_closed(path)


def write_three(table, path):
    with ThreeWriter(path, table.schema) as writer:
        writer.write_table(table)


def verify_table(folder, name, *, max_bytes=256 * 1024**2):
    schema = pa.ipc.read_schema(pa.BufferReader((folder / f"{name}.schema.arrow").read_bytes()))
    digest = hashlib.sha256(schema.serialize().to_pybytes())
    csv_rows = iter_csv(folder / f"{name}.csv", schema, max_record_bytes=max_bytes)
    json_rows = iter_json_array(folder / f"{name}.json", max_record_bytes=max_bytes)
    missing = object()
    count = 0
    total = pq.read_metadata(folder / f"{name}.parquet").num_rows
    stage = "format_readback/" + name
    progress(stage, processed=0, total=total, unit="table_rows")
    parquet = parquet_rows(folder / f"{name}.parquet", 1024, max_bytes)
    try:
        if pq.read_schema(folder / f"{name}.parquet") != schema:
            raise ValueError("Parquet schema differs from declared serialization schema")
        for expected in parquet:
            from_csv, from_json = next(csv_rows, missing), next(json_rows, missing)
            if not isinstance(from_csv, dict) or not isinstance(from_json, dict):
                raise ValueError("Three-format row count differs")
            json_typed = typed_row(from_json, schema)
            wire = {field.name: wire_value(expected[field.name], field) for field in schema}
            canonical = _json(wire)
            for actual in (from_csv, json_typed):
                restored = {field.name: wire_value(actual[field.name], field) for field in schema}
                # Decimal 자릿수와 부호 있는 0까지 타입이 있는 값으로 비교합니다.
                # Parquet 쪽은 앞에서 정확한 Arrow schema와 일치하는지 확인했습니다.
                if _json(restored) != canonical:
                    raise ValueError("Three-format values/nulls/order/types differ")
            digest.update((canonical + "\n").encode("utf-8"))
            count += 1
            if count % 10000 == 0:
                progress(stage, processed=count, total=total, unit="table_rows")
        if next(csv_rows, missing) is not missing or next(json_rows, missing) is not missing:
            raise ValueError("Extra rows in text representation")
    finally:
        close_all([csv_rows, json_rows, parquet])
    progress(stage, processed=count, total=total, unit="table_rows")
    return {"rows": count, "logical_sha256": digest.hexdigest()}


def finish_formats(folder, *, max_bytes=256 * 1024**2):
    progress("format_readback", unit="tables", total=len(TABLES))
    tables = {}
    for name in TABLES:
        tables[name] = verify_table(folder, name, max_bytes=max_bytes)
    result = {
        "codec": CODEC,
        "csv": "UTF-8 RFC4180; NULL=\\N; leading backslash escaped; nested typed JSON cells",
        "json": "one complete array; exact decimal/naive timestamp strings; typed nested arrays",
        "tables": tables,
    }
    write_json(folder / "formats.json", result)
    return result


def verify_formats(folder, *, max_bytes=256 * 1024**2):
    saved = read_json(folder / "formats.json")
    if saved.get("codec") != CODEC or set(saved["tables"]) != set(TABLES):
        raise ValueError("Incompatible three-format contract")
    for name in TABLES:
        if verify_table(folder, name, max_bytes=max_bytes) != saved["tables"][name]:
            raise ValueError("Three-format logical receipt mismatch")
