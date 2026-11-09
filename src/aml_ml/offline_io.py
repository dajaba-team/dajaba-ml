"""정렬·최종 적합 배치 크기와 독립적으로 Parquet 디코딩 크기를 제한합니다."""

import sys

import pyarrow as pa
import pyarrow.parquet as pq

from .offline_account_stream import check_budget
from .offline_measurements import event, observe
from .offline_progress import check_cancelled

DEFAULT_DECODE_ROWS = 1024
DEFAULT_WRITE_ROWS = 8192


def close_all(streams):
    """모든 스트림을 닫되 정리 실패가 원래 예외를 덮어쓰지 않게 합니다."""
    primary = sys.exception()
    errors = []
    for stream in streams:
        try:
            stream.close()
        except BaseException as exc:
            errors.append(exc)
    if errors:
        if primary is not None:
            primary.add_note("Reader cleanup also failed: " + type(errors[0]).__name__)
        else:
            raise errors[0]


def _supported(kind):
    if pa.types.is_list(kind) or pa.types.is_large_list(kind):
        return _supported(kind.value_type)
    return any(
        check(kind)
        for check in (
            pa.types.is_string,
            pa.types.is_large_string,
            pa.types.is_binary,
            pa.types.is_large_binary,
            pa.types.is_boolean,
            pa.types.is_integer,
            pa.types.is_floating,
            pa.types.is_decimal,
            pa.types.is_timestamp,
            pa.types.is_null,
        )
    )


def row_workset_bytes(row) -> int:
    def size(value) -> int:
        own = sys.getsizeof(value)
        if isinstance(value, dict):
            return own + sum(size(key) + size(item) for key, item in value.items())
        if isinstance(value, (list, tuple)):
            return own + sum(size(item) for item in value)
        return own

    return 512 + size(row)


def python_decode_bytes(batch):
    """배치 전체를 변환하지 않고 to_pylist 전에 중첩 Python 컨테이너 크기를 추정합니다.

    하위 요소의 부가 공간을 포함하여 Arrow 버퍼·오프셋으로 리스트를 순회합니다.
    값을 확인하려면 단일 Arrow 값은 이미 존재해야 하므로, 이 추정으로 네이티브
    디코더나 RSS 안전성을 보장할 수 없습니다. 지원하지 않는 타입은 거부합니다.
    """

    def column_size(array):
        if isinstance(array, pa.ChunkedArray):
            return sum(column_size(chunk) for chunk in array.chunks)
        kind = array.type
        if pa.types.is_list(kind) or pa.types.is_large_list(kind):
            first, last = array.offsets[0].as_py(), array.offsets[-1].as_py()
            child = array.values.slice(first, last - first)
            return len(array) * 72 + (last - first) * 8 + column_size(child)
        if not _supported(kind):
            raise ValueError("Unsupported Python decode type")
        return 4 * array.nbytes + len(array) * 128

    return len(batch) * (2048 + 256 * len(batch.schema)) + sum(
        column_size(batch.column(index)) for index in range(len(batch.schema))
    )


def _decode_rows(parquet, group_index, requested, max_bytes):
    if max_bytes is None:
        return requested
    schema = parquet.schema_arrow
    group = parquet.metadata.row_group(group_index)
    # 평면 문자열 열에서는 사전을 포함한 비압축 인코딩 청크 전체 크기를
    # 값 하나의 보수적인 상한으로 씁니다. 평균·표본 길이는 사용하지 않습니다.
    # 중첩 배열은 확장 방식이 달라 단일 행 처리 경로를 유지합니다.
    per_row = 2048 + len(schema) * 256
    # 중첩 Arrow 필드는 여러 Parquet 말단 열을 차지합니다. 뒤의 문자열이
    # 잘못된 열을 쓰지 않도록 말단 메타데이터에 접근하기 전에 반환합니다.
    if any(pa.types.is_list(f.type) or pa.types.is_large_list(f.type) for f in schema):
        return 1
    for index, field in enumerate(schema):
        kind = field.type
        if (
            pa.types.is_string(kind)
            or pa.types.is_large_string(kind)
            or pa.types.is_binary(kind)
            or pa.types.is_large_binary(kind)
        ):
            per_row += 4 * max(0, group.column(index).total_uncompressed_size) + 8
        else:
            per_row += 64
    return max(1, min(requested, max_bytes // max(1, per_row)))


def rows(path, batch_rows, max_bytes=None, *, start_row=0):
    """논리 행 오프셋을 읽되 건너뛸 완전한 행 그룹은 디코딩하지 않습니다.

    인코딩·Arrow·Python 추정치별 제한은 RSS 보장이 아닙니다. 네이티브 디코더의
    할당이나 더 나눌 수 없는 단일 값은 여전히 추정치를 초과할 수 있습니다.
    """
    if type(batch_rows) is not int or batch_rows < 1 or type(start_row) is not int:
        raise ValueError("Invalid decoder row batch/cursor")
    with pq.ParquetFile(path) as parquet:
        if any(not _supported(field.type) for field in parquet.schema_arrow):
            raise ValueError("Unsupported decoder schema (dictionary/extension/struct/map)")
        if not 0 <= start_row <= parquet.metadata.num_rows:
            raise ValueError("Decoder cursor outside complete file")
        skipped = 0
        for group_index in range(parquet.metadata.num_row_groups):
            check_cancelled()
            group = parquet.metadata.row_group(group_index)
            if skipped + group.num_rows <= start_row:
                skipped += group.num_rows
                continue
            context = {"stage": "read_preflight", "row_group": group_index}
            if max_bytes is not None:
                check_budget(
                    "encoded_row_group_bytes", 0, group.total_byte_size, max_bytes, context
                )
            size = _decode_rows(parquet, group_index, batch_rows, max_bytes)
            observe("requested_decode_rows", batch_rows)
            observe("selected_decode_rows", size)
            event(
                "parquet_encoded_group_bytes",
                sum(
                    group.column(index).total_compressed_size for index in range(group.num_columns)
                ),
            )
            for batch in parquet.iter_batches(batch_size=size, row_groups=[group_index]):
                check_cancelled()
                observe("returned_arrow_batch_rows", len(batch))
                event("arrow_decoded_bytes", batch.nbytes)
                batch_start = skipped
                skipped += len(batch)
                offset = max(0, start_row - batch_start)
                if offset >= len(batch):
                    continue
                batch = batch.slice(offset)
                if max_bytes is not None:
                    check_budget("decoded_batch_bytes", 0, batch.nbytes, max_bytes, context)
                index = 0
                while index < len(batch):
                    chunk = batch.slice(index)
                    estimate = python_decode_bytes(chunk)
                    while max_bytes is not None and estimate > max_bytes and len(chunk) > 1:
                        chunk = chunk.slice(0, max(1, len(chunk) // 2))
                        estimate = python_decode_bytes(chunk)
                    if max_bytes is not None:
                        check_budget(
                            "python_decode_estimate_bytes", 0, estimate, max_bytes, context
                        )
                    decoded = chunk.to_pylist()
                    observe("returned_python_chunk_rows", len(chunk))
                    for row in decoded:
                        if max_bytes is not None:
                            check_budget(
                                "decoded_row_workset_bytes",
                                0,
                                row_workset_bytes(row),
                                max_bytes,
                                context,
                            )
                        yield row
                    index += len(chunk)
                    del row, decoded, chunk
