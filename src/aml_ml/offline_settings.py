"""공개 API·CLI·manifest·원천 재연산에서 하나의 검증된 I/O 계약을 공유합니다."""

from datetime import datetime, timedelta


def execution_settings(
    *,
    max_source_rows=None,
    partitions=32,
    read_batch_rows=4096,
    decode_batch_rows=None,
    sort_run_rows=65536,
    sort_run_bytes=None,
    write_batch_rows=8192,
    merge_fan_in=8,
    max_partition_rows=100_000,
    max_partition_bytes=64 * 1024**2,
    horizon_seconds=86400,
    checkpoint_owner_rows=100_000,
):
    # read_batch_rows는 지원하는 공개 별칭이며, 정렬 조각의 크기 제한이 아닙니다.
    decode = read_batch_rows if decode_batch_rows is None else decode_batch_rows
    sort_bytes = max_partition_bytes if sort_run_bytes is None else sort_run_bytes
    values = {
        "partitions": (partitions, 1, 128),
        "read_batch_rows": (read_batch_rows, 1, 100_000),
        "decode_batch_rows": (decode, 1, 100_000),
        "sort_run_rows": (sort_run_rows, 1, 1_000_000),
        "sort_run_bytes": (sort_bytes, 1, 1024**3),
        "write_batch_rows": (write_batch_rows, 1, 100_000),
        "merge_fan_in": (merge_fan_in, 2, 8),
        "max_partition_rows": (max_partition_rows, 256, 1_000_000),
        "max_partition_bytes": (max_partition_bytes, 1, 1024**3),
        "horizon_seconds": (horizon_seconds, 86400, 365 * 86400),
        "checkpoint_owner_rows": (checkpoint_owner_rows, 1, 10_000_000),
    }
    for name, (value, lower, upper) in values.items():
        if type(value) is not int or not lower <= value <= upper:
            raise ValueError(f"Invalid offline setting/budget: {name}")
    if sort_bytes > max_partition_bytes:
        raise ValueError("Sort-run bytes must not exceed the declared workset budget")
    if max_source_rows is not None and (type(max_source_rows) is not int or max_source_rows < 1):
        raise ValueError("Invalid full-source row budget; no truncation")
    return {"max_source_rows": max_source_rows, **{key: item[0] for key, item in values.items()}}


def validate_boundaries(first: datetime, second: datetime, horizon_seconds: int):
    if type(horizon_seconds) is not int or not 86400 <= horizon_seconds <= 365 * 86400:
        raise ValueError("Explicit eligibility horizon must cover the actual R02 24h history")
    if first.tzinfo is not None or second.tzinfo is not None or first >= second:
        raise ValueError("Ordered naive source-time split boundaries are required")
    # 출력 생성 전에 두 경계의 날짜 범위 초과를 확인합니다. 짧은 검증 구간은
    # 허용하며, 적격 표본이 비는지는 이후 ledger에서 확인합니다.
    _ = first + timedelta(seconds=horizon_seconds)
    _ = second + timedelta(seconds=horizon_seconds)


def reader_budget(max_bytes: int, fan_in: int) -> int:
    """병합 입력 리더들, 출력 하나와 힙 예약분에 논리 작업 공간을 나눕니다.

    Arrow·Python·코덱·네이티브 버퍼와 구간·정렬 상태가 동시에 존재할 수 있습니다.
    이 논리 작업 공간 분배는 프로세스 RSS 상한을 뜻하지 않습니다.
    """
    return max(1, max_bytes // (fan_in + 2))
