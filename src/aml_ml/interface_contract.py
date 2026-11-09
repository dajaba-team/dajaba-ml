"""확정된 인터페이스 기본 요소이며, 점수 생성이나 미정 룰 정책은 포함하지 않습니다.

준혁님의 2026-11-02 피드백과 공통 규약 결정 16에 근거합니다.
HTTP 메시지 형식, 영속 저장과 재시도 실행은 AI·백엔드 계층의 책임입니다.
"""

from datetime import datetime, timedelta, timezone
from enum import StrEnum


class FailureCode(StrEnum):
    TRANSACTION_NOT_FOUND = "TRANSACTION_NOT_FOUND"
    HISTORY_NOT_SUPPORTED = "HISTORY_NOT_SUPPORTED"
    REQUIRED_INPUT_MISSING = "REQUIRED_INPUT_MISSING"
    COMPUTE_ERROR = "COMPUTE_ERROR"


def is_score_retryable(score_status: str, failure_code: str | None) -> bool:
    """FAILED + NULL/COMPUTE_ERROR만 실패 항목 재시도의 대상입니다.

    NULL은 일괄 요청의 503처럼 항목 응답이 없는 실패를 포함합니다.
    PENDING 작업 선택이나 재시도 실행은 하지 않습니다. 여러 실패의
    우선순위는 팀의 결정 사항이며, 이 함수에서 코드를 선택하지 않습니다.
    """
    code = FailureCode(failure_code) if failure_code is not None else None
    return score_status == "FAILED" and code in (None, FailureCode.COMPUTE_ERROR)


def format_system_timestamp(value: datetime) -> str:
    """시간대가 있는 시스템 시각을 UTC Z로 표현하며, 원천 시각을 재해석하지 않습니다."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("System timestamps require an explicit timezone")
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_system_timestamp(value: str) -> datetime:
    """created_at/produced_at의 확정된 UTC Z 표현을 검증합니다."""
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("System timestamps must use UTC Z")
    parsed = datetime.fromisoformat(value)
    if parsed.utcoffset() != timedelta(0):
        raise ValueError("System timestamps must use UTC Z")
    return parsed
