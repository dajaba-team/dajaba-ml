"""명시적으로 교체 가능한 R02 정책과 라벨을 보지 않는 오프라인 후보 생성을 제공합니다.

미정값에는 운영 기본값을 두지 않습니다. FIXTURE_ONLY 산출물은 확정 정책의
실행 조건을 통과할 수 없습니다. 원천 UID를 서비스 거래 ID로 바꾸지 않습니다.
"""

import hashlib
import json
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .alert_dataset import OfflineAlert
from .schema import epoch_us
from .sliding_window import MinuteWindow, evaluate_minute

RULE_ID = "R02_FAN_IN_24H"
OFFLINE_VERSION_PREFIX = "mldb-offline-r02-"


def _json_bytes(value: Any) -> int:
    """이스케이프 문자열·리스트를 할당하지 않고 기본 ensure_ascii JSON 크기를 구합니다."""
    if isinstance(value, str):
        size = 2
        for character in value:
            code = ord(character)
            size += (
                2
                if character in '\\"\b\f\n\r\t'
                else 6
                if code < 32 or 127 <= code <= 0xFFFF
                else 12
                if code > 0xFFFF
                else 1
            )
        return size
    if isinstance(value, dict):
        return (
            2
            + max(0, len(value) - 1) * 2
            + sum(_json_bytes(key) + 2 + _json_bytes(item) for key, item in value.items())
        )
    if isinstance(value, (list, tuple)):
        return 2 + max(0, len(value) - 1) * 2 + sum(_json_bytes(item) for item in value)
    return len(json.dumps(value, allow_nan=False))


def fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class R02Policy:
    rule_version: str | None = None
    policy_version: str | None = None
    min_counterparties: int | None = None
    window_policy: str | None = None
    recurrence_policy: str | None = None
    evidence_policy: str | None = None
    counterparty_policy: str | None = None
    self_transfer_policy: str | None = None
    decision_status: str = "UNCONFIRMED"
    confirmation_reference: str | None = None
    rule_version_origin: str | None = None

    def diagnostics(self, execution_scope: str) -> list[dict[str, str]]:
        """실행 조건과 같은 규칙으로 정책의 모든 차단 사유를 구합니다."""
        issues: list[dict[str, str]] = []

        def add(field: str, code: str, message: str) -> None:
            issues.append({"field": field, "code": code, "message": message})

        if execution_scope not in {"FIXTURE_ONLY", "CONFIRMED"}:
            add("execution_scope", "UNSUPPORTED_SCOPE", "Use FIXTURE_ONLY or CONFIRMED")
        if self.decision_status != execution_scope:
            add(
                "decision_status",
                "SCOPE_MISMATCH",
                "Pending/fixture policy cannot execute as a confirmed policy",
            )
        origin_required = execution_scope == "CONFIRMED" or self.rule_version_origin is not None
        if origin_required and self.rule_version_origin not in ("OFFLINE_SPEC", "BACKEND_EXPORT"):
            add("rule_version_origin", "REQUIRED_ORIGIN", "Declare OFFLINE_SPEC or BACKEND_EXPORT")
        if isinstance(self.rule_version, str):
            internal = self.rule_version.startswith(OFFLINE_VERSION_PREFIX)
            if self.rule_version_origin == "OFFLINE_SPEC" and not internal:
                add("rule_version", "OFFLINE_NAMESPACE", f"Use {OFFLINE_VERSION_PREFIX}<version>")
            if self.rule_version_origin == "BACKEND_EXPORT" and internal:
                add(
                    "rule_version",
                    "BACKEND_NAMESPACE",
                    "An internal version is not a backend version",
                )
        for field in ("rule_version", "policy_version", "confirmation_reference"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                add(
                    field,
                    "REQUIRED_STRING",
                    "Explicit rule/policy versions and confirmation reference required",
                )
        if type(self.min_counterparties) is not int or self.min_counterparties < 1:
            add(
                "min_counterparties",
                "REQUIRED_POSITIVE_INTEGER",
                "Confirmed minimum counterparty count required; no default",
            )
        supported = {
            "window_policy": "B7_E_PLUS_ONE_MINUTE",
            "recurrence_policy": "PREVIOUS_CLOCK_MINUTE_FALSE_TO_TRUE",
            "evidence_policy": "ALL_INCOMING_IN_WINDOW",
            "counterparty_policy": "EXACT_ACCOUNT_KEYS_ACROSS_CURRENCIES",
            "self_transfer_policy": "EXCLUDE_FLAGGED_AND_SAME_ACCOUNT",
        }
        for key, value in supported.items():
            if getattr(self, key) != value:
                add(key, "UNSUPPORTED_SEMANTICS", f"Required R02 semantics: {value}")
        return issues

    def readiness(self, execution_scope: str) -> dict:
        issues = self.diagnostics(execution_scope)
        return {
            "status": "POLICY_CONFIG_NOT_READY" if issues else "POLICY_CONFIG_READY",
            "rule_id": RULE_ID,
            "execution_scope": execution_scope,
            "issues": issues,
            "policy": None if issues else asdict(self),
            "config_hash": None if issues else self.config_hash,
            "not_verified": [
                "POLICY_PROVENANCE_AUTHENTICITY",
                "SOURCE_DATA_READINESS",
                "BACKEND_POLICY_ACTUALLY_APPLIED",
                "BACKEND_SAMPLE_AND_FULL_RULE_RESULTS",
            ],
        }

    def require(self, execution_scope: str) -> None:
        issues = self.diagnostics(execution_scope)
        if issues:
            raise ValueError("; ".join(f"{item['field']}: {item['message']}" for item in issues))

    def require_backend(self) -> None:
        self.require("CONFIRMED")
        if self.rule_version_origin != "BACKEND_EXPORT":
            raise ValueError(
                "Backend comparison requires BACKEND_EXPORT, not an offline spec version"
            )

    @property
    def config_hash(self) -> str:
        return fingerprint(asdict(self))


def load_policy(path: Path) -> tuple[R02Policy, str]:
    """같은 바이트를 파싱·해시하며, 파일 참조만으로 합의를 입증하지는 않습니다."""
    content = path.read_bytes()
    payload = json.loads(content.decode("utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("Policy file must contain a JSON object")
    return R02Policy(**payload), hashlib.sha256(content).hexdigest()


def validate_transactions(
    rows: list[dict], source_id: str, source_release: str, *, complete_source: bool = True
) -> None:
    """생성 전에 제공된 원천 전체를 확인하며, 정답 열은 사용하지 않습니다."""
    if not source_release.startswith(source_id + ":"):
        raise ValueError("Source release does not match source identity")
    seen = set()
    for row in rows:
        n = row["source_row_id"]
        if type(n) is not int or n < 0 or row["transaction_uid"] != f"{source_release}:{n}":
            raise ValueError("Original release/row/UID mismatch")
        if n in seen:
            raise ValueError("Duplicate original source row")
        seen.add(n)
        epoch_us(row["ts"])
        if type(row["is_self_transfer"]) is not bool:
            raise ValueError("Self-transfer flag must be a boolean")
        for key in ("from_account_key", "to_account_key"):
            if not isinstance(row[key], str) or not row[key]:
                raise ValueError("Exact nonempty account keys required")
    if complete_source and seen != set(range(len(rows))):
        raise ValueError("A complete contiguous original source is required")


def generate_r02(
    rows: list[dict],
    *,
    source_id: str,
    source_release: str,
    policy: R02Policy,
    execution_scope: str,
) -> list[OfflineAlert]:
    policy.require(execution_scope)
    validate_transactions(rows, source_id, source_release)
    return r02_partition_candidates(
        rows,
        source_id=source_id,
        source_release=source_release,
        policy=policy,
        execution_scope=execution_scope,
    )


def r02_partition_candidates(
    rows: list[dict],
    *,
    source_id: str,
    source_release: str,
    policy: R02Policy,
    execution_scope: str,
    owned_accounts: set[str] | None = None,
    max_evidence_references: int | None = None,
    max_payload_bytes: int | None = None,
) -> list[OfflineAlert]:
    """공통 계산부이며, 호출자는 담당 계좌의 완전한 이력을 제공해야 합니다."""
    policy.require(execution_scope)
    for limit in (max_evidence_references, max_payload_bytes):
        if limit is not None and (type(limit) is not int or limit < 1):
            raise ValueError("Positive evidence/payload workset budget required")
    if max_payload_bytes is not None and _json_bytes(asdict(policy)) > max_payload_bytes:
        raise ValueError("Rule configuration exceeds payload workset byte budget")
    config_hash = policy.config_hash
    validate_transactions(rows, source_id, source_release, complete_source=False)
    incoming: dict[str, list[dict]] = defaultdict(list)
    activity: dict[str, set[datetime]] = defaultdict(set)
    for row in rows:
        if row["is_self_transfer"] or row["from_account_key"] == row["to_account_key"]:
            continue
        minute = row["ts"].replace(second=0, microsecond=0)
        activity[row["from_account_key"]].add(minute)
        activity[row["to_account_key"]].add(minute)
        incoming[row["to_account_key"]].append(row)
    alerts = []
    evidence_references = 0
    payload_bytes = 0
    for account in sorted(incoming):
        if owned_accounts is not None and account not in owned_accounts:
            continue
        events = sorted(incoming[account], key=lambda r: (r["ts"], r["source_row_id"]))
        # evaluate_minute는 직전·현재 시계 구간을 단조 증가 순서로 요청합니다.
        active: deque[dict] = deque()
        peers: Counter = Counter()
        cursor = 0
        active_uid_bytes = 0

        def condition(window: MinuteWindow) -> bool:
            nonlocal cursor, active_uid_bytes
            while cursor < len(events) and events[cursor]["ts"] < window.end:
                item = events[cursor]
                active.append(item)
                if max_payload_bytes is not None:
                    active_uid_bytes += _json_bytes(item["transaction_uid"])
                peers[item["from_account_key"]] += 1
                cursor += 1
            while active and active[0]["ts"] < window.start:
                expired = active.popleft()
                if max_payload_bytes is not None:
                    active_uid_bytes -= _json_bytes(expired["transaction_uid"])
                key = expired["from_account_key"]
                peers[key] -= 1
                if not peers[key]:
                    del peers[key]
            return len(peers) >= policy.min_counterparties  # type: ignore[operator]

        for minute in sorted(activity[account]):
            decision = evaluate_minute(minute, has_account_transactions=True, condition=condition)
            if decision.emit:
                evidence_references += len(active)
                if (
                    max_evidence_references is not None
                    and evidence_references > max_evidence_references
                ):
                    raise ValueError("Account partition exceeds evidence-reference workset budget")
                if max_payload_bytes is not None:
                    # 근거 tuple, 검증용 set, 정렬 UID 목록이나 training_alert_uid JSON을
                    # 할당하기 전에 본문과 식별자 양쪽의 크기를 제한합니다.
                    metadata = dict(
                        source_id=source_id,
                        rule_id=RULE_ID,
                        rule_version=str(policy.rule_version),
                        rule_config_hash=config_hash,
                        account_key=account,
                        reference_ts=decision.reference_ts.isoformat(),
                        window_start=decision.window.start.isoformat(),
                        window_end=decision.window.end.isoformat(),
                        transaction_uids=(),
                        paths=(),
                    )
                    identity = dict(
                        source=source_id,
                        rule=RULE_ID,
                        rule_version=str(policy.rule_version),
                        rule_config_hash=config_hash,
                        account=account,
                        reference_ts=decision.reference_ts.isoformat(timespec="microseconds"),
                        window_start=decision.window.start.isoformat(timespec="microseconds"),
                        window_end=decision.window.end.isoformat(timespec="microseconds"),
                        evidence=(),
                        paths=(),
                    )
                    payload_bytes += max(_json_bytes(metadata), _json_bytes(identity))
                    payload_bytes += active_uid_bytes + max(0, len(active) - 1) * 2
                    if payload_bytes > max_payload_bytes:
                        raise ValueError(
                            "Account partition exceeds alert/evidence payload byte budget"
                        )
                alerts.append(
                    OfflineAlert(
                        source_id=source_id,
                        rule_id=RULE_ID,
                        rule_version=str(policy.rule_version),
                        rule_config_hash=config_hash,
                        account_key=account,
                        reference_ts=decision.reference_ts,
                        window_start=decision.window.start,
                        window_end=decision.window.end,
                        transaction_uids=tuple(item["transaction_uid"] for item in active),
                    )
                )
    return sorted(alerts, key=lambda a: (a.reference_ts, a.training_alert_uid))
