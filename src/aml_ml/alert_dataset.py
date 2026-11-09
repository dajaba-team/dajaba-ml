"""오프라인 경보 식별자, 전체 근거 기반 proxy 정답과 분할 보호를 제공합니다.

이 모듈은 룰 후보를 입력받습니다. 미정 룰 조건 생성, 서비스 거래 ID 발급,
피처 계산이나 STR 결과 추론은 수행하지 않습니다.
"""

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Mapping


@dataclass(frozen=True)
class OfflineAlert:
    source_id: str
    rule_id: str
    rule_version: str
    rule_config_hash: str
    account_key: str
    reference_ts: datetime
    window_start: datetime
    window_end: datetime
    transaction_uids: tuple[str, ...]
    paths: tuple[tuple[str, ...], ...] = ()

    def __post_init__(self):
        if not isinstance(self.rule_version, str) or not self.rule_version.strip():
            raise ValueError("rule_version must be a nonempty string; never coerce a version")
        if not all((self.source_id, self.rule_id, self.rule_version, self.account_key)):
            raise ValueError("Source, rule identity/version and exact account key are required")
        if not re.fullmatch(r"[0-9a-f]{64}", self.rule_config_hash):
            raise ValueError("A SHA-256 fingerprint of the actual rule configuration is required")
        if any(
            ts.tzinfo is not None for ts in (self.reference_ts, self.window_start, self.window_end)
        ):
            raise ValueError("Source timestamps must preserve naive source time")
        if self.window_start >= self.window_end:
            raise ValueError("Candidate window_start must precede window_end")
        if not self.transaction_uids or len(set(self.transaction_uids)) != len(
            self.transaction_uids
        ):
            raise ValueError("Full evidence must be nonempty and contain unique UIDs")
        prefix = self.source_id + ":"
        for uid in self.transaction_uids:
            if not isinstance(uid, str) or not uid.startswith(prefix):
                raise ValueError("Evidence must use source-namespaced learning UIDs, not ops IDs")
            suffix = uid[len(prefix) :]
            if not re.fullmatch(r"[0-9a-f]{64}:(0|[1-9][0-9]*)", suffix):
                raise ValueError("UID must contain the original file SHA-256 and zero-based row")
        if self.rule_id in {"R04_CYCLE", "R05_STACK"} and not self.paths:
            raise ValueError("Path-rule candidates require complete paths")
        if self.rule_id == "R04_CYCLE" and any(len(path) > 3 for path in self.paths):
            raise ValueError("R04 permits at most 3 hops including the return transaction")
        if self.paths:
            if any(not path for path in self.paths):
                raise ValueError("Empty evidence path")
            if {uid for path in self.paths for uid in path} != set(self.transaction_uids):
                raise ValueError("Path union must equal full alert evidence")

    @property
    def training_alert_uid(self) -> str:
        identity = {
            "source": self.source_id,
            "rule": self.rule_id,
            "rule_version": self.rule_version,
            "rule_config_hash": self.rule_config_hash,
            "account": self.account_key,
            "reference_ts": self.reference_ts.isoformat(timespec="microseconds"),
            "window_start": self.window_start.isoformat(timespec="microseconds"),
            "window_end": self.window_end.isoformat(timespec="microseconds"),
            "evidence": sorted(self.transaction_uids),
            "paths": sorted([list(path) for path in self.paths]),
        }
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        return "TRAIN-ALERT-" + digest


def offline_proxy_targets(
    alerts: list[OfflineAlert], labels: Mapping[str, int | None]
) -> list[dict]:
    """다른 근거가 양성이어도 누락된 라벨은 null로 유지합니다."""
    result = []
    seen = set()
    for alert in alerts:
        uid = alert.training_alert_uid
        if uid in seen:
            raise ValueError("Duplicate training alert identity")
        seen.add(uid)
        values = [labels.get(item) for item in alert.transaction_uids]
        for value in values:
            if value is not None and (type(value) is not int or value not in (0, 1)):
                raise ValueError("Source labels must be integer 0/1 or null")
        complete = all(value is not None for value in values)
        result.append(
            {
                "training_alert_uid": uid,
                "source_id": alert.source_id,
                "reference_ts": alert.reference_ts.isoformat(),
                "y_alert_proxy": int(any(value == 1 for value in values)) if complete else None,
                "target_status": "COMPLETE_PROXY_NOT_STR_TRUTH"
                if complete
                else "UNRESOLVED_EVIDENCE_LABEL",
                "target_definition": "ANY_POSITIVE_IN_FULL_ALERT_EVIDENCE",
                "evidence_count": len(values),
                "missing_label_count": sum(value is None for value in values),
                "id_namespace": "OFFLINE_SOURCE_UID",
            }
        )
    return result


def chronological_split(
    alerts: list[OfflineAlert], *, train_before: datetime, test_from: datetime
) -> dict[str, list[str]]:
    """라벨을 보지 않고 정한 명시적 경계를 사용하며, 분할 간 근거 누수를 거부합니다.

    같은 기준 시각은 같은 분할에 둡니다. 근거가 겹치면 오류로 보고하며,
    경보를 몰래 삭제하거나 승인되지 않은 제거 정책을 선택하지 않습니다.
    """
    if train_before.tzinfo is not None or test_from.tzinfo is not None:
        raise ValueError("Split boundaries must use naive source time")
    if train_before >= test_from:
        raise ValueError("train_before must precede test_from")
    if len({alert.source_id for alert in alerts}) > 1:
        raise ValueError("Split each source independently")
    splits = {name: [] for name in ("train", "validation", "test")}
    evidence_owner = {}
    seen = set()
    for alert in sorted(alerts, key=lambda item: (item.reference_ts, item.training_alert_uid)):
        uid = alert.training_alert_uid
        if uid in seen:
            raise ValueError("Duplicate training alert identity")
        seen.add(uid)
        split = (
            "train"
            if alert.reference_ts < train_before
            else ("validation" if alert.reference_ts < test_from else "test")
        )
        for transaction_uid in alert.transaction_uids:
            owner = evidence_owner.setdefault(transaction_uid, split)
            if owner != split:
                raise ValueError(
                    "Full evidence overlaps splits; an explicit purge policy is required"
                )
        splits[split].append(uid)
    return splits
