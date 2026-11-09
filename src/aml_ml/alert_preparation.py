"""호출자가 제공한 오프라인 경보 묶음을 불변 조립 근거에 연결합니다.

룰 생성, 피처 계산, 백엔드 ID 발급, 금액 합산이나 모델 학습은 하지 않습니다.
참조된 Parquet 파트만 읽습니다. 라벨은 별도의 정답 결과에 보관합니다.
"""

import bisect
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, cast

import pyarrow as pa
import pyarrow.compute as _compute
import pyarrow.parquet as pq

from .alert_dataset import OfflineAlert, chronological_split, offline_proxy_targets
from .storage import SourceInput

pc = cast(Any, _compute)


@dataclass(frozen=True)
class AlertBatch:
    alerts: list[dict]
    targets: list[dict]
    evidence: pa.Table
    splits: dict[str, list[str]]
    metadata: dict


def bind_evidence(source: SourceInput, alerts: list[OfflineAlert]) -> tuple[pa.Table, dict]:
    """정확한 릴리스+행 UID를 연결하며, 행 순번을 운영 ID로 바꾸지 않습니다."""
    requested = {uid for alert in alerts for uid in alert.transaction_uids}
    if not requested or any(alert.source_id != source.id for alert in alerts):
        raise ValueError("A nonempty batch from one assembly source is required")
    prefix = source.metadata["source_release"] + ":"
    rows = {}
    for uid in requested:
        if not uid.startswith(prefix):
            raise ValueError("Evidence UID belongs to another source release")
        row = int(uid[len(prefix) :])
        if not 0 <= row < source.metadata["input_rows"]:
            raise ValueError("Evidence UID row is outside the complete source")
        rows[row] = uid
    ends = []
    total = 0
    for part in source.parts("transactions"):
        total += part["rows"]
        ends.append(total)
    by_part: dict[int, list[int]] = {}
    for row in rows:
        by_part.setdefault(bisect.bisect_right(ends, row), []).append(row)
    tables = []
    labels = {}
    for index, row_ids in sorted(by_part.items()):
        tx = pq.read_table(source.part_path("transactions", index))
        target = pq.read_table(source.part_path("labels", index))
        ids = pa.array(row_ids, type=pa.int64())
        tx = tx.filter(pc.is_in(tx.column("source_row_id"), value_set=ids))
        target = target.filter(pc.is_in(target.column("source_row_id"), value_set=ids))
        if len(tx) != len(row_ids) or len(target) != len(row_ids):
            raise ValueError("Missing or duplicated full-evidence rows")
        for table in (tx, target):
            actual = dict(
                zip(
                    table.column("source_row_id").to_pylist(),
                    table.column("transaction_uid").to_pylist(),
                )
            )
            if actual != {row: rows[row] for row in row_ids}:
                raise ValueError("Evidence UID/row identity differs from assembly")
        for item in target.select(["transaction_uid", "source_positive"]).to_pylist():
            labels[item["transaction_uid"]] = item["source_positive"]
        tables.append(tx)
    evidence = pa.concat_tables(tables).sort_by([("source_row_id", "ascending")])
    return evidence, labels


def _check_paths(alert: OfflineAlert, by_uid: dict) -> None:
    if alert.rule_id not in {"R04_CYCLE", "R05_STACK"}:
        if alert.paths:
            raise ValueError("Paths are defined only for R04/R05")
        return
    for path in alert.paths:
        if len(path) < 2 or len(set(path)) != len(path):
            raise ValueError("A path needs at least two distinct transactions")
        hops = [by_uid[uid] for uid in path]
        if any(row["is_self_transfer"] for row in hops):
            raise ValueError("Self transfers cannot form path evidence")
        if hops[0]["from_account_key"] != alert.account_key:
            raise ValueError("Path must start at the exact assembly account key")
        for first, second in zip(hops, hops[1:]):
            if first["ts"] > second["ts"] or first["to_account_key"] != second["from_account_key"]:
                raise ValueError("Path order or account connection is invalid; never repair it")
        if alert.rule_id == "R04_CYCLE" and hops[-1]["to_account_key"] != alert.account_key:
            raise ValueError("Cycle must return to its starting account")


def prepare_alert_batch(
    source: SourceInput,
    alerts: list[OfflineAlert],
    *,
    train_before: datetime,
    test_from: datetime,
) -> AlertBatch:
    """입력 후보의 공통 전처리이며, 완성된 SCRUM-34 학습 데이터는 아닙니다.

    호출자는 승인된 룰 버전과 라벨을 보지 않고 정한 분할 경계를 제공합니다.
    reference_ts 이후나 같은 시각의 근거는 proxy 근거에 남기며 피처 입력으로
    선언하지 않습니다. 실제 24시간 문맥과 피처는 별도 전처리 단계에서 만듭니다.
    """
    splits = chronological_split(alerts, train_before=train_before, test_from=test_from)
    evidence, labels = bind_evidence(source, alerts)
    by_uid = {row["transaction_uid"]: row for row in evidence.to_pylist()}
    for alert in alerts:
        _check_paths(alert, by_uid)
        # 뒤쪽 분할의 라벨이 앞쪽 분할에 들어가면 안 됩니다. reference_ts의
        # 경계 소속은 chronological_split을 정확히 따릅니다.
        upper = (
            train_before
            if alert.reference_ts < train_before
            else (test_from if alert.reference_ts < test_from else None)
        )
        if upper is not None and any(by_uid[uid]["ts"] >= upper for uid in alert.transaction_uids):
            raise ValueError("Evidence reaches a later chronological split")
    return AlertBatch(
        alerts=[
            dict(asdict(alert), training_alert_uid=alert.training_alert_uid) for alert in alerts
        ],
        targets=offline_proxy_targets(alerts, labels),
        evidence=evidence,
        splits=splits,
        metadata=dict(
            source_release=source.metadata["source_release"],
            id_namespace="OFFLINE_SOURCE_UID",
            prediction_unit="ALERT_CANDIDATE_PREPARATION",
            full_evidence_preserved=True,
            feature_cutoff="ts < reference_ts; evidence is not a feature matrix",
            rule_generation_executed=False,
            features_computed=False,
            model_training_executed=False,
            split_overlap_policy="REJECT; no implicit purge",
        ),
    )
