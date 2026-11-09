"""서비스 ID를 생성하지 않고 로컬 백엔드 내보내기의 원천·ID 연결을 검증합니다.

로컬 인계 어댑터이며 새 백엔드 API나 R02 탐지기는 아닙니다. 검사를 통과해도
내보내기 결과에 룰의 전체 근거가 담겼다는 사실까지 입증하지는 않습니다.
"""

import argparse
import hashlib
import json
import re
from datetime import datetime, timedelta
from pathlib import Path

from .offline_contract import source_fingerprint
from .offline_rules import OFFLINE_VERSION_PREFIX, RULE_ID, R02Policy, load_policy
from .sliding_window import minute_window
from .storage import PROJECT_SOURCES, SourceInput, load_index

MAX_BIGINT = 2**63 - 1


def _backend_id(value) -> int:
    if type(value) is not int or not 1 <= value <= MAX_BIGINT:
        raise ValueError("Actual backend IDs must be positive signed BIGINT integers")
    return value


def _r02_contract(alert: dict) -> dict:
    if alert.get("rule_id") != RULE_ID:
        raise ValueError("This handoff check accepts R02 samples only")
    for key in ("alert_id", "rule_version", "account_key"):
        if not isinstance(alert.get(key), str) or not alert[key].strip():
            raise ValueError(f"R02 {key} must be a nonempty string")
    if alert["rule_version"].startswith(OFFLINE_VERSION_PREFIX):
        raise ValueError("An offline spec version cannot identify an actual backend alert")
    times = {}
    for key in ("window_start", "window_end", "reference_ts"):
        if not isinstance(alert.get(key), str):
            raise ValueError(f"R02 {key} must preserve the source timestamp string")
        times[key] = datetime.fromisoformat(alert[key])
        if times[key].tzinfo is not None:
            raise ValueError("R02 windows must preserve naive source time, not UTC system time")
    evaluation_minute = times["window_end"] - timedelta(minutes=1)
    expected = minute_window(evaluation_minute)
    if times["window_start"] != expected.start or times["reference_ts"] != expected.end:
        raise ValueError("R02 requires the B7 24h window and reference_ts=window_end")
    result = {
        key: alert[key]
        for key in (
            "alert_id",
            "rule_id",
            "rule_version",
            "account_key",
            "window_start",
            "window_end",
            "reference_ts",
        )
    }
    # 감사에서 파생한 필드이며, 백엔드 HTTP 요청에 새로 요구하는 필드가 아닙니다.
    result["evaluation_minute"] = evaluation_minute.isoformat()
    result["batch_date"] = evaluation_minute.date().isoformat()
    return result


def _currency_evidence_status(alert: dict, evidence_ids: list[int]) -> str:
    """금액 해석이나 합산 없이 입력된 통화별 ID의 합집합을 확인합니다.

    부재는 명시적으로 남깁니다. 이 참조 검사는 거래가 해당 통화에 속한다는
    사실을 입증하거나 금액을 원천 값과 대조하는 검증이 아닙니다.
    """
    if "rule_facts" not in alert:
        return "NOT_PROVIDED"
    facts = alert["rule_facts"]
    if not isinstance(facts, dict):
        raise ValueError("Supplied rule_facts must be an object")
    if "amounts_by_currency" not in facts:
        return "NOT_PROVIDED"
    groups = facts["amounts_by_currency"]
    if not isinstance(groups, list):
        raise ValueError("Supplied amounts_by_currency must be a list")
    union: set[int] = set()
    for group in groups:
        if not isinstance(group, dict):
            raise ValueError("Each currency evidence group must be an object")
        currency = group.get("currency")
        if not isinstance(currency, str) or not currency.strip():
            raise ValueError("Currency evidence must preserve a nonempty currency string")
        transaction_ids = group.get("transaction_ids")
        if not isinstance(transaction_ids, list):
            raise ValueError("Currency transaction_ids must be an array of actual backend IDs")
        ids = [_backend_id(value) for value in transaction_ids]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate transaction ID within currency evidence")
        union.update(ids)
    if union != set(evidence_ids):
        raise ValueError("Currency evidence union must equal top-level transaction_ids")
    return "MATCHES_TOP_LEVEL_IDS"


def validate_handoff(
    payload: dict, source: SourceInput, *, policy: R02Policy | None = None
) -> dict:
    """단일 원천 샘플을 기존에 검증된 조립 메타데이터에 연결합니다.

    백엔드는 ID와 원본 파일 해시·행을 제공해야 합니다. 행 번호 기준은
    명시해야 하며, 어댑터는 0부터인지 1부터인지 추측하지 않습니다.
    원천 전체 읽기, 금액 변환, 라벨 계산이나 모델 학습은 하지 않습니다.
    """
    if policy is not None:
        policy.require_backend()
    if not isinstance(payload, dict) or payload.get("id_namespace") != "BACKEND_OPS":
        raise ValueError("An explicit BACKEND_OPS export is required; candidates are not IDs")
    reference = payload.get("export_reference")
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError("Backend export provenance reference is required")
    base = payload.get("source_row_number_base")
    if type(base) is not int or base not in (0, 1):
        raise ValueError("Explicit source_row_number_base must be 0 or 1")
    if source.id not in PROJECT_SOURCES:
        raise ValueError("Only project Small/SAML sources are supported; no Large")
    release = source.metadata.get("source_release")
    if not isinstance(release, str) or not re.fullmatch(
        re.escape(source.id) + r":[0-9a-f]{64}", release
    ):
        raise ValueError("Source release must bind source identity and original SHA-256")
    count = source.metadata.get("input_rows")
    if type(count) is not int or count < 1:
        raise ValueError("A positive original source row count is required")
    source_hash = source_fingerprint(source)
    original_sha = release.split(":", 1)[1]
    records, alerts = payload.get("mapping"), payload.get("alerts")
    if not isinstance(records, list) or not records:
        raise ValueError("A nonempty backend mapping list is required")
    if not isinstance(alerts, list) or not alerts:
        raise ValueError("A nonempty backend alert sample is required")
    by_id, seen_uids = {}, set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Each mapping record must be an object")
        transaction_id = _backend_id(record.get("transaction_id"))
        if record.get("source_file_hash") != original_sha:
            raise ValueError("Backend source_file_hash differs from the assembly source")
        ordinal = record.get("source_row_number")
        if type(ordinal) is not int or not base <= ordinal < count + base:
            raise ValueError("Original source row number is outside the declared source")
        uid = f"{release}:{ordinal - base}"
        if "transaction_uid" in record and record["transaction_uid"] != uid:
            raise ValueError("Supplied learning UID differs from the original SHA/row")
        if transaction_id in by_id or uid in seen_uids:
            raise ValueError("Backend ID and original source row must be one-to-one")
        by_id[transaction_id] = uid
        seen_uids.add(uid)
    linked_alerts, seen_alerts = [], set()
    for alert_index, alert in enumerate(alerts):
        if not isinstance(alert, dict):
            raise ValueError("Each alert sample must be an object")
        contract = _r02_contract(alert)
        if policy is not None and contract["rule_version"] != policy.rule_version:
            raise ValueError(f"alerts[{alert_index}].rule_version differs from the supplied policy")
        alert_id = contract["alert_id"]
        if alert_id in seen_alerts:
            raise ValueError("Duplicate alert identity")
        seen_alerts.add(alert_id)
        evidence = alert.get("transaction_ids")
        if not isinstance(evidence, list) or not evidence:
            raise ValueError("Full evidence IDs must be supplied as a nonempty list")
        ids = [_backend_id(value) for value in evidence]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate evidence transaction ID")
        if any(value not in by_id for value in ids):
            raise ValueError("An evidence ID is missing from the backend source mapping")
        currency_status = _currency_evidence_status(alert, ids)
        linked_alerts.append(
            {
                **contract,
                "transaction_ids": ids,
                "transaction_uids": [by_id[value] for value in ids],
                "currency_evidence_status": currency_status,
            }
        )
    return {
        "status": "MAPPING_AND_EVIDENCE_REFERENCES_VALID",
        "id_namespace": "BACKEND_OPS",
        "export_reference": reference,
        "source_id": source.id,
        "source_release": release,
        "source_fingerprint": source_hash,
        "source_row_number_base": base,
        "mapping_rows": len(by_id),
        "alert_count": len(linked_alerts),
        "policy_binding": {
            "status": "MATCHES_SUPPLIED_POLICY" if policy is not None else "NOT_PROVIDED",
            "checks": ["RULE_VERSION_EQUALITY"] if policy is not None else [],
            "rule_version": policy.rule_version if policy is not None else None,
            "rule_version_origin": policy.rule_version_origin if policy is not None else None,
            "policy_version": policy.policy_version if policy is not None else None,
            "min_counterparties": policy.min_counterparties if policy is not None else None,
            "config_hash": policy.config_hash if policy is not None else None,
            "confirmation_reference": policy.confirmation_reference if policy is not None else None,
        },
        "alerts": linked_alerts,
        "not_verified": [
            "BACKEND_EXPORT_AUTHENTICITY_OR_DB_CONTENTS",
            "FULL_RULE_EVIDENCE_COMPLETENESS",
            "RULE_VERSION_AND_CONFIG_AGREEMENT",
            "BACKEND_THRESHOLD_ACTUALLY_APPLIED",
            "TRANSACTION_TIME_AND_ACCOUNT_MEMBERSHIP",
            "CURRENCY_AMOUNTS_AND_TRANSACTION_CURRENCY_MEMBERSHIP",
            "R02_RECURRENCE_AND_RULE_RESULT_EQUIVALENCE",
        ],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assembly-index", type=Path, required=True)
    parser.add_argument("--source", choices=sorted(PROJECT_SOURCES), required=True)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", type=Path, help="Received confirmed policy; local file only")
    parser.add_argument("--max-input-bytes", type=int, default=16 * 1024 * 1024)
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise ValueError("Output already exists; choose a new report path")
        policy = None
        policy_file_sha256 = None
        if args.policy is not None:
            policy, policy_file_sha256 = load_policy(args.policy)
            policy.require_backend()
        if args.max_input_bytes < 1:
            raise ValueError("Input byte budget must be positive")
        with args.handoff.open("rb") as stream:
            content = stream.read(args.max_input_bytes + 1)
        if len(content) > args.max_input_bytes:
            raise ValueError("Handoff exceeds the explicit input byte budget")
        payload = json.loads(content.decode("utf-8-sig"))
        source = load_index(args.assembly_index, [args.source])[0]
        result = validate_handoff(payload, source, policy=policy)
        if args.policy is not None:
            result["policy_binding"]["policy_file_sha256"] = policy_file_sha256
            result["policy_binding"]["policy_path"] = str(args.policy.resolve())
        result["handoff_sha256"] = hashlib.sha256(content).hexdigest()
        result["validator_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # 배타적 생성으로 다른 작성자와 경합할 때도 덮어쓰기를 막습니다.
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        print(f"HANDOFF_REFERENCES=VALID\nREPORT={args.output.resolve()}")
        return 0
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
        parser.exit(2, f"Backend handoff rejected: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
