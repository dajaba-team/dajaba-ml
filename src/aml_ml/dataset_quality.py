"""모델 피처·proxy 정답과 분리된, 크기가 제한된 집계 진단을 제공합니다."""

import copy
import math
from datetime import datetime
from typing import Any

from .offline_features import FEATURE_NAMES

SPLITS = ("train", "validation", "test")
QUALITY_VERSION = "offline-alert-quality-v1"
FLAGS = ("account_coverage_partial_24h", "account_history_absent", "account_money_unit_unknown")


class DatasetQuality:
    """고정 크기 집계로 거래 ID를 보관하지 않고, 금액 합산이나 X의 라벨 포함도 하지 않습니다."""

    def __init__(self, source_id: str):
        self.source_id = source_id
        self.splits: dict[str, dict[str, Any]] = {
            split: {
                "alerts": 0,
                "proxy_positive": 0,
                "proxy_negative": 0,
                "proxy_unresolved": 0,
                "evidence_references": 0,
                "missing_label_references": 0,
                "reference_ts_min": None,
                "reference_ts_max": None,
                "raw_feature_nulls": dict.fromkeys(FEATURE_NAMES, 0),
                "flagged_alerts": dict.fromkeys(FLAGS, 0),
            }
            for split in SPLITS
        }

    def add(self, target: dict, feature: dict) -> None:
        split = target["split"]
        if split not in self.splits or target["source_id"] != self.source_id:
            raise ValueError("Quality target source/split mismatch")
        truth = target["y_alert_proxy"]
        evidence, missing = target["evidence_count"], target["missing_label_count"]
        if (
            type(evidence) is not int
            or type(missing) is not int
            or not 0 <= missing <= evidence
            or evidence < 1
            or (truth is not None and (type(truth) is not int or truth not in (0, 1)))
            or ((truth is None) != (missing > 0))
        ):
            raise ValueError("Quality target has inconsistent proxy/evidence counts")
        if set(feature) != set(FEATURE_NAMES):
            raise ValueError("Quality feature schema differs from the account allowlist")
        stamp = datetime.fromisoformat(target["reference_ts"])
        if stamp.tzinfo is not None:
            raise ValueError("Quality timestamps must preserve naive source time")
        record = self.splits[split]
        record["alerts"] += 1
        key = "proxy_unresolved"
        if truth is not None:
            key = "proxy_positive" if truth else "proxy_negative"
        record[key] += 1
        record["evidence_references"] += evidence
        record["missing_label_references"] += missing
        # 비교 전에 ISO 마이크로초 표현으로 같은 시각의 문자열을 정규화합니다.
        timestamp = stamp.isoformat(timespec="microseconds")
        for suffix, choose in (("min", min), ("max", max)):
            key = f"reference_ts_{suffix}"
            record[key] = timestamp if record[key] is None else choose(record[key], timestamp)
        for name in FEATURE_NAMES:
            value = feature[name]
            if value is None:
                record["raw_feature_nulls"][name] += 1
            elif not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("Quality features must be finite numbers or null")
        for name in FLAGS:
            if feature[name] not in (None, 0, 1):
                raise ValueError("Quality flag must be zero, one or null")
            record["flagged_alerts"][name] += int(feature[name] == 1)

    def report(self) -> dict:
        splits = copy.deepcopy(self.splits)
        warnings = []
        for split, record in splits.items():
            count = record["alerts"]
            record["raw_feature_null_rates"] = {
                name: value / count if count else None
                for name, value in record["raw_feature_nulls"].items()
            }
            if record["proxy_unresolved"]:
                warnings.append({"split": split, "code": "UNRESOLVED_PROXY_LABELS"})
            if split in ("train", "validation") and (
                not record["proxy_positive"] or not record["proxy_negative"]
            ):
                warnings.append({"split": split, "code": "TRAINING_REQUIRES_BOTH_PROXY_CLASSES"})
        return {
            "quality_version": QUALITY_VERSION,
            "source_id": self.source_id,
            "prediction_unit": "OFFLINE_ALERT",
            "feature_context": "DIRECT_ACCOUNT_24H_BEFORE_REFERENCE",
            "target_definition": "ANY_POSITIVE_IN_FULL_ALERT_EVIDENCE_NOT_STR_TRUTH",
            "evidence_count_unit": "REFERENCES_PER_ALERT_NOT_DISTINCT_TRANSACTIONS",
            "raw_feature_names": list(FEATURE_NAMES),
            "splits": splits,
            "warnings": warnings,
            "service_id_mapping_verified": False,
            "model_training_executed": False,
        }
