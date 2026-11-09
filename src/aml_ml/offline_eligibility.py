"""라벨을 보지 않고 시간 기준 적격 여부를 판정하며, 후보·정답 전체 기록을 보존합니다."""

from collections import Counter
from datetime import timedelta

from .offline_settings import validate_boundaries

ELIGIBILITY_VERSION = "r02-label-blind-purge-v1"
SPLITS = ("train", "validation", "test")


class DatasetNotReady(ValueError):
    """후보 근거는 완전하지만 지도 학습 전처리를 마칠 수 없는 상태입니다."""


def original_split(reference, first, second):
    return "train" if reference < first else "validation" if reference < second else "test"


def ledger_row(alert, target, *, train_before, test_from, horizon_seconds=86400):
    validate_boundaries(train_before, test_from, horizon_seconds)
    horizon = timedelta(seconds=horizon_seconds)
    if horizon < timedelta(hours=24) or alert.reference_ts - alert.window_start > horizon:
        raise ValueError("Eligibility horizon is shorter than the actual history/evidence")
    split = original_split(alert.reference_ts, train_before, test_from)
    reasons = []
    if split == "validation" and alert.reference_ts < train_before + horizon:
        reasons.append("VALIDATION_HORIZON_PURGE")
    if split == "test" and alert.reference_ts < test_from + horizon:
        reasons.append("TEST_HORIZON_PURGE")
    time_eligible = not reasons
    truth = target["y_alert_proxy"]
    complete = target["missing_label_count"] == 0
    if complete != (truth is not None) or (
        truth is not None and (type(truth) is not int or truth not in (0, 1))
    ):
        raise ValueError("Eligibility target completeness is inconsistent")
    if not complete:
        reasons.append("UNRESOLVED_EVIDENCE_LABEL")
    if target["training_alert_uid"] != alert.training_alert_uid:
        raise ValueError("Eligibility alert/target UID mismatch")
    return {
        "training_alert_uid": alert.training_alert_uid,
        "source_id": alert.source_id,
        "reference_ts": alert.reference_ts.isoformat(),
        "original_split": split,
        "time_eligible": time_eligible,
        "target_complete": complete,
        "supervised_eligible": time_eligible and complete,
        "exclusion_reasons": reasons,
        "y_alert_proxy": truth,
        "evidence_uids": list(alert.transaction_uids),
        "horizon_seconds": horizon_seconds,
        "train_before": train_before.isoformat(),
        "test_from": test_from.isoformat(),
        "eligibility_version": ELIGIBILITY_VERSION,
    }


class EligibilityCounts:
    """제한된 집계를 유지하며, 사유가 겹쳐도 경보를 중복 차감하지 않습니다."""

    def __init__(self):
        self.counts = {
            split: {kind: Counter() for kind in ("all", "eligible", "excluded")} for split in SPLITS
        }
        self.reasons = {split: Counter() for split in SPLITS}

    def add(self, row):
        split = row["original_split"]
        truth = row["y_alert_proxy"]
        label = "unresolved" if truth is None else "positive" if truth == 1 else "negative"
        kind = "eligible" if row["supervised_eligible"] else "excluded"
        for subset in ("all", kind):
            self.counts[split][subset]["rows"] += 1
            self.counts[split][subset][label] += 1
        for reason in set(row["exclusion_reasons"]):
            self.reasons[split][reason] += 1

    def report(self):
        total = sum(self.counts[s]["all"]["rows"] for s in SPLITS)
        eligible = sum(self.counts[s]["eligible"]["rows"] for s in SPLITS)
        splits, blockers = {}, ["MODEL_METRIC_MINIMUM_SUPPORT_POLICY_UNSET"]
        for split in SPLITS:
            record = {}
            for subset in ("all", "eligible", "excluded"):
                counts = self.counts[split][subset]
                rows = counts["rows"]
                known = counts["positive"] + counts["negative"]
                record[subset] = {
                    key: counts[key] for key in ("rows", "positive", "negative", "unresolved")
                }
                record[subset]["positive_fraction_all"] = (
                    counts["positive"] / rows if rows else None
                )
                record[subset]["positive_fraction_known"] = (
                    counts["positive"] / known if known else None
                )
            if record["all"]["rows"] != record["eligible"]["rows"] + record["excluded"]["rows"]:
                raise ValueError("Eligibility ledger total does not balance")
            record["original_split_fraction"] = record["all"]["rows"] / total if total else None
            record["eligible_split_fraction"] = (
                record["eligible"]["rows"] / eligible if eligible else None
            )
            record["exclusion_reason_counts"] = dict(sorted(self.reasons[split].items()))
            if not record["eligible"]["rows"]:
                blockers.append(f"EMPTY_ELIGIBLE_{split.upper()}")
            if not all(record["eligible"][label] for label in ("positive", "negative")):
                blockers.append(f"MISSING_ELIGIBLE_CLASS_{split.upper()}")
            splits[split] = record
        return {
            "policy_version": ELIGIBILITY_VERSION,
            "total_alerts": total,
            "eligible_alerts": eligible,
            "excluded_alerts": total - eligible,
            "splits": splits,
            "model_evaluation_readiness": "NOT_READY",
            "readiness_reasons": blockers,
            "minimum_support_policy": None,
        }
