"""적합이나 test 접근 전에 모델·평가에 필요한 표본 기준을 명시하도록 요구합니다."""

import copy

from .offline_rules import fingerprint
from .storage import read_json


def require_evaluation_support(dataset, support_policy, model_kind):
    """최솟값 추정, 자동 완화나 실행 범위 우회를 허용하지 않습니다.

    호출자는 학습을 시작하기 전에 모델·지표에 필요한 표본 기준을 정해야 합니다.
    합성 테스트는 용도가 명시된 자체 합성 기준을 전달합니다. 이 기준이 실제
    데이터의 기본값이나 팀의 승인 근거가 되지는 않습니다.
    """
    if not isinstance(support_policy, dict):
        raise ValueError("NOT_READY: explicit model/metric minimum support policy is required")
    policy = copy.deepcopy(support_policy)
    if (
        set(policy) != {"version", "model_kind", "decision_reference", "metric_set", "minimums"}
        or policy["version"] != "offline-evaluation-support-v1"
        or policy["model_kind"] != model_kind
        or not isinstance(policy["decision_reference"], str)
        or not policy["decision_reference"].strip()
        or policy["metric_set"] != "PR_AUC_PRECISION_RECALL_F1_TOP_K_V1"
        or not isinstance(policy["minimums"], dict)
        or set(policy["minimums"]) != {"train", "validation", "test"}
    ):
        raise ValueError("Invalid explicit evaluation support policy")
    for split, minimums in policy["minimums"].items():
        if not isinstance(minimums, dict) or set(minimums) != {"rows", "positive", "negative"}:
            raise ValueError("Explicit row and class minimums are required for every split")
        if any(type(value) is not int or value < 1 for value in minimums.values()):
            raise ValueError("Declared metric set requires positive row/class support minimums")
        if minimums["rows"] < minimums["positive"] + minimums["negative"]:
            raise ValueError("Minimum row count is inconsistent with declared class minimums")
    quality = read_json(dataset / "quality.json")
    ledger = quality.get("eligibility")
    if not isinstance(ledger, dict):
        raise ValueError("NOT_READY: eligible population ledger is required")
    observed = {}
    for split, minimums in policy["minimums"].items():
        counts = ledger["splits"][split]["eligible"]
        observed[split] = {key: counts[key] for key in minimums}
        if any(counts[key] < minimum for key, minimum in minimums.items()):
            raise ValueError(f"NOT_READY: {split} is below declared evaluation support minimums")
    return {
        "status": "READY_FOR_DECLARED_SUPPORT",
        "policy": policy,
        "policy_sha256": fingerprint(policy),
        "observed_counts": observed,
        "dataset_quality_sha256": fingerprint(quality),
    }
