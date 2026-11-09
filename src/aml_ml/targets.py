"""오프라인 경보 proxy 라벨 전용이며, 피처 어댑터에서는 불러오지 않습니다."""

import pandas as pd


def alert_proxy_targets(
    alerts: list[dict], mapping: pd.DataFrame, labels: pd.DataFrame, *, id_namespace: str
) -> pd.DataFrame:
    if id_namespace != "BACKEND_OPS":
        raise ValueError(
            "An explicit actual-backend-ID mapping is required; candidate row IDs are not ops IDs"
        )
    if not {"transaction_id", "transaction_uid"} <= set(mapping.columns) or not {
        "transaction_uid",
        "source_positive",
    } <= set(labels.columns):
        raise ValueError("Missing mapping/label columns")
    if mapping.transaction_id.duplicated().any() or labels.transaction_uid.duplicated().any():
        raise ValueError("Ambiguous mapping or source label identity")
    by_id = dict(zip(mapping.transaction_id, mapping.transaction_uid))
    by_uid = dict(zip(labels.transaction_uid, labels.source_positive))
    result = []
    for alert in alerts:
        evidence = alert.get("evidence_transaction_ids", [])
        values = [by_uid.get(by_id.get(i)) for i in evidence]
        complete = bool(values) and all(v in (0, 1) for v in values)
        result.append(
            dict(
                alert_id=alert["alert_id"],
                y_alert_proxy=int(any(v == 1 for v in values)) if complete else None,
                target_status="COMPLETE_PROXY_NOT_STR_TRUTH"
                if complete
                else "UNRESOLVED_EVIDENCE_LABEL",
                target_definition="ANY_POSITIVE_IN_EVIDENCE",
                id_namespace=id_namespace,
            )
        )
    return pd.DataFrame(result)
