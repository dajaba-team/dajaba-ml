"""공통 합성 오프라인 fixture이며, 각 테스트에 새 원천 데이터를 제공합니다."""

import hashlib
from datetime import datetime
from decimal import Decimal
from importlib.resources import files

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from aml_ml.offline_rules import R02Policy
from aml_ml.storage import SourceInput, sha256, write_json

pytest.register_assert_rewrite("offline_test_data")


@pytest.fixture
def policy():
    # 이 값은 합성 fixture이며 팀의 운영 임계값이 아닙니다.
    return R02Policy(
        rule_version="fixture-r02",
        policy_version="fixture-policy-v1",
        min_counterparties=2,
        window_policy="B7_E_PLUS_ONE_MINUTE",
        recurrence_policy="PREVIOUS_CLOCK_MINUTE_FALSE_TO_TRUE",
        evidence_policy="ALL_INCOMING_IN_WINDOW",
        counterparty_policy="EXACT_ACCOUNT_KEYS_ACROSS_CURRENCIES",
        self_transfer_policy="EXCLUDE_FLAGGED_AND_SAME_ACCOUNT",
        decision_status="FIXTURE_ONLY",
        confirmation_reference="SYNTHETIC_TEST_NOT_TEAM_AGREEMENT",
    )


@pytest.fixture
def source(tmp_path):
    folder = tmp_path / "assembly" / "ibm_hi_small"
    folder.mkdir(parents=True)
    release = "ibm_hi_small:" + "a" * 64
    rows, labels = [], []
    for day in (1, 3, 5):
        # 모든 송신 거래는 한 분 안에 들어오며, 두 클래스는 과거 계좌 행동이 다릅니다.
        for account, count in (("P", 3), ("Q", 2)):
            for peer in range(count):
                n = len(rows)
                rows.append(
                    {
                        "transaction_uid": f"{release}:{n}",
                        "source_row_id": n,
                        "ts": datetime(2022, 1, day, 12, 0, peer),
                        "from_account_key": f"sender-{peer}",
                        "to_account_key": account,
                        "is_self_transfer": False,
                        "payment_format": "Wire",
                        "amount_paid": Decimal(str((peer + 1) * (10 if account == "P" else 1))),
                        "amount_received": Decimal(str((peer + 1) * (10 if account == "P" else 1))),
                        "currency_paid_raw": "US Dollar",
                        "currency_received_raw": "US Dollar",
                        "currency_paid_bucket": "USD",
                        "currency_received_bucket": "USD",
                    }
                )
                labels.append(
                    {
                        "transaction_uid": f"{release}:{n}",
                        "source_row_id": n,
                        "source_positive": 1 if account == "P" and peer == 1 else 0,
                    }
                )
    tx = pa.Table.from_pylist(rows)
    truth = pa.Table.from_pylist(labels)
    manifests, datasets = {}, {}
    for key, table in (("ml_transactions", tx), ("ml_labels", truth)):
        path = folder / key
        path.mkdir()
        p = path / "part-0.parquet"
        pq.write_table(table, p)
        manifest = {
            "status": "VERIFIED",
            "source_id": "ibm_hi_small",
            "source_kind": "ibm",
            "source_release": release,
            "purpose": "ml",
            "scope": "FULL_SOURCE",
            "rows": len(rows),
            "parts": [{"rows": len(rows), "files": [{"path": p.name, "sha256": sha256(p)}]}],
            "aliases_sha256": hashlib.sha256(
                files("aml_ml").joinpath("currencies.json").read_bytes()
            ).hexdigest(),
        }
        write_json(path / "manifest.json", manifest)
        manifests[key] = manifest
        datasets[key] = {"path": key, "rows": len(rows)}
    metadata = {
        "source_id": "ibm_hi_small",
        "status": "VERIFIED",
        "source_release": release,
        "input_rows": len(rows),
        "ml_output_rows": len(rows),
        "datasets": datasets,
    }
    write_json(folder / "source_manifest.json", metadata)
    run = folder.parent
    write_json(
        run / "run.json",
        {
            "contract_version": "dajaba-assembly-v2",
            "status": "COMPLETED",
            "sources": {"ibm_hi_small": {"status": "VERIFIED", "folder": folder.name}},
        },
    )
    write_json(
        tmp_path / "index.json",
        {
            "ml_runs": {"ibm_hi_small": "assembly"},
            "run_manifest_sha256": {"ibm_hi_small": sha256(run / "run.json")},
        },
    )
    return SourceInput(
        "ibm_hi_small",
        "ibm",
        folder,
        metadata,
        manifests["ml_transactions"],
        manifests["ml_labels"],
    )
