"""사용자가 합성 데이터로 실행하는 제한된 실측입니다. 실제 원천은 읽지 않습니다.

입력은 8,064행, 예상 경보는 768개이며 N10도 명시적인 합성 시험용 설정입니다.
작은 파이프라인과 직렬화 비용을 관측하며 운영 속도·용량·모델 성능으로 해석하지 않습니다.
"""

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import datetime
from decimal import Decimal
from importlib.resources import files
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .offline_benchmark import run_plan
from .offline_pipeline import build_dataset
from .offline_progress import Progress, cancelled_result, progress, reporting
from .offline_resume import reject_artifact_ancestor
from .offline_rules import R02Policy
from .offline_settings import execution_settings
from .storage import load_index, read_json, sha256, write_json

SOURCE_ROWS = 8064
ALERT_ROWS = 768
GROUPS = 128
SETTINGS = execution_settings(
    max_source_rows=SOURCE_ROWS,
    partitions=4,
    decode_batch_rows=1,
    sort_run_rows=256,
    sort_run_bytes=1024**2,
    write_batch_rows=128,
    merge_fan_in=4,
    max_partition_rows=256,
    max_partition_bytes=16 * 1024**2,
)


def _inventory(folder):
    return {
        path.relative_to(folder).as_posix(): sha256(path)
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }


def _fixture(root):
    """이번 실행 폴더 안에 고정 크기 합성 원천과 시험 전용 정책을 만듭니다."""
    source_id = "ibm_hi_small"
    release = source_id + ":" + hashlib.sha256(b"synthetic-n10-8064-v1").hexdigest()
    transactions, labels = [], []
    for group in range(GROUPS):
        for day in (1, 3, 5):
            for account, count in (("P", 11), ("Q", 10)):
                for peer in range(count):
                    ordinal = len(transactions)
                    uid = f"{release}:{ordinal}"
                    amount = Decimal((peer + 1) * (10 if account == "P" else 1))
                    transactions.append(
                        {
                            "transaction_uid": uid,
                            "source_row_id": ordinal,
                            "ts": datetime(2022, 1, day, 12, 0, peer),
                            "from_account_key": f"synthetic-{group}-sender-{peer}",
                            "to_account_key": f"synthetic-{group}-{account}",
                            "is_self_transfer": False,
                            "payment_format": "Wire",
                            "amount_paid": amount,
                            "amount_received": amount,
                            "currency_paid_raw": "US Dollar",
                            "currency_received_raw": "US Dollar",
                            "currency_paid_bucket": "USD",
                            "currency_received_bucket": "USD",
                        }
                    )
                    labels.append(
                        {
                            "transaction_uid": uid,
                            "source_row_id": ordinal,
                            "source_positive": int(account == "P" and peer == 1),
                        }
                    )
    if len(transactions) != SOURCE_ROWS or len(labels) != SOURCE_ROWS:
        raise ValueError("Synthetic fixture row count drift")
    schema = pa.schema(
        [
            ("transaction_uid", pa.string()),
            ("source_row_id", pa.int64()),
            ("ts", pa.timestamp("us")),
            ("from_account_key", pa.string()),
            ("to_account_key", pa.string()),
            ("is_self_transfer", pa.bool_()),
            ("payment_format", pa.string()),
            ("amount_paid", pa.decimal128(24, 6)),
            ("amount_received", pa.decimal128(24, 6)),
            ("currency_paid_raw", pa.string()),
            ("currency_received_raw", pa.string()),
            ("currency_paid_bucket", pa.string()),
            ("currency_received_bucket", pa.string()),
        ]
    )
    tx = pa.Table.from_pylist(transactions, schema=schema)
    truth = pa.Table.from_pylist(labels)
    run = root / "source-run"
    source_folder = run / source_id
    datasets = {}
    aliases = hashlib.sha256(files("aml_ml").joinpath("currencies.json").read_bytes()).hexdigest()
    for key, table in (("ml_transactions", tx), ("ml_labels", truth)):
        folder = source_folder / key
        folder.mkdir(parents=True, exist_ok=False)
        parts = []
        for start in range(0, SOURCE_ROWS, 512):
            part = table.slice(start, 512)
            path = folder / f"part-{start:05d}.parquet"
            pq.write_table(part, path, row_group_size=256)
            if pq.read_metadata(path).num_rows != len(part):
                raise ValueError("Synthetic part footer row mismatch")
            parts.append(
                {"rows": len(part), "files": [{"path": path.name, "sha256": sha256(path)}]}
            )
        write_json(
            folder / "manifest.json",
            {
                "status": "VERIFIED",
                "source_id": source_id,
                "source_kind": "ibm",
                "source_release": release,
                "purpose": "ml",
                "scope": "FULL_SOURCE",
                "rows": SOURCE_ROWS,
                "parts": parts,
                "aliases_sha256": aliases,
                "synthetic_fixture": True,
            },
        )
        datasets[key] = {"path": key, "rows": SOURCE_ROWS}
    write_json(
        source_folder / "source_manifest.json",
        {
            "source_id": source_id,
            "status": "VERIFIED",
            "source_release": release,
            "input_rows": SOURCE_ROWS,
            "ml_output_rows": SOURCE_ROWS,
            "datasets": datasets,
            "synthetic_fixture": True,
        },
    )
    write_json(
        run / "run.json",
        {
            "contract_version": "dajaba-assembly-v2",
            "status": "COMPLETED",
            "sources": {source_id: {"status": "VERIFIED", "folder": source_id}},
            "synthetic_fixture": True,
        },
    )
    index = root / "index.json"
    write_json(
        index,
        {
            "ml_runs": {source_id: "source-run"},
            "run_manifest_sha256": {source_id: sha256(run / "run.json")},
            "synthetic_fixture": True,
        },
    )
    policy = R02Policy(
        rule_version="fixture-n10-bounded-8064",
        policy_version="synthetic-microbenchmark-v1",
        min_counterparties=10,
        window_policy="B7_E_PLUS_ONE_MINUTE",
        recurrence_policy="PREVIOUS_CLOCK_MINUTE_FALSE_TO_TRUE",
        evidence_policy="ALL_INCOMING_IN_WINDOW",
        counterparty_policy="EXACT_ACCOUNT_KEYS_ACROSS_CURRENCIES",
        self_transfer_policy="EXCLUDE_FLAGGED_AND_SAME_ACCOUNT",
        decision_status="FIXTURE_ONLY",
        confirmation_reference="SYNTHETIC_BOUNDED_MEASUREMENT_NOT_TEAM_RULE_DEPLOYMENT",
    )
    write_json(root / "policy.json", asdict(policy))
    return load_index(index, [source_id])[0], policy


def run(output):
    artifact_root = Path(__file__).resolve().parents[3] / "artifacts"
    if artifact_root.is_symlink() or artifact_root.is_junction():
        raise ValueError("Project artifacts root must not be a link/junction")
    if output.exists() or output.is_symlink() or output.is_junction():
        raise FileExistsError("Synthetic benchmark requires a new output directory")
    output = output.resolve()
    if output == artifact_root or not output.is_relative_to(artifact_root.resolve()):
        raise ValueError("Bounded synthetic measurement output must be below project artifacts")
    reject_artifact_ancestor(output)
    output.mkdir(parents=True, exist_ok=False)
    with Progress(output / "progress", console=True) as reporter:
        with reporting(reporter):
            source, policy = _fixture(output / "fixture")
            before = _inventory(output / "fixture")
            write_json(output / "fixture_before.json", before)
            dataset = output / "reference-dataset"
            result = build_dataset(
                source,
                policy,
                dataset,
                train_before=datetime(2022, 1, 2),
                test_from=datetime(2022, 1, 4),
                execution_scope="FIXTURE_ONLY",
                **SETTINGS,
            )
            expected_rows = {split: 256 for split in ("train", "validation", "test")}
            if result["status"] != "VERIFIED" or result["rows"] != expected_rows:
                raise ValueError("Synthetic reference dataset row contract changed")
            if pq.read_metadata(dataset / "alerts.parquet").num_rows != ALERT_ROWS:
                raise ValueError("Synthetic candidate count changed")
            base = {
                "assembly_index": "fixture/index.json",
                "policy": "fixture/policy.json",
                "source": source.id,
                "scope": "FIXTURE_ONLY",
                "settings": SETTINGS,
                "repetitions": 2,
            }
            plans = {
                "preprocess": dict(
                    base,
                    mode="preprocess",
                    train_before="2022-01-02T00:00:00",
                    test_from="2022-01-04T00:00:00",
                    cases=[
                        {"name": "decode-1", "overrides": {}},
                        {"name": "decode-64", "overrides": {"decode_batch_rows": 64}},
                    ],
                ),
                "serialization": dict(
                    base,
                    mode="serialization",
                    dataset="reference-dataset",
                    cases=[
                        {"name": form, "format": form, "overrides": {}}
                        for form in ("csv", "json", "parquet", "three")
                    ],
                ),
            }
            comparisons = {}
            for mode, plan in plans.items():
                progress("synthetic_benchmark/" + mode, unit="isolated_cases")
                path = output / f"{mode}.plan.json"
                write_json(path, plan)
                comparisons[mode] = run_plan(path, output / mode)
            after = _inventory(output / "fixture")
            write_json(output / "fixture_after.json", after)
            if before != after:
                raise ValueError("Synthetic input changed during measurement")
            summary = {
                "status": "BOUNDED_SYNTHETIC_BENCHMARK_COMPLETED",
                "source_rows": SOURCE_ROWS,
                "candidate_alerts": ALERT_ROWS,
                "eligible_rows": expected_rows,
                "source_parts": 16,
                "repetitions": 2,
                "preprocess_cases": 4,
                "serialization_cases": 8,
                "settings": SETTINGS,
                "comparisons": comparisons,
                "reference_dataset_fingerprint": read_json(dataset / "manifest.json")[
                    "dataset_fingerprint"
                ],
                "production_source_read": False,
                "model_training_executed": False,
                "limitation": "Synthetic microbenchmark only; no production speedup/capacity/ETA conclusion",
            }
            write_json(output / "summary.json", summary)
            reporter.finish("BUILT")
            return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = run(args.output)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0
    except KeyboardInterrupt:
        return cancelled_result()
    except Exception as exc:
        print(json.dumps({"status": "FAILED", "error": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
