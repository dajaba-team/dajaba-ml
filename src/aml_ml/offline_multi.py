"""Isolated multi-rule preprocessing: immutable sources, new output, no model training.

SQL candidate generation is label-blind. Shared 24h feature and train-only
preprocessing kernels are retained. All rule candidates use versioned, actual-evidence-disjoint temporal eligibility.
This new layout intentionally does not masquerade as the old R02-only dataset.
"""

import argparse
import copy
import json
import time
from datetime import datetime
from pathlib import Path

import duckdb

from .offline_contract import source_fingerprint
from .offline_features import FEATURE_NAMES, OFFLINE_FEATURE_VERSION
from .offline_multi_dataset import RAW_SCHEMA as RAW_SCHEMA
from .offline_multi_dataset import export_dataset as export_dataset
from .offline_multi_dataset import feature_table as feature_table
from .offline_multi_eligibility import POLICY_VERSION
from .offline_multi_io import BATCH as BATCH
from .offline_multi_io import append as append
from .offline_multi_io import export_query as export_query
from .offline_multi_io import load_tables as load_tables
from .offline_multi_io import query_batches as query_batches
from .offline_multi_metadata import FACT_SCHEMA as FACT_SCHEMA
from .offline_multi_metadata import META_SCHEMA as META_SCHEMA
from .offline_multi_metadata import candidate_metadata as candidate_metadata
from .offline_multi_progress import log, phase
from .offline_multi_recovery import restore_candidate_prefix as restore_candidate_prefix
from .storage import load_index, read_json, sha256, write_json

LAYOUT = "MULTI_RULE_SQL_SOURCE_UID_V1"


def build_source(
    source,
    policy,
    output,
    *,
    train_before,
    test_from,
    memory_limit="4GB",
    threads=2,
    recover_from=None,
):
    from .offline_rule_sql import generate_candidates

    if policy["eligibility_policy"]["version"] != POLICY_VERSION:
        raise ValueError("Explicit supported eligibility policy version required")
    if output.exists():
        raise FileExistsError("Use a new directory; old results must remain untouched")
    output.mkdir(parents=True)
    started = time.monotonic()
    binding = source_fingerprint(source)
    effective = copy.deepcopy(policy)
    unsupported = []
    if source.kind != "ibm" and "R07" in effective["enabled_rules"]:
        effective["enabled_rules"].remove("R07")
        unsupported.append(
            dict(rule_id="R07_STRUCTURING", reason="NO_CONFIRMED_DIRECTIONAL_AMOUNT_RECEIVED")
        )
    write_json(output / "policy.json", effective)
    try:
        with phase(output, "recovery_preflight", source_id=source.id):
            reused = (
                restore_candidate_prefix(source, effective, recover_from, output)
                if recover_from else None
            )
    except BaseException as exc:
        write_json(output / "failure.json", {
            "status": "FAILED", "stage": "RECOVERY_PREFLIGHT",
            "error_type": type(exc).__name__, "message": str(exc),
            "model_training_executed": False, "partial_outputs_preserved": True,
        })
        raise
    con = duckdb.connect(str(output / "preprocessing.duckdb"))
    try:
        con.execute("SET memory_limit=?", [memory_limit])
        con.execute("SET threads=?", [threads])
        con.execute("SET preserve_insertion_order=false")
        con.execute("SET temp_directory=?", [str(output / "sql_spill")])
        con.execute("SET max_temp_directory_size='32GB'")
        if reused:
            con.execute("ALTER TABLE rule_candidates RENAME TO rule_candidates_kept")
            remaining = copy.deepcopy(effective)
            remaining["enabled_rules"] = [
                rule for rule in effective["enabled_rules"] if rule not in ("R01", "R02")
            ]
            log(output, "recover_candidates", reused_rules=reused,
                remaining_rules=remaining["enabled_rules"])
            with phase(output, "generate_candidates", source_id=source.id):
                rule_counts = generate_candidates(
                    con, remaining, source.id, source.metadata["source_release"],
                    progress=lambda stage, **items: log(output, stage, **items),
                )
            con.execute("INSERT INTO rule_candidates SELECT * FROM rule_candidates_kept")
            con.execute("DROP TABLE rule_candidates_kept")
            rule_counts.update(reused)
        else:
            with phase(output, "load_and_verify_source", source_id=source.id):
                load_tables(con, source, output)
            with phase(output, "generate_candidates", source_id=source.id,
                       enabled_rules=effective["enabled_rules"]):
                rule_counts = generate_candidates(
                    con, effective, source.id, source.metadata["source_release"],
                    progress=lambda stage, **items: log(output, stage, **items),
                )
        log(output, "candidates_completed", rule_counts=rule_counts)
        with phase(output, "metadata_evidence_and_split_checks", source_id=source.id):
            eligibility, support = candidate_metadata(
                con, source, effective, output, train_before, test_from
            )
        log(
            output,
            "metadata_completed",
            alerts=eligibility["total_alerts"],
            eligible=eligibility["eligible_alerts"],
        )
        with phase(output, 'feature_history', source_id=source.id):
            feature_table(con, source, output)
        quality = dict(
            eligibility=eligibility,
            support_by_rule_split=support,
            unsupported_rules=unsupported,
            model_training_executed=False,
        )
        with phase(output, 'export_and_format_verification', source_id=source.id):
            result = export_dataset(con, source, output, quality)
        if source_fingerprint(source) != binding:
            raise ValueError("Source manifests changed during generation")
        with phase(output, 'database_checkpoint', source_id=source.id):
            con.execute("CHECKPOINT")
        manifest = dict(
            **result,
            layout=LAYOUT,
            source_id=source.id,
            source_release=source.metadata["source_release"],
            source_fingerprint=binding,
            input_rows=source.metadata["input_rows"],
            policy=effective,
            rule_counts=rule_counts,
            unsupported_rules=unsupported,
            support_by_rule_split=support,
            train_before=train_before.isoformat(),
            test_from=test_from.isoformat(),
            eligibility_policy=effective["eligibility_policy"],
            feature_version=OFFLINE_FEATURE_VERSION,
            raw_feature_columns=list(FEATURE_NAMES),
            id_namespace="OFFLINE_SOURCE_UID_NOT_BACKEND_TRANSACTION_ID",
            model_training_executed=False,
            service_id_mapping_verified=False,
            verification_scope="INPUT_HASHES_IDENTITIES_EVIDENCE_WINDOWS_SPLIT_ISOLATION_FORMAT_ROUNDTRIP",
            full_independent_source_replay=False,
            elapsed_seconds=round(time.monotonic() - started, 3),
            code_sha256={
                name: sha256(Path(__file__).parent / name)
                for name in (
                    "offline_multi.py",
                    "offline_multi_io.py",
                    "offline_multi_metadata.py",
                    "offline_multi_dataset.py",
                    "offline_multi_recovery.py",
                    "offline_rule_sql.py",
                    "offline_multi_eligibility.py",
                    "offline_multi_progress.py",
                    "history.py",
                    "offline_features.py",
                    "preprocessing.py",
                    "offline_serialization.py",
                )
            },
        )
        con.close()
        con = None
        with phase(output, "hash_final_outputs", source_id=source.id):
            manifest["files"] = {
                p.name: sha256(p)
                for p in sorted(output.iterdir())
                if p.is_file() and p.name not in (
                    "progress.json", "progress.json.tmp", "progress.jsonl", "manifest.json"
                )
            }
        write_json(output / "manifest.json", manifest)
        log(
            output,
            "completed",
            status=result["status"],
            elapsed_seconds=manifest["elapsed_seconds"],
        )
        return manifest
    except BaseException as exc:
        write_json(
            output / "failure.json",
            dict(
                status="FAILED",
                error_type=type(exc).__name__,
                message=str(exc),
                model_training_executed=False,
                partial_outputs_preserved=True,
            ),
        )
        raise
    finally:
        if con is not None:
            con.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", action="append")
    parser.add_argument("--recover-source", action="append", default=[])
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    config = read_json(config_path)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Output root already exists; choose a fresh run")
    sources = load_index((config_path.parent / config["assembly_index"]).resolve(), args.source)
    output.mkdir(parents=True)
    write_json(output / "execution.json", config)
    recovery = dict(item.split("=", 1) for item in args.recover_source)
    if set(recovery) - {s.id for s in sources}:
        raise ValueError("Unknown recovery source")
    results = {}
    try:
        for source_number, source in enumerate(sources, 1):
            log(output, 'source_started', source_id=source.id,
                source_number=source_number, source_count=len(sources))
            split = next(s for s in config["sources"] if s["id"] == source.id)
            result = build_source(
                source,
                config["policy"],
                output / source.id,
                train_before=datetime.fromisoformat(split["train_before"]),
                test_from=datetime.fromisoformat(split["test_from"]),
                memory_limit=config["memory_limit"],
                threads=config["threads"],
                recover_from=Path(recovery[source.id]) if source.id in recovery else None,
            )
            results[source.id] = dict(
                status=result["status"],
                rule_counts=result["rule_counts"],
                rows=result["rows"],
                unsupported_rules=result["unsupported_rules"],
            )
            write_json(
                output / "run.json",
                dict(status="RUNNING", sources=results, model_training_executed=False),
            )
            log(output, "source_completed", source_id=source.id, status=result["status"])
            if result["status"] != "COMPLETED":
                raise RuntimeError(f"{source.id} is {result['status']}; later sources not started")
        write_json(
            output / "run.json",
            dict(
                status=(
                    "COMPLETED"
                    if all(r["status"] == "COMPLETED" for r in results.values())
                    else "COMPLETED_WITH_NOT_READY"
                ),
                sources=results,
                model_training_executed=False,
            ),
        )
    except BaseException as exc:
        write_json(
            output / "run.json",
            dict(
                status="FAILED",
                sources=results,
                error_type=type(exc).__name__,
                message=str(exc),
                model_training_executed=False,
            ),
        )
        raise
    print(json.dumps(results, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
