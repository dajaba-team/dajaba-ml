"""Recover only a verified closed R01/R02 candidate prefix without changing its source."""

import duckdb

from .offline_contract import source_fingerprint
from .storage import read_json, sha256, write_json


def restore_candidate_prefix(source, policy, previous, output):
    """Reuse only the verified R01/R02 prefix of this task's pre-metadata OOM.

    Copy, never move or mutate, the failed database. The loader's immutable input
    manifests are rechecked; unknown tables/stages/policies fail closed.
    """
    import shutil

    previous = previous.resolve()
    failed = read_json(previous / "failure.json")
    if failed.get("error_type") != "OutOfMemoryException":
        raise ValueError("Recovery accepts only the recorded pre-metadata memory failure")
    prior_policy = read_json(previous / "policy.json")
    if (previous / "manifest.json").exists():
        raise ValueError("Old output is already complete; do not recover it as a prefix")
    # Only raw R01/R02 candidates are reused. Uncomputed R06 status/version and
    # later eligibility metadata may change, but reused-rule meaning must not.
    prefix = ("R01", "R02")
    if (
        prior_policy.get("semantics_version") != policy.get("semantics_version")
        or prior_policy.get("rule_version_origin") != policy.get("rule_version_origin")
        or not set(prefix).issubset(policy.get("enabled_rules", []))
        or not set(prefix).issubset(prior_policy.get("enabled_rules", []))
        or any(
            prior_policy.get("rules", {}).get(rule) != policy.get("rules", {}).get(rule)
            or prior_policy.get("rule_versions", {}).get(rule)
            != policy.get("rule_versions", {}).get(rule)
            for rule in prefix
        )
    ):
        raise ValueError("Recovery R01/R02 params, versions, or semantics differ")
    path = previous / "preprocessing.duckdb"
    if path.with_suffix(".duckdb.wal").exists():
        raise ValueError("Recovery requires a closed failed database without a WAL")
    digest = sha256(path)
    with duckdb.connect(str(path), read_only=True) as old:
        tables = {r[0] for r in old.execute("SHOW TABLES").fetchall()}
        if tables != {"input_labels", "input_transactions", "labels", "rule_candidates", "tx"}:
            raise ValueError("Unexpected recovery stage; do not guess partial-rule completeness")
        rules = dict(
            old.execute("SELECT rule_id,count(*) FROM rule_candidates GROUP BY rule_id").fetchall()
        )
        if set(rules) != {"R01_FAN_OUT_24H", "R02_FAN_IN_24H"}:
            raise ValueError("Only completed R01/R02 prefix can be reused")
        expected = source.metadata["input_rows"]
        for table in ("tx", "labels"):
            result = old.execute(
                f"""
                SELECT count(*),count(DISTINCT source_row_id),min(source_row_id),
                  max(source_row_id),count(*) FILTER (
                    WHERE transaction_uid IS DISTINCT FROM ? || ':' || source_row_id)
                FROM {table}
            """,
                [source.metadata["source_release"]],
            ).fetchone()
            if result != (expected, expected, 0, expected - 1, 0):
                raise ValueError("Recovery source identity mismatch")
    # Every input part is independently hash-checked; no expensive candidate rerun.
    for which in ("transactions", "labels"):
        for i in range(len(source.parts(which))):
            source.part_path(which, i)
    shutil.copyfile(path, output / "preprocessing.duckdb")
    if sha256(output / "preprocessing.duckdb") != digest or sha256(path) != digest:
        raise ValueError("Recovery copy or original database changed")
    receipt = dict(
        prior_folder=str(previous),
        prior_database_sha256=digest,
        source_fingerprint=source_fingerprint(source),
        reused_rules=rules,
        original_preserved=True,
    )
    write_json(output / "recovery_receipt.json", receipt)
    return rules
