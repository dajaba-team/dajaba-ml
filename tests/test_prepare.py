import csv
import json
import math
import sys
from datetime import timedelta
from importlib.resources import files
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from assemble_db.config import load as load_assembly
from assemble_db.pipeline import execute as assemble
from dbcheck.config import CONTRACT_VERSION
from dbcheck.config import load as load_check
from dbcheck.distribution import IBM, SAML
from dbcheck.pipeline import execute as check

from aml_ml.prepare import execute, load_config
from aml_ml.storage import load_source
from aml_ml.verify import main as verify_main
from aml_ml.verify import verify_source


@pytest.fixture
def raw_preparation_config(tmp_path):
    default = Path(__file__).parent / "fixtures" / "legacy" / "default.json"
    return json.loads(default.read_text(encoding="utf-8")), tmp_path / "config.json"


@pytest.mark.parametrize("standardize", [True, False])
def test_load_config_preserves_boolean_standardize(raw_preparation_config, standardize):
    data, path = raw_preparation_config
    data["standardize"] = standardize
    path.write_text(json.dumps(data), encoding="utf-8")
    assert load_config(path).standardize is standardize


@pytest.mark.parametrize("standardize", ["false", "true", "", 0, 1, 0.0, 1.0, None, [], {}])
def test_load_config_rejects_nonboolean_standardize(raw_preparation_config, standardize):
    data, path = raw_preparation_config
    data["standardize"] = standardize
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="standardize must be a boolean"):
        load_config(path)


def test_load_config_missing_standardize_keeps_key_error(raw_preparation_config):
    data, path = raw_preparation_config
    del data["standardize"]
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(KeyError, match="standardize"):
        load_config(path)


@pytest.fixture
def prepared_input(tmp_path):
    inputs = tmp_path / "input"
    inputs.mkdir()
    rows = []
    subjects = [
        ("001", "A", "2", "B"),
        ("001", "A", "2", "B"),
        ("001", "A", "001", "A"),
        ("2", "B", "001", "A"),
        ("001", "A", "001", "C"),
        ("001", "A", "001", "C"),
        ("001", "C", "001", "A"),
        ("001", "C", "001", "A"),
        ("001", "A", "2", "B"),
        ("001", "A", "2", "B"),
        ("2", "B", "001", "C"),
        ("2", "B", "001", "C"),
    ]
    for i, (bank, account, other_bank, other) in enumerate(subjects):
        rows.append(
            [
                f"2022/09/01 {i // 2:02d}:00",
                bank,
                account,
                other_bank,
                other,
                str(10 + i),
                "US Dollar",
                str(10 + i),
                "US Dollar",
                "Wire" if i < 8 else "ACH",
                str(i % 2),
            ]
        )
    for path, header, data in [
        (inputs / "tx.csv", IBM, rows),
        (
            inputs / "accounts.csv",
            ["Bank Name", "Bank ID", "Account Number", "Entity ID", "Entity Name"],
            [
                ["Bank", "1", "A", "EA", "A"],
                ["Bank", "2", "B", "EB", "B"],
                ["Bank", "1", "C", "EC", "C"],
            ],
        ),
    ]:
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(data)
    (tmp_path / "currencies.json").write_bytes(
        files("aml_ml").joinpath("currencies.json").read_bytes()
    )
    (tmp_path / "check.toml").write_text(
        f'''schema_version="{CONTRACT_VERSION}"
output_root="reports"
currency_map="currencies.json"
[[sources]]
id="ibm_hi_small"
kind="ibm"
path="input/tx.csv"
accounts="input/accounts.csv"
label_semantics="AMLWORLD_ILLICIT_FLOW"
''',
        encoding="utf-8",
    )
    validation, ok = check(load_check(tmp_path / "check.toml"))
    assert ok
    report = json.loads((validation / "ibm_hi_small/validation_report.json").read_text())
    warnings = [r["rule"] for r in report["checks"] if r["status"] == "WARN"]
    (tmp_path / "assembly.toml").write_text(
        f"""schema_version="dajaba-assembly-v2"
dbcheck_config="check.toml"
output_root="assembled"
batch_rows=3
acknowledged_warnings={json.dumps(warnings)}
""",
        encoding="utf-8",
    )
    run, ok = assemble(load_assembly(tmp_path / "assembly.toml"), validation, purpose="ml")
    assert ok
    (tmp_path / "index.json").write_text(
        json.dumps({"ml_runs": {"ibm_hi_small": str(run)}}), encoding="utf-8"
    )
    config = {
        "version": "aml-preparation-v1",
        "assembly_index": "index.json",
        "output_root": "features",
        "partitions": 2,
        "batch_rows": 3,
        "train_fraction": 0.6,
        "validation_fraction": 0.2,
        "primary_windows": ["24h"],
        "auxiliary_windows": ["1h", "7d"],
        "exclude_self_transfers": True,
        "standardize": False,
        "max_categories": 128,
        "prediction_unit": "TRANSACTION_AUXILIARY",
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return load_config(tmp_path / "config.json"), run


def test_real_pipeline_preserves_rows_and_labels_excludes_ties_self_and_future(
    prepared_input, monkeypatch
):
    config, _ = prepared_input
    root, ok = execute(config)
    assert ok
    folder = root / "ibm_hi_small"
    manifest = json.loads((folder / "manifest.json").read_text())
    assert (
        manifest["input_rows"] == 12 and sum(v["rows"] for v in manifest["splits"].values()) == 12
    )
    assert sum(v["positive"] for v in manifest["splits"].values()) == 6
    raw = (
        pa.concat_tables([pq.read_table(p) for p in (folder / "raw").rglob("*.parquet")])
        .to_pandas()
        .set_index("source_row_id")
    )
    assert "source_positive" not in raw.columns and "source_typology" not in raw.columns
    assert raw.loc[0, "sender_out_count_24h"] == raw.loc[1, "sender_out_count_24h"] == 0
    # 두 시간 시점에는 앞선 출금 관측 두 개가 있고, 한 시간 시점의 자기 이체는 제외됩니다.
    assert raw.loc[4, "sender_out_count_24h"] == 2 and raw.loc[4, "sender_in_count_24h"] == 1
    assert raw.loc[4, "sender_out_amount_log1p_24h"] == pytest.approx(math.log1p(21))
    for info in manifest["splits"].values():
        assert info["rows"] > 0
    target_splits = {}
    for split in ["train", "validation", "test"]:
        target = pa.concat_tables(
            [pq.read_table(p) for p in (folder / "targets" / split).glob("*.parquet")]
        ).to_pandas()
        for ts in target.ts:
            assert ts not in target_splits or target_splits[ts] == split
            target_splits[ts] = split
        for part in manifest["splits"][split]["parts"]:
            matrix = pq.read_table(folder / "processed" / split / part["name"])
            assert matrix.schema.names == manifest["feature_columns"]
            assert not {"source_row_id", "transaction_uid", "source_positive", "ts"} & set(
                matrix.schema.names
            )
            assert not any("_1h" in name or "_7d" in name for name in matrix.schema.names)
    assert manifest["primary_alert_dataset_generated"] is False
    run_metadata = json.loads((root / "run.json").read_text())
    assert run_metadata["assembly_index"] == str(config.index)
    monkeypatch.setattr(sys, "argv", ["aml_ml.verify", str(root)])
    verify_main()  # 과거 전역 기본값 대신 이번 실행에 기록된 인덱스를 사용합니다.


def test_causal_features_match_independent_history_across_all_splits(prepared_input):
    config, assembly_run = prepared_input
    root, ok = execute(config)
    assert ok
    folder = root / "ibm_hi_small"
    source = load_source(assembly_run, "ibm_hi_small")
    original = [
        row
        for i in range(len(source.parts("transactions")))
        for row in pq.read_table(source.part_path("transactions", i)).to_pylist()
    ]
    by_id = {row["source_row_id"]: row for row in original}
    seen = set()
    for split in ("train", "validation", "test"):
        parts = list((folder / "raw" / split).glob("*.parquet"))
        assert parts
        for part in parts:
            for actual in pq.read_table(part).to_pylist():
                assert actual["source_row_id"] not in seen
                seen.add(actual["source_row_id"])
                current = by_id[actual["source_row_id"]]
                for side, account_column, currency_column in (
                    ("sender", "from_account_key", "currency_paid_bucket"),
                    ("receiver", "to_account_key", "currency_received_bucket"),
                ):
                    account, unit = current[account_column], current[currency_column]
                    for window, hours in (("1h", 1), ("24h", 24)):
                        past = [
                            row
                            for row in original
                            if current["ts"] - timedelta(hours=hours) <= row["ts"] < current["ts"]
                            and not row["is_self_transfer"]
                        ]
                        for direction, account_key, other_key, amount, currency in (
                            (
                                "out",
                                "from_account_key",
                                "to_account_key",
                                "amount_paid",
                                "currency_paid_bucket",
                            ),
                            (
                                "in",
                                "to_account_key",
                                "from_account_key",
                                "amount_received",
                                "currency_received_bucket",
                            ),
                        ):
                            matches = [row for row in past if row[account_key] == account]
                            assert actual[f"{side}_{direction}_count_{window}"] == len(matches)
                            assert actual[f"{side}_{direction}_counterparties_{window}"] == len(
                                {row[other_key] for row in matches}
                            )
                            total = sum(row[amount] for row in matches if row[currency] == unit)
                            assert actual[
                                f"{side}_{direction}_amount_log1p_{window}"
                            ] == pytest.approx(math.log1p(float(total)))
    assert seen == set(by_id)


def test_changed_input_part_is_rejected(prepared_input):
    config, run = prepared_input
    source = load_source(run, "ibm_hi_small")
    part = source.part_path("transactions", 0)
    with part.open("ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError, match="changed"):
        source.part_path("transactions", 0)


def test_independent_reload_verifies_then_rejects_modified_output(prepared_input):
    config, assembly_run = prepared_input
    root, ok = execute(config)
    assert ok
    folder = root / "ibm_hi_small"
    assert verify_source(folder, assembly_run)["status"] == "VERIFIED"
    part = next((folder / "processed/train").glob("*.parquet"))
    with part.open("ab") as stream:
        stream.write(b"modified")
    with pytest.raises(ValueError, match="changed"):
        verify_source(folder, assembly_run)


def test_saml_preserves_topology_without_inventing_directional_money(prepared_input):
    config, _ = prepared_input
    parent = config.path.parent
    with (parent / "input/saml.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(SAML)
        writer.writerows(
            [
                [
                    f"{i:02d}:00:00",
                    "2022-01-01",
                    "000A",
                    "000B",
                    str(10 + i),
                    "US Dollar",
                    "Euro" if i == 3 else "US Dollar",
                    "UK",
                    "USA",
                    "ACH",
                    str(i % 2),
                    "Layering" if i % 2 else "Normal",
                ]
                for i in range(8)
            ]
        )
    (parent / "check.toml").write_text(
        f'''schema_version="{CONTRACT_VERSION}"
output_root="reports"
currency_map="currencies.json"
[[sources]]
id="saml_d"
kind="saml"
path="input/saml.csv"
label_semantics="SOURCE_DESIGNATED_SUSPICIOUS"
amount_precision=38
amount_scale=16
''',
        encoding="utf-8",
    )
    validation, ok = check(load_check(parent / "check.toml"))
    assert ok
    report = json.loads((validation / "saml_d/validation_report.json").read_text())
    warnings = [r["rule"] for r in report["checks"] if r["status"] == "WARN"]
    (parent / "assembly.toml").write_text(
        f"""schema_version="dajaba-assembly-v2"
dbcheck_config="check.toml"
output_root="assembled"
batch_rows=3
acknowledged_warnings={json.dumps(warnings)}
""",
        encoding="utf-8",
    )
    assembly_run, ok = assemble(load_assembly(parent / "assembly.toml"), validation, purpose="ml")
    assert ok
    config.index.write_text(
        json.dumps({"ml_runs": {"saml_d": str(assembly_run)}}), encoding="utf-8"
    )
    root, ok = execute(config)
    assert ok
    folder = root / "saml_d"
    raw = (
        pa.concat_tables([pq.read_table(p) for p in (folder / "raw").rglob("*.parquet")])
        .to_pandas()
        .set_index("source_row_id")
    )
    assert raw["observed_paid_log1p"].isna().all()
    assert raw["observed_received_log1p"].isna().all()
    assert raw["sender_out_amount_log1p_24h"].isna().all()
    assert raw.loc[3, "observed_single_log1p"] != raw.loc[3, "observed_single_log1p"]
    assert raw.loc[4, "observed_single_log1p"] == pytest.approx(math.log1p(14))
    assert raw.loc[4, "sender_out_count_24h"] == 4
    assert verify_source(folder, assembly_run)["rows"] == 8
