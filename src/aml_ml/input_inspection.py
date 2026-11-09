"""전체 검증 없이 manifest·경로·첫 파트의 스키마만 제한적으로 확인합니다."""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .storage import load_index


def inspect_inputs(index: Path, selected: list[str] | None = None) -> dict:
    sources = {}
    for source in load_index(index, selected):
        tables = {}
        for key, manifest in (
            ("ml_transactions", source.transactions),
            ("ml_labels", source.labels),
        ):
            folder = source.folder / source.metadata["datasets"][key]["path"]
            with pa.memory_map(str(folder / "schema.arrow"), "r") as stream:
                declared = pa.ipc.read_schema(stream)
            required = {
                "transaction_uid": pa.string(),
                "source_row_id": pa.int64(),
                **(
                    {
                        "ts": pa.timestamp("us"),
                        "from_account_key": pa.string(),
                        "to_account_key": pa.string(),
                        "is_self_transfer": pa.bool_(),
                        "payment_format": pa.string(),
                        **(
                            {
                                "currency_paid_bucket": pa.string(),
                                "currency_received_bucket": pa.string(),
                            }
                            if source.kind == "ibm"
                            else {
                                "currency_paid_raw": pa.string(),
                                "currency_received_raw": pa.string(),
                                "money_operations_allowed": pa.bool_(),
                            }
                        ),
                    }
                    if key == "ml_transactions"
                    else {"source_positive": pa.int8()}
                ),
            }
            if any(
                name not in declared.names or declared.field(name).type != kind
                for name, kind in required.items()
            ):
                raise ValueError("Schema does not satisfy the current ML input contract")
            if key == "ml_transactions":
                amount_names = (
                    ["amount_paid", "amount_received"] if source.kind == "ibm" else ["amount"]
                )
                if any(
                    name not in declared.names or not pa.types.is_decimal(declared.field(name).type)
                    for name in amount_names
                ):
                    raise ValueError("ML amounts must retain exact Decimal types")
            paths = []
            for part in manifest["parts"]:
                files = [f for f in part["files"] if f["path"].endswith(".parquet")]
                if len(files) != 1:
                    raise ValueError("Expected one Parquet per part")
                path = (folder / files[0]["path"]).resolve()
                if path.parent != folder.resolve() or not path.is_file():
                    raise ValueError("Missing or unsafe ML part")
                paths.append(path)
            if not paths or not declared.equals(pq.read_schema(paths[0])):
                raise ValueError("Declared schema differs from first Parquet part")
            tables[key] = dict(rows=manifest["rows"], parts=len(paths), schema=str(declared))
        sources[source.id] = dict(
            folder=str(source.folder),
            source_release=source.metadata["source_release"],
            tables=tables,
        )
    return dict(
        status="INPUT_METADATA_AND_FIRST_SCHEMA_OK",
        full_data_verified=False,
        inspection_scope="MANIFESTS_ALL_PART_PATHS_DECLARED_SCHEMA_AND_FIRST_PARQUET_SCHEMA",
        sources=sources,
    )
