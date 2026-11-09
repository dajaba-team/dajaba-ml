"""완료된 M5 출력을 독립적으로 다시 읽고 모든 원천 행을 대조합니다."""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyarrow as pa
import pyarrow.compute as _compute
import pyarrow.parquet as pq

from .preprocessing import Preprocessor
from .schema import FORBIDDEN
from .storage import load_index, load_source, read_json, sha256, write_json

pc = cast(Any, _compute)


def verify_source(folder: Path, assembly_run: Path) -> dict:
    manifest = read_json(folder / "manifest.json")
    source = load_source(assembly_run, manifest["source_id"])
    if manifest["status"] != "COMPLETED":
        raise ValueError("Only completed output may be verified")
    expected = {name: [] for name in ["transaction_uid", "ts", "source_positive"]}
    for i in range(len(source.parts("transactions"))):
        transactions = pq.read_table(
            source.part_path("transactions", i), columns=["transaction_uid", "ts"]
        )
        labels = pq.read_table(source.part_path("labels", i), columns=["source_positive"])
        for name in expected:
            expected[name].append(
                (labels if name == "source_positive" else transactions).column(name)
            )
    original = {
        name: pa.chunked_array([chunk for col in columns for chunk in col.chunks])
        for name, columns in expected.items()
    }
    count = source.metadata["input_rows"]
    seen = np.zeros(count, dtype=np.uint8)
    processor = Preprocessor.load(folder / "preprocessing.json")
    state = processor.state
    assert state is not None
    if (
        state["feature_columns"] != manifest["feature_columns"]
        or set(state["feature_columns"]) & FORBIDDEN
    ):
        raise ValueError("Model feature allowlist mismatch")
    if any("_1h" in name or "_7d" in name for name in state["feature_columns"]):
        raise ValueError("Diagnostic windows entered the primary model")
    refit = Preprocessor(standardize=state["standardize"])
    # 보존된 어휘 중 가장 긴 것으로 실제 학습 상한을 복원합니다.
    # 변경 가능한 외부 설정이나 validation/test 범주는 읽지 않습니다.
    refit.max_categories = max(1, max(len(v["mapping"]) for v in state["categorical"].values()))
    first, second = manifest["split_plan"]["train_before"], manifest["split_plan"]["test_from"]
    bounds, report = {}, {}
    first, second = datetime.fromisoformat(first), datetime.fromisoformat(second)
    for split in ["train", "validation", "test"]:
        rows = positive = 0
        low = high = None
        for part in manifest["splits"][split]["parts"]:
            tables = {}
            for kind, digest in [
                ("raw", "raw_sha256"),
                ("targets", "target_sha256"),
                ("processed", "processed_sha256"),
            ]:
                path = (folder / kind / split / part["name"]).resolve()
                if path.parent != (folder / kind / split).resolve() or sha256(path) != part[digest]:
                    raise ValueError(f"Output changed or unsafe path: {path}")
                tables[kind] = pq.read_table(path)
            raw, targets, matrix = tables["raw"], tables["targets"], tables["processed"]
            if any(len(t) != part["rows"] for t in tables.values()):
                raise ValueError("Part row count mismatch")
            ids = targets.column("source_row_id").to_numpy()
            if (
                (ids < 0).any()
                or (ids >= count).any()
                or len(np.unique(ids)) != len(ids)
                or seen[ids].any()
            ):
                raise ValueError("Duplicate or invalid original row")
            seen[ids] = 1
            indices = pa.array(ids)
            for name in ["transaction_uid", "ts", "source_positive"]:
                expected_values = pc.cast(
                    pc.take(original[name], indices), targets.schema.field(name).type, safe=True
                )
                if not targets.column(name).equals(expected_values):
                    raise ValueError(f"Original row/label correspondence changed: {name}")
            for name in ["transaction_uid", "ts", "source_row_id"]:
                if not raw.column(name).equals(targets.column(name)):
                    raise ValueError("Raw/target row order mismatch")
            if "source_positive" in raw.schema.names or not matrix.equals(processor.transform(raw)):
                raise ValueError("Features changed, leaked target or frozen transform mismatch")
            if split == "train":
                refit.partial_fit(raw, split="train")
            minimum, maximum = pc.min_max(targets.column("ts")).as_py().values()
            low = min(low, minimum) if low is not None else minimum
            high = max(high, maximum) if high is not None else maximum
            rows += len(raw)
            positive += pc.sum(targets.column("source_positive")).as_py()
        if (
            not rows
            or rows != manifest["splits"][split]["rows"]
            or positive != manifest["splits"][split]["positive"]
        ):
            raise ValueError("Split count/target reconciliation failed")
        if low is None or high is None:
            raise ValueError("Missing chronological bounds")
        bounds[split] = low, high
        report[split] = dict(
            rows=rows, positive=positive, min_ts=low.isoformat(), max_ts=high.isoformat()
        )
    if (
        not seen.all()
        or not bounds["train"][1]
        < first
        <= bounds["validation"][0]
        <= bounds["validation"][1]
        < second
        <= bounds["test"][0]
    ):
        raise ValueError("Missing source rows or overlapping chronological splits")
    if refit.freeze() != state or state["fit_rows"] != report["train"]["rows"]:
        raise ValueError("Saved preprocessing differs from independent train-only refit")
    result = dict(
        status="VERIFIED",
        rows=count,
        splits=report,
        feature_count=len(state["feature_columns"]),
        original_row_labels_exact=True,
        frozen_transform_exact=True,
        train_only_refit_exact=True,
        duplicate_rows=0,
    )
    write_json(folder / "verification.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument(
        "--assembly-index",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    run = read_json(args.run / "run.json")
    if run["status"] != "COMPLETED":
        raise ValueError("Preparation run is not completed")
    index_path = (
        args.assembly_index
        or Path(
            run.get(
                "assembly_index",
                Path(__file__).resolve().parents[3] / "artifacts/redesign/phase3/LATEST.json",
            )
        )
    ).resolve()
    if sha256(index_path) != run["assembly_index_sha256"]:
        raise ValueError("Assembly input index differs from this preparation run")
    load_index(index_path, list(run["sources"]))
    index = read_json(index_path)
    for sid in run["sources"]:
        result = verify_source(
            args.run / sid, (index_path.parent / index["ml_runs"][sid]).resolve()
        )
        print(f"{sid}: {result['status']} {result['rows']:,} rows", flush=True)


if __name__ == "__main__":
    main()
