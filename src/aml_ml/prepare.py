"""디스크 분할로 과거 기반 거래 보조 데이터를 전처리하며, 모델 학습은 하지 않습니다."""

import bisect
import hashlib
import importlib.metadata
import logging
import math
import platform
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from .history import AccountHistory, Event
from .preprocessing import Preprocessor
from .schema import (
    ACCOUNT_FEATURES,
    CATEGORICAL,
    FEATURE_VERSION,
    NUMERIC,
    currency_bucket,
    epoch_us,
    log_amount,
)
from .storage import PartitionSink, SourceInput, load_index, read_json, sha256, write_json

LOG = logging.getLogger("aml_ml.prepare")
BASE_SCHEMA = pa.schema(
    [
        ("transaction_uid", pa.string()),
        ("source_row_id", pa.int64()),
        ("ts", pa.timestamp("us")),
        ("_target", pa.int8()),
    ]
    + [
        (n, pa.float64())
        for n in [
            "observed_paid_log1p",
            "observed_received_log1p",
            "observed_single_log1p",
            "self_transfer",
        ]
    ]
    + [(n, pa.string()) for n in CATEGORICAL]
)
SIDE_SCHEMA = pa.schema(
    [("source_row_id", pa.int64()), ("side", pa.int8())]
    + [(n, pa.float64()) for n in ACCOUNT_FEATURES]
)


@dataclass(frozen=True)
class Config:
    path: Path
    index: Path
    output: Path
    partitions: int
    batch_rows: int
    train: float
    validation: float
    standardize: bool
    max_categories: int


def load_config(path: Path) -> Config:
    path = path.resolve()
    data = read_json(path)
    allowed = {
        "version",
        "assembly_index",
        "output_root",
        "partitions",
        "batch_rows",
        "train_fraction",
        "validation_fraction",
        "primary_windows",
        "auxiliary_windows",
        "exclude_self_transfers",
        "standardize",
        "max_categories",
        "prediction_unit",
        "main_alert_training_status",
    }
    if set(data) - allowed or data.get("version") != "aml-preparation-v1":
        raise ValueError("Unsupported preparation configuration")
    if data.get("primary_windows") != ["24h"] or data.get("exclude_self_transfers") is not True:
        raise ValueError(
            "Current reviewed feature policy is 24h-primary with self transfers excluded"
        )
    if data.get("prediction_unit") != "TRANSACTION_AUXILIARY":
        raise ValueError("Confirmed alert data must use the separate alert contract")
    train, valid = data["train_fraction"], data["validation_fraction"]
    if type(train) not in (float, int) or type(valid) not in (float, int):
        raise ValueError("Chronological split fractions must be numbers")
    if not 0 < train < 1 or not 0 < valid < 1 or train + valid >= 1:
        raise ValueError("Invalid chronological split fractions")
    parts, batch = data["partitions"], data["batch_rows"]
    if (
        type(parts) is not int
        or not 1 <= parts <= 128
        or type(batch) is not int
        or not 1 <= batch <= 100000
    ):
        raise ValueError("Invalid bounded partition/batch size")
    maximum = data["max_categories"]
    if type(maximum) is not int or not 1 <= maximum <= 1024:
        raise ValueError("Invalid categorical cap")
    standardize = data["standardize"]
    if type(standardize) is not bool:
        raise ValueError("standardize must be a boolean")
    return Config(
        path,
        (path.parent / data["assembly_index"]).resolve(),
        (path.parent / data["output_root"]).resolve(),
        parts,
        batch,
        float(train),
        float(valid),
        standardize,
        maximum,
    )


def split_plan(source: SourceInput, config: Config) -> dict:
    paths = [
        str(source.part_path("transactions", i)) for i in range(len(source.parts("transactions")))
    ]
    histogram = (
        pl.scan_parquet(paths)
        .select("ts")
        .group_by("ts")
        .agg(pl.len().cast(pl.UInt64).alias("n"))
        .sort("ts")
        .collect(engine="streaming")
    )
    if histogram.height < 3:
        raise ValueError("At least three distinct timestamps are required")
    cumulative = histogram["n"].cum_sum()
    total = int(cumulative[-1])
    if total != source.metadata["input_rows"]:
        raise ValueError("Timestamp count mismatch")
    first = bisect.bisect_left(cumulative, math.ceil(total * config.train)) + 1
    first = max(1, min(first, histogram.height - 2))
    second = (
        bisect.bisect_left(cumulative, math.ceil(total * (config.train + config.validation))) + 1
    )
    second = max(first + 1, min(second, histogram.height - 1))
    counts = [
        int(cumulative[first - 1]),
        int(cumulative[second - 1] - cumulative[first - 1]),
        int(total - cumulative[second - 1]),
    ]
    return dict(
        method="WEIGHTED_TIMESTAMP_COHORT_60_20_20",
        label_blind=True,
        same_timestamp_atomic=True,
        train_before=histogram["ts"][first].isoformat(),
        test_from=histogram["ts"][second].isoformat(),
        coverage_start=histogram["ts"][0].isoformat(),
        source_end=histogram["ts"][-1].isoformat(),
        expected_rows=dict(zip(["train", "validation", "test"], counts)),
        actual_fractions={k: v / total for k, v in zip(["train", "validation", "test"], counts)},
        requested_fractions=dict(
            train=config.train,
            validation=config.validation,
            test=1 - config.train - config.validation,
        ),
        note="Whole timestamp cohorts are assigned together; time order, not random rows. Large cohorts can shift ratios.",
    )


def require_space(folder: Path):
    if shutil.disk_usage(folder).free < 5 * 1024**3:
        raise OSError("Preparation stopped: less than 5 GiB free")


def event_schema(source: SourceInput) -> pa.Schema:
    amount_type = (
        pq.read_schema(source.part_path("transactions", 0))
        .field("amount_paid" if source.kind == "ibm" else "amount")
        .type
    )
    return pa.schema(
        [
            ("account", pa.string()),
            ("other", pa.string()),
            ("ts_us", pa.int64()),
            ("source_row_id", pa.int64()),
            ("side", pa.int8()),
            ("amount", amount_type),
            ("unit", pa.string()),
            ("eligible", pa.bool_()),
        ]
    )


def partition_inputs(
    source: SourceInput, config: Config, folder: Path
) -> tuple[list[int], list[int]]:
    event_sink = PartitionSink(folder / "events", event_schema(source), config.partitions)
    base_sink = PartitionSink(folder / "base", BASE_SCHEMA, config.partitions)
    cache = {}
    processed = 0
    try:
        for i in range(len(source.parts("transactions"))):
            require_space(folder)
            tx = pq.read_table(source.part_path("transactions", i))
            labels = pq.read_table(
                source.part_path("labels", i),
                columns=["transaction_uid", "source_row_id", "source_positive"],
            )
            for key in ["transaction_uid", "source_row_id"]:
                if not tx.column(key).equals(labels.column(key)):
                    raise ValueError("Transaction/label rows are not aligned")
            targets = labels.column("source_positive").to_pylist()
            rows = tx.to_pylist()
            event_rows = []
            event_parts = []
            base_rows = []
            base_parts = []
            for record, target in zip(rows, targets):
                row_id = record["source_row_id"]
                if row_id != processed or target not in (0, 1):
                    raise ValueError("Original row sequence/target changed")
                processed += 1
                ts = epoch_us(record["ts"])
                if source.kind == "ibm":
                    paid, received = record["amount_paid"], record["amount_received"]
                    paid_unit, received_unit = (
                        record["currency_paid_bucket"],
                        record["currency_received_bucket"],
                    )
                    single = None
                else:
                    # 단일 Amount는 실제 방향별 금액 두 개의 근거가 아닙니다.
                    paid = received = None
                    paid_unit = currency_bucket(source.id, record["currency_paid_raw"])
                    received_unit = currency_bucket(source.id, record["currency_received_raw"])
                    single = record["amount"] if record["money_operations_allowed"] else None
                base_rows.append(
                    dict(
                        transaction_uid=record["transaction_uid"],
                        source_row_id=row_id,
                        ts=record["ts"],
                        _target=target,
                        observed_paid_log1p=log_amount(paid),
                        observed_received_log1p=log_amount(received),
                        observed_single_log1p=log_amount(single),
                        self_transfer=float(record["is_self_transfer"]),
                        payment_format=record["payment_format"],
                        currency_paid_bucket=paid_unit,
                        currency_received_bucket=received_unit,
                    )
                )
                base_parts.append(row_id % config.partitions)
                for side, account, other, amount, unit in [
                    (0, record["from_account_key"], record["to_account_key"], paid, paid_unit),
                    (
                        1,
                        record["to_account_key"],
                        record["from_account_key"],
                        received,
                        received_unit,
                    ),
                ]:
                    if account not in cache:
                        cache[account] = (
                            int.from_bytes(
                                hashlib.blake2b(account.encode(), digest_size=4).digest(), "big"
                            )
                            % config.partitions
                        )
                    event_rows.append(
                        dict(
                            account=account,
                            other=other,
                            ts_us=ts,
                            source_row_id=row_id,
                            side=side,
                            amount=amount,
                            unit=unit if amount is not None else None,
                            eligible=not record["is_self_transfer"],
                        )
                    )
                    event_parts.append(cache[account])
            event_sink.scatter_rows(event_rows, event_parts)
            base_sink.scatter_rows(base_rows, base_parts)
            if processed % 500000 < len(rows):
                LOG.info(
                    "%s: partitioned %s/%s",
                    source.id,
                    f"{processed:,}",
                    f"{source.metadata['input_rows']:,}",
                )
    finally:
        event_sink.close()
        base_sink.close()
    if processed != source.metadata["input_rows"]:
        raise ValueError("Input count changed")
    return event_sink.counts, base_sink.counts


def history_partitions(
    source: SourceInput, config: Config, folder: Path, counts: list[int], plan: dict
) -> list[int]:
    sink = PartitionSink(folder / "side_features", SIDE_SCHEMA, config.partitions)
    coverage = epoch_us(datetime.fromisoformat(plan["coverage_start"]))
    buffer = []
    buffer_parts = []

    def flush():
        if buffer:
            sink.scatter_rows(buffer, buffer_parts)
            buffer.clear()
            buffer_parts.clear()

    def process(group: list[dict], history: AccountHistory):
        if not group:
            return
        as_of = group[0]["ts_us"]
        history.expire(as_of)
        snapshots = {}
        for item in group:
            unit = item["unit"]
            if unit not in snapshots:
                snapshots[unit] = history.snapshot(as_of, unit, coverage)
            buffer.append(
                dict(
                    source_row_id=item["source_row_id"],
                    side=item["side"],
                    **dict(zip(ACCOUNT_FEATURES, snapshots[unit])),
                )
            )
            buffer_parts.append(item["source_row_id"] % config.partitions)
            if len(buffer) >= config.batch_rows:
                flush()
        # as_of와 정확히 같은 시각의 관측은 모든 조회를 마친 뒤에만 추가합니다.
        for item in group:
            if item["eligible"]:
                history.add(Event(as_of, item["side"], item["other"], item["amount"], item["unit"]))

    try:
        for index, count in enumerate(counts):
            if not count:
                continue
            require_space(folder)
            path = folder / "events" / f"part-{index:05d}.parquet"
            events = (
                pl.scan_parquet(path)
                .sort(["account", "ts_us", "source_row_id", "side"])
                .collect(engine="streaming")
                .to_arrow()
            )
            account = None
            timestamp = None
            group = []
            history = AccountHistory()
            for batch in events.to_batches(max_chunksize=config.batch_rows):
                for item in batch.to_pylist():
                    if item["account"] != account or item["ts_us"] != timestamp:
                        process(group, history)
                        group = []
                        if item["account"] != account:
                            history = AccountHistory()
                        account, timestamp = item["account"], item["ts_us"]
                    group.append(item)
            process(group, history)
            flush()
            LOG.info(
                "%s: causal account partition %s/%s (%s events)",
                source.id,
                index + 1,
                config.partitions,
                f"{count:,}",
            )
    finally:
        sink.close()
    if sum(sink.counts) != 2 * source.metadata["input_rows"]:
        raise ValueError("Side-feature row reconciliation failed")
    return sink.counts


def split_rows(
    source: SourceInput, config: Config, folder: Path, base_counts: list[int], plan: dict
) -> dict:
    first = datetime.fromisoformat(plan["train_before"])
    second = datetime.fromisoformat(plan["test_from"])
    stats: dict[str, dict[str, Any]] = {
        name: dict(rows=0, positive=0, parts=[]) for name in ["train", "validation", "test"]
    }
    for split in stats:
        (folder / "raw" / split).mkdir(parents=True)
        (folder / "targets" / split).mkdir(parents=True)
    for index, count in enumerate(base_counts):
        if not count:
            continue
        base = pl.read_parquet(folder / "base" / f"part-{index:05d}.parquet")
        side = pl.read_parquet(folder / "side_features" / f"part-{index:05d}.parquet")
        for code, prefix in [(0, "sender"), (1, "receiver")]:
            subset = (
                side.filter(pl.col("side") == code)
                .drop("side")
                .rename({n: f"{prefix}_{n}" for n in ACCOUNT_FEATURES})
            )
            if subset.height != base.height:
                raise ValueError("Missing sender/receiver features")
            base = base.join(subset, on="source_row_id", how="left", validate="1:1")
        base = base.sort(["ts", "source_row_id"])
        if base.filter(
            pl.col("sender_out_count_24h").is_null() | pl.col("receiver_out_count_24h").is_null()
        ).height:
            raise ValueError("A source row lost its causal feature query")
        conditions = {
            "train": pl.col("ts") < first,
            "validation": (pl.col("ts") >= first) & (pl.col("ts") < second),
            "test": pl.col("ts") >= second,
        }
        for split, condition in conditions.items():
            subset = base.filter(condition)
            if not len(subset):
                continue
            raw = subset.select(
                ["transaction_uid", "source_row_id", "ts", *NUMERIC, *CATEGORICAL]
            ).to_arrow()
            target = subset.select(
                [
                    "transaction_uid",
                    "source_row_id",
                    "ts",
                    pl.col("_target").alias("source_positive"),
                ]
            ).to_arrow()
            for start in range(0, len(raw), config.batch_rows):
                part = f"part-{len(stats[split]['parts']):06d}.parquet"
                data = raw.slice(start, config.batch_rows)
                labels = target.slice(start, config.batch_rows)
                pq.write_table(data, folder / "raw" / split / part, compression="zstd")
                pq.write_table(labels, folder / "targets" / split / part, compression="zstd")
                stats[split]["rows"] += len(data)
                stats[split]["positive"] += sum(labels.column("source_positive").to_pylist())
                stats[split]["parts"].append(
                    dict(
                        name=part,
                        rows=len(data),
                        raw_sha256=sha256(folder / "raw" / split / part),
                        target_sha256=sha256(folder / "targets" / split / part),
                    )
                )
    if {k: v["rows"] for k, v in stats.items()} != plan["expected_rows"]:
        raise ValueError("Chronological split counts differ from frozen plan")
    return stats


def fit_transform(folder: Path, config: Config, splits: dict) -> dict:
    processor = Preprocessor(standardize=config.standardize, max_categories=config.max_categories)
    for part in splits["train"]["parts"]:
        processor.partial_fit(pq.read_table(folder / "raw/train" / part["name"]), split="train")
    state = processor.freeze()
    processor.save(folder / "preprocessing.json")
    for split, info in splits.items():
        (folder / "processed" / split).mkdir(parents=True)
        for part in info["parts"]:
            data = pq.read_table(folder / "raw" / split / part["name"])
            matrix = processor.transform(data)
            if len(matrix) != part["rows"] or set(matrix.schema.names) != set(
                state["feature_columns"]
            ):
                raise ValueError("Processed feature contract drift")
            path = folder / "processed" / split / part["name"]
            pq.write_table(matrix, path, compression="zstd")
            part["processed_sha256"] = sha256(path)
    return state


def prepare_source(source: SourceInput, config: Config, folder: Path) -> dict:
    folder.mkdir(parents=True, exist_ok=False)
    plan = split_plan(source, config)
    write_json(folder / "split_plan.json", plan)
    write_json(folder / "progress.json", dict(stage="PARTITION_INPUTS", source=source.id))
    counts, base_counts = partition_inputs(source, config, folder)
    write_json(folder / "progress.json", dict(stage="CAUSAL_HISTORY", source=source.id))
    history_partitions(source, config, folder, counts, plan)
    write_json(folder / "progress.json", dict(stage="SPLIT_AND_FIT_TRAIN_ONLY", source=source.id))
    splits = split_rows(source, config, folder, base_counts, plan)
    state = fit_transform(folder, config, splits)
    if (
        sum(v["rows"] for v in splits.values()) != source.metadata["input_rows"]
        or sum(v["positive"] for v in splits.values()) != source.metadata["positive_rows"]
    ):
        raise ValueError("Input/output target reconciliation failed")
    manifest = dict(
        status="COMPLETED",
        source_id=source.id,
        source_release=source.metadata["source_release"],
        feature_version=FEATURE_VERSION,
        prediction_unit="TRANSACTION_AUXILIARY",
        primary_alert_dataset_generated=False,
        split_plan=plan,
        splits=splits,
        input_rows=source.metadata["input_rows"],
        feature_columns=state["feature_columns"],
        raw_features=list(NUMERIC + CATEGORICAL),
        model_fit_scope="TRAIN_ONLY",
        model_training_executed=False,
        label_input_used_for_features=False,
        self_transfers_excluded_from_history=True,
        same_timestamp_history_excluded=True,
        incoming_outgoing_saml_amounts="DISABLED_UNKNOWN_DIRECTION; single observed amount retained when unit known",
        test_set_status="FROZEN_NO_MODEL_SELECTION",
        source_inputs_modified=False,
    )
    write_json(folder / "manifest.json", manifest)
    return manifest


def execute(config: Config, selected: list[str] | None = None) -> tuple[Path, bool]:
    inputs = load_index(config.index, selected)
    for source in inputs:
        protected = [Path(p) for p in source.metadata["input_fingerprints"]] + [source.folder]
        if any(
            config.output == p or config.output.is_relative_to(p if p.is_dir() else p.parent)
            for p in protected
        ):
            raise ValueError(
                "Preparation outputs must stay outside originals and assembly source folders"
            )
    root = config.output / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    )
    root.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = dict(
        status="RUNNING",
        feature_version=FEATURE_VERSION,
        prediction_unit="TRANSACTION_AUXILIARY",
        config_sha256=sha256(config.path),
        code_files={p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
        python=platform.python_version(),
        packages={n: importlib.metadata.version(n) for n in ["pyarrow", "polars", "pandas"]},
        assembly_index_sha256=sha256(config.index),
        assembly_index=str(config.index),
        sources={},
        model_training_executed=False,
    )
    write_json(root / "run.json", manifest)
    for source in inputs:
        try:
            result = prepare_source(source, config, root / source.id)
            manifest["sources"][source.id] = dict(
                status="COMPLETED", folder=source.id, rows=result["input_rows"]
            )
        except Exception as exc:
            LOG.exception("%s: preparation failed; independent sources continue", source.id)
            write_json(
                root / source.id / "failure.json", dict(type=type(exc).__name__, error=str(exc))
            )
            manifest["sources"][source.id] = dict(status="FAILED", error=str(exc), folder=source.id)
        write_json(root / "run.json", manifest)
    success = all(s["status"] == "COMPLETED" for s in manifest["sources"].values())
    manifest["status"] = "COMPLETED" if success else "FAILED"
    write_json(root / "run.json", manifest)
    return root, success
