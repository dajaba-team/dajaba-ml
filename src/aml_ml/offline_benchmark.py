"""사용자가 별도 process에서 실행하는 실측입니다. 전처리와 직렬화 비교를 분리합니다."""

import argparse
import csv
import hashlib
import json
import platform
import re
import subprocess
import sys
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .offline_contract import contract, validate_dataset_integrity
from .offline_io import close_all, row_workset_bytes, rows
from .offline_measurements import Measurement, event, milestone, temporary_closed, temporary_root
from .offline_rules import _json_bytes, fingerprint, load_policy
from .offline_serialization import TABLES, ThreeWriter, _csv_cell, _json, wire_value
from .offline_settings import execution_settings, validate_boundaries
from .storage import load_index, read_json, sha256, write_json

VARIABLES = frozenset(
    {
        "decode_batch_rows",
        "sort_run_rows",
        "sort_run_bytes",
        "write_batch_rows",
        "merge_fan_in",
    }
)


class ParquetOnlyWriter:
    """동일한 최종 256행 batch로 Parquet만 써서 텍스트 직렬화 비용을 분리합니다."""

    def __init__(self, path, schema, *, max_bytes):
        self.path, self.schema, self.max_bytes = path, schema, max_bytes
        self.stack = ExitStack()

    def __enter__(self):
        try:
            stream = self.stack.enter_context(self.path.open("xb"))
            self.writer = self.stack.enter_context(pq.ParquetWriter(stream, self.schema))
            return self
        except BaseException:
            close_all([self.stack])
            raise

    def write_table(self, table):
        if table.schema != self.schema or table.nbytes > self.max_bytes:
            raise ValueError("Benchmark Parquet sink schema/budget mismatch")
        for batch in table.to_batches(max_chunksize=256):
            self.writer.write_table(pa.Table.from_batches([batch], schema=self.schema))

    def __exit__(self, kind, value, traceback):
        try:
            self.stack.close()
        except BaseException as exc:
            if value is None:
                raise
            value.add_note("Benchmark writer close also failed: " + type(exc).__name__)
        if kind is None:
            event("parquet_closed_write_bytes", self.path.stat().st_size)
            temporary_closed(self.path)


def logical_tables(folder, settings):
    result = {}
    for name in TABLES:
        path = folder / f"{name}.parquet"
        schema = pq.read_schema(path)
        digest = hashlib.sha256(schema.serialize().to_pybytes())
        count = 0
        iterator = rows(path, settings["decode_batch_rows"], settings["max_partition_bytes"])
        try:
            for row in iterator:
                encoded = _json({f.name: wire_value(row[f.name], f) for f in schema})
                digest.update((encoded + "\n").encode("utf-8"))
                count += 1
        finally:
            iterator.close()
        result[name] = {"rows": count, "logical_sha256": digest.hexdigest()}
    for name in ("splits.json", "quality.json", "preprocessing.json"):
        result[name] = {"logical_sha256": fingerprint(read_json(folder / name))}
    return result


def _preprocess(job, source, policy, settings, output):
    from .offline_streaming import _partition_source, _scratch, _shards, _write_outputs

    first, second = (
        datetime.fromisoformat(job["train_before"]),
        datetime.fromisoformat(job["test_from"]),
    )
    validate_boundaries(first, second, settings["horizon_seconds"])
    binding = contract(
        source, policy, job["scope"], first, second, horizon_seconds=settings["horizon_seconds"]
    )
    data = output / "parquet-candidates"
    with Measurement() as measurement:
        with _scratch(output) as work:
            sorter, schema, coverage, stats = _partition_source(source, work, settings)
            records, evidence = _shards(
                source,
                policy,
                job["scope"],
                work,
                sorter,
                schema,
                coverage,
                first,
                second,
                settings,
                stats,
            )
            data.mkdir(exist_ok=False)
            _write_outputs(
                data,
                work,
                records,
                evidence,
                schema,
                settings["max_partition_bytes"],
                source.id,
                settings=settings,
                writer_type=ParquetOnlyWriter,
            )
            milestone("build_complete")
    # 논리 값 재조회는 생성 측정 시간 밖에서 합니다. 실측 산출물은 학습 승인 자료가 아닙니다.
    logical = logical_tables(data, settings)
    for which in ("transactions", "labels"):
        for index in range(len(source.parts(which))):
            source.part_path(which, index)
    if (
        contract(
            source, policy, job["scope"], first, second, horizon_seconds=settings["horizon_seconds"]
        )
        != binding
    ):
        raise ValueError("Benchmark input/code changed during measurement")
    return {
        "measurement": measurement.report,
        "contract": binding,
        "logical_result": logical,
        "source_schemas": {
            which: hashlib.sha256(
                pq.read_schema(source.part_path(which, 0)).serialize().to_pybytes()
            ).hexdigest()
            for which in ("transactions", "labels")
        },
        "scope": "PREPROCESS_THROUGH_FIXED256_FIT_AND_PARQUET_OUTPUT_NO_TEXT_CODEC_NO_REPLAY",
    }


def _serialize_table(source_path, output, name, mode, settings):
    schema = pq.read_schema(source_path)
    max_bytes = settings["max_partition_bytes"]
    iterator = rows(source_path, settings["decode_batch_rows"], max_bytes)
    with ExitStack() as stack:
        stack.callback(iterator.close)
        writer = None
        text = None
        csv_writer = None
        if mode in {"parquet", "three"}:
            writer_type = ThreeWriter if mode == "three" else ParquetOnlyWriter
            writer = stack.enter_context(
                writer_type(output / f"{name}.parquet", schema, max_bytes=max_bytes)
            )
        else:
            text = stack.enter_context(
                (output / f"{name}.{mode}").open("x", encoding="utf-8", newline="")
            )
            if mode == "json":
                text.write("[")
            else:
                csv_writer = csv.writer(text, lineterminator="\n")
                csv_writer.writerow(schema.names)
        buffer, count, buffered_bytes = [], 0, 0
        for row in iterator:
            milestone("first_final_row")
            if writer is not None:
                size = row_workset_bytes(row)
                if size > max_bytes:
                    raise ValueError("Benchmark single-row workset exceeds budget")
                if buffer and (len(buffer) == 256 or buffered_bytes + size > max_bytes):
                    writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
                    buffer.clear()
                    buffered_bytes = 0
                buffer.append(row)
                buffered_bytes += size
            else:
                wire = {f.name: wire_value(row[f.name], f) for f in schema}
                if _json_bytes(wire) > max_bytes:
                    raise ValueError("Benchmark serialized record exceeds byte budget")
                if mode == "json" and text is not None:
                    text.write(("," if count else "") + _json(wire))
                elif csv_writer is not None:
                    csv_writer.writerow([_csv_cell(wire[f.name]) for f in schema])
            count += 1
        if writer is not None and buffer:
            writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
        if mode == "json" and text is not None:
            text.write("]")
    if mode in {"csv", "json"}:
        event(f"{mode}_closed_write_bytes", (output / f"{name}.{mode}").stat().st_size)
        temporary_closed(output / f"{name}.{mode}")


def _serialization(job, source, policy, settings, output):
    dataset = Path(job["dataset"])
    integrity = validate_dataset_integrity(dataset, source, policy, execution_scope=job["scope"])
    manifest = read_json(dataset / "manifest.json")
    logical = logical_tables(dataset, settings)
    with Measurement() as measurement:
        temporary_root(output)
        for name in TABLES:
            _serialize_table(dataset / f"{name}.parquet", output, name, job["format"], settings)
        milestone("build_complete")
    # 모든 출력 형식의 재조회는 쓰기 시간 측정이 끝난 뒤 별도로 수행합니다.
    from .offline_serialization import iter_csv, iter_json_array, typed_row

    for name in TABLES:
        schema = pq.read_schema(dataset / f"{name}.parquet")
        for form in ("csv", "json", "parquet") if job["format"] == "three" else (job["format"],):
            path = output / f"{name}.{form}"
            if form == "parquet":
                iterator = rows(
                    path, settings["decode_batch_rows"], settings["max_partition_bytes"]
                )
            elif form == "csv":
                iterator = iter_csv(path, schema, max_record_bytes=settings["max_partition_bytes"])
            else:
                iterator = iter_json_array(path, max_record_bytes=settings["max_partition_bytes"])
            digest = hashlib.sha256(schema.serialize().to_pybytes())
            count = 0
            try:
                for item in iterator:
                    if form == "json":
                        item = typed_row(item, schema)
                    digest.update(
                        (
                            _json({f.name: wire_value(item[f.name], f) for f in schema}) + "\n"
                        ).encode("utf-8")
                    )
                    count += 1
            finally:
                iterator.close()
            if {"rows": count, "logical_sha256": digest.hexdigest()} != logical[name]:
                raise ValueError("Benchmark codec output is not logically equivalent")
    after = validate_dataset_integrity(dataset, source, policy, execution_scope=job["scope"])
    if after["manifest_sha256"] != integrity["manifest_sha256"]:
        raise ValueError("Benchmark input changed")
    return {
        "measurement": measurement.report,
        "contract": manifest["contract"],
        "logical_result": logical,
        "input_manifest_sha256": integrity["manifest_sha256"],
        "scope": "FIXED_PARQUET_DECODE_PLUS_SELECTED_CODEC_WRITES_READBACK_OUTSIDE_TIMER",
    }


def worker(job_path):
    job_path = job_path.resolve()
    job = read_json(job_path)
    for key in ("output", "policy", "assembly_index", "dataset"):
        if key in job:
            job[key] = str((job_path.parent / job[key]).resolve())
    output = Path(job["output"])
    source, policy = preflight_paths(job, job_path.parent, output)
    settings = execution_settings(**job["settings"])
    benchmark_digest = sha256(Path(__file__))
    environment = {"python": platform.python_version(), "pyarrow": pa.__version__}
    output.mkdir(parents=True, exist_ok=False)
    result = (_preprocess if job["mode"] == "preprocess" else _serialization)(
        job, source, policy, settings, output
    )
    if benchmark_digest != sha256(Path(__file__)) or environment != {
        "python": platform.python_version(),
        "pyarrow": pa.__version__,
    }:
        raise ValueError("Benchmark implementation/environment changed during measurement")
    result.update(
        status="BENCHMARK_COMPLETED",
        settings=settings,
        model_training_executed=False,
        benchmark_code_sha256=benchmark_digest,
        environment=environment,
    )
    write_json(output / "benchmark.json", result)
    return result


def preflight_paths(plan, base_dir, output):
    """하위 worker도 출력 폴더를 만들기 전에 입력·출력 경로를 읽기 전용으로 검사합니다."""
    if output.exists() or output.is_symlink():
        raise FileExistsError("Benchmark output must be new")
    index = (base_dir / plan["assembly_index"]).resolve()
    policy_path = (base_dir / plan["policy"]).resolve()
    policy, _ = load_policy(policy_path)
    policy.require(plan["scope"])
    source = load_index(index, [plan["source"]])[0]
    protected = [source.folder.resolve(), index, policy_path]
    for ancestor in source.folder.parents:
        if (ancestor / "run.json").is_file():
            protected.append(ancestor)
            break
    if plan["mode"] == "serialization":
        protected.append((base_dir / plan["dataset"]).resolve())
    elif plan["mode"] != "preprocess":
        raise ValueError("Unknown benchmark mode")
    destination = output.resolve()
    if any(
        destination.is_relative_to(path) or path.is_relative_to(destination) for path in protected
    ):
        raise ValueError("Benchmark output overlaps protected source/dataset/config input")
    return source, policy


def run_plan(plan_path, output):
    plan = read_json(plan_path)
    mode = plan["mode"]
    if mode not in {"preprocess", "serialization"}:
        raise ValueError("Separate preprocess or serialization benchmark mode required")
    base = execution_settings(**plan["settings"])
    repeats = plan["repetitions"]
    if type(repeats) is not int or not 1 <= repeats <= 10:
        raise ValueError("Explicit 1..10 benchmark repetitions required")
    cases = plan["cases"]
    names = [case["name"] for case in cases]
    if (
        not names
        or len(set(names)) != len(names)
        or any(not re.fullmatch(r"[a-z0-9_-]{1,40}", n) for n in names)
    ):
        raise ValueError("Unique safe benchmark case names required")
    for case in cases:
        changes = case.get("overrides", {})
        if mode == "preprocess" and (set(changes) - VARIABLES or len(changes) > 1):
            raise ValueError("One I/O variable per case; semantic/memory policy stays fixed")
        if mode == "serialization" and (
            changes or case.get("format") not in {"csv", "json", "parquet", "three"}
        ):
            raise ValueError("Codec comparison must keep identical preprocessing/settings")
        execution_settings(**dict(base, **changes))
    preflight_paths(plan, plan_path.parent, output)
    output.mkdir(parents=True, exist_ok=False)
    reference = None
    results = []
    for repetition in range(repeats):
        ordered = cases[repetition % len(cases) :] + cases[: repetition % len(cases)]
        for case in ordered:
            name = f"{repetition:02d}-{case['name']}"
            job = {key: plan[key] for key in ("mode", "source", "scope")}
            for key in ("policy", "assembly_index", "dataset"):
                if key in plan:
                    job[key] = str((plan_path.parent / plan[key]).resolve())
            if mode == "preprocess":
                job.update(train_before=plan["train_before"], test_from=plan["test_from"])
            else:
                job["format"] = case["format"]
            job.update(
                output=str((output / name).resolve()),
                settings=dict(base, **case.get("overrides", {})),
            )
            job_path = output / f"{name}.job.json"
            write_json(job_path, job)
            started = datetime.now(timezone.utc).isoformat()
            with (
                (output / f"{name}.stdout.txt").open("x") as stdout,
                (output / f"{name}.stderr.txt").open("x") as stderr,
            ):
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "aml_ml.offline_benchmark",
                        "worker",
                        "--job",
                        str(job_path),
                    ],
                    stdout=stdout,
                    stderr=stderr,
                    check=False,
                )
            write_json(
                output / f"{name}.result.json",
                {
                    "case": name,
                    "started_at_utc": started,
                    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                    "exit_code": completed.returncode,
                    "job": job_path.name,
                    "stdout": f"{name}.stdout.txt",
                    "stderr": f"{name}.stderr.txt",
                },
            )
            if completed.returncode != 0:
                raise ValueError(f"Benchmark {name} failed; no speedup conclusion is valid")
            result = read_json(output / name / "benchmark.json")
            comparable = {key: result[key] for key in ("contract", "logical_result")}
            for key in (
                "source_schemas",
                "input_manifest_sha256",
                "benchmark_code_sha256",
                "environment",
            ):
                if key in result:
                    comparable[key] = result[key]
            if reference is None:
                reference = comparable
            elif comparable != reference:
                raise ValueError(
                    "Benchmark cases changed source/policy/schema/eligibility/logical result"
                )
            results.append({"case": name, "measurement": result["measurement"]})
    report = {
        "status": "BENCHMARK_COMPLETED",
        "mode": mode,
        "cases": results,
        "comparable_result_sha256": fingerprint(reference),
        "cache_state": "UNCONFIRMED",
        "physical_io": "UNCONFIRMED",
        "speedup": None,
        "note": "Same-code controlled I/O/codec comparison; historical failures are not baselines",
    }
    write_json(output / "comparison.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "worker"))
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--job", type=Path)
    args = parser.parse_args(argv)
    if args.command == "worker" and args.job is not None:
        result = worker(args.job)
    elif args.command == "run" and args.plan is not None and args.output is not None:
        result = run_plan(args.plan.resolve(), args.output.resolve())
    else:
        parser.error("run needs --plan/--output; worker needs --job")
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
