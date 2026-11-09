"""독립 I/O 제어, 할당 경계와 공정한 실측을 위한 사전 확인을 검사합니다."""

from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from offline_test_data import prepare

from aml_ml import offline_benchmark as benchmark
from aml_ml import offline_checkpoint, offline_io, offline_streaming
from aml_ml.offline_measurements import Measurement, temporary_root
from aml_ml.offline_settings import execution_settings, validate_boundaries
from aml_ml.storage import read_json, sha256, write_json


def test_transient_sorter_rejects_restore_before_checkpoint_read(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Workspace must be present before checkpoint validation")

    monkeypatch.setattr(offline_checkpoint, "verify_frontier", forbidden)
    sorter = offline_streaming._sorter(
        tmp_path, pa.schema([("n", pa.int64())]), lambda row: row["n"], "rows", execution_settings()
    )
    with pytest.raises(ValueError, match="Only durable sorters"):
        sorter.restore({})
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("first", [datetime(9999, 12, 31), datetime(2022, 1, 1)])
def test_both_split_boundaries_keep_overflow_validation(first):
    with pytest.raises(OverflowError):
        validate_boundaries(first, datetime.max, 86400)


def test_short_validation_interval_does_not_relax_or_reject_horizon():
    assert validate_boundaries(datetime(2022, 1, 1), datetime(2022, 1, 1, 1), 86400) is None


@pytest.mark.parametrize("fan_in", [2, 3, 8])
def test_nine_or_more_runs_ties_and_multiple_merge_passes_close_readers(
    tmp_path, monkeypatch, fan_in
):
    schema = pa.schema([("ts", pa.int64()), ("ordinal", pa.int64())])
    paths = []
    for index in range(19):
        path = tmp_path / f"{index}.parquet"
        pq.write_table(pa.table({"ts": [1], "ordinal": [index]}, schema=schema), path)
        paths.append(path)
    original = offline_streaming._rows
    readers = {"active": 0, "peak": 0}

    def tracked(*args, **kwargs):
        readers["active"] += 1
        readers["peak"] = max(readers["peak"], readers["active"])
        try:
            yield from original(*args, **kwargs)
        finally:
            readers["active"] -= 1

    monkeypatch.setattr(offline_streaming, "_rows", tracked)
    merged = offline_streaming._bounded_merge(
        list(reversed(paths)),
        lambda row: (row["ts"], row["ordinal"]),
        work=tmp_path,
        schema=schema,
        prefix="many",
        max_bytes=1024**2,
        fan_in=fan_in,
        decode_rows=2,
        write_rows=3,
    )
    assert [row["ordinal"] for row in merged] == list(range(19))
    assert readers["active"] == 0 and readers["peak"] <= fan_in
    assert len(list(tmp_path.glob("many-merge-*.parquet"))) > 1


def test_sort_accumulation_limit_does_not_shrink_merge_single_row_budget(tmp_path):
    settings = execution_settings(
        sort_run_rows=3, sort_run_bytes=10_000, max_partition_bytes=1024**2, write_batch_rows=2
    )
    schema = pa.schema([("n", pa.int64()), ("text", pa.string())])
    sorter = offline_streaming._sorter(tmp_path, schema, lambda row: row["n"], "rows", settings)
    for index in reversed(range(31)):
        sorter.add({"n": index, "text": "x" * 1500}, {"stage": "fixture"})
    merged = offline_streaming._bounded_merge(
        sorter.paths(),
        lambda row: row["n"],
        work=tmp_path,
        schema=schema,
        prefix="final",
        max_bytes=settings["max_partition_bytes"],
        **offline_streaming._merge_options(settings),
    )
    assert [row["n"] for row in merged] == list(range(31))
    assert sorter.peak_rows <= 3 and sorter.peak_bytes <= 10_000


def test_nested_python_expansion_is_rejected_before_to_pylist(tmp_path, monkeypatch):
    path = tmp_path / "nested.parquet"
    pq.write_table(pa.table({"items": [[0] * 10_000]}), path)
    reader_type = pq.ParquetFile

    class Batch:
        def __init__(self, batch):
            self.batch, self.nbytes, self.schema = batch, batch.nbytes, batch.schema

        def __len__(self):
            return len(self.batch)

        def column(self, index):
            return self.batch.column(index)

        def slice(self, offset, length=None):
            return Batch(self.batch.slice(offset, length))

        def to_pylist(self):
            pytest.fail("Nested Python conversion happened before the recursive budget check")

    class Reader:
        def __init__(self, filename):
            self.reader = reader_type(filename)
            self.schema_arrow, self.metadata = self.reader.schema_arrow, self.reader.metadata

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.reader.close()

        def iter_batches(self, **kwargs):
            for batch in self.reader.iter_batches(**kwargs):
                yield Batch(batch)

    monkeypatch.setattr(offline_io.pq, "ParquetFile", Reader)
    with pytest.raises(ValueError, match="python_decode_estimate_bytes"):
        list(offline_io.rows(path, 1024, max_bytes=128 * 1024))


def test_independent_settings_reach_manifest_and_replay(source, policy, tmp_path):
    output = tmp_path / "dataset"
    prepare(
        source,
        policy,
        output,
        read_batch_rows=99,
        decode_batch_rows=2,
        sort_run_rows=1,
        sort_run_bytes=65536,
        write_batch_rows=3,
        merge_fan_in=2,
    )
    settings = read_json(output / "manifest.json")["execution_settings"]
    assert [
        settings[key]
        for key in (
            "read_batch_rows",
            "decode_batch_rows",
            "sort_run_rows",
            "sort_run_bytes",
            "write_batch_rows",
            "merge_fan_in",
        )
    ] == [99, 2, 1, 65536, 3, 2]
    assert read_json(output / "manifest.json")["work_stats"]["owned_rows_processed"] == 30


@pytest.mark.parametrize("place", ["source", "dataset"])
def test_benchmark_output_overlap_is_rejected_before_any_mutation(source, policy, tmp_path, place):
    policy_path = tmp_path / "policy.json"
    write_json(policy_path, asdict(policy))
    dataset = tmp_path / "frozen-dataset"
    dataset.mkdir()
    (dataset / "keep.txt").write_text("untouched", encoding="utf-8")
    output = (source.folder if place == "source" else dataset) / "new-benchmark"
    plan = {
        "mode": "serialization",
        "dataset": str(dataset),
        "assembly_index": "index.json",
        "policy": str(policy_path),
        "source": source.id,
        "scope": "FIXTURE_ONLY",
        "settings": {},
        "repetitions": 1,
        "cases": [{"name": "json", "format": "json"}],
    }
    path = tmp_path / "plan.json"
    write_json(path, plan)
    before = {str(p): sha256(p) for p in source.folder.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="overlaps protected"):
        benchmark.run_plan(path, output)
    assert not output.exists()
    assert [p.name for p in dataset.iterdir()] == ["keep.txt"]
    assert {str(p): sha256(p) for p in source.folder.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("form", ["csv", "json", "parquet", "three"])
def test_codec_benchmark_tracks_closed_files_for_every_format(tmp_path, form):
    source = tmp_path / "input.parquet"
    pq.write_table(pa.table({"id": [9007199254740993], "text": ["한글"]}), source)
    output = tmp_path / "output"
    output.mkdir()
    with Measurement(interval=10) as measurement:
        temporary_root(output)
        benchmark._serialize_table(source, output, "table", form, execution_settings())
    assert measurement.report is not None
    assert measurement.report["closed_file_temporary_disk_peak_bytes"] > 0
    assert measurement.report["physical_io_bytes"] is None
    if form in {"json", "three"}:
        assert read_json(output / "table.json") == [{"id": 9007199254740993, "text": "한글"}]


def test_reader_cleanup_closes_every_stream_and_preserves_primary_failure():
    closed = []

    class Reader:
        def __init__(self, number):
            self.number = number

        def close(self):
            closed.append(self.number)
            if self.number == 1:
                raise OSError("close fixture")

    with pytest.raises(ValueError, match="primary fixture"):
        try:
            raise ValueError("primary fixture")
        finally:
            offline_io.close_all([Reader(1), Reader(2)])
    assert closed == [1, 2]
    closed.clear()
    with pytest.raises(OSError, match="close fixture"):
        offline_io.close_all([Reader(1), Reader(2)])
    assert closed == [1, 2]


def test_direct_benchmark_worker_resolves_all_paths_from_job_not_cwd(
    source, policy, tmp_path, monkeypatch
):
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    write_json(tmp_path / "policy.json", asdict(policy))
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    job = jobs / "job.json"
    write_json(
        job,
        {
            "mode": "serialization",
            "dataset": "../dataset",
            "assembly_index": "../index.json",
            "policy": "../policy.json",
            "source": source.id,
            "scope": "FIXTURE_ONLY",
            "settings": {},
            "format": "json",
            "output": "../benchmark-result",
        },
    )
    other = tmp_path / "unrelated-cwd"
    other.mkdir()
    monkeypatch.chdir(other)

    def observe(resolved, *_):
        assert Path(resolved["dataset"]) == dataset
        assert Path(resolved["assembly_index"]) == tmp_path / "index.json"
        return {}

    monkeypatch.setattr(benchmark, "_serialization", observe)
    benchmark.worker(job)
    assert (tmp_path / "benchmark-result" / "benchmark.json").is_file()


def test_benchmark_code_change_during_worker_cannot_publish_result(
    source, policy, tmp_path, monkeypatch
):
    write_json(tmp_path / "policy.json", asdict(policy))
    job = tmp_path / "job.json"
    write_json(
        job,
        {
            "mode": "preprocess",
            "assembly_index": "index.json",
            "policy": "policy.json",
            "source": source.id,
            "scope": "FIXTURE_ONLY",
            "settings": {},
            "output": "result",
            "train_before": "2022-01-02",
            "test_from": "2022-01-04",
        },
    )
    original_hash = benchmark.sha256
    changed = []

    def hash_file(path):
        if Path(path) == Path(benchmark.__file__) and changed:
            return "0" * 64
        return original_hash(path)

    def pretend_work(*args):
        changed.append(True)
        return {}

    monkeypatch.setattr(benchmark, "sha256", hash_file)
    monkeypatch.setattr(benchmark, "_preprocess", pretend_work)
    with pytest.raises(ValueError, match="implementation/environment changed"):
        benchmark.worker(job)
    assert not (tmp_path / "result" / "benchmark.json").exists()


def test_codec_benchmark_rechecks_original_input_after_measurement(
    source, policy, tmp_path, monkeypatch
):
    dataset = tmp_path / "dataset"
    prepare(source, policy, dataset)
    output = tmp_path / "comparison"
    output.mkdir()
    original = benchmark._serialize_table
    mutated = []

    def change_after_write(*args, **kwargs):
        original(*args, **kwargs)
        if not mutated:
            with (dataset / "raw.json").open("a", encoding="utf-8") as stream:
                stream.write(" ")
            mutated.append(True)

    monkeypatch.setattr(benchmark, "_serialize_table", change_after_write)
    with pytest.raises(ValueError, match="hash"):
        benchmark._serialization(
            {"dataset": str(dataset), "scope": "FIXTURE_ONLY", "format": "json"},
            source,
            policy,
            execution_settings(),
            output,
        )
    assert mutated and not (output / "benchmark.json").exists()
