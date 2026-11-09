"""검증된 Parquet 입력과 크기를 제한한 계좌·행 분할 쓰기를 제공합니다."""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .offline_measurements import event
from .offline_progress import check_cancelled

PROJECT_SOURCES = frozenset({"ibm_hi_small", "ibm_li_small", "saml_d"})


def load_index(index_path: Path, selected: list[str] | None = None) -> list["SourceInput"]:
    """데이터 전체를 읽지 않고 인덱스 기준 상대 경로에서 고정된 원천 실행을 찾습니다."""
    index_path = index_path.resolve()
    index = read_json(index_path)
    if not index.get("ml_runs") or set(index["ml_runs"]) - PROJECT_SOURCES:
        raise ValueError("Only HI-Small, LI-Small and SAML-D belong to this project; no Large")
    ids = selected or list(index["ml_runs"])
    if set(ids) - set(index["ml_runs"]):
        raise ValueError("Unknown source")
    inputs = []
    for sid in ids:
        run = (index_path.parent / index["ml_runs"][sid]).resolve()
        expected = index.get("run_manifest_sha256", {}).get(sid)
        if expected is not None and sha256(run / "run.json") != expected:
            raise ValueError("Pinned assembly run manifest changed")
        inputs.append(load_source(run, sid))
    return inputs


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            check_cancelled()
            digest.update(block)
            event("hash_logical_read_bytes", len(block))
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    temp.replace(path)


@dataclass(frozen=True)
class SourceInput:
    id: str
    kind: str
    folder: Path
    metadata: dict
    transactions: dict
    labels: dict

    def parts(self, which: str) -> list[dict]:
        manifest = self.transactions if which == "transactions" else self.labels
        return manifest["parts"]

    def part_path(self, which: str, index: int) -> Path:
        manifest = self.transactions if which == "transactions" else self.labels
        key = "ml_transactions" if which == "transactions" else "ml_labels"
        folder = (self.folder / self.metadata["datasets"][key]["path"]).resolve()
        if not folder.is_relative_to(self.folder):
            raise ValueError("Unsafe dataset path")
        files = [f for f in manifest["parts"][index]["files"] if f["path"].endswith(".parquet")]
        if len(files) != 1:
            raise ValueError("Expected exactly one Parquet per part")
        path = (folder / files[0]["path"]).resolve()
        if path.parent != folder or sha256(path) != files[0]["sha256"]:
            raise ValueError("ML input part changed or uses an unsafe path")
        return path


def load_source(run: Path, source_id: str) -> SourceInput:
    run = run.resolve()
    manifest = read_json(run / "run.json")
    if (
        not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", source_id)
        or manifest.get("contract_version") != "dajaba-assembly-v2"
        or manifest.get("status") != "COMPLETED"
    ):
        raise ValueError("An explicit completed source assembly run is required")
    item = manifest["sources"][source_id]
    if item["status"] != "VERIFIED":
        raise ValueError("Unverified source")
    folder = (run / item["folder"]).resolve()
    if not folder.is_relative_to(run):
        raise ValueError("Unsafe source folder")
    source = read_json(folder / "source_manifest.json")
    if (
        source["source_id"] != source_id
        or source["status"] != "VERIFIED"
        or source["ml_output_rows"] != source["input_rows"]
        or source["input_rows"] <= 0
    ):
        raise ValueError("ML requires the full source, never the random backend excerpt")
    tables = {}
    for key in ["ml_transactions", "ml_labels"]:
        path = (folder / source["datasets"][key]["path"]).resolve()
        if not path.is_relative_to(folder):
            raise ValueError("Unsafe table folder")
        tables[key] = read_json(path / "manifest.json")
        if tables[key]["status"] != "VERIFIED" or tables[key]["rows"] != source["input_rows"]:
            raise ValueError("Incomplete ML table")
    tx, labels = tables["ml_transactions"], tables["ml_labels"]
    for table in (tx, labels):
        if (
            table.get("source_id") != source_id
            or table.get("source_release") != source["source_release"]
            or table.get("purpose") != "ml"
            or table.get("scope") != "FULL_SOURCE"
            or sum(p["rows"] for p in table["parts"]) != source["input_rows"]
        ):
            raise ValueError("ML table source/release/scope mismatch")
    if len(tx["parts"]) != len(labels["parts"]) or [p["rows"] for p in tx["parts"]] != [
        p["rows"] for p in labels["parts"]
    ]:
        raise ValueError("ML transaction/label alignment is not established")
    from importlib.resources import files

    alias_sha = hashlib.sha256(files("aml_ml").joinpath("currencies.json").read_bytes()).hexdigest()
    if tx["aliases_sha256"] != alias_sha:
        raise ValueError("Frozen package currency map differs from the prepared source")
    return SourceInput(source_id, tx["source_kind"], folder, source, tx, labels)


class PartitionSink:
    def __init__(self, folder: Path, schema: pa.Schema, partitions: int):
        self.folder, self.schema, self.partitions = folder, schema, partitions
        folder.mkdir(parents=True, exist_ok=True)
        self.writers: dict[int, pq.ParquetWriter] = {}
        self.counts = [0] * partitions

    def write(self, index: int, table: pa.Table):
        if not len(table):
            return
        if not 0 <= index < self.partitions or table.schema != self.schema:
            raise ValueError("Partition schema/index mismatch")
        if index not in self.writers:
            self.writers[index] = pq.ParquetWriter(
                self.path(index), self.schema, compression="zstd"
            )
        self.writers[index].write_table(table)
        self.counts[index] += len(table)

    def path(self, index: int) -> Path:
        return self.folder / f"part-{index:05d}.parquet"

    def close(self):
        for writer in self.writers.values():
            writer.close()

    def scatter_rows(self, rows: list[dict], indices: list[int]):
        groups: dict[int, list[dict]] = {}
        for row, index in zip(rows, indices):
            groups.setdefault(index, []).append(row)
        for index, items in groups.items():
            self.write(index, pa.Table.from_pylist(items, schema=self.schema))
