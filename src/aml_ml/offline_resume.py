"""완료한 source part·계정 단위 체크포인트와 별도 원천 재검증을 연결합니다.

저장된 generation이 가리키는 닫힌 파일만 다시 사용합니다.
적합과 출력은 새 시도에서 다시 만들며 미완료 적합 통계는 복구하지 않습니다.
"""

import copy
import os
import platform
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .offline_account_stream import account_alerts
from .offline_checkpoint import MergeCursor, Workspace, frontier_files, schema_hash
from .offline_contract import validate_dataset_integrity
from .offline_eligibility import DatasetNotReady
from .offline_io import close_all, row_workset_bytes, rows
from .offline_progress import progress, stage_scope
from .offline_publish import publish_directory
from .offline_settings import reader_budget
from .storage import read_json, sha256, write_json


def _existing_device(path):
    while not path.exists():
        if path.parent == path:
            raise ValueError("Output/workspace filesystem is unavailable")
        path = path.parent
    return path.stat().st_dev


def reject_artifact_ancestor(path):
    """기존 산출물 안에 새 하위 폴더를 추가해 기존 파일 목록을 바꾸지 않게 합니다."""
    for ancestor in path.resolve().parents:
        if any(
            (ancestor / name).is_file() for name in ("manifest.json", "run.json", "failure.json")
        ):
            raise ValueError("New control/output directory is inside an existing artifact")


def _validate_invocation_inputs(context):
    if context is None:
        return
    for path_key, hash_key in (
        ("assembly_index", "assembly_index_sha256"),
        ("policy_path", "policy_file_sha256"),
    ):
        path = Path(context[path_key])
        if not path.is_absolute() or sha256(path) != context[hash_key]:
            raise ValueError("Pinned invocation policy/index bytes changed")


def _preflight(source, output, workspace, resume):
    for path in (output, workspace):
        if path.is_symlink() or path.is_junction():
            raise ValueError("Output/workspace must not be a link or junction")
    output, workspace = output.resolve(), workspace.resolve()
    reject_artifact_ancestor(output)
    reject_artifact_ancestor(workspace)
    if output.is_relative_to(workspace) or workspace.is_relative_to(output):
        raise ValueError("Output and checkpoint workspace must be separate sibling trees")
    if output.anchor.casefold() != workspace.anchor.casefold() or _existing_device(
        output
    ) != _existing_device(workspace):
        raise ValueError("Atomic publication requires output/workspace on the same filesystem")
    protected = source.folder.resolve()
    for parent in (protected, *protected.parents):
        if (parent / "run.json").is_file():
            protected = parent
            break
    for path in (output, workspace):
        if path.is_relative_to(protected) or protected.is_relative_to(path):
            raise ValueError("Output/workspace overlaps the pinned assembly source")
    if resume:
        Workspace.require_checkpoint(workspace)
    elif output.exists():
        raise FileExistsError("Existing output is preserved; use a new run")
    return output, workspace


def _closed_files(folder):
    receipts = {}
    for path in sorted(folder.iterdir()):
        if path.is_symlink() or not path.is_file() or ".partial" in path.name:
            raise ValueError("Final output contains an incomplete or unsafe entry")
        with path.open("r+b") as stream:
            os.fsync(stream.fileno())
        receipts[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    return receipts


def _check_closed(folder, receipts):
    if not folder.is_dir() or folder.is_symlink() or folder.is_junction():
        raise ValueError("Committed output directory is missing or unsafe")
    if {path.name for path in folder.iterdir()} != set(receipts):
        raise ValueError("Committed final file set changed")
    for name, expected in receipts.items():
        path = folder / name
        if (
            Path(name).name != name
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != expected["bytes"]
            or sha256(path) != expected["sha256"]
        ):
            raise ValueError("Committed final file receipt changed")


def _owner_schema(tx_schema):
    return pa.schema(
        list(tx_schema)
        + [
            ("_owner", pa.string()),
            ("_bucket", pa.int32()),
            ("_offline_label", pa.int8()),
        ]
    )


def _source_phase(ws, source, settings):
    from .offline_streaming import _partition_source

    sorter, tx_schema, coverage, stats = _partition_source(
        source,
        ws.attempt,
        settings,
        workspace=ws,
    )
    names = frontier_files(sorter.snapshot())
    ws.commit(
        "OWNERS",
        {
            "owners": names,
            "coverage": coverage,
            "stats": stats,
            "tx_schema_sha256": schema_hash(tx_schema),
        },
        names,
    )


def _reduce_owners(ws, settings, tx_schema):
    from .offline_streaming import _merge, _owner_key, _sorter

    data = copy.deepcopy(ws.state["data"])
    fan_in = settings["merge_fan_in"]
    sorter = _sorter(
        ws.attempt,
        _owner_schema(tx_schema),
        _owner_key,
        "owner-merge",
        settings,
        ws,
    )
    while len(data["owners"]) > fan_in:
        inputs = data["owners"][:fan_in]
        progress("owner_merge", processed=0, total=len(data["owners"]), unit="runs")
        merged = _merge(
            [ws.path(name) for name in inputs],
            _owner_key,
            max_bytes=settings["max_partition_bytes"],
            fan_in=fan_in,
            decode_rows=settings["decode_batch_rows"],
        )
        try:
            path = sorter._write(merged)
        finally:
            close_all([merged])
        data["owners"] = data["owners"][fan_in:] + [path.relative_to(ws.folder).as_posix()]
        ws.commit("OWNERS", data, data["owners"])
    data["offsets"] = [0] * len(data["owners"])
    data["last_owner"] = None
    data["records"] = data["evidence"] = None
    ws.commit("ACCOUNTS", data, data["owners"])


def _account_phase(ws, source, policy, first, second, settings, tx_schema):
    from .offline_streaming import RECORD_SCHEMA, _consume_emissions, _owner_key, _sorter

    data = copy.deepcopy(ws.state["data"])
    stats = data["stats"]
    if stats["owned_rows_processed"] != sum(data["offsets"]):
        raise ValueError("Owner statistics differ from committed ACK offsets")
    records = _sorter(
        ws.attempt,
        RECORD_SCHEMA,
        lambda r: (r["reference_ts"], r["training_alert_uid"]),
        "records",
        settings,
        ws,
    )
    evidence = _sorter(
        ws.attempt,
        tx_schema,
        lambda r: r["source_row_id"],
        "evidence-rows",
        settings,
        ws,
    )
    for sorter, key in ((records, "records"), (evidence, "evidence")):
        if data[key] is not None:
            sorter.restore(data[key])
    counts = [ws.receipts[name]["rows"] for name in data["owners"]]
    if sum(counts) != stats["owned_rows"]:
        raise ValueError("Owner merge frontier does not cover all derived owners")
    cursor = MergeCursor(
        [ws.path(name) for name in data["owners"]],
        _owner_key,
        reader=rows,
        offsets=data["offsets"],
        row_counts=counts,
        fan_in=settings["merge_fan_in"],
        max_bytes=reader_budget(settings["max_partition_bytes"], settings["merge_fan_in"]),
        batch_rows=settings["decode_batch_rows"],
    )
    last_committed = stats["owned_rows_processed"]
    resume_owner = None if data["last_owner"] is None else tuple(data["last_owner"])

    def guarded_rows():
        for row in cursor:
            if resume_owner is not None and (row["_bucket"], row["_owner"]) <= resume_owner:
                raise ValueError("Resume frontier enters an already completed account")
            yield row

    def commit_owner(bucket, owner):
        nonlocal last_committed
        data["last_owner"] = [bucket, owner]
        progress(
            "account_windows",
            processed=stats["owned_rows_processed"],
            total=stats["owned_rows"],
            unit="owned_rows",
        )
        if stats["owned_rows_processed"] - last_committed < settings["checkpoint_owner_rows"]:
            return
        data["records"], data["evidence"] = records.snapshot(), evidence.snapshot()
        data["offsets"] = list(cursor.offsets)
        if sum(data["offsets"]) != stats["owned_rows_processed"]:
            raise ValueError("Owner ACK checkpoint row accounting differs")
        keep = data["owners"] + frontier_files(data["records"]) + frontier_files(data["evidence"])
        ws.commit("ACCOUNTS", data, keep)
        last_committed = stats["owned_rows_processed"]

    ordered = guarded_rows()
    emissions = account_alerts(
        ordered,
        source,
        policy,
        coverage=data["coverage"],
        train_before=first,
        test_from=second,
        settings=settings,
        row_size=row_workset_bytes,
        stats=stats,
        acknowledge=cursor.acknowledge,
        owner_done=commit_owner,
    )
    try:
        _consume_emissions(
            emissions,
            records,
            evidence,
            tx_schema,
            stats,
            settings["max_partition_bytes"],
        )
        cursor.require_complete()
    finally:
        close_all([emissions, ordered, cursor])
    if stats["owned_rows_processed"] != stats["owned_rows"]:
        raise ValueError("Incomplete owner traversal")
    data["records"], data["evidence"] = records.snapshot(), evidence.snapshot()
    data["offsets"] = list(cursor.offsets)
    stats["max_record_sort_buffer_bytes"] = records.peak_bytes
    stats["max_evidence_sort_buffer_bytes"] = evidence.peak_bytes
    keep = frontier_files(data["records"]) + frontier_files(data["evidence"])
    data["owners"] = []
    ws.commit("OUTPUT", data, keep)


def _output_phase(ws, source, settings, binding, tx_schema):
    from .offline_streaming import _manifest, _write_outputs

    data = copy.deepcopy(ws.state["data"])
    stage = ws.attempt / "dataset"
    stage.mkdir(exist_ok=False)
    keep = frontier_files(data["records"]) + frontier_files(data["evidence"])
    try:
        counts, columns = _write_outputs(
            stage,
            ws.attempt,
            [ws.path(name) for name in frontier_files(data["records"])],
            [ws.path(name) for name in frontier_files(data["evidence"])],
            tx_schema,
            settings["max_partition_bytes"],
            source.id,
            settings=settings,
        )
    except DatasetNotReady as exc:
        write_json(
            stage / "failure.json",
            {
                "status": "NOT_READY",
                "reason": str(exc),
                "contract": binding,
                "complete_candidates_preserved": True,
                "training_executed": False,
            },
        )
        data.update(stage=stage.relative_to(ws.folder).as_posix(), files=_closed_files(stage))
        ws.commit("NOT_READY", data, keep)
        raise DatasetNotReady(f"{exc}; complete candidates: {stage}") from exc
    manifest = _manifest(stage, binding, settings, data["stats"], counts, columns)
    write_json(stage / "manifest.json", manifest)
    data.update(stage=stage.relative_to(ws.folder).as_posix(), files=_closed_files(stage))
    # 이동 전에 닫힌 출력 전체를 기록합니다. 독립 검증 완료 전에는 중간 입력을 보존합니다.
    ws.commit("PUBLISH_INTENT", data, keep)


def _publish(ws, output):
    data = copy.deepcopy(ws.state["data"])
    stage = ws.path(data["stage"])
    if stage.exists() == output.exists():
        raise ValueError("Publication expects exactly one of committed stage or output")
    if stage.exists():
        _check_closed(stage, data["files"])
        output.parent.mkdir(parents=True, exist_ok=True)
        # Windows에서는 기존 목적지로 rename하지 못합니다. 다른 OS에서도 실행마다 새 경로를 씁니다.
        if output.exists() or output.is_symlink():
            raise FileExistsError("Publication destination already exists")
        publish_directory(stage, output)
    _check_closed(output, data["files"])
    ws.commit("PUBLISHED", data, list(ws.receipts))


def build_durable_dataset(
    source,
    policy,
    output,
    *,
    train_before,
    test_from,
    execution_scope,
    settings,
    binding,
    checkpoint_dir,
    resume,
    verify,
    checkpoint_context=None,
):
    output, checkpoint_dir = _preflight(source, output, checkpoint_dir, resume)
    _validate_invocation_inputs(checkpoint_context)
    tx_schema = pq.read_schema(source.part_path("transactions", 0))
    label_schema = pq.read_schema(source.part_path("labels", 0))
    run_binding = {
        "contract": binding,
        "settings": settings,
        "output": str(output),
        "tx_schema_sha256": schema_hash(tx_schema),
        "label_schema_sha256": schema_hash(label_schema),
        "python": platform.python_version(),
        "pyarrow": pa.__version__,
        "mode": "VERIFIED" if verify else "BUILD_ONLY",
        "invocation_inputs": checkpoint_context,
    }
    with Workspace(checkpoint_dir, run_binding, resume=resume) as ws:
        if output.exists() and ws.state["phase"] not in {
            "PUBLISH_INTENT",
            "PUBLISHED",
            "VERIFIED",
            "BUILT",
        }:
            raise FileExistsError("Existing output is unrelated to the committed publication")
        if ws.state["phase"] not in {"SOURCE", "VERIFIED", "BUILT"}:
            for i in range(len(source.parts("transactions"))):
                source.part_path("transactions", i)
                source.part_path("labels", i)
                progress(
                    "resume_source_hash",
                    processed=i + 1,
                    total=len(source.parts("transactions")),
                    unit="parts",
                )
        if ws.state["phase"] == "SOURCE":
            _source_phase(ws, source, settings)
        if ws.state["phase"] == "OWNERS":
            _reduce_owners(ws, settings, tx_schema)
        if ws.state["phase"] == "ACCOUNTS":
            _account_phase(ws, source, policy, train_before, test_from, settings, tx_schema)
        if ws.state["phase"] == "OUTPUT":
            _output_phase(ws, source, settings, binding, tx_schema)
        if ws.state["phase"] == "NOT_READY":
            stage = ws.path(ws.state["data"]["stage"])
            _check_closed(stage, ws.state["data"]["files"])
            raise DatasetNotReady(f"NOT_READY: complete candidates preserved at {stage}")
        if ws.state["phase"] == "PUBLISH_INTENT":
            _validate_invocation_inputs(checkpoint_context)
            _publish(ws, output)
        if ws.state["phase"] == "PUBLISHED":
            _check_closed(output, ws.state["data"]["files"])
            if verify:
                from .offline_streaming import verify_streaming_dataset

                with stage_scope("independent_replay/"):
                    result = verify_streaming_dataset(
                        output,
                        source,
                        policy,
                        execution_scope=execution_scope,
                        replay_workspace=ws.folder / "replay-work",
                        replay_output=ws.folder / "replay-dataset",
                    )
            else:
                validate_dataset_integrity(output, source, policy, execution_scope=execution_scope)
                result = read_json(output / "manifest.json")
            data = copy.deepcopy(ws.state["data"])
            data["result"] = result
            data["manifest_sha256"] = sha256(output / "manifest.json")
            _validate_invocation_inputs(checkpoint_context)
            ws.commit("VERIFIED" if verify else "BUILT", data, [])
        if ws.state["phase"] not in {"VERIFIED", "BUILT"}:
            raise ValueError("Unknown checkpoint phase")
        data = ws.state["data"]
        integrity = validate_dataset_integrity(
            output, source, policy, execution_scope=execution_scope
        )
        if integrity["manifest_sha256"] != data["manifest_sha256"]:
            raise ValueError("Completed checkpoint output was modified")
        _validate_invocation_inputs(checkpoint_context)
        return data["result"]
