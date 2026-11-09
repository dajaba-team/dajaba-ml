"""오프라인 R02의 prepare/verify/train/verify-model 명령 진입점입니다."""

import argparse
import json
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from .offline_eligibility import DatasetNotReady
from .offline_pipeline import build_dataset, verify_dataset
from .offline_progress import Progress, cancelled_result, reporting
from .offline_rules import load_policy
from .storage import load_index, read_json, sha256

XGBOOST_OPTIONS = (
    "seed",
    "num_boost_round",
    "max_depth",
    "learning_rate",
    "min_child_weight",
    "l2",
    "early_stopping_rounds",
)
MODEL_ALGORITHMS = {
    "NUMPY_LOGISTIC_PROXY_BASELINE": "numpy-logistic",
    "XGBOOST_BINARY_LOGISTIC_PROXY": "xgboost",
}


def _training_options(args):
    algorithm = args.algorithm or "numpy-logistic"
    options = {
        name: getattr(args, name) for name in XGBOOST_OPTIONS if getattr(args, name) is not None
    }
    if args.command != "train" and args.algorithm is not None:
        raise ValueError("--algorithm is only valid for train")
    if options and (args.command != "train" or algorithm != "xgboost"):
        raise ValueError("XGBoost options require train --algorithm xgboost")
    if args.command == "train" and algorithm == "xgboost" and args.seed is None:
        raise ValueError("XGBoost training requires an explicit --seed")
    return algorithm, options


def _training_module(algorithm):
    # 정책 확인 후에만 호출하므로 선택적 학습 모듈은 필요할 때 불러옵니다.
    if algorithm == "xgboost":
        from . import offline_xgboost

        return offline_xgboost
    if algorithm == "numpy-logistic":
        from . import offline_training

        return offline_training
    raise ValueError("Unsupported offline training algorithm")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["check-policy", "prepare", "verify", "train", "verify-model"]
    )
    parser.add_argument("--assembly-index", type=Path)
    parser.add_argument("--source")
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--scope", choices=["FIXTURE_ONLY", "CONFIRMED"], required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--model-version")
    parser.add_argument("--support-policy", type=Path)
    parser.add_argument("--k", type=int)
    parser.add_argument(
        "--algorithm",
        choices=["numpy-logistic", "xgboost"],
        help="Training algorithm (default: numpy-logistic); verify-model uses saved metadata",
    )
    parser.add_argument("--seed", type=int, help="Required explicit XGBoost training seed")
    parser.add_argument("--num-boost-round", type=int)
    parser.add_argument("--max-depth", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--min-child-weight", type=float)
    parser.add_argument("--l2", type=float)
    parser.add_argument("--early-stopping-rounds", type=int)
    parser.add_argument("--train-before", type=datetime.fromisoformat)
    parser.add_argument("--test-from", type=datetime.fromisoformat)
    parser.add_argument("--max-source-rows", type=int)
    parser.add_argument("--partitions", type=int, default=32)
    parser.add_argument("--read-batch-rows", type=int, default=4096)
    parser.add_argument("--decode-batch-rows", type=int)
    parser.add_argument("--sort-run-rows", type=int, default=65536)
    parser.add_argument("--sort-run-bytes", type=int)
    parser.add_argument("--write-batch-rows", type=int, default=8192)
    parser.add_argument("--merge-fan-in", type=int, default=8)
    parser.add_argument("--horizon-seconds", type=int, default=86400)
    parser.add_argument("--checkpoint-owner-rows", type=int, default=100_000)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-dir", type=Path)
    parser.add_argument("--heartbeat-seconds", type=float, default=30)
    parser.add_argument("--cancel-file", type=Path)
    parser.add_argument("--max-partition-rows", type=int, default=100_000)
    parser.add_argument("--max-partition-bytes", type=int, default=64 * 1024**2)
    args = parser.parse_args(argv)
    try:
        policy, policy_file_sha256 = load_policy(args.policy)
        if args.command != "prepare" and (
            args.resume
            or args.checkpoint_dir
            or args.progress_dir
            or args.cancel_file
            or args.heartbeat_seconds != 30
        ):
            raise ValueError("Checkpoint/progress/cancel options require prepare")
        if args.command == "check-policy":
            _training_options(args)
            result = policy.readiness(args.scope)
            result["policy_file_sha256"] = policy_file_sha256
            result["policy_path"] = str(args.policy.resolve())
            print(json.dumps(result, ensure_ascii=False, allow_nan=False))
            return 2 if result["issues"] else 0
        # 원천 읽기나 출력 전에 미확정 정책을 거부합니다.
        policy.require(args.scope)
        algorithm, training_options = _training_options(args)
        if args.assembly_index is None or args.source is None:
            raise ValueError("Execution requires --assembly-index and --source")
        source = load_index(args.assembly_index, [args.source])[0]
        index_sha256 = sha256(args.assembly_index)
        with ExitStack() as stack:
            reporter = None
            if args.command == "prepare":
                if args.output is None:
                    raise ValueError("prepare requires a new output directory")
                if args.checkpoint_dir is None:
                    if args.resume:
                        raise ValueError("--resume requires an explicit --checkpoint-dir")
                    args.checkpoint_dir = args.output.with_name(args.output.name + ".work")
                progress_dir = args.progress_dir or args.output.with_name(
                    args.output.name + ".progress-" + uuid4().hex
                )
                from .offline_resume import _preflight, reject_artifact_ancestor

                _preflight(source, args.output, args.checkpoint_dir, args.resume)
                protected = source.folder.resolve()
                for parent in (protected, *protected.parents):
                    if (parent / "run.json").is_file():
                        protected = parent
                        break
                # 진단 결과는 최종 산출물 및 원천 데이터와 분리합니다.
                if any(
                    progress_dir.resolve().is_relative_to(path.resolve())
                    or path.resolve().is_relative_to(progress_dir.resolve())
                    for path in (
                        args.output,
                        args.checkpoint_dir,
                        protected,
                        args.policy,
                        args.assembly_index,
                    )
                ):
                    raise ValueError("Progress directory must be outside output/workspace")
                for path in (args.output, args.checkpoint_dir):
                    if any(
                        path.resolve().is_relative_to(item.resolve())
                        or item.resolve().is_relative_to(path.resolve())
                        for item in (args.policy, args.assembly_index)
                    ):
                        raise ValueError("Output/workspace overlaps policy/index input")
                if args.cancel_file is not None:
                    flag = args.cancel_file.resolve()
                    if any(
                        flag.is_relative_to(path.resolve())
                        for path in (
                            protected,
                            args.output,
                            args.checkpoint_dir,
                            progress_dir,
                            args.policy,
                            args.assembly_index,
                        )
                    ):
                        raise ValueError(
                            "Cancellation flag must be outside data/control directories"
                        )
                    reject_artifact_ancestor(flag)
                reject_artifact_ancestor(progress_dir)
                reporter = stack.enter_context(
                    Progress(
                        progress_dir,
                        interval=args.heartbeat_seconds,
                        console=True,
                        cancel_file=args.cancel_file,
                    )
                )
                stack.enter_context(reporting(reporter))
            try:
                result = _execute(
                    args,
                    policy,
                    source,
                    policy_file_sha256,
                    index_sha256,
                    algorithm,
                    training_options,
                )
            except DatasetNotReady:
                if reporter is not None:
                    reporter.finish("NOT_READY", error_type="DatasetNotReady")
                raise
            if (
                sha256(args.policy) != policy_file_sha256
                or sha256(args.assembly_index) != index_sha256
            ):
                raise ValueError("Policy/index file changed during execution")
            if reporter is not None:
                reporter.finish("VERIFIED" if result.get("status") == "VERIFIED" else "BUILT")
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0
    except KeyboardInterrupt:
        return cancelled_result()
    except Exception as exc:
        status = "NOT_READY" if isinstance(exc, DatasetNotReady) else "FAILED"
        print(json.dumps({"status": status, "error": str(exc)}, ensure_ascii=False))
        return 2


def _execute(args, policy, source, policy_file_sha256, index_sha256, algorithm, training_options):
    common = dict(execution_scope=args.scope, max_source_rows=args.max_source_rows)
    if args.command == "prepare":
        if args.output is None or args.train_before is None or args.test_from is None:
            raise ValueError(
                "prepare requires new output and explicit source-time split boundaries"
            )
        return build_dataset(
            source,
            policy,
            args.output,
            train_before=args.train_before,
            test_from=args.test_from,
            partitions=args.partitions,
            read_batch_rows=args.read_batch_rows,
            max_partition_rows=args.max_partition_rows,
            max_partition_bytes=args.max_partition_bytes,
            decode_batch_rows=args.decode_batch_rows,
            sort_run_rows=args.sort_run_rows,
            sort_run_bytes=args.sort_run_bytes,
            write_batch_rows=args.write_batch_rows,
            merge_fan_in=args.merge_fan_in,
            horizon_seconds=args.horizon_seconds,
            checkpoint_owner_rows=args.checkpoint_owner_rows,
            checkpoint_dir=args.checkpoint_dir,
            resume=args.resume,
            checkpoint_context={
                "assembly_index": str(args.assembly_index.resolve()),
                "assembly_index_sha256": index_sha256,
                "policy_path": str(args.policy.resolve()),
                "policy_file_sha256": policy_file_sha256,
            },
            **common,
        )
    if args.dataset is None:
        raise ValueError("An explicit offline dataset is required")
    if args.command == "verify":
        return verify_dataset(args.dataset, source, policy, **common)
    if args.command == "train":
        if args.output is None or args.model_version is None or args.k is None:
            raise ValueError("train requires new output, model version and K")
        support = (
            {}
            if args.support_policy is None
            else {
                "support_policy": read_json(args.support_policy),
            }
        )
        return _training_module(algorithm).train_evaluate(
            args.dataset,
            source,
            policy,
            args.output,
            model_version=args.model_version,
            k=args.k,
            execution_scope=args.scope,
            max_source_rows=100_000 if args.max_source_rows is None else args.max_source_rows,
            **support,
            **training_options,
        )
    if args.model is None:
        raise ValueError("An explicit model directory is required")
    kind = read_json(args.model / "model.json").get("model_kind")
    if kind not in MODEL_ALGORITHMS:
        raise ValueError("Missing or unsupported saved offline model kind")
    return _training_module(MODEL_ALGORITHMS[kind]).verify_model(
        args.model,
        args.dataset,
        source,
        policy,
        execution_scope=args.scope,
        max_source_rows=100_000 if args.max_source_rows is None else args.max_source_rows,
    )


if __name__ == "__main__":
    raise SystemExit(main())
