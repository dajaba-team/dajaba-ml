"""M5 데이터 전처리만 수행하며, 탐지기 학습이나 백엔드 서비스 수정은 하지 않습니다."""

import argparse
import json
import logging
from pathlib import Path

from .input_inspection import inspect_inputs
from .prepare import execute, load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("prepare")
    build.add_argument(
        "--config", type=Path, default=Path(__file__).resolve().parents[2] / "configs/default.json"
    )
    build.add_argument("--source", action="append")
    inspect = commands.add_parser(
        "inspect-inputs", help="Read metadata/schema only; no full verify"
    )
    inspect.add_argument(
        "--config", type=Path, default=Path(__file__).resolve().parents[2] / "configs/default.json"
    )
    inspect.add_argument("--source", action="append")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        if args.command == "inspect-inputs":
            print(
                json.dumps(
                    inspect_inputs(load_config(args.config).index, args.source),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        root, ok = execute(load_config(args.config), args.source)
        print(f"PREPARATION_DIRECTORY={root}", flush=True)
        return 0 if ok else 2
    except KeyboardInterrupt:
        logging.error("Interrupted preparation is incomplete; rerun into a new output folder")
        return 130
    except (OSError, ValueError, KeyError) as exc:
        logging.error("Preparation/config error: %s", exc)
        return 2
