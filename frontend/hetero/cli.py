"""Command line entrypoint for the M0/M1 control-plane scaffold."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

from . import __version__
from .schema import ConfigError, load_and_validate_config


def _input_key(config: dict[str, object]) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gpu-atlas-heterosim")
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate a v1 experiment")
    validate.add_argument("--config", required=True, type=Path)

    run = subparsers.add_parser("run", help="validate a run request")
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        config = load_and_validate_config(args.config)
    except ConfigError as error:
        print(f"configuration error: {error}")
        return 2

    key = _input_key(config)
    if args.command == "validate":
        print(f"valid hetero-sim/v1 configuration; simulation_input_key={key}")
        return 0
    if args.dry_run:
        print(f"dry-run validated; simulation_input_key={key}")
        return 0
    print("runtime execution is not implemented in the M0/M1 scaffold; use --dry-run")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())

