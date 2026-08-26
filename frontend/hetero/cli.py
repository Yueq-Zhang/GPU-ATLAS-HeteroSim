"""Command line entrypoint for the M0/M1 control-plane scaffold."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from . import __version__
from .backends import AccelSimBackend, AccelSimBackendConfig, AccelSimBackendError
from .runner import execute_run, simulation_input_key
from .runtime_bridge import RuntimeUnavailableError
from .schema import ConfigError, load_and_validate_config
from .trace_manifest import TraceManifest, TraceManifestError


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gpu-atlas-heterosim")
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate a v1 experiment")
    validate.add_argument("--config", required=True, type=Path)

    run = subparsers.add_parser("run", help="validate a run request")
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--runs-root", type=Path)

    qualify_gpu = subparsers.add_parser(
        "qualify-gpu", help="compare the Accel-Sim adapter with its native baseline"
    )
    qualify_gpu.add_argument("--backend-config", required=True, type=Path)
    qualify_gpu.add_argument("--trace-manifest", required=True, type=Path)
    qualify_gpu.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "qualify-gpu":
        try:
            backend_config = AccelSimBackendConfig.load(args.backend_config)
            manifest = TraceManifest.load(args.trace_manifest)
            record = AccelSimBackend(backend_config).qualify(manifest, args.output)
        except (AccelSimBackendError, TraceManifestError) as error:
            print(f"GPU qualification error: {error}")
            return 5
        print(f"GPU qualification passed; record={record}")
        return 0

    try:
        config = load_and_validate_config(args.config)
    except ConfigError as error:
        print(f"configuration error: {error}")
        return 2

    key = simulation_input_key(config)
    if args.command == "validate":
        print(f"valid hetero-sim/v1 configuration; simulation_input_key={key}")
        return 0
    if args.dry_run:
        print(f"dry-run validated; simulation_input_key={key}")
        return 0
    project_root = Path(__file__).resolve().parents[2]
    try:
        run_dir = execute_run(config, project_root, args.runs_root)
    except RuntimeUnavailableError as error:
        print(f"runtime error: {error}")
        return 3
    except (ValueError, RuntimeError) as error:
        print(f"simulation error: {error}")
        return 4
    execution_mode = config["simulation"].get("execution_mode", "scheduler_validation")
    print(
        f"{str(execution_mode).replace('_', '-')} run completed; "
        f"simulation_input_key={key}; run_dir={run_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
