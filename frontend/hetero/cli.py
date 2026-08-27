"""Command line entrypoint for the M0/M1 control-plane scaffold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from . import __version__
from .backends import (
    AccelSimBackend,
    AccelSimBackendConfig,
    AccelSimBackendError,
    AtlasArtifact,
    AtlasBackend,
    AtlasBackendConfig,
    AtlasBackendError,
    MemoryBridgeError,
    Ramulator2Backend,
    Ramulator2BackendConfig,
    Ramulator2BackendError,
    run_jsonl_bridge,
)
from .runner import execute_run, simulation_input_key
from .dse import DseError, run_dse
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

    dse = subparsers.add_parser("dse", help="run deterministic outer-loop DSE")
    dse.add_argument("--config", required=True, type=Path)
    dse.add_argument("--search", required=True, type=Path)
    dse.add_argument("--output-root", required=True, type=Path)

    bridge = subparsers.add_parser(
        "bridge-memory", help="translate trace addresses and service JSONL requests"
    )
    bridge.add_argument("--trace-manifest", required=True, type=Path)
    bridge.add_argument("--buffer-bindings", required=True, type=Path)
    bridge.add_argument("--memory-config", required=True, type=Path)
    bridge.add_argument("--requests", required=True, type=Path)
    bridge.add_argument("--responses", required=True, type=Path)

    qualify_memory = subparsers.add_parser(
        "qualify-memory", help="run deterministic standalone Ramulator2 replay twice"
    )
    qualify_memory.add_argument("--backend-config", required=True, type=Path)
    qualify_memory.add_argument("--requests", required=True, type=Path)
    qualify_memory.add_argument("--output", required=True, type=Path)

    qualify_gpu = subparsers.add_parser(
        "qualify-gpu", help="compare the Accel-Sim adapter with its native baseline"
    )
    qualify_gpu.add_argument("--backend-config", required=True, type=Path)
    qualify_gpu.add_argument("--trace-manifest", required=True, type=Path)
    qualify_gpu.add_argument("--output", required=True, type=Path)
    qualify_atlas = subparsers.add_parser(
        "qualify-atlas", help="check deterministic ATLAS adapter equivalence"
    )
    qualify_atlas.add_argument("--backend-config", required=True, type=Path)
    qualify_atlas.add_argument("--chip-config", required=True, type=Path)
    qualify_atlas.add_argument("--operator-list", required=True, type=Path)
    qualify_atlas.add_argument("--placement-map", required=True, type=Path)
    qualify_atlas.add_argument("--output", required=True, type=Path)
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
    if args.command == "qualify-atlas":
        try:
            backend_config = AtlasBackendConfig.load(args.backend_config)
            artifact = AtlasArtifact(args.operator_list, args.placement_map)
            record = AtlasBackend(backend_config).qualify(
                args.chip_config, artifact, args.output
            )
        except AtlasBackendError as error:
            print(f"ATLAS qualification error: {error}")
            return 6
        print(f"ATLAS qualification passed; record={record}")
        return 0
    if args.command == "bridge-memory":
        try:
            result = run_jsonl_bridge(
                args.trace_manifest,
                args.buffer_bindings,
                args.memory_config,
                args.requests,
                args.responses,
            )
        except (OSError, ValueError, MemoryBridgeError) as error:
            print(f"memory bridge error: {error}")
            return 8
        print(
            "memory bridge completed; "
            f"requests={result['parent_requests_completed']}; "
            f"responses={args.responses}"
        )
        return 0
    if args.command == "qualify-memory":
        try:
            payload = json.loads(args.requests.read_text(encoding="utf-8"))
            if not isinstance(payload, list) or not payload:
                raise Ramulator2BackendError("request input must be a non-empty array")
            record = Ramulator2Backend(
                Ramulator2BackendConfig.load(args.backend_config)
            ).qualify(payload, args.output)
        except (OSError, ValueError, Ramulator2BackendError) as error:
            print(f"Ramulator2 qualification error: {error}")
            return 9
        print(f"Ramulator2 qualification passed; record={record}")
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
    if args.command == "dse":
        try:
            search = json.loads(args.search.read_text(encoding="utf-8"))
            if not isinstance(search, dict):
                raise DseError("DSE search root must be an object")
            project_root = Path(__file__).resolve().parents[2]
            report = run_dse(
                config, search, project_root, args.output_root
            )
        except (OSError, ValueError, RuntimeError) as error:
            print(f"DSE error: {error}")
            return 7
        print(f"DSE completed; report={report}")
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
