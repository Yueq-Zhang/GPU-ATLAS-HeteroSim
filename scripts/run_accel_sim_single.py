#!/usr/bin/env python3
"""Run one pinned Accel-Sim qualification leg.

This helper is intended for remote parallel validation.  Run the native and
adapter legs in separate output directories, then invoke ``qualify-gpu`` with
``--resume-completed-runs`` to validate and compare their completed records.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from frontend.hetero.backends.accel_sim import (
    AccelSimBackend,
    AccelSimBackendConfig,
    AccelSimBackendError,
)
from frontend.hetero.trace_manifest import TraceManifest, TraceManifestError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend-config", required=True, type=Path)
    parser.add_argument("--trace-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        config = AccelSimBackendConfig.load(args.backend_config)
        manifest = TraceManifest.load(args.trace_manifest)
        result = AccelSimBackend(config).run(manifest, args.output)
    except (AccelSimBackendError, TraceManifestError) as error:
        print(f"single Accel-Sim run failed: {error}")
        return 5

    print(
        json.dumps(
            {
                "status": "passed",
                "output": str(result.output_directory),
                "cycles": result.cycles,
                "instructions": result.instructions,
                "external_memory_stats": result.external_memory_stats,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
