#!/usr/bin/env python3
"""Build a sealed P17 GPU-operator simulator measurement catalog."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frontend.hetero.gpu_operator_calibration import (  # noqa: E402
    build_native_vram_simulator_catalog,
    build_simulator_measurement_catalog,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--capabilities",
        type=Path,
        default=ROOT / "configs/hetero/operator_capabilities/"
        "tinyllama_prefill_layer0_bs1_ctx16.json",
    )
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--core-frequency-hz", type=int, default=1_132_000_000)
    parser.add_argument("--memory-topology", default="external_shared_3ddram")
    parser.add_argument(
        "--qualification-root",
        type=Path,
        help="Use native-VRAM Accel-Sim double-run records from this directory",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.qualification_root is None:
        payload = build_simulator_measurement_catalog(
            args.capabilities,
            args.repository_root,
            core_frequency_hz=args.core_frequency_hz,
            memory_topology=args.memory_topology,
        )
    else:
        payload = build_native_vram_simulator_catalog(
            args.capabilities,
            args.repository_root,
            args.qualification_root,
            core_frequency_hz=args.core_frequency_hz,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"P17 simulator catalog written: {args.output} "
        f"({payload['operator_count']} operators)"
    )


if __name__ == "__main__":
    main()
