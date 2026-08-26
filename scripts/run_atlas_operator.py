#!/usr/bin/env python3
"""Run one prepared ATLAS operator bundle and emit machine-readable statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from atlasim import Chip


def _stats(stats: object) -> dict[str, int | float]:
    fields = (
        "e2e_cycles",
        "matrix_cycles",
        "vector_cycles",
        "buffer_cycles",
        "dram_cycles",
        "noc_cycles",
        "e2e_energy",
        "controller_energy",
        "matrix_energy",
        "vector_energy",
        "buffer_energy",
        "dram_energy",
        "noc_energy",
        "flop_count",
        "memory_access_bytes",
        "compute_non_overlap_cycles",
        "matrix_bubble_on_chip_cycles",
        "matrix_bubble_dram_cycles",
    )
    return {field: getattr(stats, field) for field in fields}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chip", required=True)
    parser.add_argument("--operators", required=True)
    parser.add_argument("--placement", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    chip = Chip(args.chip, args.operators, args.placement)
    performance = chip.simulate()
    payload = {
        "schema_version": "hetero-atlas-native-stats/v1",
        "chip_frequency_mhz": performance.chip_frequency,
        "chip_bandwidth_native": performance.chip_bw,
        "core_bandwidth_native": performance.core_bw,
        "e2e_stats": _stats(performance.e2e_stats),
        "operator_stats": {
            str(name): _stats(stats)
            for name, stats in performance.operator_stats
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
