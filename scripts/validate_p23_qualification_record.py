#!/usr/bin/env python3
"""Fail-closed validation for one P23 double-run qualification record."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def validate(path: Path) -> dict[str, int]:
    record = json.loads(path.read_text(encoding="utf-8"))
    comparison = dict(record.get("comparison", {}))
    ownership = dict(record.get("timing_ownership", {}))
    cycles = comparison.get("gpu_tot_sim_cycle")
    instructions = comparison.get("gpu_tot_sim_insn")
    memories = comparison.get("external_memory_stats")
    if (
        record.get("status") != "passed"
        or not isinstance(cycles, list)
        or len(cycles) != 2
        or cycles[0] != cycles[1]
        or not isinstance(instructions, list)
        or len(instructions) != 2
        or instructions[0] != instructions[1]
        or not isinstance(memories, list)
        or len(memories) != 2
        or memories[0] != memories[1]
        or ownership.get("duration_mode") != "coupled"
        or ownership.get("external_ramulator2") != "shared3d.ramulator2"
        or ownership.get("gpu_local_dram") is not None
    ):
        raise ValueError(f"invalid P23 deterministic qualification: {path}")
    stats = dict(memories[0])
    parents = int(stats.get("gpu_parents", -1))
    children = int(stats.get("gpu_children", -1))
    if (
        int(stats.get("instances", -1)) != 1
        or parents <= 0
        or children <= 0
        or int(stats.get("gpu_completed", -1)) != parents
        or int(stats.get("completed", -1)) != parents
        or int(stats.get("children_sent", -1)) != children
        or int(stats.get("children_completed", -1)) != children
        or int(stats.get("durable_completed", -1)) != parents
        or int(stats.get("address_translated", 0)) <= 0
        or int(stats.get("address_unmapped", -1)) != 0
        or int(stats.get("atlas_parents", -1)) != 0
        or int(stats.get("atlas_children", -1)) != 0
        or int(stats.get("atlas_completed", -1)) != 0
        or int(stats.get("outstanding", -1)) != 0
    ):
        raise ValueError(f"invalid P23 request conservation: {path}")
    return {
        "cycles": int(cycles[0]),
        "instructions": int(instructions[0]),
        "parents": parents,
        "children": children,
        "translated": int(stats["address_translated"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("record", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.record), sort_keys=True))


if __name__ == "__main__":
    main()
