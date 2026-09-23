#!/usr/bin/env python3
"""Qualify deterministic GPU and ATLAS P31 request-cycle Artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frontend.hetero.framework_shadow_runtime import (
    qualify_p33_cycle_replays,
)


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root is not an object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-run-a", required=True, type=Path)
    parser.add_argument("--gpu-run-b", required=True, type=Path)
    parser.add_argument("--atlas-run-a", required=True, type=Path)
    parser.add_argument("--atlas-run-b", required=True, type=Path)
    parser.add_argument("--p31-qualification", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    p31 = _load(args.p31_qualification)
    record = qualify_p33_cycle_replays(
        [_load(args.gpu_run_a), _load(args.gpu_run_b)],
        [_load(args.atlas_run_a), _load(args.atlas_run_b)],
        framework_simulation_key=str(p31.get("p30_simulation_key", "")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(record["qualification"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
