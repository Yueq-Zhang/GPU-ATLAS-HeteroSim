#!/usr/bin/env python3
"""Run one P31 ATLAS Global-PA trace through the live Ramulator2 bridge."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frontend.hetero.framework_shadow_runtime import (
    run_atlas_cycle_replay,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("schema_version") != "hetero-p33-atlas-cycle-replay-config/v1":
        raise ValueError("unsupported P33 ATLAS cycle configuration")
    record = run_atlas_cycle_replay(
        ROOT, config["memory_service"], args.trace.resolve()
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "request_count": record["request_count"],
                "completion_cycle": record["ramulator2"]["gpu_cycles"],
                "status": "passed",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
