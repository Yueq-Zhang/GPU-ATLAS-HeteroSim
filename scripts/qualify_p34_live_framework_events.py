#!/usr/bin/env python3
"""Qualify real vLLM and TensorRT-LLM event adapters by double run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frontend.hetero.framework_live_adapters import (
    qualify_live_framework_pair,
)


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root is not an object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-run-a", required=True, type=Path)
    parser.add_argument("--vllm-run-b", required=True, type=Path)
    parser.add_argument("--trtllm-run-a", required=True, type=Path)
    parser.add_argument("--trtllm-run-b", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    record = qualify_live_framework_pair(
        [_load(args.vllm_run_a), _load(args.vllm_run_b)],
        [_load(args.trtllm_run_a), _load(args.trtllm_run_b)],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(record["qualification"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
