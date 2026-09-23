#!/usr/bin/env python3
"""Build the P32 non-additive online Shadow Artifact timeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frontend.hetero.framework_shadow_runtime import (
    build_p32_online_shadow_timeline,
)


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root is not an object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p30-qualification", required=True, type=Path)
    parser.add_argument("--p31-qualification", required=True, type=Path)
    parser.add_argument("--p33-qualification", required=True, type=Path)
    parser.add_argument("--gpu-binding", required=True, type=Path)
    parser.add_argument("--atlas-artifact", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    record = build_p32_online_shadow_timeline(
        _load(args.p30_qualification),
        _load(args.p31_qualification),
        _load(args.p33_qualification),
        _load(args.gpu_binding),
        _load(args.atlas_artifact),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(record["qualification"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
