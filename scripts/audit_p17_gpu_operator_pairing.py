#!/usr/bin/env python3
"""Fail-closed exact-operator native/simulator pairing audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frontend.hetero.gpu_operator_calibration import (  # noqa: E402
    audit_gpu_operator_pairing,
)


def _payload(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"catalog root must be an object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", required=True, type=Path)
    parser.add_argument("--simulator", required=True, type=Path)
    parser.add_argument(
        "--capabilities",
        type=Path,
        default=ROOT / "configs/hetero/operator_capabilities/"
        "tinyllama_prefill_layer0_bs1_ctx16.json",
    )
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--max-relative-error", type=float, default=0.15)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    audit = audit_gpu_operator_pairing(
        _payload(args.native),
        _payload(args.simulator),
        args.capabilities,
        args.repository_root,
        max_relative_error=args.max_relative_error,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"P17 GPU pairing audit written: {args.output}; "
        f"paired={audit['paired_operator_count']}/"
        f"{audit['required_operator_count']}"
    )


if __name__ == "__main__":
    main()
