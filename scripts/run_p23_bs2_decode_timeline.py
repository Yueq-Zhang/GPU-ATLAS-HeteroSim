#!/usr/bin/env python3
"""Execute one P23 exact-BS=2, one-layer Decode timeline leg."""

from __future__ import annotations

import argparse
from pathlib import Path

from frontend.hetero.p23_batched_decode import preflight_timeline, run_timeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/hetero/experiments/p23_tinyllama_decode1_1layer_bs2_real_trace.json"
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.preflight_only:
        print(preflight_timeline(args.config, args.output))
        return
    result = run_timeline(args.config, args.output)
    print(result)


if __name__ == "__main__":
    main()
