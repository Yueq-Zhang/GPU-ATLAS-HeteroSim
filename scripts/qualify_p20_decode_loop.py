#!/usr/bin/env python3
"""Write one fail-closed P20 Decode-loop double-run record."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frontend.hetero.decode_loop_qualification import qualify_p20_decode_loop_pair


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--leg1", type=Path, required=True)
    parser.add_argument("--leg2", type=Path, required=True)
    parser.add_argument("--layers", type=int, choices=(1, 22), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = qualify_p20_decode_loop_pair(args.leg1, args.leg2, args.layers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
