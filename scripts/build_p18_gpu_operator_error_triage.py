#!/usr/bin/env python3
"""Build the P18 GPU operator calibration-error baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frontend.hetero.gpu_operator_error_triage import (  # noqa: E402
    build_gpu_operator_error_triage,
)


def _payload(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _locator(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pairing-audit",
        type=Path,
        default=ROOT
        / "validation/p17/gpu_operator_pairing/native_vram_pairing_audit.json",
    )
    parser.add_argument(
        "--simulator-catalog",
        type=Path,
        default=ROOT
        / "validation/p17/gpu_operator_pairing/simulator_native_vram.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "validation/p18/gpu_operator_error_triage.json",
    )
    args = parser.parse_args()

    report = build_gpu_operator_error_triage(
        _payload(args.pairing_audit), _payload(args.simulator_catalog)
    )
    report["inputs"] = {
        "pairing_audit": _locator(args.pairing_audit),
        "pairing_audit_sha256": _sha256(args.pairing_audit),
        "simulator_catalog": _locator(args.simulator_catalog),
        "simulator_catalog_sha256": _sha256(args.simulator_catalog),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = report["summary"]
    print(
        f"P18 GPU error triage written: {args.output}; "
        f"paired={summary['paired_operator_count']}/{summary['operator_count']}; "
        f"mean_error={summary['mean_absolute_relative_error']:.6f}"
    )


if __name__ == "__main__":
    main()
