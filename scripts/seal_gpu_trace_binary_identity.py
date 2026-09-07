#!/usr/bin/env python3
"""Seal executed SASS binary identity from NVBit Trace headers into metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from collections.abc import Iterable


REQUIRED_HEADERS = (
    "kernel name",
    "grid dim",
    "block dim",
    "shmem",
    "nregs",
    "binary version",
)


def _resolve_binary_contract(
    binary_versions: list[int],
    *,
    required_binary_sm: int | None,
    allowed_binary_sms: list[int],
    replay_target_sm: int | None,
) -> tuple[int, str]:
    """Resolve one replay target without hiding mixed executed binaries.

    Accel-Sim 2.0 maps both SM80 and SM86 Trace headers to its Ampere opcode
    table.  Mixed binaries are therefore permitted only when the caller opts
    into that exact family and records one explicit replay configuration.
    """

    if required_binary_sm is not None and allowed_binary_sms:
        raise ValueError(
            "--require-binary-sm and --allow-binary-sm are mutually exclusive"
        )
    if allowed_binary_sms:
        allowed = set(allowed_binary_sms)
        actual = set(binary_versions)
        if not actual.issubset(allowed):
            raise ValueError(
                f"SASS binary versions {binary_versions} exceed allowed set "
                f"{sorted(allowed)}"
            )
        if not actual.issubset({80, 86}) or not allowed.issubset({80, 86}):
            raise ValueError("mixed-binary replay is limited to SM80/SM86 Ampere")
        if replay_target_sm not in {80, 86}:
            raise ValueError(
                "mixed Ampere replay requires --replay-target-sm 80 or 86"
            )
        return replay_target_sm, "ampere_sm80_sm86_opcode_compatible"
    if len(binary_versions) != 1:
        raise ValueError(
            f"mixed SASS binary versions are unsupported: {binary_versions}"
        )
    binary_sm = binary_versions[0]
    if required_binary_sm is not None and binary_sm != required_binary_sm:
        raise ValueError(
            f"executed binary is SM{binary_sm}, expected SM{required_binary_sm}"
        )
    if replay_target_sm is not None and replay_target_sm != binary_sm:
        raise ValueError(
            f"replay target SM{replay_target_sm} does not match executed SM{binary_sm}"
        )
    return binary_sm, "single_binary_exact"


def _descriptor(lines: Iterable[str], trace: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for index, line in enumerate(lines):
        if index >= 128:
            break
        stripped = line.strip()
        if not stripped.startswith("-") or "=" not in stripped:
            continue
        name, value = stripped[1:].split("=", maxsplit=1)
        name = name.strip()
        if name in REQUIRED_HEADERS:
            values[name] = value.strip()
        if all(name in values for name in REQUIRED_HEADERS):
            break
    missing = [name for name in REQUIRED_HEADERS if name not in values]
    if missing:
        raise ValueError(f"Trace header is incomplete for {trace}: {missing}")
    return {name: values[name] for name in REQUIRED_HEADERS}


def _read_descriptor(trace: Path) -> dict[str, str]:
    if trace.suffix == ".tracez":
        process = subprocess.Popen(
            ["zstdcat", str(trace)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert process.stdout is not None
        try:
            header = process.stdout.read(64 * 1024).decode(
                "utf-8", errors="replace"
            )
            return _descriptor(header.splitlines(), trace)
        finally:
            process.terminate()
            process.wait(timeout=10)
    with trace.open("r", encoding="utf-8", errors="replace") as stream:
        return _descriptor(stream, trace)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--kernels-list", required=True, type=Path)
    parser.add_argument("--expected-device-sm", required=True, type=int)
    parser.add_argument("--require-binary-sm", type=int)
    parser.add_argument("--allow-binary-sm", action="append", type=int, default=[])
    parser.add_argument("--replay-target-sm", type=int)
    args = parser.parse_args()

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    compilation = metadata.get("compilation")
    if not isinstance(compilation, dict):
        raise ValueError("metadata compilation identity is absent")
    recorded_device_sm = int(
        compilation.get("capture_device_sm", compilation["target_sm"])
    )
    if recorded_device_sm != args.expected_device_sm:
        raise ValueError(
            f"capture device SM {recorded_device_sm} does not match expected "
            f"SM {args.expected_device_sm}"
        )

    descriptors: list[dict[str, str]] = []
    for raw in args.kernels_list.read_text(encoding="utf-8").splitlines():
        entry = raw.strip()
        if not entry:
            continue
        trace = (args.kernels_list.parent / entry).resolve()
        if not trace.is_file():
            raise FileNotFoundError(trace)
        descriptors.append(_read_descriptor(trace))
    if not descriptors:
        raise ValueError("cannot seal an empty kernel sequence")
    binary_versions = sorted({int(item["binary version"]) for item in descriptors})
    replay_target_sm, compatibility = _resolve_binary_contract(
        binary_versions,
        required_binary_sm=args.require_binary_sm,
        allowed_binary_sms=args.allow_binary_sm,
        replay_target_sm=args.replay_target_sm,
    )

    compilation["capture_device_sm"] = recorded_device_sm
    compilation["target_sm"] = replay_target_sm
    compilation["binary_identity"] = {
        "source": "nvbit_trace_headers",
        "binary_versions": binary_versions,
        "mixed_binary_versions": len(binary_versions) > 1,
        "replay_target_sm": replay_target_sm,
        "replay_compatibility": compatibility,
        "kernel_launch_count": len(descriptors),
        "kernel_sequence_sha256": _canonical_sha256(descriptors),
    }
    args.metadata.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(replay_target_sm)


if __name__ == "__main__":
    main()
