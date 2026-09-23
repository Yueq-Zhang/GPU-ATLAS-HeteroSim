#!/usr/bin/env python3
"""Qualify two real Hugging Face online-observation runs for P30."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _validate_bundle(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "hetero-p30-huggingface-online-bundle/v1":
        raise ValueError(f"invalid P30 bundle: {path}")
    raw = _mapping(payload.get("raw_observation"), "raw observation")
    normalized = _mapping(payload.get("normalized_runtime"), "normalized runtime")
    binding = _mapping(payload.get("online_binding"), "online binding")
    qualification = _mapping(payload.get("qualification"), "qualification")
    if raw.get("schema_version") != "hetero-huggingface-runtime-observation/v1":
        raise ValueError("raw runtime schema is invalid")
    if normalized.get("schema_version") != "hetero-huggingface-runtime-normalized/v1":
        raise ValueError("normalized runtime schema is invalid")
    if binding.get("schema_version") != "hetero-huggingface-online-shadow-binding/v1":
        raise ValueError("online binding schema is invalid")
    required_flags = (
        "actual_huggingface_request_executed",
        "actual_runtime_callbacks_observed",
        "tensor_alias_and_global_pa_bound",
        "p28_simulation_key_connected",
    )
    if any(qualification.get(flag) is not True for flag in required_flags):
        raise ValueError("P30 runtime qualification flags are incomplete")
    if qualification.get("performance_claim_allowed") is not False:
        raise ValueError("P30 must remain non-performance-qualified")

    identity = _mapping(normalized.get("identity"), "identity")
    events = identity.get("events")
    runtime_events = normalized.get("runtime_events")
    storages = _mapping(normalized.get("memory_map"), "memory map").get("storages")
    tensor_bindings = _mapping(
        normalized.get("tensor_global_pa_bindings"), "tensor bindings"
    )
    if not isinstance(events, Sequence) or not isinstance(runtime_events, Sequence):
        raise ValueError("P30 event arrays are missing")
    if not isinstance(storages, Sequence) or not storages or not tensor_bindings:
        raise ValueError("P30 runtime memory map is empty")
    storage_rows = []
    for item in storages:
        storage = _mapping(item, "storage")
        begin = int(storage["global_address"])
        end = begin + int(storage["size_bytes"])
        if begin < 0 or end <= begin:
            raise ValueError("invalid P30 Global PA storage")
        storage_rows.append((begin, end, str(storage["storage_id"])))
    storage_rows.sort()
    for left, right in zip(storage_rows, storage_rows[1:]):
        if left[1] > right[0]:
            raise ValueError("P30 Global PA storages overlap")
    device = _mapping(raw.get("device"), "device")
    if device.get("name") != "NVIDIA GeForce RTX 4090" or device.get(
        "compute_capability"
    ) != [8, 9]:
        raise ValueError("P30 must execute on the remote RTX 4090/SM89")
    return {
        "bundle_sha256": _sha256(path),
        "identity_sha256": normalized["identity_sha256"],
        "allocation_observation_sha256": normalized["allocation_observation_sha256"],
        "memory_map_sha256": normalized["memory_map_sha256"],
        "simulation_key": binding["simulation_key"],
        "result": raw["result"],
        "event_count": len(events),
        "module_count": len(identity["module_sequence"]),
        "storage_count": len(storage_rows),
        "tensor_binding_count": len(tensor_bindings),
        "alias_group_count": len(normalized["alias_groups"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-a", required=True, type=Path)
    parser.add_argument("--run-b", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    runs = [
        _validate_bundle(args.run_a.resolve()),
        _validate_bundle(args.run_b.resolve()),
    ]
    stable_fields = (
        "identity_sha256",
        "simulation_key",
        "result",
        "event_count",
        "module_count",
    )
    if any(runs[0][field] != runs[1][field] for field in stable_fields):
        raise ValueError("P30 semantic double-run identity is not deterministic")
    record = {
        "schema_version": "hetero-p30-huggingface-online-qualification/v1",
        "status": "passed",
        "execution_host": "remote_rtx4090_sm89",
        "workload": {
            "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "revision": "fe8a4ea1ffedaf415f4da2f062534de366a451e6",
            "batch_size": 1,
            "prompt_tokens": 16,
            "decode_tokens": 1,
            "dtype": "fp16",
        },
        "double_run": {
            "semantic_identity_equal": True,
            "simulation_key_equal": True,
            "semantic_result_equal": True,
            "allocator_workspace_may_vary": True,
            "runs": runs,
        },
        "qualification": {
            "actual_huggingface_request_executed": True,
            "module_callbacks_connected": True,
            "tensor_views_and_aliases_observed": True,
            "runtime_global_pa_bound": True,
            "p28_simulation_key_connected": True,
            "sass_trace_generated": False,
            "performance_claim_allowed": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
