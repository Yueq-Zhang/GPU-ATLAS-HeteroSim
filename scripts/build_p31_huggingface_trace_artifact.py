#!/usr/bin/env python3
"""Seal one framework-selected Layer-0 NVBit trace and bind it to Global PA."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from frontend.hetero.online_address_binding import (  # noqa: E402
    PackedRangeRebasePolicy,
    materialize_online_address_bindings,
)
from frontend.hetero.trace_manifest import TraceManifest  # noqa: E402


HEADERS = ("kernel name", "grid dim", "block dim", "shmem", "nregs", "binary version")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _alignment(address: int, maximum: int = 256) -> int:
    return maximum if address == 0 else min(maximum, address & -address)


def _descriptor(lines: Iterable[str], trace: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for index, line in enumerate(lines):
        if index >= 128:
            break
        stripped = line.strip()
        if not stripped.startswith("-") or "=" not in stripped:
            continue
        key, value = stripped[1:].split("=", maxsplit=1)
        key = key.strip()
        if key in HEADERS:
            values[key] = value.strip()
        if all(key in values for key in HEADERS):
            break
    missing = [key for key in HEADERS if key not in values]
    if missing:
        raise ValueError(f"trace header is incomplete for {trace}: {missing}")
    return {key: values[key] for key in HEADERS}


def _read_descriptor(trace: Path) -> dict[str, str]:
    if trace.suffix == ".tracez":
        process = subprocess.Popen(
            ["zstdcat", str(trace)], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        assert process.stdout is not None
        try:
            header = process.stdout.read(64 * 1024).decode("utf-8", errors="replace")
            return _descriptor(header.splitlines(), trace)
        finally:
            process.terminate()
            process.wait(timeout=10)
    with trace.open("r", encoding="utf-8", errors="replace") as stream:
        return _descriptor(stream, trace)


def _file(path: Path, kind: str) -> dict[str, object]:
    return {
        "kind": kind,
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _load_trace_files(kernels_list: Path) -> tuple[list[Path], list[dict[str, str]]]:
    traces: list[Path] = []
    for raw in kernels_list.read_text(encoding="utf-8").splitlines():
        entry = raw.strip()
        if entry:
            trace = (kernels_list.parent / entry).resolve()
            if not trace.is_file():
                raise FileNotFoundError(trace)
            traces.append(trace)
    if not traces:
        raise ValueError("framework Layer-0 capture contains no kernels")
    return traces, [_read_descriptor(trace) for trace in traces]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--kernels-list", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--trace-manifest-output", required=True, type=Path)
    parser.add_argument("--binding-directory", required=True, type=Path)
    args = parser.parse_args()
    bundle_path = args.bundle.resolve()
    kernels_list = args.kernels_list.resolve()
    output = args.output.resolve()
    trace_manifest_path = args.trace_manifest_output.resolve()
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    if bundle.get("schema_version") != "hetero-p30-huggingface-online-bundle/v1":
        raise ValueError("P31 requires a P30 Hugging Face online bundle")
    raw = bundle["raw_observation"]
    capture = raw["capture"]
    if (
        capture.get("capture_control") != "nvbit_exported_instrumentation_api"
        or capture.get("nvbit_profiler_phase") != "decode_step"
        or capture.get("nvbit_instrumentation_range_emitted") is not True
    ):
        raise ValueError("P31 bundle did not delimit the Decode Layer-0 NVBit window")
    device = raw["device"]
    if device.get("name") != "NVIDIA GeForce RTX 4090" or device.get(
        "compute_capability"
    ) != [8, 9]:
        raise ValueError("P31 SASS must be captured on the remote RTX 4090/SM89")
    allocator = raw.get("capture_allocator")
    if not isinstance(allocator, Mapping) or not allocator.get("ranges"):
        raise ValueError("P31 requires full-window CUDA allocator ranges")
    ranges = sorted(
        (
            int(item["address"]),
            int(item["address"]) + int(item["size_bytes"]),
        )
        for item in allocator["ranges"]
    )
    for index, ((begin, end), following) in enumerate(
        zip(ranges, ranges[1:] + [(1 << 64, 1 << 64)])
    ):
        if begin < 0 or end <= begin or end > following[0]:
            raise ValueError(f"invalid or overlapping allocator range {index}")

    traces, descriptors = _load_trace_files(kernels_list)
    binary_versions = sorted({int(item["binary version"]) for item in descriptors})
    if not set(binary_versions).issubset({80, 86}):
        raise ValueError(
            f"Accel-Sim 2.0 has no qualified replay contract for {binary_versions}"
        )
    replay_target_sm = 86
    trace_id = (
        "tinyllama.1_1b.huggingface.layer0.decode_step.bs1.ctx16.q1.kv17."
        "fp16.remote_sm89.ampere_replay_sm86.accel_sim_v2"
    )
    compilation = {
        "framework": raw["framework"],
        "capture_device_sm": 89,
        "binary_identity": {
            "source": "nvbit_trace_headers",
            "binary_versions": binary_versions,
            "mixed_binary_versions": len(binary_versions) > 1,
            "replay_target_sm": replay_target_sm,
            "replay_compatibility": (
                "single_binary_exact"
                if binary_versions == [86]
                else "ampere_sm80_sm86_opcode_compatible"
            ),
            "kernel_launch_count": len(descriptors),
            "kernel_sequence_sha256": _canonical_sha256(descriptors),
        },
        "framework_simulation_key": bundle["online_binding"]["simulation_key"],
    }
    address_ranges = [
        {
            "capture_allocation_id": f"hf.layer0.allocator.{index}.capture0",
            "trace_base": hex(begin),
            "size_bytes": end - begin,
            "tensor_id": f"hf.layer0.opaque_allocator_{index}",
            "tensor_offset_bytes": 0,
            "capture_epoch": 0,
            "backing_allocation_id": f"hf.layer0.allocator.{index}.capture0",
            "view_offset_bytes": 0,
            "alignment_bytes": _alignment(begin),
            "shape": [end - begin],
            "layout": "opaque_pytorch_allocator_range",
        }
        for index, (begin, end) in enumerate(ranges)
    ]
    trace_manifest = {
        "schema_version": "hetero-trace-manifest/v1",
        "trace_id": trace_id,
        "trace_semantics": "functional",
        "replay_safe": False,
        "qualification_record": None,
        "kernels_list": str(kernels_list),
        "capture": {
            "tool": "NVBit",
            "tool_version": "1.8",
            "host": "remote_rtx4090_sm89",
            "phase": "decode_step",
            "module_path": "model.layers.0",
            "control": "nvbit_exported_instrumentation_api",
            "allocator_coverage": "full_runtime_window_segments_and_alloc_events",
        },
        "compilation": compilation,
        "address_ranges": address_ranges,
    }
    trace_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    trace_manifest_path.write_text(
        json.dumps(trace_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = TraceManifest.load(trace_manifest_path)
    binding = materialize_online_address_bindings(
        manifest,
        PackedRangeRebasePolicy(
            mode="range_rebase_packed_manifest",
            memory_space_id="shared0.dram3d",
            physical_base_bytes=0,
            capacity_bytes=1 << 32,
            alignment_bytes=256,
            require_nonzero_translations=True,
        ),
        args.binding_directory.resolve(),
    )
    files = [
        _file(bundle_path, "p30_online_bundle"),
        _file(kernels_list, "kernels_list"),
    ]
    files.extend(_file(trace, "sass_instruction_trace") for trace in traces)
    files.extend(
        (
            _file(trace_manifest_path, "trace_manifest"),
            _file(Path(binding["table_path"]), "online_address_bindings"),
            _file(
                args.binding_directory.resolve() / "online_address_binding.json",
                "online_address_binding_record",
            ),
        )
    )
    tensors = [
        {
            "tensor_id": item["tensor_id"],
            "role": "opaque_runtime_allocator",
            "trace_base": begin,
            "size_bytes": end - begin,
            "shape": [end - begin],
            "strides": [1],
            "dtype": "byte",
            "layout": item["layout"],
            "alignment_bytes": item["alignment_bytes"],
        }
        for item, (begin, end) in zip(address_ranges, ranges)
    ]
    artifact = {
        "schema_version": "hetero-operator-artifact/v1",
        "artifact_id": trace_id,
        "source_contract": {
            "model": raw["model"]["name"],
            "model_spec_name": "TinyLlama-1.1B",
            "checkpoint_revision": raw["model"]["revision"],
            "operator": "huggingface_layer0",
            "implementation": "transformers_native_layer0_forward",
            "phase": "decode_step",
            "layer_id": 0,
            "batch_size": 1,
            "context_length": 16,
            "q_len": 1,
            "kv_length": 17,
            "dtype": "fp16",
        },
        "backend": {
            "kind": "accel_sim",
            "tool": "NVBit",
            "tool_version": "1.8",
            "accel_sim_version": "2.0.0",
            "gpu": device["name"],
            "driver": raw["framework"].get("cuda_runtime", "unknown"),
            "capture_device_sm": 89,
            "target_sm": replay_target_sm,
            "binary_versions": binary_versions,
        },
        "execution_contract": {
            "trace_semantics": "functional",
            "memory_traffic": "full_instruction_trace",
            "supports_stall_resume": False,
            "compute_memory_coupled": False,
            "global_pa_binding_ready": True,
            "request_cycle_ready": False,
            "replay_safe_across_memory_candidates": False,
        },
        "address_contract": {
            "capture_address": "trace_address",
            "normalized_address": "tensor_id_plus_offset",
            "global_pa_binding": "required_at_simulation",
            "virtual_memory_mode": "identity_untranslated",
            "dram_mapping": "candidate_specific_after_global_pa",
            "capture_allocator_coverage": "target_window_pytorch_allocator",
        },
        "qualification": {
            "status": "framework_selected_sass_captured_pending_cycle_qualification",
            "performance_eligible": False,
            "qualification_record": None,
            "limitations": [
                "Capture device SM89 is distinct from actual SASS binary versions.",
                "Global PA binding is materialized but request-cycle replay is pending.",
                "No calibrated performance claim is permitted.",
            ],
        },
        "tensors": tensors,
        "files": files,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "artifact": str(output),
                "trace_id": trace_id,
                "kernel_launch_count": len(traces),
                "binary_versions": binary_versions,
                "allocator_range_count": len(ranges),
                "global_pa_range_count": binding["range_count"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
