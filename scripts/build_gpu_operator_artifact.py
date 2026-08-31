#!/usr/bin/env python3
"""Build a strict P15 GPU operator artifact manifest from NVBit output."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from frontend.hetero.capture_allocation_ranges import subtract_address_ranges


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file(path: Path, base: Path, kind: str) -> dict[str, object]:
    try:
        rendered = str(path.relative_to(base))
    except ValueError:
        rendered = str(path)
    return {
        "kind": kind,
        "path": rendered,
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _capture_range_alignment(address: int, maximum: int = 256) -> int:
    """Return a truthful power-of-two alignment for a derived range start."""

    if address < 0 or maximum <= 0 or maximum & (maximum - 1):
        raise ValueError("address must be non-negative and maximum a power of two")
    if address == 0:
        return maximum
    return min(maximum, address & -address)


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--kernels-list", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--trace-manifest-output", type=Path)
    parser.add_argument("--qualification-record", type=Path)
    parser.add_argument("--gpu", default="NVIDIA GeForce RTX 3070")
    parser.add_argument("--driver", default="591.86")
    parser.add_argument("--target-sm", type=int, default=86)
    args = parser.parse_args()

    metadata_path = args.metadata.resolve()
    kernels_list = args.kernels_list.resolve()
    output = args.output.resolve()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != "heterosim-exact-llm-operator/v2":
        raise ValueError("metadata must use heterosim-exact-llm-operator/v2")
    if not kernels_list.is_file():
        raise FileNotFoundError(kernels_list)
    trace_files: list[Path] = []
    for line in kernels_list.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if not entry:
            continue
        trace_path = (kernels_list.parent / entry).resolve()
        if not trace_path.is_file():
            raise FileNotFoundError(trace_path)
        trace_files.append(trace_path)
    operator = str(metadata["operator"])
    context = int(metadata["context_length"])
    batch = int(metadata["batch_size"])
    tensors = [
        {
            "tensor_id": item["tensor_id"],
            "role": item["role"],
            "trace_base": item["address"],
            "size_bytes": item["size_bytes"],
            "shape": item["shape"],
            "strides": item["strides"],
            "dtype": item["dtype"],
            "layout": item["layout"],
            "alignment_bytes": item["alignment_bytes"],
        }
        for item in metadata["tensors"]
    ]
    allocator = metadata.get("capture_allocator")
    workspace_ranges: tuple[tuple[int, int], ...] = ()
    allocator_source: str | None = None
    trace_tensor_sizes = {
        str(item["tensor_id"]): int(item["size_bytes"])
        for item in metadata["tensors"]
    }
    if allocator is not None:
        if not isinstance(allocator, dict):
            raise ValueError("unsupported capture allocator metadata")
        allocator_source = str(allocator.get("source"))
        if allocator_source not in {
            "pytorch_cuda_caching_allocator_target_window",
            "pytorch_cuda_caching_allocator_target_window_plus_tensor_segments",
        }:
            raise ValueError("unsupported capture allocator metadata")
        raw_ranges = allocator.get("ranges")
        if not isinstance(raw_ranges, list):
            raise ValueError("capture allocator ranges must be an array")
        allocator_ranges = tuple(
            (int(item["address"]), int(item["address"]) + int(item["size_bytes"]))
            for item in raw_ranges
        )
        semantic_ranges: list[tuple[int, int]] = []
        for item in metadata["tensors"]:
            tensor_id = str(item["tensor_id"])
            begin = int(item["address"])
            logical_end = begin + int(item["size_bytes"])
            transaction_end = _align_up(logical_end, 32)
            if not any(
                range_begin <= begin and transaction_end <= range_end
                for range_begin, range_end in allocator_ranges
            ):
                raise ValueError(
                    f"allocator range cannot cover 32-byte transactions for {tensor_id}"
                )
            trace_tensor_sizes[tensor_id] = transaction_end - begin
            semantic_ranges.append((begin, transaction_end))
        ordered_semantic = sorted(semantic_ranges)
        for left, right in zip(ordered_semantic, ordered_semantic[1:]):
            if left[1] > right[0]:
                raise ValueError(
                    "32-byte transaction coverage overlaps semantic tensor ranges"
                )
        workspace_ranges = subtract_address_ranges(
            allocator_ranges,
            semantic_ranges,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    base = output.parent
    files = [_file(metadata_path, base, "operator_metadata")]
    files.append(_file(kernels_list, base, "kernels_list"))
    files.extend(_file(path, base, "instruction_trace") for path in trace_files)
    runtime_state = not trace_files
    if runtime_state and operator != "kv_append":
        raise ValueError(
            "an empty kernels list is only valid for the explicit "
            "kv_append state operation"
        )
    artifact_id = (
        f"tinyllama.1_1b.layer0.{operator}.prefill.bs{batch}_ctx{context}.fp16."
        + ("runtime_state_v1" if runtime_state else "sm86.accel_sim_v2")
    )
    trace_manifest_path = (
        args.trace_manifest_output.resolve() if args.trace_manifest_output else None
    )
    if runtime_state and trace_manifest_path is not None:
        raise ValueError("runtime_state does not have an Accel-Sim trace manifest")
    if trace_manifest_path is not None:
        trace_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        trace_manifest = {
            "schema_version": "hetero-trace-manifest/v1",
            "trace_id": artifact_id,
            "trace_semantics": "functional",
            "replay_safe": False,
            "qualification_record": None,
            "kernels_list": str(kernels_list),
            "capture": {
                "source": "P15 shape-locked operator capture",
                "tool": "NVBit",
                "version": "1.8",
                "model": metadata["model"],
                "model_revision": metadata["revision"],
                "operator": operator,
                "phase": metadata["phase"],
                "shape": {
                    "batch_size": batch,
                    "context_length": context,
                    "q_len": metadata["q_len"],
                    "kv_length": metadata["kv_length"],
                },
                "dtype": metadata["dtype"],
                "scope": metadata["scope"],
            },
            "compilation": {
                "pytorch": "2.3.0+cu121",
                "cuda_runtime": "12.1",
                "target_sm": args.target_sm,
                "implementation": metadata["implementation"],
            },
            "address_ranges": [
                {
                    "capture_allocation_id": f"{item['tensor_id']}.capture0",
                    "trace_base": item["address"],
                    "size_bytes": trace_tensor_sizes[str(item["tensor_id"])],
                    "tensor_id": item["tensor_id"],
                    "tensor_offset_bytes": 0,
                    "capture_epoch": 0,
                    "backing_allocation_id": f"{item['tensor_id']}.capture0",
                    "view_offset_bytes": 0,
                    "alignment_bytes": item["alignment_bytes"],
                    "shape": item["shape"],
                    "layout": item["layout"],
                }
                for item in metadata["tensors"]
            ]
            + [
                {
                    "capture_allocation_id": (
                        f"tinyllama.layer0.{operator}.opaque_workspace_{index}.capture0"
                    ),
                    "trace_base": begin,
                    "size_bytes": end - begin,
                    "tensor_id": (
                        f"tinyllama.layer0.{operator}.opaque_workspace_{index}"
                    ),
                    "tensor_offset_bytes": 0,
                    "capture_epoch": 0,
                    "backing_allocation_id": (
                        f"tinyllama.layer0.{operator}.opaque_workspace_{index}.capture0"
                    ),
                    "view_offset_bytes": 0,
                    "alignment_bytes": _capture_range_alignment(begin),
                    "shape": [end - begin],
                    "layout": "opaque_allocator_range",
                }
                for index, (begin, end) in enumerate(workspace_ranges)
            ],
        }
        trace_manifest_path.write_text(
            json.dumps(trace_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        files.append(_file(trace_manifest_path, base, "accel_sim_trace_manifest"))
    qualification_path = (
        args.qualification_record.resolve() if args.qualification_record else None
    )
    if qualification_path is not None:
        if runtime_state:
            raise ValueError("runtime_state cannot use an Accel-Sim qualification")
        qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
        if (
            qualification.get("schema_version")
            != "hetero-accel-sim-qualification/v1"
            or qualification.get("status") != "passed"
            or qualification.get("trace_id") != artifact_id
        ):
            raise ValueError("Accel-Sim qualification record does not match artifact")
        files.append(_file(qualification_path, base, "accel_sim_qualification_record"))
    manifest: dict[str, Any] = {
        "schema_version": "hetero-operator-artifact/v1",
        "artifact_id": artifact_id,
        "source_contract": {
            "model": metadata["model"],
            "model_spec_name": metadata["model_spec_name"],
            "checkpoint_revision": metadata["revision"],
            "operator": operator,
            "implementation": metadata["implementation"],
            "phase": metadata["phase"],
            "layer_id": metadata["layer_id"],
            "batch_size": batch,
            "context_length": context,
            "q_len": metadata["q_len"],
            "kv_length": metadata["kv_length"],
            "dtype": metadata["dtype"],
        },
        "backend": {
            "kind": "runtime_state" if runtime_state else "accel_sim",
            "tool": "NVBit",
            "tool_version": "1.8",
            "accel_sim_version": "2.0.0",
            "gpu": args.gpu,
            "driver": args.driver,
            "target_sm": args.target_sm,
        },
        "execution_contract": {
            "trace_semantics": "none" if runtime_state else "functional",
            "memory_traffic": "not_extracted",
            "supports_stall_resume": False,
            "compute_memory_coupled": False,
            "global_pa_binding_ready": False,
            "request_cycle_ready": False,
            "replay_safe_across_memory_candidates": False,
        },
        "address_contract": {
            "capture_address": (
                "cuda_allocation_address_no_instruction_trace"
                if runtime_state
                else "trace_address"
            ),
            "normalized_address": "tensor_id_plus_offset",
            "global_pa_binding": "required_at_simulation",
            "virtual_memory_mode": "identity_untranslated",
            "dram_mapping": "candidate_specific_after_global_pa",
            "capture_allocator_coverage": (
                (
                    "target_window_pytorch_allocator_plus_tensor_segments"
                    if allocator_source
                    == (
                        "pytorch_cuda_caching_allocator_target_window_plus_"
                        "tensor_segments"
                    )
                    else "target_window_pytorch_allocator"
                )
                if allocator is not None else "semantic_tensors_only"
            ),
        },
        "qualification": {
            "status": (
                "semantic_state_operation_pending_full_memory_lowering"
                if runtime_state
                else (
                    "standalone_accel_sim_qualified_pending_shared_request_cycle"
                    if qualification_path
                    else "capture_only_pending_cycle_qualification"
                )
            ),
            "performance_eligible": False,
            "qualification_record": (
                str(qualification_path) if qualification_path else None
            ),
        },
        "tensors": tensors,
        "files": files,
    }
    output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "artifact_id": artifact_id,
                "output": str(output),
                "trace_manifest": (
                    str(trace_manifest_path) if trace_manifest_path else None
                ),
                "trace_files": len(trace_files),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
