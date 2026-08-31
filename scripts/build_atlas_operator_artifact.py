#!/usr/bin/env python3
"""Wrap one generated ATLAS bundle in the strict P15 artifact contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atlas-bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--qualification-record", type=Path)
    args = parser.parse_args()

    bundle = args.atlas_bundle.resolve()
    source_manifest_path = bundle / "artifact_manifest.json"
    operator_path = bundle / "operator_description.yaml"
    placement_path = bundle / "data_placement.yaml"
    for path in (source_manifest_path, operator_path, placement_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source = source_manifest["source_contract"]
    if source.get("phase") != "prefill":
        raise ValueError("P15 wrapper currently accepts Prefill ATLAS bundles only")
    shape = source["shape"]
    local = source_manifest["per_core_local_address_space"]
    element_size = 2
    input_size = int(shape["M"]) * int(shape["K"]) * element_size
    output_size = (
        int(shape["M"])
        * int(source_manifest["lowering"]["per_core_shape"]["N"])
        * element_size
    )
    weight_size = (
        int(shape["K"])
        * int(source_manifest["lowering"]["per_core_shape"]["N"])
        * element_size
    )
    operator = str(source["operator"])
    tensors = [
        {
            "tensor_id": f"tinyllama.layer0.{operator}.input.per_core_template",
            "role": "input",
            "trace_base": int(local["input_base"]),
            "size_bytes": input_size,
            "shape": [int(shape["M"]), int(shape["K"])],
            "strides": [int(shape["K"]), 1],
            "dtype": "float16",
            "layout": "row_major_per_core_template",
            "alignment_bytes": 16384,
        },
        {
            "tensor_id": f"tinyllama.layer0.{operator}.output.per_core_template",
            "role": "output",
            "trace_base": int(local["output_base"]),
            "size_bytes": output_size,
            "shape": [
                int(shape["M"]),
                int(source_manifest["lowering"]["per_core_shape"]["N"]),
            ],
            "strides": [
                int(source_manifest["lowering"]["per_core_shape"]["N"]),
                1,
            ],
            "dtype": "float16",
            "layout": "row_major_per_core_template",
            "alignment_bytes": 16384,
        },
        {
            "tensor_id": f"tinyllama.layer0.{operator}.weight.per_core_template",
            "role": "parameter",
            "trace_base": int(local["weight_base"]),
            "size_bytes": weight_size,
            "shape": [
                int(shape["K"]),
                int(source_manifest["lowering"]["per_core_shape"]["N"]),
            ],
            "strides": [1, int(shape["K"])],
            "dtype": "float16",
            "layout": "column_major_per_core_template",
            "alignment_bytes": 16384,
        },
    ]
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    base = output.parent
    qualification_path = (
        args.qualification_record.resolve() if args.qualification_record else None
    )
    if qualification_path is not None:
        record = json.loads(qualification_path.read_text(encoding="utf-8"))
        if (
            record.get("schema_version") != "hetero-atlas-qualification/v1"
            or record.get("status") != "passed"
        ):
            raise ValueError("ATLAS qualification record is not passed")
    manifest: dict[str, Any] = {
        "schema_version": "hetero-operator-artifact/v1",
        "artifact_id": source_manifest["artifact_id"],
        "source_contract": {
            "model": source["model"],
            "model_spec_name": "TinyLlama-1.1B",
            "checkpoint_revision": source["checkpoint_revision"],
            "operator": operator,
            "implementation": source_manifest["lowering"]["rule"],
            "phase": "prefill",
            "layer_id": 0,
            "batch_size": int(source["batch_size"]),
            "context_length": int(source["context_length"]),
            "q_len": int(source["q_len"]),
            "kv_length": int(source["kv_length"]),
            "dtype": "fp16",
        },
        "backend": {
            "kind": "atlasim",
            "frontend": "ATLAS edge GEMM lowering",
            "core_count": int(source_manifest["lowering"]["core_count"]),
            "tile": source_manifest["lowering"]["tile"],
        },
        "execution_contract": {
            "trace_semantics": "declarative_task_graph",
            "memory_traffic": "full",
            "supports_stall_resume": True,
            "compute_memory_coupled": False,
            "global_pa_binding_ready": True,
            "request_cycle_ready": False,
            "replay_safe_across_memory_candidates": False,
        },
        "address_contract": {
            "capture_address": "atlas_per_core_local_address",
            "normalized_address": "tensor_id_plus_offset",
            "global_pa_binding": "required_at_simulation",
            "virtual_memory_mode": "identity_untranslated",
            "dram_mapping": "candidate_specific_after_global_pa",
        },
        "qualification": {
            "status": (
                "standalone_atlas_qualified_pending_shared_request_cycle"
                if qualification_path
                else "generated_pending_full_chip_request_cycle_qualification"
            ),
            "performance_eligible": False,
            "qualification_record": (
                str(qualification_path) if qualification_path else None
            ),
        },
        "tensors": tensors,
        "files": [
            _file(source_manifest_path, base, "atlas_source_manifest"),
            _file(operator_path, base, "atlas_operator_description"),
            _file(placement_path, base, "atlas_data_placement"),
            *(
                [_file(qualification_path, base, "atlas_qualification_record")]
                if qualification_path
                else []
            ),
        ],
    }
    output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"artifact_id": manifest["artifact_id"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
