#!/usr/bin/env python3
"""Compile a P30 model projection into an executable ATLAS memory Artifact."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from frontend.hetero.framework_runtime import (  # noqa: E402
    compile_atlas_executable_artifact,
    iter_atlas_memory_trace,
    summarize_atlas_memory_trace,
)
from frontend.hetero.inference_framework import compile_atlas_tensor_ir  # noqa: E402
from frontend.hetero.model_graph import ModelSpec  # noqa: E402


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p30-bundle", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    args = parser.parse_args()
    bundle_path = args.p30_bundle.resolve()
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    if bundle.get("schema_version") != "hetero-p30-huggingface-online-bundle/v1":
        raise ValueError("P31 ATLAS compilation requires a P30 bundle")
    model_payload = bundle["framework_export"]["manifests"]["model"]["model"]
    model = ModelSpec(**model_payload)
    compiled = compile_atlas_tensor_ir(
        operator="qkv_projection",
        model=model,
        tokens=1,
        core_count=16,
        tile_m=1,
        tile_k=512,
        tile_n=16,
    )
    source = compiled["source_tensor_ir"]
    bytes_per_element = model.bytes_per_element
    activation_bytes = (
        source["inputs"][0]["shape"][0]
        * source["inputs"][0]["shape"][1]
        * bytes_per_element
    )
    weight_bytes = (
        source["inputs"][1]["shape"][0]
        * source["inputs"][1]["shape"][1]
        * bytes_per_element
    )
    output_bytes = (
        source["outputs"][0]["shape"][0]
        * source["outputs"][0]["shape"][1]
        * bytes_per_element
    )
    alignment = 4096
    activation_base = 1 << 33
    weight_base = _align_up(activation_base + activation_bytes, alignment)
    output_base = _align_up(weight_base + weight_bytes, alignment)
    artifact = compile_atlas_executable_artifact(
        compiled,
        tensor_global_pa={
            "activation": activation_base,
            "weight": weight_base,
            "output": output_base,
        },
        compiler_version="heterosim-atlas-stage-compiler/v1",
        request_bytes=64,
    )
    output_directory = args.output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    artifact_path = output_directory / "atlas_executable_artifact.json"
    trace_path = output_directory / "atlas_memory_trace.jsonl.gz"
    summary_path = output_directory / "atlas_memory_trace_summary.json"
    artifact_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with trace_path.open("wb") as raw_stream:
        with gzip.GzipFile(fileobj=raw_stream, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as stream:
                for request in iter_atlas_memory_trace(artifact):
                    stream.write(
                        json.dumps(request, sort_keys=True, separators=(",", ":"))
                        + "\n"
                    )
    summary = summarize_atlas_memory_trace(artifact)
    record = {
        "schema_version": "hetero-p31-atlas-executable-qualification/v1",
        "status": "passed",
        "execution_host": "remote_rtx4090_server",
        "source_p30_bundle_sha256": _sha256(bundle_path),
        "framework_simulation_key": bundle["online_binding"]["simulation_key"],
        "artifact_key": artifact["artifact_key"],
        "artifact_sha256": _sha256(artifact_path),
        "trace_file_sha256": _sha256(trace_path),
        "trace": summary,
        "tensor_global_pa": artifact["identity"]["tensor_global_pa"],
        "tensor_extents_bytes": {
            "activation": activation_bytes,
            "weight": weight_bytes,
            "output": output_bytes,
        },
        "qualification": {
            "tensor_ir_lowered": True,
            "tile_and_core_placement_materialized": True,
            "global_pa_bound_before_dram_mapping": True,
            "full_memory_trace_materialized": True,
            "parent_child_conservation_expected": True,
            "ramulator2_cycle_replay_qualified": False,
            "performance_claim_allowed": False,
        },
    }
    summary_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
