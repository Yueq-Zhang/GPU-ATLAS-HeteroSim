#!/usr/bin/env python3
"""Generate a shape-locked ATLAS GEMM bundle for one decoder projection.

The lowering follows ATLAS's edge GEMM frontend in
``frontend/atlang/simulator/edge/gemm.py``.  The output dimension is column
sharded across logic-die cores; every core receives the same input vector and
one disjoint weight/output slice.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROW_BYTES = 16_384


def _align(value: int, alignment: int = ROW_BYTES) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _dram_task(
    *,
    name: str,
    is_write: bool,
    extent: list[int],
    stride_add: list[int],
    offset_add: list[int],
    init_iter: int,
    stride_iter: int,
    total_iter: int,
) -> dict[str, Any]:
    return {
        "name": name,
        "is_write": is_write,
        "access_base": [0, 0],
        "access_extent": extent,
        "access_stride_add": stride_add,
        "access_offset_add": offset_add,
        "init_iter": init_iter,
        "stride_iter": stride_iter,
        "total_iter": total_iter,
    }


def build_bundle(
    *,
    m_dim: int,
    k_dim: int,
    n_dim: int,
    core_count: int,
    tile_m: int,
    tile_k: int,
    tile_n: int,
    element_size: int,
    artifact_id: str = (
        "tinyllama.1_1b.layer0.q_proj.decode.bs1_ctx1024.fp16.atlas_edge"
    ),
    source_operator: str = "model.layers.0.self_attn.q_proj",
    phase: str = "decode_step",
    batch_size: int = 1,
    context_length: int = 1024,
    bundle_name: str = "q_proj",
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    values = (m_dim, k_dim, n_dim, core_count, tile_m, tile_k, tile_n, element_size)
    if any(value <= 0 for value in values):
        raise ValueError("all dimensions, counts and sizes must be positive")
    if n_dim % core_count:
        raise ValueError("N must be evenly divisible by core_count")

    per_core_n = n_dim // core_count
    if m_dim % tile_m or k_dim % tile_k or per_core_n % tile_n:
        raise ValueError("M, K and per-core N must be divisible by their tile sizes")

    n_m = m_dim // tile_m
    n_k = k_dim // tile_k
    n_n = per_core_n // tile_n
    iterations = n_m * n_k * n_n

    if phase not in {"decode_step", "prefill"}:
        raise ValueError("phase must be decode_step or prefill")
    if batch_size <= 0 or context_length <= 0:
        raise ValueError("batch_size and context_length must be positive")
    if not artifact_id or not source_operator or not bundle_name:
        raise ValueError("artifact/operator/bundle names must be non-empty")

    input_name = f"input_layer0_{bundle_name}"
    output_name = f"output_layer0_{bundle_name}"
    weight_name = f"weight_layer0_{bundle_name}_slice"

    mac_count = tile_m * tile_k * tile_n
    output_tile_bytes = element_size * tile_m * tile_n
    operator = {
        "operator": [
            {
                "name": f"tinyllama_layer0_{bundle_name}_{phase}",
                "type": "gemm",
                "iteration": iterations,
                "execution": {
                    "matrix": [{"name": "gemm_tile", "mac_count": mac_count}],
                    "vector": [
                        {
                            "name": "gemm_tile_accumulation",
                            "vec_count": tile_m * tile_n,
                        }
                    ],
                    "buffer_load": [
                        {
                            "name": "gemm_tile_load",
                            "byte_count": element_size
                            * (tile_m * tile_k + tile_k * tile_n),
                            "is_write": False,
                        },
                        {
                            "name": "gemm_tile_accumulation_load",
                            "byte_count": output_tile_bytes,
                            "is_write": False,
                        },
                    ],
                    "buffer_store": [
                        {
                            "name": "gemm_tile_store",
                            "byte_count": output_tile_bytes,
                            "is_write": True,
                        },
                        {
                            "name": "gemm_tile_accumulation_store",
                            "byte_count": output_tile_bytes,
                            "is_write": True,
                        },
                    ],
                    "dram": [
                        _dram_task(
                            name=input_name,
                            is_write=False,
                            extent=[tile_m, tile_k],
                            stride_add=[n_k * n_n, 1],
                            offset_add=[tile_m, tile_k],
                            init_iter=0,
                            stride_iter=1 if n_k > 1 else n_n,
                            total_iter=iterations if n_k > 1 else n_m,
                        ),
                        _dram_task(
                            name=weight_name,
                            is_write=False,
                            extent=[tile_k, tile_n],
                            stride_add=[1, n_k],
                            offset_add=[tile_k, tile_n],
                            init_iter=0,
                            stride_iter=1,
                            total_iter=iterations if n_k * n_n > 1 else 1,
                        ),
                        _dram_task(
                            name=output_name,
                            is_write=True,
                            extent=[tile_m, tile_n],
                            stride_add=[n_k * n_n, n_k],
                            offset_add=[tile_m, tile_n],
                            init_iter=n_k + 1,
                            stride_iter=n_k,
                            total_iter=n_m * n_n,
                        ),
                    ],
                },
            }
        ]
    }

    input_bytes = m_dim * k_dim * element_size
    per_core_output_bytes = m_dim * per_core_n * element_size
    per_core_weight_bytes = k_dim * per_core_n * element_size
    input_base = 0
    output_base = _align(input_base + input_bytes)
    weight_base = _align(output_base + per_core_output_bytes)

    per_core_tensors = []
    for core_id in range(core_count):
        per_core_tensors.append(
            {
                "core_id": core_id,
                "tensor": [
                    {
                        "name": input_name,
                        "base_addr": input_base,
                        "shape": [m_dim, k_dim],
                        "strides": [k_dim, 1],
                        "element_size": element_size,
                    },
                    {
                        "name": output_name,
                        "base_addr": output_base,
                        "shape": [m_dim, per_core_n],
                        "strides": [per_core_n, 1],
                        "element_size": element_size,
                    },
                    {
                        "name": weight_name,
                        "base_addr": weight_base,
                        "shape": [k_dim, per_core_n],
                        "strides": [1, k_dim],
                        "element_size": element_size,
                    },
                ],
            }
        )
    placement = {"core_tensor": per_core_tensors}

    per_core_input_dram_bytes = iterations * tile_m * tile_k * element_size
    per_core_weight_dram_bytes = iterations * tile_k * tile_n * element_size
    per_core_output_dram_bytes = n_m * n_n * tile_m * tile_n * element_size
    source_contract: dict[str, Any] = {
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "checkpoint_revision": "fe8a4ea1ffedaf415f4da2f062534de366a451e6",
        "operator": source_operator,
        "phase": phase,
        "batch_size": batch_size,
        "dtype": "float16",
        "shape": {"M": m_dim, "K": k_dim, "N": n_dim},
    }
    if phase == "decode_step":
        source_contract["initial_kv_length"] = context_length
    else:
        source_contract.update(
            {
                "context_length": context_length,
                "q_len": m_dim,
                "kv_length": context_length,
            }
        )
    manifest = {
        "schema_version": "hetero-atlas-artifact/v1",
        "artifact_id": artifact_id,
        "source_contract": source_contract,
        "lowering": {
            "rule": "ATLAS frontend/atlang/simulator/edge/gemm.py fixed tiling",
            "core_count": core_count,
            "sharding": "N-column partition, replicated input",
            "per_core_shape": {"M": m_dim, "K": k_dim, "N": per_core_n},
            "tile": {"M": tile_m, "K": tile_k, "N": tile_n},
            "iterations_per_core": iterations,
        },
        "expected_totals": {
            "mac_count": m_dim * k_dim * n_dim,
            "matrix_flop_count": 2 * m_dim * k_dim * n_dim,
            "vector_accumulation_op_count": core_count
            * iterations
            * tile_m
            * tile_n,
            "atlas_reported_operation_count": 2 * m_dim * k_dim * n_dim
            + core_count * iterations * tile_m * tile_n,
            "logical_input_bytes": input_bytes,
            "logical_weight_bytes": k_dim * n_dim * element_size,
            "logical_output_bytes": m_dim * n_dim * element_size,
            "atlas_dram_request_bytes": core_count
            * (
                per_core_input_dram_bytes
                + per_core_weight_dram_bytes
                + per_core_output_dram_bytes
            ),
        },
        "per_core_local_address_space": {
            "input_base": input_base,
            "output_base": output_base,
            "weight_base": weight_base,
            "span_bytes": weight_base + per_core_weight_bytes,
        },
        "global_column_slices": [
            {
                "core_id": core_id,
                "n_begin": core_id * per_core_n,
                "n_end": (core_id + 1) * per_core_n,
            }
            for core_id in range(core_count)
        ],
    }
    return operator, placement, manifest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--m", type=int, default=1)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--n", type=int, default=2048)
    parser.add_argument("--cores", type=int, default=16)
    parser.add_argument("--tile-m", type=int, default=1)
    parser.add_argument("--tile-k", type=int, default=512)
    parser.add_argument("--tile-n", type=int, default=16)
    parser.add_argument("--element-size", type=int, default=2)
    parser.add_argument("--artifact-id")
    parser.add_argument("--source-operator", default="model.layers.0.self_attn.q_proj")
    parser.add_argument("--phase", choices=("decode_step", "prefill"), default="decode_step")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--bundle-name", default="q_proj")
    args = parser.parse_args()

    operator, placement, manifest = build_bundle(
        m_dim=args.m,
        k_dim=args.k,
        n_dim=args.n,
        core_count=args.cores,
        tile_m=args.tile_m,
        tile_k=args.tile_k,
        tile_n=args.tile_n,
        element_size=args.element_size,
        artifact_id=args.artifact_id
        or "tinyllama.1_1b.layer0.q_proj.decode.bs1_ctx1024.fp16.atlas_edge",
        source_operator=args.source_operator,
        phase=args.phase,
        batch_size=args.batch_size,
        context_length=args.context_length,
        bundle_name=args.bundle_name,
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    operator_path = output / "operator_description.yaml"
    placement_path = output / "data_placement.yaml"
    manifest_path = output / "artifact_manifest.json"
    operator_path.write_text(yaml.safe_dump(operator, sort_keys=False), encoding="utf-8")
    placement_path.write_text(yaml.safe_dump(placement, sort_keys=False), encoding="utf-8")
    manifest["files"] = {
        "operator_description.yaml": _sha256(operator_path),
        "data_placement.yaml": _sha256(placement_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
