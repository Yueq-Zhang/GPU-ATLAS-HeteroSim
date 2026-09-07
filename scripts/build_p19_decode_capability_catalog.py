#!/usr/bin/env python3
"""Build auditable P19 Decode capability catalogs for one and 22 layers."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frontend.hetero.model_graph import (
    ModelSpec,
    RequestSpec,
    build_request_graph,
)

CHECKPOINT = "fe8a4ea1ffedaf415f4da2f062534de366a451e6"
LAYER_OPERATORS = {
    "attention_norm",
    "qkv_projection",
    "rope",
    "kv_append",
    "causal_attention",
    "output_projection",
    "residual_add",
    "mlp_norm",
    "gate_up_projection",
    "silu_multiply",
    "down_projection",
}
CONTROL_OPERATORS = {"request_start", "request_finish"}


def _model(layers: int) -> ModelSpec:
    return ModelSpec(
        name="TinyLlama-1.1B",
        hidden_size=2048,
        intermediate_size=5632,
        num_layers=layers,
        num_attention_heads=32,
        num_kv_heads=4,
        head_dim=64,
        vocab_size=32000,
        tied_embeddings=False,
        input_embedding_mode="token_ids",
        materialize_parameters=True,
        checkpoint_revision=CHECKPOINT,
    )


def build_catalog(layers: int) -> dict[str, object]:
    model = _model(layers)
    request = RequestSpec(
        "TINYLLAMA11B-DECODE-R0",
        16,
        1,
        execution_scope="decode_step",
        initial_kv_length=16,
    )
    graph = build_request_graph(model, request)
    counts = Counter(node.op for node in graph.nodes)
    shapes: dict[str, object] = {
        "control_ctx16": {
            "model_contract_ref": "tinyllama_1_1b_fe8a4e_fp16",
            "phase": "control",
            "layer_id": 0,
            "batch_size": 1,
            "context_length": 16,
            "q_len": 1,
            "kv_length": 17,
        }
    }
    for layer_id in range(layers):
        shapes[f"decode_l{layer_id}_q1_kv17"] = {
            "model_contract_ref": "tinyllama_1_1b_fe8a4e_fp16",
            "phase": "decode_step",
            "layer_id": layer_id,
            "batch_size": 1,
            "context_length": 16,
            "q_len": 1,
            "kv_length": 17,
        }

    operator_types: list[dict[str, object]] = []
    for operator in sorted(counts):
        if operator in CONTROL_OPERATORS:
            backend = "event_marker"
            status = "graph_causality_tested"
            fidelity = "event_only"
            shape_refs = ["control_ctx16"]
            notes = "Causal host boundary; excluded from device performance."
        else:
            backend = "cycle_replay"
            status = "runtime_cycle_contract_tested"
            fidelity = "runtime_cycle_contract"
            if operator in LAYER_OPERATORS:
                shape_refs = [
                    f"decode_l{layer_id}_q1_kv17" for layer_id in range(layers)
                ]
            else:
                shape_refs = ["decode_l0_q1_kv17"]
            notes = (
                "P19 functional request-cycle contract with Global PA and one live "
                "Ramulator2. Compute is not an Accel-Sim instruction trace and is not "
                "performance qualified."
            )
        operator_types.append(
            {
                "operator_type": operator,
                "instances_in_reference_graph": counts[operator],
                "task_kind": next(
                    node.kind.value for node in graph.nodes if node.op == operator
                ),
                "backend_kind": backend,
                "implementation_status": "implemented",
                "test_status": status,
                "cycle_fidelity": fidelity,
                "request_cycle_ready": False,
                "performance_eligible": False,
                "shape_policy": "exact_only",
                "shape_contract_refs": shape_refs,
                "artifact_refs": (
                    []
                    if operator in CONTROL_OPERATORS
                    else [
                        "configs/hetero/cycle_artifacts/tinyllama11b_request_cycle_fp16_v1.json"
                    ]
                ),
                "notes": notes,
            }
        )

    return {
        "schema_version": "hetero-operator-capability-catalog/v1",
        "catalog_id": f"tinyllama.1_1b.decode.bs1_ctx16.layers{layers}.p19",
        "reference_graph": (
            "configs/hetero/experiments/"
            f"p19_tinyllama_decode_{layers}layer_bs1_ctx16_request_cycle.json"
            if layers == 1
            else "configs/hetero/experiments/"
            "p19_tinyllama_decode_22layer_bs1_ctx16_request_cycle.json"
        ),
        "claim_boundary": {
            "performance_claim_allowed": False,
            "reason": (
                "P19 qualifies Decode graph, KV lifecycle, Global PA, scheduling and "
                "live-memory causality only. GPU compute uses an uncalibrated tiled "
                "cycle contract rather than an Accel-Sim instruction trace."
            ),
        },
        "model_contracts": {
            "tinyllama_1_1b_fe8a4e_fp16": {
                "model_spec_name": model.name,
                "checkpoint_revision": CHECKPOINT,
                "hidden_size": model.hidden_size,
                "intermediate_size": model.intermediate_size,
                "num_attention_heads": model.num_attention_heads,
                "num_kv_heads": model.num_kv_heads,
                "head_dim": model.head_dim,
                "vocab_size": model.vocab_size,
                "dtype": model.dtype,
            }
        },
        "shape_contracts": shapes,
        "operator_types": operator_types,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, choices=(1, 22), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_catalog(args.layers)
    output = args.output
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()
