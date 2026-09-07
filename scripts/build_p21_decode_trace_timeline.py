#!/usr/bin/env python3
"""Build P21 four-token Decode configs from 56 qualified operator artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frontend.hetero.model_graph import (  # noqa: E402
    ModelSpec,
    RequestSpec,
    build_request_graph,
)
from frontend.hetero.operator_artifact import (  # noqa: E402
    OperatorArtifactManifest,
)

CHECKPOINT = "fe8a4ea1ffedaf415f4da2f062534de366a451e6"
OPERATORS = (
    "token_embedding",
    "attention_norm",
    "qkv_projection",
    "rope",
    "causal_attention",
    "output_projection",
    "residual_add",
    "mlp_norm",
    "gate_up_projection",
    "silu_multiply",
    "down_projection",
    "final_norm",
    "lm_head",
    "sampling",
)
LAYER_OPERATORS = frozenset(
    {
        "attention_norm",
        "qkv_projection",
        "rope",
        "causal_attention",
        "output_projection",
        "residual_add",
        "mlp_norm",
        "gate_up_projection",
        "silu_multiply",
        "down_projection",
    }
)
RUNTIME_OPERATORS = (
    "request_start",
    "kv_allocate",
    "kv_append",
    "request_finish",
    "kv_release",
)


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


def _binding(
    tensor_id: str, source: str, index: int, offset: int = 0, *, shadow: bool = False
) -> dict[str, object]:
    record: dict[str, object] = {
        "tensor_id": tensor_id,
        "source": source,
        "index": index,
        "value_offset_bytes": offset,
    }
    if shadow:
        record["binding_mode"] = "external_input_widened_shadow"
    return record


def _value_bindings(operator: str, kv_length: int) -> list[dict[str, object]]:
    prefix = f"tinyllama.decode.kv{kv_length}.layer0.{operator}"
    if operator == "token_embedding":
        return [
            _binding(f"{prefix}.token_ids", "input", 0, shadow=True),
            _binding(f"{prefix}.weight", "input", 1),
            _binding(f"{prefix}.output", "output", 0),
        ]
    if operator in {"attention_norm", "mlp_norm", "final_norm"}:
        return [
            _binding(f"{prefix}.input", "input", 0),
            _binding(f"{prefix}.weight", "input", 1),
            _binding(f"{prefix}.output", "output", 0),
        ]
    if operator == "qkv_projection":
        return [
            _binding(f"{prefix}.input", "input", 0),
            _binding(f"{prefix}.q_weight", "input", 1),
            _binding(f"{prefix}.k_weight", "input", 1, 8_388_608),
            _binding(f"{prefix}.v_weight", "input", 1, 9_437_184),
            _binding(f"{prefix}.output", "output", 0),
        ]
    if operator == "rope":
        return [
            _binding(f"{prefix}.query", "input", 0),
            _binding(f"{prefix}.key", "input", 0, 4096),
            _binding(f"{prefix}.query_output", "output", 0),
            _binding(f"{prefix}.key_output", "output", 0, 4096),
        ]
    if operator == "causal_attention":
        return [
            _binding(f"{prefix}.query", "input", 0),
            _binding(f"{prefix}.key", "input", 1, shadow=True),
            _binding(f"{prefix}.value", "input", 2, shadow=True),
            _binding(f"{prefix}.output", "output", 0),
        ]
    if operator in {"output_projection", "down_projection", "lm_head"}:
        output_name = "logits" if operator == "lm_head" else "output"
        return [
            _binding(f"{prefix}.input", "input", 0),
            _binding(f"{prefix}.weight", "input", 1),
            _binding(f"{prefix}.{output_name}", "output", 0),
        ]
    if operator == "residual_add":
        return [
            _binding(f"{prefix}.input", "input", 0),
            _binding(f"{prefix}.residual", "input", 1),
            _binding(f"{prefix}.output", "output", 0),
        ]
    if operator == "gate_up_projection":
        return [
            _binding(f"{prefix}.input", "input", 0),
            _binding(f"{prefix}.gate_weight", "input", 1),
            _binding(f"{prefix}.up_weight", "input", 1, 23_068_672),
            _binding(f"{prefix}.output", "output", 0),
        ]
    if operator == "silu_multiply":
        return [
            _binding(f"{prefix}.gate", "input", 0),
            _binding(f"{prefix}.up", "input", 0, 11_264),
            _binding(f"{prefix}.output", "output", 0),
        ]
    if operator == "sampling":
        return [
            _binding(f"{prefix}.logits", "input", 0),
            _binding(f"{prefix}.token", "output", 0),
        ]
    raise ValueError(f"P21 value bindings are not defined for {operator}")


def _qualified_artifact(root: Path, operator: str, kv_length: int) -> Path:
    return root / (
        f"tinyllama_decode_bs1_ctx16_kv{kv_length}_{operator}_sm89_"
        "shared_hbdram_range_rebase.json"
    )


def _trace_manifest(root: Path, operator: str, kv_length: int) -> Path:
    return root / (
        f"tinyllama_decode_bs1_ctx16_kv{kv_length}_{operator}_sm89_trace.json"
    )


def _validate_inputs(source_root: Path, coupled_root: Path) -> None:
    for kv_length in range(17, 21):
        for operator in OPERATORS:
            trace = _trace_manifest(source_root, operator, kv_length)
            coupled = _qualified_artifact(coupled_root, operator, kv_length)
            if not trace.is_file() or not coupled.is_file():
                raise FileNotFoundError(trace if not trace.is_file() else coupled)
            artifact = OperatorArtifactManifest.load(coupled)
            source = artifact.payload["source_contract"]
            if (
                not artifact.request_cycle_ready
                or source.get("operator") != operator
                or source.get("phase") != "decode_step"
                or source.get("q_len") != 1
                or source.get("kv_length") != kv_length
                or artifact.payload["qualification"].get("performance_eligible")
                is not False
            ):
                raise ValueError(f"P21 artifact is not qualified: {coupled}")


def build_experiment(
    layers: int, source_root: Path, coupled_root: Path
) -> dict[str, object]:
    trace_bindings: list[dict[str, object]] = []
    for step_id, kv_length in enumerate(range(17, 21)):
        for operator in OPERATORS:
            trace_bindings.append(
                {
                    "selector": {
                        "phase": "decode",
                        "op": operator,
                        "step_id": step_id,
                    },
                    "trace_manifest": str(
                        _trace_manifest(source_root, operator, kv_length)
                    ),
                    "operator_artifact": str(
                        _qualified_artifact(coupled_root, operator, kv_length)
                    ),
                    "compatibility": "exact_operator",
                    "contract_overrides": {"layer_id": 0},
                    "value_bindings": _value_bindings(operator, kv_length),
                }
            )
    return {
        "schema_version": "hetero-sim/v1",
        "experiment": {
            "name": f"p21_tinyllama_decode4_{layers}layer_bs1_ctx16_real_trace",
            "seed": 1,
            "generation_mode": "trace_locked",
        },
        "simulation": {
            "coupling": "operator_event",
            "execution_mode": "operator_event",
            "validation_policy": "required",
        },
        "system": {
            "profile": "model3_gpu_native_3ddram",
            "links": {
                "shared3d.explicit_noncoherent": {
                    "wire_bandwidth_Bps": 12_800_000_000,
                    "latency_fs": 10_000_000,
                    "header_bytes": 16,
                    "resource_id": "unused.p21.gpu_only",
                    "parameter_source": "unused_no_cross_device_route",
                }
            },
        },
        "backends": {
            "gpu": {
                "kind": "accel_sim",
                "requested_timing_mode": "coupled",
                "config_ref": (
                    "configs/hetero/backends/"
                    "gpu_accelsim_rtx3070_ramulator2_hbdram_edge_16ch_"
                    "range_rebase.json"
                ),
                "require_request_cycle_ready": True,
                "resource_bindings": {
                    "gpu_core": "gpu0.core",
                    "gpu_l1": "gpu0.l1",
                    "gpu_l2": "gpu0.l2",
                    "gpu_noc": "gpu0.noc",
                    "shared_3d_dram": "shared0.dram3d",
                },
                "trace_bindings": trace_bindings,
                "fallback_kind": "none",
                "runtime_task_model_ref": (
                    "configs/hetero/runtime_tasks/"
                    "tinyllama_decode4_bs1_ctx16_uncalibrated.json"
                ),
                "runtime_task_operators": list(RUNTIME_OPERATORS),
                "parameter_source": (
                    "P21 remote RTX4090 capture of executed SM86 binaries; "
                    "Accel-Sim plus range-rebased Ramulator2 functional timing"
                ),
            },
            "atlas": {"kind": "none"},
            "host": {"kind": "none"},
        },
        "model": {
            "ref": (
                f"configs/hetero/models/tinyllama_1_1b_decode_{layers}layer_fp16.json"
            )
        },
        "workload": {
            "ref": "configs/hetero/workloads/tinyllama11b_bs1_ctx16_decode4.json"
        },
        "scheduling": {"ref": "configs/hetero/schedulers/decode_bs1_ctx16.json"},
        "placement": {"ref": "configs/hetero/placements/gpu_only_shared_3ddram.json"},
        "address": {
            "ref": "configs/hetero/addresses/tinyllama11b_prefill_global_pa.json"
        },
        "metrics": {"ref": "configs/hetero/metrics/e2e_validation.json"},
    }


def build_capability(layers: int, coupled_root: Path) -> dict[str, object]:
    model = _model(layers)
    graph = build_request_graph(
        model,
        RequestSpec(
            "TINYLLAMA11B-DECODE4-R0",
            16,
            4,
            execution_scope="decode_loop",
            initial_kv_length=16,
        ),
    )
    counts = Counter(node.op for node in graph.nodes)
    operator_types: list[dict[str, object]] = []
    for operator in sorted(counts):
        traced = operator in OPERATORS
        artifact_refs = (
            [
                str(_qualified_artifact(coupled_root, operator, kv_length))
                for kv_length in range(17, 21)
            ]
            if traced
            else [
                "configs/hetero/runtime_tasks/"
                "tinyllama_decode4_bs1_ctx16_uncalibrated.json"
            ]
        )
        operator_types.append(
            {
                "operator_type": operator,
                "instances_in_reference_graph": counts[operator],
                "backend_kind": "accel_sim" if traced else "runtime_task_model",
                "implementation_status": "implemented",
                "test_status": (
                    "range_rebase_double_run_qualified"
                    if traced
                    else "functional_runtime_contract_tested"
                ),
                "cycle_fidelity": (
                    "cycle_simulated_instruction_request_coupled"
                    if traced
                    else "uncalibrated_runtime_cycle_contract"
                ),
                "request_cycle_ready": traced,
                "performance_eligible": False,
                "shape_policy": "exact_kv17_18_19_20_only",
                "artifact_refs": artifact_refs,
                "notes": (
                    "Physical capture device SM89 executed an SM86 binary sequence."
                    if traced
                    else "Control, allocation, KV-copy, or release task without SM Trace."
                ),
            }
        )
    return {
        "schema_version": "hetero-operator-capability-catalog/v1",
        "catalog_id": f"tinyllama.1_1b.decode4.bs1_ctx16.layers{layers}.p21",
        "reference_graph": (
            "configs/hetero/experiments/"
            f"p21_tinyllama_decode4_{layers}layer_bs1_ctx16_real_trace.json"
        ),
        "claim_boundary": {
            "performance_claim_allowed": False,
            "reason": (
                "Functional instruction/request closure is not RTX4090 or RTX3070 "
                "performance calibration. Runtime state contracts remain uncalibrated."
            ),
        },
        "model_contract": {
            "model_spec_name": model.name,
            "checkpoint_revision": CHECKPOINT,
            "batch_size": 1,
            "initial_context_length": 16,
            "q_len": 1,
            "kv_lengths": [17, 18, 19, 20],
            "layers": layers,
            "dtype": model.dtype,
        },
        "operator_types": operator_types,
    }


def _write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", required=True, type=int, choices=(1, 22))
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("configs/hetero/operator_artifacts/p21_sm89_decode"),
    )
    parser.add_argument(
        "--coupled-root",
        type=Path,
        default=Path("configs/hetero/operator_artifacts/p21_sm89_decode_coupled"),
    )
    parser.add_argument("--experiment-output", required=True, type=Path)
    parser.add_argument("--capability-output", required=True, type=Path)
    args = parser.parse_args()

    _validate_inputs(args.source_root, args.coupled_root)
    _write(
        args.experiment_output,
        build_experiment(args.layers, args.source_root, args.coupled_root),
    )
    _write(args.capability_output, build_capability(args.layers, args.coupled_root))
    print(args.experiment_output)
    print(args.capability_output)


if __name__ == "__main__":
    main()
