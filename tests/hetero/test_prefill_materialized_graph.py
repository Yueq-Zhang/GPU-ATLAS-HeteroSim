from pathlib import Path

from frontend.hetero.global_memory_map import (
    build_global_memory_map,
    sampled_requests_for_values,
)
from frontend.hetero.model_graph import ModelSpec, RequestSpec, build_request_graph
from frontend.hetero.placement import place_nodes
from frontend.hetero.prefill_cycle_artifact import PrefillCycleCatalog
from frontend.hetero.runner import _execution_graph


ROOT = Path(__file__).resolve().parents[2]


def _model(layers: int = 1) -> ModelSpec:
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
    )


def test_one_layer_prefill_materializes_complete_tensor_contract() -> None:
    graph = build_request_graph(_model(), RequestSpec("R0", 16, 1))
    assert len(graph.nodes) == 20
    assert {node.op for node in graph.nodes} == {
        "request_start",
        "kv_allocate",
        "token_embedding",
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
        "final_norm",
        "lm_head",
        "sampling",
        "request_finish",
        "kv_release",
    }
    values = {value.value_id: value for value in graph.values}
    assert values["R0.prefill.s0.l0.attention.projection.out"].shape_expr == (
        16,
        2560,
    )
    assert values["R0.prefill.s0.l0.mlp.gate_up.out"].shape_expr == (16, 11264)
    assert values["R0.prefill.s0.l0.mlp.activation.out"].shape_expr == (16, 5632)
    assert values["R0.prefill.s0.final_norm.out"].shape_expr == (1, 2048)
    assert values["model.layers.0.qkv.weight"].storage_class.value == "parameter"
    kv_append = next(node for node in graph.nodes if node.op == "kv_append")
    assert set(kv_append.read_values) & set(kv_append.write_values) == {
        "R0.kv.l0.k",
        "R0.kv.l0.v",
    }
    residuals = [node for node in graph.nodes if node.op == "residual_add"]
    assert len(residuals) == 2
    assert all(len(node.read_values) == 2 for node in residuals)


def test_p11_cycle_catalog_covers_every_prefill_operator_on_both_devices() -> None:
    graph = build_request_graph(_model(), RequestSpec("R0", 16, 1))
    catalog = PrefillCycleCatalog.load(
        ROOT / "configs/hetero/cycle_artifacts/tinyllama11b_prefill_fp16_v1.json"
    )
    required = {node.op for node in graph.nodes}
    assert required <= catalog.supported_ops("gpu0")
    assert required <= catalog.supported_ops("atlas0.compute")
    assert len(required) == 19


def test_p14_global_pa_map_fits_four_gib_and_sampling_conserves_bytes() -> None:
    model = _model(layers=22)
    request = RequestSpec("R0", 1024, 1)
    graph = build_request_graph(model, request)
    decisions = place_nodes(
        graph.nodes,
        {"mode": "manual", "default_target": "gpu0", "rules": []},
    )
    execution, _, _ = _execution_graph(
        [(graph, decisions, request)],
        "model3_gpu_native_3ddram",
        "copy",
        model,
        {"gpu": {"kind": "cycle_replay"}, "atlas": {"kind": "none"}, "host": {"kind": "none"}},
        {},
        "prefill_cycle",
    )
    allocations, payload = build_global_memory_map(
        execution, "shared0.dram3d", 4 * 1024**3, 64
    )
    assert payload["allocation_count"] == 448
    assert payload["allocated_bytes"] == 3_957_580_290
    assert payload["allocated_bytes"] < payload["capacity_bytes"]
    assert payload["non_overlapping"] is True
    first_task = next(
        task for task in execution["tasks"] if task["op"] == "token_embedding"
    )
    samples = sampled_requests_for_values(
        str(first_task["task_id"]),
        "gpu0",
        first_task["input_values"],
        allocations,
        "read",
        64,
        4,
    )
    represented: dict[str, int] = {}
    for sample in samples:
        represented[str(sample["value_id"])] = represented.get(
            str(sample["value_id"]), 0
        ) + int(sample["represented_bytes"])
    assert represented == {
        str(value["value_id"]): int(value["size_bytes"])
        for value in first_task["input_values"]
    }
