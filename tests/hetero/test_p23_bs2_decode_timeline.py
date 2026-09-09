import json
from collections import Counter
from pathlib import Path

from frontend.hetero.model_graph import graph_to_dict, model_spec_from_config
from frontend.hetero.p23_batched_decode import (
    BATCH_REQUEST_ID,
    GPU_OPERATORS,
    MEMBER_REQUEST_IDS,
    build_graph,
    graph_summary,
    value_bindings,
)
from scripts.validate_p23_sealed_catalog import validate_catalog


def _model():
    return model_spec_from_config(
        json.loads(
            Path(
                "configs/hetero/models/tinyllama_1_1b_decode_1layer_fp16.json"
            ).read_text(encoding="utf-8")
        )
    )


def test_p23_graph_is_one_fused_bs2_layer_with_exact_task_inventory() -> None:
    graph = build_graph(_model())
    summary = graph_summary(graph)
    counts = Counter(node.op for node in graph.nodes)
    assert summary["task_count"] == 20
    assert summary["gpu_task_count"] == 15
    assert summary["gpu_operator_type_count"] == 14
    assert summary["runtime_task_count"] == 5
    assert counts["residual_add"] == 2
    assert counts["kv_append"] == 1
    assert set(GPU_OPERATORS) <= set(counts)
    assert all(node.attributes["batch_size"] == 2 for node in graph.nodes)
    assert all(
        node.attributes["batch_member_request_ids"] == list(MEMBER_REQUEST_IDS)
        for node in graph.nodes
    )


def test_p23_graph_has_separate_batch_member_kv_extent_and_split_qkv_values() -> None:
    graph = build_graph(_model())
    values = {value.value_id: value for value in graph.values}
    assert values[f"{BATCH_REQUEST_ID}.kv.l0.k"].shape_expr == (2, "kv_tokens", 4, 64)
    assert values[f"{BATCH_REQUEST_ID}.kv.l0.v"].shape_expr == (2, "kv_tokens", 4, 64)
    assert values[f"{BATCH_REQUEST_ID}.q"].shape_expr == (2, 1, 2048)
    assert values[f"{BATCH_REQUEST_ID}.k_new"].shape_expr == (2, 1, 256)
    assert values[f"{BATCH_REQUEST_ID}.v_new"].shape_expr == (2, 1, 256)
    kv_append = next(node for node in graph.nodes if node.op == "kv_append")
    assert len(kv_append.read_values) == 5
    assert len(kv_append.write_values) == 3


def test_p23_bindings_preserve_split_qkv_and_gate_up_semantics() -> None:
    qkv = value_bindings("qkv_projection")
    gate_up = value_bindings("gate_up_projection")
    assert [(item["source"], item["index"]) for item in qkv[-3:]] == [
        ("output", 0),
        ("output", 1),
        ("output", 2),
    ]
    assert [(item["source"], item["index"]) for item in gate_up[-2:]] == [
        ("output", 0),
        ("output", 1),
    ]


def test_p23_model_graph_is_json_serializable() -> None:
    graph = build_graph(_model())
    payload = json.loads(json.dumps(graph_to_dict(graph)))
    assert payload["schema_version"] == graph.schema_version
    assert len(payload["nodes"]) == 20
    assert len(payload["values"]) == len(graph.values)


def test_p23_repository_seal_is_complete() -> None:
    summary = validate_catalog(
        Path("validation/p23/ready_catalog.json"), Path.cwd()
    )
    assert summary == {
        "status": "passed",
        "operator_count": 14,
        "timeline_qualified": True,
        "double_run_signature_equal": True,
        "total_task_instances": 20,
        "makespan_fs": 34843748683165,
        "performance_claim_allowed": False,
    }
