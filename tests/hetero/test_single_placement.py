from dataclasses import replace

import pytest

from frontend.hetero.execution_plan import (
    SinglePlacementError,
    build_single_placement_plan,
)
from frontend.hetero.model_graph import ModelSpec, RequestSpec, build_request_graph
from frontend.hetero.placement import place_nodes


def _graph_and_decisions():
    model = ModelSpec("tiny", 128, 256, 1, 4, 2, 32, 256)
    graph = build_request_graph(
        model,
        RequestSpec(
            "R0",
            prompt_length=1,
            output_length=1,
            execution_scope="decode_step",
            initial_kv_length=16,
        ),
    )
    decisions = place_nodes(
        graph.nodes,
        {"mode": "manual", "default_target": "gpu0", "rules": []},
    )
    return model, graph, decisions


def test_single_placement_rejects_missing_and_duplicate_decisions() -> None:
    model, graph, decisions = _graph_and_decisions()
    with pytest.raises(SinglePlacementError, match="exactly one decision"):
        build_single_placement_plan(
            graph,
            decisions[:-1],
            "model3_gpu_native_3ddram",
            "copy",
            model,
        )
    with pytest.raises(SinglePlacementError, match="exactly one decision"):
        build_single_placement_plan(
            graph,
            [*decisions, decisions[0]],
            "model3_gpu_native_3ddram",
            "copy",
            model,
        )


def test_cross_device_consumer_routes_every_input_value_with_versions() -> None:
    model, graph, decisions = _graph_and_decisions()
    decisions = [
        replace(decision, target_device="atlas0.compute")
        if decision.node_id == "R0.decode.s0.l0.attention.core"
        else decision
        for decision in decisions
    ]
    plan = build_single_placement_plan(
        graph,
        decisions,
        "model3_gpu_native_3ddram",
        "copy",
        model,
    )

    attention_task = "task.R0.decode.s0.l0.attention.core"
    incoming = [
        route for route in plan.routes if route.consumer_task_id == attention_task
    ]
    assert {route.value_id for route in incoming} == {
        "R0.decode.s0.l0.attention.kv_append.out",
        "R0.kv.l0.k",
        "R0.kv.l0.v",
    }
    assert {route.value_version for route in incoming} == {1}
    assert all(route.lowering.kind.value == "synchronization" for route in incoming)
    assert all(
        route.lowering.actions
        == ("writeback", "release_fence", "invalidate", "acquire_fence")
        for route in incoming
    )
    planned_attention = next(
        node for node in plan.nodes if node.task_id == attention_task
    )
    assert {route.task_id for route in incoming} <= set(
        planned_attention.dependencies
    )
    assert plan.conservation["each_logical_node_exactly_once"] is True


def test_kv_append_is_versioned_read_modify_write() -> None:
    model, graph, decisions = _graph_and_decisions()
    plan = build_single_placement_plan(
        graph,
        decisions,
        "model3_gpu_native_3ddram",
        "copy",
        model,
    )
    kv_append = next(
        node
        for node in plan.nodes
        if node.node.node_id == "R0.decode.s0.l0.attention.kv_append"
    )
    inputs = {item["value_id"]: item["version"] for item in kv_append.input_values}
    outputs = {
        item["value_id"]: item["version"] for item in kv_append.output_values
    }
    assert inputs["R0.kv.l0.k"] == inputs["R0.kv.l0.v"] == 0
    assert outputs["R0.kv.l0.k"] == outputs["R0.kv.l0.v"] == 1
