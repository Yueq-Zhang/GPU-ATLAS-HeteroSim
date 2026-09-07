from copy import deepcopy
from pathlib import Path

import pytest

from frontend.hetero.decode_lifecycle import build_decode_loop_kv_lifecycle
from frontend.hetero.decode_loop_qualification import qualify_p20_decode_loop_pair
from frontend.hetero.global_memory_map import build_global_memory_map
from frontend.hetero.model_graph import (
    build_request_graph,
    graph_counters,
    model_spec_from_config,
    request_specs_from_config,
)
from frontend.hetero.operator_capability import OperatorCapabilityCatalog
from frontend.hetero.placement import place_nodes
from frontend.hetero.runner import _execution_graph
from frontend.hetero.schema import (
    ConfigError,
    load_and_validate_config,
    validate_config,
)

ROOT = Path(__file__).resolve().parents[2]


def _config(layers: int) -> dict[str, object]:
    return load_and_validate_config(
        ROOT
        / "configs/hetero/experiments"
        / f"p20_tinyllama_decode4_{layers}layer_bs1_ctx16_request_cycle.json"
    )


def _graph(layers: int):
    config = _config(layers)
    model = model_spec_from_config(dict(config["model"]))
    request = request_specs_from_config(
        [dict(config["workload"]["requests"][0])]
    )[0]
    graph = build_request_graph(model, request)
    return config, model, request, graph


@pytest.mark.parametrize("layers,expected_tasks", [(1, 68), (22, 1076)])
def test_p20_decode_loop_builds_four_autoregressive_forwards(
    layers: int, expected_tasks: int
) -> None:
    _, model, request, graph = _graph(layers)
    assert len(graph.nodes) == expected_tasks == 4 + 4 * (4 + 12 * layers)
    assert {node.phase.value for node in graph.nodes} == {"control", "decode"}
    assert len([node for node in graph.nodes if node.op == "kv_append"]) == 4 * layers
    assert len([node for node in graph.nodes if node.op == "sampling"]) == 4

    for step_id in range(4):
        step_nodes = [
            node
            for node in graph.nodes
            if node.phase.value == "decode" and node.step_id == step_id
        ]
        assert step_nodes
        assert all(node.attributes["q_len"] == 1 for node in step_nodes)
        assert all(
            node.attributes["past_kv_len"] == 16 + step_id
            for node in step_nodes
        )
        assert all(
            node.attributes["attention_kv_len"] == 17 + step_id
            for node in step_nodes
        )
        embedding = next(node for node in step_nodes if node.op == "token_embedding")
        expected_token = (
            "TINYLLAMA11B-DECODE4-R0.decode_token_id"
            if step_id == 0
            else f"TINYLLAMA11B-DECODE4-R0.token.{step_id - 1}"
        )
        assert expected_token in embedding.read_values

    counters = graph_counters(model, request)
    assert counters.prefill_forwards == 0
    assert counters.decode_forwards == 4
    assert counters.lm_head == counters.sampling == 4
    assert counters.kv_append_pairs == 4 * layers
    assert counters.kv_range_writes == 8 * layers
    assert counters.final_committed_kv_len == 20


@pytest.mark.parametrize("layers", [1, 22])
def test_p20_decode_loop_versions_and_global_pa_ranges(layers: int) -> None:
    config, model, request, graph = _graph(layers)
    decisions = place_nodes(graph.nodes, dict(config["placement"]))
    execution, _, _ = _execution_graph(
        [(graph, decisions, request)],
        str(config["system"]["profile"]),
        str(config["system"].get("access_policy", "copy")),
        model,
        dict(config["backends"]),
        dict(config["system"].get("links", {})),
        "request_cycle",
    )
    allocations, memory_map = build_global_memory_map(
        execution, "shared0.dram3d", 4 * 1024**3, 64
    )
    lifecycle = build_decode_loop_kv_lifecycle(
        graph, model, request, allocations
    )

    assert memory_map["non_overlapping"] is True
    assert lifecycle["schema_version"] == "hetero-decode-kv-lifecycle/v2"
    assert lifecycle["generated_tokens"] == 4
    assert lifecycle["initial_kv_length"] == 16
    assert lifecycle["final_kv_length"] == 20
    assert lifecycle["bytes_per_kv_token_per_tensor"] == 512
    assert lifecycle["total_appended_kv_bytes"] == 2 * layers * 4 * 512
    assert lifecycle["all_steps_autoregressive"] is True
    assert lifecycle["global_pa_bound"] is True

    expected_offsets = [8192, 8704, 9216, 9728]
    for step, expected_offset in zip(lifecycle["steps"], expected_offsets):
        for layer in step["layers"]:
            for value in layer["values"]:
                assert value["append_range"]["offset_bytes"] == expected_offset
                assert value["committed_output_version"] == step["step_id"] + 1
                assert (
                    value["global_pa"]["append_address"]
                    == value["global_pa"]["base_address"] + expected_offset
                )

    final_records = {
        item["value_id"]: item["version"]
        for item in execution["residency_plan"]["final_records"]
    }
    kv_values = [
        value["value_id"]
        for layer in lifecycle["layers"]
        for value in layer["values"]
    ]
    assert len(kv_values) == 2 * layers
    assert all(final_records[value_id] == 4 for value_id in kv_values)


def test_p20_decode_loop_has_explicit_cross_step_token_dependency() -> None:
    _, _, _, graph = _graph(1)
    by_id = {node.node_id: node for node in graph.nodes}
    for step_id in range(1, 4):
        embedding = by_id[f"TINYLLAMA11B-DECODE4-R0.decode.s{step_id}.embedding"]
        previous_sampling = (
            f"TINYLLAMA11B-DECODE4-R0.decode.s{step_id - 1}.sampling"
        )
        assert previous_sampling in embedding.dependencies
        assert f"TINYLLAMA11B-DECODE4-R0.token.{step_id - 1}" in embedding.read_values


@pytest.mark.parametrize("layers", [1, 22])
def test_p20_decode_loop_configs_are_strictly_valid(layers: int) -> None:
    config = _config(layers)
    request = config["workload"]["requests"][0]
    assert request["execution_scope"] == "decode_loop"
    assert request["initial_kv_length"] == 16
    assert request["output_length"] == 4
    assert config["model"]["materialize_parameters"] is True


def test_decode_step_remains_single_token_and_decode_loop_requires_initial_kv() -> None:
    config = _config(1)
    broken = deepcopy(config)
    broken["workload"]["requests"][0]["execution_scope"] = "decode_step"
    with pytest.raises(ConfigError, match="decode_step requires output_length=1"):
        validate_config(broken)
    broken = deepcopy(config)
    broken["workload"]["requests"][0]["initial_kv_length"] = 0
    with pytest.raises(ConfigError, match="decode_loop requires initial_kv_length"):
        validate_config(broken)


@pytest.mark.parametrize("layers,expected_tasks", [(1, 68), (22, 1076)])
def test_p20_capability_catalog_covers_every_step_shape_without_performance_claim(
    layers: int, expected_tasks: int
) -> None:
    catalog = OperatorCapabilityCatalog.load(
        ROOT
        / "configs/hetero/operator_capabilities"
        / f"tinyllama_decode4_{layers}layer_bs1_ctx16_p20.json"
    )
    assert sum(
        item.instances_in_reference_graph for item in catalog.operators.values()
    ) == expected_tasks
    assert len(catalog.operators) == 19
    assert all(
        item.performance_eligible is False for item in catalog.operators.values()
    )
    assert all(item.request_cycle_ready is False for item in catalog.operators.values())
    decode_shapes = [
        shape
        for shape in catalog.shape_contracts.values()
        if shape.phase == "decode_step"
    ]
    assert len(decode_shapes) == 4 * layers
    assert {shape.kv_length for shape in decode_shapes} == {17, 18, 19, 20}
    assert all(shape.q_len == 1 for shape in decode_shapes)


@pytest.mark.parametrize(
    "case,experiment,key,layers,expected_tasks,expected_parents",
    [
        (
            "one_layer",
            "p20_tinyllama_decode4_1layer_bs1_ctx16_request_cycle",
            "bb9e796205b813f8dbb74bee001e0b05f1ba56416b4d47d02ed0b6cf87d60040",
            1,
            68,
            760,
        ),
        (
            "twenty_two_layer",
            "p20_tinyllama_decode4_22layer_bs1_ctx16_request_cycle",
            "214db18d7815e0c829fd40fc568f6fb897f203f09bb01f6d578e2da568f406b4",
            22,
            1076,
            13528,
        ),
    ],
)
def test_p20_saved_double_runs_pass_fail_closed_qualification(
    case: str,
    experiment: str,
    key: str,
    layers: int,
    expected_tasks: int,
    expected_parents: int,
) -> None:
    base = ROOT / "validation/p20" / case
    record = qualify_p20_decode_loop_pair(
        base / "leg1" / experiment / key,
        base / "leg2" / experiment / key,
        layers,
    )
    assert record["status"] == "passed"
    assert record["double_run_deterministic"] is True
    assert record["legs"][0]["task_count"] == expected_tasks
    assert record["legs"][0]["accepted_parent_requests"] == expected_parents
    assert record["legs"][0]["kv_versions_monotonic"] is True
    assert record["claim_boundary"]["performance_claim_allowed"] is False
