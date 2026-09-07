from copy import deepcopy
from pathlib import Path

import pytest

from frontend.hetero.decode_lifecycle import build_decode_kv_lifecycle
from frontend.hetero.global_memory_map import build_global_memory_map
from frontend.hetero.model_graph import (
    build_request_graph,
    graph_counters,
    model_spec_from_config,
    request_specs_from_config,
)
from frontend.hetero.online_operator_runtime import OnlineDispatchSpec
from frontend.hetero.operator_capability import OperatorCapabilityCatalog
from frontend.hetero.placement import place_nodes
from frontend.hetero.request_cycle_artifact import RequestCycleCatalog
from frontend.hetero.request_cycle_qualification import qualify_p19_decode_pair
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
        / f"p19_tinyllama_decode_{layers}layer_bs1_ctx16_request_cycle.json"
    )


def _graph(layers: int):
    config = _config(layers)
    model = model_spec_from_config(dict(config["model"]))
    request = request_specs_from_config([dict(config["workload"]["requests"][0])])[0]
    graph = build_request_graph(model, request)
    return config, model, request, graph


@pytest.mark.parametrize("layers,expected_tasks", [(1, 20), (22, 272)])
def test_p19_decode_graph_is_a_single_materialized_forward(
    layers: int, expected_tasks: int
) -> None:
    _, model, request, graph = _graph(layers)
    assert len(graph.nodes) == expected_tasks == 8 + 12 * layers
    assert {node.phase.value for node in graph.nodes} == {"control", "decode"}
    assert len([node for node in graph.nodes if node.op == "kv_append"]) == layers
    assert (
        len([node for node in graph.nodes if node.op == "causal_attention"]) == layers
    )
    for node in graph.nodes:
        if node.phase.value == "decode":
            assert node.attributes["q_len"] == 1
            assert node.attributes["past_kv_len"] == 16
            assert node.attributes["attention_kv_len"] == 17
    counters = graph_counters(model, request)
    assert counters.prefill_forwards == 0
    assert counters.decode_forwards == 1
    assert counters.kv_append_pairs == layers
    assert counters.kv_range_writes == 2 * layers
    assert counters.final_committed_kv_len == 17


@pytest.mark.parametrize("layers", [1, 22])
def test_p19_decode_kv_lifecycle_binds_nonoverlapping_global_pa(layers: int) -> None:
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
    lifecycle = build_decode_kv_lifecycle(graph, model, request, allocations)

    assert memory_map["non_overlapping"] is True
    assert lifecycle["layer_count"] == layers
    assert lifecycle["initial_kv_length"] == 16
    assert lifecycle["final_kv_length"] == 17
    assert lifecycle["bytes_per_kv_token_per_tensor"] == 512
    assert lifecycle["total_initial_valid_kv_bytes"] == 2 * layers * 8192
    assert lifecycle["total_appended_kv_bytes"] == 2 * layers * 512
    assert lifecycle["global_pa_bound"] is True
    assert lifecycle["kv_release_after_request_finish"] is True
    values = [value for layer in lifecycle["layers"] for value in layer["values"]]
    assert len(values) == 2 * layers
    assert all(value["append_range"]["offset_bytes"] == 8192 for value in values)
    assert all(
        value["global_pa"]["append_address"]
        == value["global_pa"]["base_address"] + 8192
        for value in values
    )


def test_request_cycle_catalog_dispatches_a_decode_operator() -> None:
    _, model, _, graph = _graph(1)
    catalog = RequestCycleCatalog.load(
        ROOT
        / "configs/hetero/cycle_artifacts"
        / "tinyllama11b_request_cycle_fp16_v1.json"
    )
    node = next(node for node in graph.nodes if node.op == "causal_attention")
    plan = catalog.plan(
        OnlineDispatchSpec(
            task_id=f"task.{node.node_id}",
            backend_key="gpu",
            node=node,
            model=model,
            device_id="gpu0",
        ),
        1_200_000_000,
    )
    assert catalog.supported_phases() == {"prefill", "decode", "control"}
    assert plan.global_compute_cycles > 0
    assert plan.artifact["kind"] == "request_tiled_cycle_contract"
    assert plan.fidelity["performance_eligible"] is False
    assert plan.fidelity["trace_coverage"] == 0.0


@pytest.mark.parametrize("layers,expected_tasks", [(1, 20), (22, 272)])
def test_p19_capability_catalog_records_every_exact_shape_without_performance_claim(
    layers: int, expected_tasks: int
) -> None:
    catalog = OperatorCapabilityCatalog.load(
        ROOT
        / "configs/hetero/operator_capabilities"
        / f"tinyllama_decode_{layers}layer_bs1_ctx16_p19.json"
    )
    assert (
        sum(item.instances_in_reference_graph for item in catalog.operators.values())
        == expected_tasks
    )
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
    assert len(decode_shapes) == layers
    assert {shape.layer_id for shape in decode_shapes} == set(range(layers))
    assert all(shape.q_len == 1 and shape.kv_length == 17 for shape in decode_shapes)


@pytest.mark.parametrize("layers", [1, 22])
def test_p19_request_cycle_configs_are_strictly_valid(layers: int) -> None:
    config = _config(layers)
    assert config["simulation"] == {
        "coupling": "request_cycle",
        "execution_mode": "request_cycle",
        "validation_policy": "required",
    }
    request = config["workload"]["requests"][0]
    assert request["execution_scope"] == "decode_step"
    assert request["initial_kv_length"] == 16
    assert request["output_length"] == 1
    assert config["model"]["materialize_parameters"] is True


def test_request_cycle_rejects_multi_token_and_nonmaterialized_parameters() -> None:
    config = _config(1)
    broken = deepcopy(config)
    broken["workload"]["requests"][0]["output_length"] = 2
    with pytest.raises(ConfigError, match="output_length=1"):
        validate_config(broken)
    broken = deepcopy(config)
    broken["model"]["materialize_parameters"] = False
    with pytest.raises(ConfigError, match="materialize_parameters=true"):
        validate_config(broken)


@pytest.mark.parametrize(
    "case,experiment,key,layers,expected_tasks,expected_parents",
    [
        (
            "one_layer",
            "p19_tinyllama_decode_1layer_bs1_ctx16_request_cycle",
            "326a377f896d82b7d83020daa0b982b53dc3691a443abb70e40a399b486f2039",
            1,
            20,
            190,
        ),
        (
            "twenty_two_layer",
            "p19_tinyllama_decode_22layer_bs1_ctx16_request_cycle",
            "bc41de9d8e3af16e47fdc3902834bf22ee9b727d156eee969f3d6be2f3ce1087",
            22,
            272,
            3382,
        ),
    ],
)
def test_p19_saved_double_runs_pass_fail_closed_qualification(
    case: str,
    experiment: str,
    key: str,
    layers: int,
    expected_tasks: int,
    expected_parents: int,
) -> None:
    base = ROOT / "validation/p19" / case
    record = qualify_p19_decode_pair(
        base / "leg1" / experiment / key,
        base / "leg2" / experiment / key,
        layers,
    )
    assert record["status"] == "passed"
    assert record["double_run_deterministic"] is True
    assert record["legs"][0]["task_count"] == expected_tasks
    assert record["legs"][0]["accepted_parent_requests"] == expected_parents
    assert record["claim_boundary"]["performance_claim_allowed"] is False
