from copy import deepcopy

import pytest

from frontend.hetero.ir import ModelNode, NodeKind, Phase
from frontend.hetero.model_graph import ModelSpec
from frontend.hetero.online_operator_runtime import (
    OnlineDispatchSpec,
    OnlineOperatorRuntimeError,
    run_online_operator_dag,
)
from frontend.hetero.operator_event import BackendTaskResult


def _node(node_id: str, device: str) -> OnlineDispatchSpec:
    node = ModelNode(
        node_id=node_id,
        kind=NodeKind.COMPUTE,
        op="golden",
        phase=Phase.DECODE,
        layer_id=0,
        step_id=0,
    )
    model = ModelSpec("tiny", 128, 256, 1, 4, 2, 32, 256)
    return OnlineDispatchSpec(
        task_id=f"task.{node_id}",
        backend_key="gpu" if device == "gpu0" else "atlas",
        node=node,
        model=model,
        device_id=device,
    )


def _graph() -> dict[str, object]:
    return {
        "tasks": [
            {
                "task_id": "task.gpu",
                "task_kind": "device",
                "device_id": "gpu0",
                "resource_id": "gpu0",
                "dependencies": [],
                "release_time_fs": 0,
                "input_values": [],
                "output_values": [{"value_id": "x", "version": 1}],
            },
            {
                "task_id": "task.atlas",
                "task_kind": "device",
                "device_id": "atlas0.compute",
                "resource_id": "atlas0.compute",
                "dependencies": ["task.gpu", "route.x"],
                "release_time_fs": 0,
                "input_values": [{"value_id": "x", "version": 1}],
                "output_values": [{"value_id": "y", "version": 1}],
            },
        ],
        "routes": [
            {
                "task_id": "route.x",
                "task_kind": "route",
                "resource_id": "shared3d.sync",
                "dependencies": ["task.gpu"],
                "release_time_fs": 0,
                "duration_fs": 10,
                "value_id": "x",
                "value_version": 1,
                "producer_device": "gpu0",
                "consumer_device": "atlas0.compute",
                "kind": "synchronization",
                "actions": [
                    "writeback",
                    "release_fence",
                    "invalidate",
                    "acquire_fence",
                ],
            }
        ],
        "residency_plan": {"events": []},
    }


def _dispatch(spec: OnlineDispatchSpec) -> BackendTaskResult:
    duration = 5 if spec.device_id == "gpu0" else 7
    return BackendTaskResult(
        backend_id=f"{spec.backend_key}.cycle_golden",
        duration_fs=duration,
        resource_id=spec.device_id,
        timing_contract={"duration_semantics": "total"},
        fidelity={"compute_fidelity": "cycle_simulated_golden"},
        statistics={"cycles": duration},
        artifact={"kind": "golden"},
    )


def test_backend_launch_waits_for_route_completion_and_validates_version() -> None:
    graph = _graph()
    result = run_online_operator_dag(
        graph,
        {"task.gpu": _node("gpu", "gpu0"), "task.atlas": _node("atlas", "atlas0.compute")},
        _dispatch,
    )
    timings = {item["task_id"]: item for item in result["tasks"]}
    assert timings["task.gpu"]["completion_time_fs"] == 5
    assert timings["route.x"]["start_time_fs"] == 5
    assert timings["route.x"]["completion_time_fs"] == 15
    assert timings["task.atlas"]["start_time_fs"] == 15
    assert graph["tasks"][1]["backend_launch_time_fs"] == 15  # type: ignore[index]
    assert graph["tasks"][1]["validated_input_versions"] == [  # type: ignore[index]
        {"value_id": "x", "version": 1}
    ]
    assert result["backend_dispatch_count"] == 2
    assert result["version_checks"] == 2
    assert result["final_versions"] == {"x": 1, "y": 1}


def test_stale_consumer_version_fails_before_backend_launch() -> None:
    graph = deepcopy(_graph())
    graph["tasks"][1]["input_values"][0]["version"] = 0  # type: ignore[index]
    with pytest.raises(OnlineOperatorRuntimeError, match="stale or unavailable"):
        run_online_operator_dag(
            graph,
            {"task.gpu": _node("gpu", "gpu0"), "task.atlas": _node("atlas", "atlas0.compute")},
            _dispatch,
        )
