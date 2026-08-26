"""M1 executable slice: graph, placement, C++ scheduling and C++ KV allocation."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from .model_graph import (
    build_request_graph,
    graph_counters,
    model_spec_from_config,
    request_specs_from_config,
)
from .placement import place_nodes
from .runtime_bridge import allocate_paged_kv, simulate_token_barrier
from .topology import lower_cross_device_dependency, primary_3ddram


def simulation_input_key(config: Mapping[str, object]) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _jsonable(value: Any) -> Any:
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _git_revision(project_root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _execution_graph(
    graphs: list[tuple[object, list[object]]],
    profile: str,
    access_policy: str,
) -> dict[str, object]:
    tasks: list[dict[str, object]] = []
    routes: list[dict[str, object]] = []
    for graph_object, decisions_object in graphs:
        graph = graph_object
        decisions = decisions_object
        by_node = {decision.node_id: decision for decision in decisions}
        previous_device: str | None = None
        previous_task: str | None = None
        for node in graph.nodes:
            decision = by_node[node.node_id]
            dependencies: list[str] = []
            if previous_task is not None:
                dependencies.append(previous_task)
            if previous_device is not None and previous_device != decision.target_device:
                lowering = lower_cross_device_dependency(
                    profile, previous_device, decision.target_device, access_policy
                )
                route_task = f"route.{len(routes)}"
                routes.append(
                    {
                        "task_id": route_task,
                        "dependencies": [previous_task],
                        "producer_device": previous_device,
                        "consumer_device": decision.target_device,
                        **asdict(lowering),
                    }
                )
                dependencies = [route_task]
            task_id = f"task.{node.node_id}"
            tasks.append(
                {
                    "task_id": task_id,
                    "template_node_id": node.node_id,
                    "task_kind": "device",
                    "phase": node.phase,
                    "layer_id": node.layer_id,
                    "step_id": node.step_id,
                    "device_id": decision.target_device,
                    "backend_id": "gpu.roofline"
                    if decision.target_device == "gpu0"
                    else "atlas.analytical",
                    "dependencies": dependencies,
                    "fidelity": {
                        "compute_fidelity": "unavailable",
                        "memory_fidelity": "unavailable",
                        "link_fidelity": "event_modeled"
                        if dependencies and routes
                        else "unavailable",
                        "scheduler_fidelity": "event_modeled",
                        "extrapolated_fraction": 0.0,
                        "trace_coverage": 0.0,
                    },
                }
            )
            previous_device = decision.target_device
            previous_task = task_id
    return {
        "schema_version": "hetero-execution-graph/v1",
        "tasks": tasks,
        "routes": routes,
    }


def _metrics(
    runtime_result: dict[str, object], requests: list[dict[str, object]]
) -> dict[str, object]:
    inputs = {str(request["request_id"]): request for request in requests}
    request_metrics: list[dict[str, object]] = []
    for raw in runtime_result["requests"]:  # type: ignore[index]
        result = dict(raw)
        request_id = str(result["request_id"])
        arrival = int(inputs[request_id].get("arrival_time_fs", 0))
        ready = [int(value) for value in result["token_ready_time_fs"]]
        intervals = [right - left for left, right in zip(ready, ready[1:])]
        request_metrics.append(
            {
                "request_id": request_id,
                "ttft_fs": ready[0] - arrival,
                "tpot_fs": sum(intervals) / len(intervals) if intervals else None,
                "itl_fs": intervals,
                "e2e_user_fs": ready[-1] - arrival,
                "retire_latency_fs": int(result["finish_time_fs"]) - arrival,
                "generated_length": result["generated_length"],
                "final_committed_kv_len": result["committed_kv_length"],
            }
        )
    return {
        "schema_version": "hetero-metrics/v1",
        "run_status": "scheduler_validation",
        "performance_claim_allowed": False,
        "requests": request_metrics,
        "fidelity": {
            "compute_fidelity": "unavailable",
            "memory_fidelity": "unavailable",
            "link_fidelity": "unavailable",
            "scheduler_fidelity": "event_modeled",
            "extrapolated_fraction": 0.0,
            "trace_coverage": 0.0,
        },
    }


def execute_run(
    config: dict[str, object],
    project_root: Path,
    runs_root: Path | None = None,
) -> Path:
    model_config = dict(config["model"])  # type: ignore[arg-type]
    workload = dict(config["workload"])  # type: ignore[arg-type]
    request_configs = [dict(item) for item in workload["requests"]]
    scheduling = dict(config["scheduling"])  # type: ignore[arg-type]
    placement_config = dict(config["placement"])  # type: ignore[arg-type]
    address = dict(config["address"])  # type: ignore[arg-type]
    system = dict(config["system"])  # type: ignore[arg-type]
    profile = str(system["profile"])
    access_policy = str(system.get("access_policy", "copy"))

    model = model_spec_from_config(model_config)
    requests = request_specs_from_config(request_configs)
    graph_payloads: list[dict[str, object]] = []
    placed_graphs: list[tuple[object, list[object]]] = []
    for request in requests:
        graph = build_request_graph(model, request)
        decisions = place_nodes(graph.nodes, placement_config, active_batch=len(requests))
        placed_graphs.append((graph, decisions))
        graph_payloads.append(
            {
                "request_id": request.request_id,
                "graph": asdict(graph),
                "counters": asdict(graph_counters(model, request)),
                "placement": [asdict(decision) for decision in decisions],
            }
        )

    memory_space_id = primary_3ddram(profile)
    bindings = allocate_paged_kv(
        request_configs, model_config, address, memory_space_id
    )
    runtime_result = simulate_token_barrier(request_configs, scheduling)
    execution_graph = _execution_graph(placed_graphs, profile, access_policy)
    metrics = _metrics(runtime_result, request_configs)

    key = simulation_input_key(config)
    experiment = dict(config["experiment"])  # type: ignore[arg-type]
    root = runs_root or project_root / "runs"
    run_dir = root / str(experiment["name"]) / key
    run_dir.mkdir(parents=True, exist_ok=True)

    _write_json(run_dir / "resolved_config.yaml", config)
    dependency_lock = project_root / "dependency_lock.yaml"
    (run_dir / "dependency_lock.yaml").write_text(
        dependency_lock.read_text(encoding="utf-8"), encoding="utf-8"
    )
    _write_json(
        run_dir / "provenance.json",
        {
            "schema_version": "hetero-provenance/v1",
            "simulator_revision": _git_revision(project_root),
            "simulation_input_key": key,
            "runtime_owner": "cpp.TokenBarrierScheduler",
            "address_owner": "cpp.PagedKvAllocator",
        },
    )
    _write_json(
        run_dir / "model_graph.json",
        {"schema_version": "hetero-model-graph-bundle/v1", "requests": graph_payloads},
    )
    _write_json(run_dir / "execution_graph.json", execution_graph)
    _write_json(run_dir / "buffer_bindings.json", bindings)
    _write_json(
        run_dir / "trace_manifest.json",
        {
            "schema_version": "hetero-trace-manifest/v1",
            "trace_semantics": "functional",
            "replay_safe": False,
            "qualification_record": None,
            "captures": [],
        },
    )
    _write_json(run_dir / "metrics.json", metrics)
    with (run_dir / "event_log.jsonl").open("w", encoding="utf-8") as stream:
        for epoch in runtime_result["epochs"]:  # type: ignore[index]
            stream.write(json.dumps(epoch, sort_keys=True) + "\n")
    return run_dir
