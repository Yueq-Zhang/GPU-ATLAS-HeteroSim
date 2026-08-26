"""M1 executable slice: graph, placement, C++ scheduling and C++ KV allocation."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from .analytical import estimate_link_duration_fs, estimate_node_cost
from .model_graph import (
    build_request_graph,
    graph_counters,
    model_spec_from_config,
    request_specs_from_config,
)
from .placement import place_nodes
from .operator_event import OperatorEventDispatcher
from .runtime_bridge import allocate_paged_kv, run_task_dag, simulate_token_barrier
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
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        return f"{revision}-dirty" if dirty else revision
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _execution_graph(
    graphs: list[tuple[object, list[object], object]],
    profile: str,
    access_policy: str,
    model: object,
    backends: Mapping[str, object],
    links: Mapping[str, object],
    execution_mode: str,
    dispatcher: OperatorEventDispatcher | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    timed_execution = execution_mode in {"analytical_preview", "operator_event"}
    tasks: list[dict[str, object]] = []
    routes: list[dict[str, object]] = []
    runtime_tasks: list[dict[str, object]] = []
    for graph_object, decisions_object, request_object in graphs:
        graph = graph_object
        decisions = decisions_object
        request = request_object
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
                route_record = {
                    "task_id": route_task,
                    "request_id": request.request_id,
                    "dependencies": [previous_task],
                    "producer_device": previous_device,
                    "consumer_device": decision.target_device,
                    **asdict(lowering),
                }
                if timed_execution:
                    if lowering.route_id not in links:
                        raise ValueError(
                            f"missing analytical parameters for route {lowering.route_id}"
                        )
                    link = links[lowering.route_id]
                    if not isinstance(link, Mapping):
                        raise ValueError(f"route {lowering.route_id} must be an object")
                    q_len = int(node.attributes.get("q_len", 1))
                    payload_bytes = q_len * model.hidden_size * model.bytes_per_element
                    duration_fs = estimate_link_duration_fs(payload_bytes, link)
                    route_record.update(
                        {
                            "resource_id": str(link["resource_id"]),
                            "payload_bytes": payload_bytes,
                            "duration_fs": duration_fs,
                            "analytical_parameters": dict(link),
                        }
                    )
                    runtime_tasks.append(
                        {
                            "task_id": route_task,
                            "resource_id": str(link["resource_id"]),
                            "dependencies": [previous_task],
                            "release_time_fs": 0,
                            "duration_fs": duration_fs,
                        }
                    )
                routes.append(route_record)
                dependencies = [route_task]
            task_id = f"task.{node.node_id}"
            backend_key = (
                "gpu"
                if decision.target_device == "gpu0"
                else "atlas"
                if decision.target_device == "atlas0.compute"
                else "host"
            )
            backend = backends[backend_key]
            if not isinstance(backend, Mapping):
                raise ValueError(f"backend {backend_key} must be an object")
            task_record = {
                "task_id": task_id,
                "request_id": request.request_id,
                "template_node_id": node.node_id,
                "task_kind": "device",
                "phase": node.phase,
                "op": node.op,
                "operator_group": node.attributes.get("operator_group"),
                "layer_id": node.layer_id,
                "step_id": node.step_id,
                "device_id": decision.target_device,
                "backend_id": f"{backend_key}.{backend['kind']}",
                "dependencies": dependencies,
                "fidelity": {
                    "compute_fidelity": "unavailable",
                    "memory_fidelity": "unavailable",
                    "link_fidelity": "unavailable",
                    "scheduler_fidelity": "event_modeled",
                    "extrapolated_fraction": 0.0,
                    "trace_coverage": 0.0,
                },
            }
            if execution_mode == "analytical_preview":
                cost = estimate_node_cost(node, model, backend)
                task_record.update(
                    {
                        "resource_id": decision.target_device,
                        "duration_fs": cost.duration_fs,
                        "analytical_cost": cost.to_dict(),
                        "analytical_parameters": dict(backend),
                        "fidelity": {
                            "compute_fidelity": "analytical",
                            "memory_fidelity": "analytical",
                            "link_fidelity": "not_applicable",
                            "scheduler_fidelity": "event_modeled",
                            "extrapolated_fraction": 1.0,
                            "trace_coverage": 0.0,
                        },
                    }
                )
                runtime_tasks.append(
                    {
                        "task_id": task_id,
                        "resource_id": decision.target_device,
                        "dependencies": dependencies,
                        "release_time_fs": request.arrival_time_fs
                        if previous_task is None
                        else 0,
                        "duration_fs": cost.duration_fs,
                    }
                )
            elif execution_mode == "operator_event":
                if dispatcher is None:
                    raise ValueError("operator_event execution requires a Backend dispatcher")
                result = dispatcher.dispatch(
                    backend_key, node, model, decision.target_device
                )
                task_record.update(
                    {
                        "backend_id": result.backend_id,
                        "resource_id": result.resource_id,
                        "duration_fs": result.duration_fs,
                        "timing_contract": dict(result.timing_contract),
                        "backend_statistics": dict(result.statistics),
                        "compiled_artifact": dict(result.artifact),
                        "fidelity": dict(result.fidelity),
                    }
                )
                runtime_tasks.append(
                    {
                        "task_id": task_id,
                        "resource_id": result.resource_id,
                        "dependencies": dependencies,
                        "release_time_fs": request.arrival_time_fs
                        if previous_task is None
                        else 0,
                        "duration_fs": result.duration_fs,
                    }
                )
            tasks.append(task_record)
            previous_device = decision.target_device
            previous_task = task_id
    return (
        {
            "schema_version": "hetero-execution-graph/v1",
            "tasks": tasks,
            "routes": routes,
        },
        runtime_tasks,
    )


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


def _timed_metrics(
    runtime_result: Mapping[str, object],
    execution_graph: Mapping[str, object],
    requests: list[dict[str, object]],
    run_status: str,
) -> dict[str, object]:
    timings = {
        str(item["task_id"]): item
        for item in runtime_result["tasks"]  # type: ignore[index]
    }
    device_tasks = list(execution_graph["tasks"])  # type: ignore[arg-type]
    request_metrics: list[dict[str, object]] = []
    for request in requests:
        request_id = str(request["request_id"])
        sampling = sorted(
            (
                task
                for task in device_tasks
                if task["request_id"] == request_id
                and str(task["template_node_id"]).endswith(".sampling")
            ),
            key=lambda task: int(task["step_id"]),
        )
        ready = [
            int(timings[str(task["task_id"])]["completion_time_fs"])
            for task in sampling
        ]
        intervals = [right - left for left, right in zip(ready, ready[1:])]
        retire_task = next(
            task
            for task in device_tasks
            if task["request_id"] == request_id
            and str(task["template_node_id"]).endswith(".kv_release")
        )
        arrival = int(request.get("arrival_time_fs", 0))
        request_metrics.append(
            {
                "request_id": request_id,
                "ttft_fs": ready[0] - arrival,
                "tpot_fs": sum(intervals) / len(intervals) if intervals else None,
                "itl_fs": intervals,
                "e2e_user_fs": ready[-1] - arrival,
                "retire_latency_fs": int(
                    timings[str(retire_task["task_id"])]["completion_time_fs"]
                )
                - arrival,
                "generated_length": len(ready),
                "final_committed_kv_len": int(request["prompt_length"])
                + int(request["output_length"])
                - 1,
            }
        )
    fidelities = [dict(task["fidelity"]) for task in device_tasks]

    def aggregate(field: str) -> str:
        values = {str(item[field]) for item in fidelities}
        return next(iter(values)) if len(values) == 1 else "mixed"

    count = len(fidelities)
    fidelity = {
        "compute_fidelity": aggregate("compute_fidelity"),
        "memory_fidelity": aggregate("memory_fidelity"),
        "link_fidelity": "analytical" if execution_graph["routes"] else "not_applicable",
        "scheduler_fidelity": "event_modeled",
        "extrapolated_fraction": sum(
            float(item["extrapolated_fraction"]) for item in fidelities
        )
        / count,
        "trace_coverage": sum(float(item["trace_coverage"]) for item in fidelities)
        / count,
        "artifact_coverage": sum(
            float(item.get("artifact_coverage", 0.0)) for item in fidelities
        )
        / count,
    }
    return {
        "schema_version": "hetero-metrics/v1",
        "run_status": run_status,
        "performance_claim_allowed": False,
        "makespan_fs": runtime_result["makespan_fs"],
        "requests": request_metrics,
        "fidelity": fidelity,
    }


def execute_run(
    config: dict[str, object],
    project_root: Path,
    runs_root: Path | None = None,
) -> Path:
    key = simulation_input_key(config)
    experiment = dict(config["experiment"])  # type: ignore[arg-type]
    root = runs_root or project_root / "runs"
    run_dir = root / str(experiment["name"]) / key
    run_dir.mkdir(parents=True, exist_ok=True)

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
    placed_graphs: list[tuple[object, list[object], object]] = []
    for request in requests:
        graph = build_request_graph(model, request)
        decisions = place_nodes(graph.nodes, placement_config, active_batch=len(requests))
        placed_graphs.append((graph, decisions, request))
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
    simulation = dict(config["simulation"])  # type: ignore[arg-type]
    execution_mode = str(simulation.get("execution_mode", "scheduler_validation"))
    if execution_mode == "full_runtime":
        raise ValueError("full_runtime is reserved but not implemented")
    backends = dict(config["backends"])  # type: ignore[arg-type]
    dispatcher = (
        OperatorEventDispatcher(project_root, run_dir / "backend_runs", backends)
        if execution_mode == "operator_event"
        else None
    )
    links = system.get("links", {})
    if not isinstance(links, Mapping):
        raise ValueError("system.links must be an object")
    execution_graph, runtime_tasks = _execution_graph(
        placed_graphs,
        profile,
        access_policy,
        model,
        backends,
        links,
        execution_mode,
        dispatcher,
    )
    if execution_mode in {"analytical_preview", "operator_event"}:
        runtime_result = run_task_dag(runtime_tasks)
        metrics = _timed_metrics(
            runtime_result, execution_graph, request_configs, execution_mode
        )
        timing_by_id = {
            str(item["task_id"]): item
            for item in runtime_result["tasks"]  # type: ignore[index]
        }
        for record in [*execution_graph["tasks"], *execution_graph["routes"]]:  # type: ignore[index]
            record["timing"] = timing_by_id[str(record["task_id"])]
    else:
        runtime_result = simulate_token_barrier(request_configs, scheduling)
        metrics = _metrics(runtime_result, request_configs)

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
            "runtime_owner": "cpp.GlobalEventRuntime"
            if execution_mode in {"analytical_preview", "operator_event"}
            else "cpp.TokenBarrierScheduler",
            "address_owner": "cpp.PagedKvAllocator",
            "backend_dispatch": dispatcher.provenance() if dispatcher else None,
        },
    )
    _write_json(
        run_dir / "model_graph.json",
        {"schema_version": "hetero-model-graph-bundle/v1", "requests": graph_payloads},
    )
    _write_json(run_dir / "execution_graph.json", execution_graph)
    _write_json(run_dir / "buffer_bindings.json", bindings)
    trace_payload = (
        dispatcher.trace_bundle()
        if dispatcher
        else {
            "schema_version": "hetero-trace-manifest/v1",
            "trace_id": f"unavailable.{key}",
            "trace_semantics": "none",
            "replay_safe": False,
            "qualification_record": None,
            "kernels_list": None,
            "capture": {"status": "no_cycle_trace", "execution_mode": execution_mode},
            "compilation": {"status": "not_materialized"},
            "address_ranges": [],
        }
    )
    _write_json(run_dir / "trace_manifest.json", trace_payload)
    _write_json(run_dir / "metrics.json", metrics)
    with (run_dir / "event_log.jsonl").open("w", encoding="utf-8") as stream:
        event_records = (
            runtime_result["tasks"]
            if execution_mode in {"analytical_preview", "operator_event"}
            else runtime_result["epochs"]
        )  # type: ignore[index]
        for event in event_records:
            stream.write(json.dumps(event, sort_keys=True) + "\n")
    return run_dir
