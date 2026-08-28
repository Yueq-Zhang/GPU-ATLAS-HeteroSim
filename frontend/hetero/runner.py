"""M1 executable slice: graph, placement, C++ scheduling and C++ KV allocation."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from .analytical import estimate_link_duration_fs, estimate_node_cost
from .batching import build_batch_plan
from .execution_plan import build_single_placement_plan, route_to_dict
from .global_memory_map import build_global_memory_map
from .live_ramulator2 import LiveRamulator2Bridge
from .memory_system import (
    build_dynamic_kv_lifecycle,
    run_reference_coupled_dag,
)
from .model_graph import (
    ModelSpec,
    build_request_graph,
    graph_counters,
    model_spec_from_config,
    request_specs_from_config,
)
from .online_operator_runtime import (
    OnlineDispatchSpec,
    run_online_operator_dag,
)
from .placement import place_nodes
from .operator_event import OperatorEventDispatcher
from .prefill_cycle_artifact import PrefillCycleDispatcher
from .prefill_cycle_runtime import run_prefill_cycle_dag
from .runtime_bridge import allocate_paged_kv, run_task_dag, simulate_token_barrier
from .topology import primary_3ddram


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


def _validate_gpu_only_shared_3d_baseline(
    execution_graph: Mapping[str, object],
    memory_config: Mapping[str, object] | None,
) -> dict[str, object] | None:
    """Enforce the no-Logic-Die-contention baseline at the derived task level."""

    if not isinstance(memory_config, Mapping) or memory_config.get(
        "access_mode", "shared_gpu_atlas"
    ) != "gpu_only":
        return None
    tasks = execution_graph.get("tasks")
    routes = execution_graph.get("routes")
    if not isinstance(tasks, list) or not isinstance(routes, list):
        raise ValueError("execution graph must contain task and route arrays")
    non_gpu_tasks = [
        str(task.get("task_id"))
        for task in tasks
        if not isinstance(task, Mapping) or task.get("device_id") != "gpu0"
    ]
    if non_gpu_tasks:
        raise ValueError(
            "gpu_only shared 3D baseline contains non-GPU tasks: "
            + ", ".join(non_gpu_tasks[:8])
        )
    if routes:
        raise ValueError(
            "gpu_only shared 3D baseline unexpectedly contains cross-device routes"
        )
    initiators = memory_config.get("initiator_order")
    if initiators != ["gpu0"]:
        raise ValueError(
            "gpu_only shared 3D baseline requires initiator_order=['gpu0']"
        )
    return {
        "mode": "gpu_only",
        "enabled": False,
        "gpu_tasks": len(tasks),
        "logic_die_tasks": 0,
        "gpu_memory_requests": 0,
        "logic_die_memory_requests": 0,
    }


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
    model: ModelSpec,
    backends: Mapping[str, object],
    links: Mapping[str, object],
    execution_mode: str,
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    dict[str, OnlineDispatchSpec],
]:
    timed_execution = execution_mode in {
        "analytical_preview",
        "operator_event",
        "full_runtime",
        "prefill_cycle",
    }
    tasks: list[dict[str, object]] = []
    routes: list[dict[str, object]] = []
    runtime_tasks: list[dict[str, object]] = []
    residency_events: list[dict[str, object]] = []
    final_residency: list[dict[str, object]] = []
    request_conservation: list[dict[str, object]] = []
    operator_dispatch_specs: dict[str, OnlineDispatchSpec] = {}
    for graph_object, decisions_object, request_object in graphs:
        graph = graph_object
        decisions = decisions_object
        request = request_object
        plan = build_single_placement_plan(
            graph, decisions, profile, access_policy, model
        )
        request_conservation.append(
            {"request_id": request.request_id, **dict(plan.conservation)}
        )
        residency_events.extend(plan.residency_events)
        final_residency.extend(plan.final_records)
        routes_by_consumer: dict[str, list[object]] = {}
        for planned_route in plan.routes:
            routes_by_consumer.setdefault(
                planned_route.consumer_task_id, []
            ).append(planned_route)

        for planned_node in plan.nodes:
            node = planned_node.node
            decision = planned_node.decision
            dependencies = list(planned_node.dependencies)
            for planned_route in routes_by_consumer.get(planned_node.task_id, []):
                route_record = {
                    "request_id": request.request_id,
                    "task_kind": "route",
                    "release_time_fs": 0,
                    **route_to_dict(planned_route),
                }
                if timed_execution:
                    route_id = route_record["route_id"]
                    if route_id not in links:
                        raise ValueError(
                            f"missing analytical parameters for route {route_id}"
                        )
                    link = links[route_id]
                    if not isinstance(link, Mapping):
                        raise ValueError(f"route {route_id} must be an object")
                    payload_bytes = int(route_record["payload_bytes"])
                    duration_fs = (
                        max(1, int(link.get("latency_fs", 0)))
                        if execution_mode == "prefill_cycle"
                        and str(
                            getattr(
                                route_record.get("kind"),
                                "value",
                                route_record.get("kind"),
                            )
                        )
                        == "synchronization"
                        else estimate_link_duration_fs(payload_bytes, link)
                    )
                    route_record.update(
                        {
                            "resource_id": str(link["resource_id"]),
                            "payload_bytes": payload_bytes,
                            "duration_fs": duration_fs,
                            "analytical_parameters": dict(link),
                            "route_timing_semantics": (
                                "live_durable_fence_and_consumer_acquire_probe"
                                if execution_mode == "prefill_cycle"
                                and str(
                                    getattr(
                                        route_record.get("kind"),
                                        "value",
                                        route_record.get("kind"),
                                    )
                                )
                                == "synchronization"
                                else "payload_serialization"
                            ),
                        }
                    )
                    runtime_tasks.append(
                        {
                            "task_id": planned_route.task_id,
                            "resource_id": str(link["resource_id"]),
                            "dependencies": list(planned_route.dependencies),
                            "release_time_fs": 0,
                            "duration_fs": duration_fs,
                        }
                    )
                routes.append(route_record)
            task_id = planned_node.task_id
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
                "read_values": list(node.read_values),
                "write_values": list(node.write_values),
                "input_values": list(planned_node.input_values),
                "output_values": list(planned_node.output_values),
                "device_id": decision.target_device,
                "backend_id": f"{backend_key}.{backend['kind']}",
                "dependencies": dependencies,
                "resource_id": decision.target_device,
                "release_time_fs": request.arrival_time_fs
                if not node.dependencies
                else 0,
                "fidelity": {
                    "compute_fidelity": "unavailable",
                    "memory_fidelity": "unavailable",
                    "link_fidelity": "unavailable",
                    "scheduler_fidelity": "event_modeled",
                    "extrapolated_fraction": 0.0,
                    "trace_coverage": 0.0,
                },
            }
            if execution_mode in {"analytical_preview", "full_runtime"}:
                cost = estimate_node_cost(node, model, backend)
                duration_fs = (
                    max(1, cost.compute_time_fs)
                    if execution_mode == "full_runtime"
                    else cost.duration_fs
                )
                task_record.update(
                    {
                        "resource_id": decision.target_device,
                        "duration_fs": duration_fs,
                        "analytical_cost": cost.to_dict(),
                        "analytical_parameters": dict(backend),
                        "fidelity": {
                            "compute_fidelity": "analytical",
                            "memory_fidelity": (
                                "event_modeled"
                                if execution_mode == "full_runtime"
                                else "analytical"
                            ),
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
                        if not node.dependencies
                        else 0,
                        "duration_fs": duration_fs,
                    }
                )
            elif execution_mode in {"operator_event", "prefill_cycle"}:
                operator_dispatch_specs[task_id] = OnlineDispatchSpec(
                    task_id=task_id,
                    backend_key=backend_key,
                    node=node,
                    model=model,
                    device_id=decision.target_device,
                )
            tasks.append(task_record)
    logical_nodes = sum(
        int(item["logical_node_count"]) for item in request_conservation
    )
    exact_once = all(
        bool(item["each_logical_node_exactly_once"])
        for item in request_conservation
    ) and logical_nodes == len(tasks)
    return (
        {
            "schema_version": "hetero-execution-graph/v1",
            "tasks": tasks,
            "routes": routes,
            "placement_contract": {
                "schema_version": "hetero-single-placement/v1",
                "semantics": "one_logical_node_one_device_one_backend_dispatch",
                "requests": request_conservation,
                "logical_node_count": logical_nodes,
                "materialized_device_task_count": len(tasks),
                "backend_dispatch_count": 0
                if execution_mode in {"operator_event", "prefill_cycle"}
                else None,
                "each_logical_node_exactly_once": exact_once,
            },
            "residency_plan": {
                "schema_version": "hetero-residency-plan/v1",
                "coherence": "explicit_noncoherent",
                "initialization_policy": "first_consumer_binding",
                "events": residency_events,
                "final_records": final_residency,
            },
        },
        runtime_tasks,
        operator_dispatch_specs,
    )


def _materialize_residency(
    execution_graph: Mapping[str, object],
    timing_by_id: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Attach runtime timestamps to the already validated residency plan."""

    plan = execution_graph.get("residency_plan")
    if not isinstance(plan, Mapping):
        raise ValueError("execution graph is missing residency_plan")
    raw_events = plan.get("events")
    if not isinstance(raw_events, list):
        raise ValueError("residency plan events must be an array")
    events: list[dict[str, object]] = []
    for raw in raw_events:
        if not isinstance(raw, Mapping):
            raise ValueError("residency event must be an object")
        trigger_task = str(raw["trigger_task_id"])
        timing = timing_by_id[trigger_task]
        trigger = str(raw["trigger"])
        time_fs = (
            int(timing["start_time_fs"])
            if trigger == "task_start"
            else int(timing["completion_time_fs"])
        )
        events.append({**dict(raw), "time_fs": time_fs})
    events.sort(key=lambda item: (int(item["time_fs"]), int(item["sequence"])))
    return {
        "schema_version": "hetero-residency/v2",
        "coherence": plan["coherence"],
        "initialization_policy": plan["initialization_policy"],
        "records": plan["final_records"],
        "events": events,
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
                "final_committed_kv_len": (
                    int(request.get("initial_kv_length", 0)) + 1
                    if request.get("execution_scope", "full_request") == "decode_step"
                    else int(request["prompt_length"])
                    + int(request["output_length"])
                    - 1
                ),
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
    allocation_requests = []
    for request in request_configs:
        allocation_request = dict(request)
        if allocation_request.get("execution_scope", "full_request") == "decode_step":
            allocation_request["prompt_length"] = (
                int(allocation_request["initial_kv_length"]) + 1
            )
            allocation_request["output_length"] = 1
        allocation_request.pop("execution_scope", None)
        allocation_request.pop("initial_kv_length", None)
        allocation_requests.append(allocation_request)
    bindings = allocate_paged_kv(
        allocation_requests, model_config, address, memory_space_id
    )
    simulation = dict(config["simulation"])  # type: ignore[arg-type]
    execution_mode = str(simulation.get("execution_mode", "scheduler_validation"))
    backends = dict(config["backends"])  # type: ignore[arg-type]
    dispatcher = (
        OperatorEventDispatcher(project_root, run_dir / "backend_runs", backends)
        if execution_mode == "operator_event"
        else None
    )
    links = system.get("links", {})
    if not isinstance(links, Mapping):
        raise ValueError("system.links must be an object")
    execution_graph, runtime_tasks, operator_dispatch_specs = _execution_graph(
        placed_graphs,
        profile,
        access_policy,
        model,
        backends,
        links,
        execution_mode,
    )
    batch_plan: dict[str, object] | None = None
    memory_lifecycle: dict[str, object] | None = None
    link_statistics: dict[str, object] | None = None
    memory_statistics: dict[str, object] | None = None
    residency_payload: dict[str, object] | None = None
    online_dispatch_payload: dict[str, object] | None = None
    request_cycle_payload: dict[str, object] | None = None
    global_memory_map_payload: dict[str, object] | None = None
    prefill_coverage_payload: dict[str, object] | None = None
    shared_config: Mapping[str, object] | None = None
    shared_reference_active = False
    if execution_mode in {"full_runtime", "prefill_cycle"}:
        memory_services = system.get("memory_services", {})
        if not isinstance(memory_services, Mapping):
            raise ValueError("system.memory_services must be an object")
        candidate = memory_services.get(memory_space_id)
        if isinstance(candidate, Mapping):
            shared_config = candidate
            shared_reference_active = candidate.get("kind") == "shared_3d_reference"
            if candidate.get("kind") == "ramulator2" and execution_mode == "full_runtime":
                raise ValueError(
                    "full_runtime live Ramulator2 coupling is not qualified; "
                    "use 'qualify-memory' for standalone replay or configure "
                    "kind=shared_3d_reference"
                )
    competition_summary = _validate_gpu_only_shared_3d_baseline(
        execution_graph, shared_config
    )
    if execution_mode in {
        "analytical_preview",
        "operator_event",
        "full_runtime",
        "prefill_cycle",
    }:
        if execution_mode == "full_runtime":
            runtime_result, link_statistics, memory_statistics = (
                run_reference_coupled_dag(
                    runtime_tasks,
                    execution_graph["tasks"],  # type: ignore[arg-type]
                    execution_graph["routes"],  # type: ignore[arg-type]
                    links,
                    shared_config if shared_reference_active else None,
                    memory_space_id,
                )
            )
        elif execution_mode == "operator_event":
            if dispatcher is None:
                raise ValueError(
                    "operator_event execution requires a Backend dispatcher"
                )
            runtime_result = run_online_operator_dag(
                execution_graph,
                operator_dispatch_specs,
                lambda spec: dispatcher.dispatch(
                    spec.backend_key,
                    spec.node,
                    spec.model,
                    spec.device_id,
                ),
            )
            placement_contract = execution_graph["placement_contract"]
            if not isinstance(placement_contract, dict):
                raise ValueError("placement_contract must be an object")
            placement_contract["backend_dispatch_count"] = runtime_result[
                "backend_dispatch_count"
            ]
            placement_contract["online_dispatch_gate"] = {
                "schema_version": runtime_result["schema_version"],
                "version_checks": runtime_result["version_checks"],
                "launch_count": len(runtime_result["launch_log"]),
                "backend_launches_after_dependencies": True,
            }
            online_dispatch_payload = {
                "schema_version": runtime_result["schema_version"],
                "backend_dispatch_count": runtime_result[
                    "backend_dispatch_count"
                ],
                "version_checks": runtime_result["version_checks"],
                "launch_log": runtime_result["launch_log"],
                "final_versions": runtime_result["final_versions"],
            }
        elif execution_mode == "prefill_cycle":
            if not isinstance(shared_config, Mapping) or shared_config.get(
                "kind"
            ) != "ramulator2":
                raise ValueError("prefill_cycle requires a live Ramulator2 service")
            global_clock_hz = int(shared_config["gpu_clock_hz"])
            prefill_dispatcher = PrefillCycleDispatcher(
                project_root, backends, global_clock_hz
            )
            allocations, global_memory_map_payload = build_global_memory_map(
                execution_graph,
                memory_space_id,
                int(
                    address.get(
                        "global_pa_capacity_bytes", address["kv_capacity_bytes"]
                    )
                ),
                int(address.get("allocation_alignment_bytes", 64)),
            )
            bridge = LiveRamulator2Bridge(project_root, shared_config)
            runtime_result = run_prefill_cycle_dag(
                execution_graph,
                operator_dispatch_specs,
                prefill_dispatcher,
                bridge,
                allocations,
                global_clock_hz=global_clock_hz,
                transaction_bytes=int(shared_config["transaction_bytes"]),
                max_samples_per_value=int(shared_config["max_samples_per_value"]),
            )
            memory_statistics = dict(runtime_result["memory_statistics"])
            prefill_coverage_payload = dict(runtime_result["artifact_coverage"])
            request_cycle_payload = {
                "schema_version": runtime_result["schema_version"],
                "backend_dispatch_count": runtime_result[
                    "backend_dispatch_count"
                ],
                "version_checks": runtime_result["version_checks"],
                "launch_log": runtime_result["launch_log"],
                "final_versions": runtime_result["final_versions"],
                "memory_trace": runtime_result["memory_trace"],
            }
            placement_contract = execution_graph["placement_contract"]
            if not isinstance(placement_contract, dict):
                raise ValueError("placement_contract must be an object")
            placement_contract["backend_dispatch_count"] = runtime_result[
                "backend_dispatch_count"
            ]
            placement_contract["request_cycle_gate"] = {
                "schema_version": runtime_result["schema_version"],
                "version_checks": runtime_result["version_checks"],
                "one_live_ramulator2": True,
                "zero_outstanding": memory_statistics["outstanding"] == 0,
                "all_artifacts_covered": prefill_coverage_payload[
                    "all_tasks_covered"
                ],
            }
        else:
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
            record["effective_duration_fs"] = (
                int(record["timing"]["completion_time_fs"])
                - int(record["timing"]["start_time_fs"])
            )
        residency_payload = _materialize_residency(execution_graph, timing_by_id)
        if execution_mode == "prefill_cycle":
            metrics["run_status"] = "prefill_cycle_deployment"
            metrics["implementation_status"] = "implemented_unqualified"
            metrics["fidelity"]["scheduler_fidelity"] = "cycle_event"  # type: ignore[index]
            metrics["fidelity"]["memory_fidelity"] = (  # type: ignore[index]
                "live_ramulator2_sampled_requests"
            )
            metrics["fidelity"]["link_fidelity"] = (  # type: ignore[index]
                "cycle_modeled" if execution_graph["routes"] else "external_gpu_link"
            )
            metrics["memory"] = memory_statistics
            metrics["prefill_artifact_coverage"] = prefill_coverage_payload
            if competition_summary is not None:
                initiators = memory_statistics.get("initiators", {})
                if not isinstance(initiators, Mapping):
                    raise RuntimeError("live memory initiator statistics must be an object")
                gpu_stats = initiators.get("gpu0", {})
                atlas_stats = initiators.get("atlas0.compute", {})
                if not isinstance(gpu_stats, Mapping) or not isinstance(
                    atlas_stats, Mapping
                ):
                    raise RuntimeError("live memory initiator entries must be objects")
                competition_summary["gpu_memory_requests"] = int(
                    gpu_stats.get("parents", 0)
                )
                competition_summary["logic_die_memory_requests"] = int(
                    atlas_stats.get("parents", 0)
                )
                if competition_summary["logic_die_memory_requests"] != 0:
                    raise RuntimeError(
                        "gpu_only baseline observed Logic Die memory requests"
                    )
                metrics["gpu_logic_die_competition"] = competition_summary
        if execution_mode == "full_runtime":
            scheduler_result = simulate_token_barrier(request_configs, scheduling)
            batch_plan = build_batch_plan(
                scheduler_result,
                placed_graphs[0][0].nodes,  # type: ignore[union-attr]
                placement_config,
            )
            memory_lifecycle = build_dynamic_kv_lifecycle(
                request_configs,
                scheduler_result,
                model_config,
                address,
                memory_space_id,
            )
            bindings["allocation_mode"] = "dynamic_first_fit_with_release"
            bindings["dynamic_lifecycle_schema"] = memory_lifecycle["schema_version"]
            bindings["peak_used_bytes"] = memory_lifecycle["memory_spaces"][0][  # type: ignore[index]
                "peak_bytes"
            ]
            metrics["run_status"] = "full_runtime_reference"
            metrics["implementation_status"] = "implemented_unqualified"
            metrics["fidelity"]["link_fidelity"] = "event_modeled"  # type: ignore[index]
            metrics["fidelity"]["memory_fidelity"] = "event_modeled"
            metrics["batch"] = {
                "epoch_count": len(batch_plan["epochs"]),
                "device_subbatch_count": len(batch_plan["device_subbatches"]),
                "effective_tokens": batch_plan["effective_tokens"],
                "padded_tokens": batch_plan["padded_tokens"],
            }
            metrics["memory"] = memory_statistics
            metrics["links"] = link_statistics
            if competition_summary is not None:
                memory_competition = memory_statistics.get(
                    "gpu_logic_die_competition", {}
                )
                if not isinstance(memory_competition, Mapping):
                    raise RuntimeError("memory competition summary must be an object")
                competition_summary["gpu_memory_requests"] = int(
                    memory_competition.get("gpu_requests", 0)
                )
                competition_summary["logic_die_memory_requests"] = int(
                    memory_competition.get("logic_die_requests", 0)
                )
                if competition_summary["logic_die_memory_requests"] != 0:
                    raise RuntimeError(
                        "gpu_only baseline observed Logic Die memory requests"
                    )
                metrics["gpu_logic_die_competition"] = competition_summary
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
            "runtime_owner": "python.OnlineOperatorRuntime"
            if execution_mode == "operator_event"
            else "python.PrefillCycleRuntime"
            if execution_mode == "prefill_cycle"
            else "cpp.GlobalEventRuntime"
            if execution_mode in {"analytical_preview", "full_runtime"}
            else "cpp.TokenBarrierScheduler",
            "address_owner": (
                "cpp.RuntimeMemoryPlanner"
                if execution_mode == "full_runtime"
                else "python.GlobalPhysicalAddressAllocator"
                if execution_mode == "prefill_cycle"
                else "cpp.PagedKvAllocator"
            ),
            "memory_timing_owner": (
                "shared3d.live_ramulator2"
                if execution_mode == "prefill_cycle"
                else (
                    "shared3d.memory_service"
                    if execution_mode == "full_runtime"
                    and profile == "model3_gpu_native_3ddram"
                    and isinstance(shared_config, Mapping)
                    and shared_config.get("kind") == "shared_3d_reference"
                    else None
                )
            ),
            "backend_dispatch": dispatcher.provenance() if dispatcher else None,
            "prefill_cycle": {
                "one_live_ramulator2": True,
                "request_sampling": "evenly_spaced_bounded",
                "performance_eligible": False,
            }
            if execution_mode == "prefill_cycle"
            else None,
        },
    )
    _write_json(
        run_dir / "model_graph.json",
        {"schema_version": "hetero-model-graph-bundle/v1", "requests": graph_payloads},
    )
    _write_json(run_dir / "execution_graph.json", execution_graph)
    _write_json(run_dir / "buffer_bindings.json", bindings)
    if batch_plan is not None:
        _write_json(run_dir / "batch_plan.json", batch_plan)
    if memory_lifecycle is not None:
        _write_json(run_dir / "memory_lifecycle.json", memory_lifecycle)
    if link_statistics is not None:
        _write_json(run_dir / "link_statistics.json", link_statistics)
    if memory_statistics is not None:
        _write_json(run_dir / "memory_statistics.json", memory_statistics)
    if residency_payload is not None:
        _write_json(run_dir / "residency.json", residency_payload)
    if online_dispatch_payload is not None:
        _write_json(run_dir / "online_dispatch.json", online_dispatch_payload)
    if request_cycle_payload is not None:
        _write_json(run_dir / "request_cycle_trace.json", request_cycle_payload)
    if global_memory_map_payload is not None:
        _write_json(run_dir / "global_memory_map.json", global_memory_map_payload)
    if prefill_coverage_payload is not None:
        _write_json(
            run_dir / "prefill_artifact_coverage.json", prefill_coverage_payload
        )
    trace_payload = (
        dispatcher.trace_bundle()
        if dispatcher
        else {
            "schema_version": "hetero-prefill-cycle-trace/v1",
            "trace_semantics": "bounded_value_range_sampling",
            "replay_safe": False,
            "qualification_record": None,
            "capture": {
                "status": "no_instruction_trace",
                "execution_mode": execution_mode,
                "cycle_artifact_coverage": prefill_coverage_payload,
            },
            "address_ranges": (
                global_memory_map_payload["ranges"]
                if global_memory_map_payload is not None
                else []
            ),
        }
        if execution_mode == "prefill_cycle"
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
            if execution_mode
            in {"analytical_preview", "operator_event", "full_runtime", "prefill_cycle"}
            else runtime_result["epochs"]
        )  # type: ignore[index]
        for event in event_records:
            stream.write(json.dumps(event, sort_keys=True) + "\n")
    return run_dir
