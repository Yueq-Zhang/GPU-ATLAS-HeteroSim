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
from .memory_system import (
    CanonicalRange,
    ResidencyManager,
    build_dynamic_kv_lifecycle,
    run_reference_coupled_dag,
)
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
    model: object,
    backends: Mapping[str, object],
    links: Mapping[str, object],
    execution_mode: str,
    dispatcher: OperatorEventDispatcher | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    timed_execution = execution_mode in {
        "analytical_preview",
        "operator_event",
        "full_runtime",
    }
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
        previous_value_id: str | None = None
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
                    "value_id": previous_value_id or f"{request.request_id}.control",
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
                "read_values": list(node.read_values),
                "write_values": list(node.write_values),
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
                        if previous_task is None
                        else 0,
                        "duration_fs": duration_fs,
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
            previous_value_id = (
                str(node.write_values[-1]) if node.write_values else previous_value_id
            )
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
    batch_plan: dict[str, object] | None = None
    memory_lifecycle: dict[str, object] | None = None
    link_statistics: dict[str, object] | None = None
    memory_statistics: dict[str, object] | None = None
    residency_payload: dict[str, object] | None = None
    shared_config: Mapping[str, object] | None = None
    shared_reference_active = False
    if execution_mode == "full_runtime":
        memory_services = system.get("memory_services", {})
        if not isinstance(memory_services, Mapping):
            raise ValueError("system.memory_services must be an object")
        candidate = memory_services.get(memory_space_id)
        if isinstance(candidate, Mapping):
            shared_config = candidate
            shared_reference_active = candidate.get("kind") == "shared_3d_reference"
            if candidate.get("kind") == "ramulator2":
                raise ValueError(
                    "full_runtime live Ramulator2 coupling is not qualified; "
                    "use 'qualify-memory' for standalone replay or configure "
                    "kind=shared_3d_reference"
                )
    competition_summary = _validate_gpu_only_shared_3d_baseline(
        execution_graph, shared_config
    )
    if execution_mode in {"analytical_preview", "operator_event", "full_runtime"}:
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
            residency = ResidencyManager()
            registered: set[str] = set()
            for route in execution_graph["routes"]:  # type: ignore[assignment]
                value_id = str(route["value_id"])
                if value_id not in registered:
                    residency.register(
                        value_id,
                        str(route["source_space"]),
                        str(route["producer_device"]),
                        1,
                    )
                    registered.add(value_id)
                route_kind = (
                    route["kind"].value
                    if hasattr(route["kind"], "value")
                    else str(route["kind"])
                )
                action = {
                    "transfer": "copy",
                    "migration": "migrate",
                    "remote_access": "remote",
                    "synchronization": "synchronize",
                    "local_dependency": "synchronize",
                }[route_kind]
                route_timing = timing_by_id[str(route["task_id"])]
                residency.transition(
                    CanonicalRange(
                        value_id,
                        0,
                        0,
                        int(route.get("payload_bytes", 1)),
                    ),
                    str(route["destination_space"]),
                    str(route["consumer_device"]),
                    action,
                    int(route_timing["completion_time_fs"]),
                )
            residency_payload = {
                "schema_version": "hetero-residency/v1",
                "coherence": "explicit_noncoherent",
                "records": residency.snapshot(),
                "events": residency.events,
            }
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
            "runtime_owner": "cpp.GlobalEventRuntime"
            if execution_mode in {"analytical_preview", "operator_event", "full_runtime"}
            else "cpp.TokenBarrierScheduler",
            "address_owner": (
                "cpp.RuntimeMemoryPlanner"
                if execution_mode == "full_runtime"
                else "cpp.PagedKvAllocator"
            ),
            "memory_timing_owner": (
                "shared3d.memory_service"
                if execution_mode == "full_runtime"
                and profile == "model3_gpu_native_3ddram"
                and isinstance(shared_config, Mapping)
                and shared_config.get("kind") == "shared_3d_reference"
                else None
            ),
            "backend_dispatch": dispatcher.provenance() if dispatcher else None,
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
            if execution_mode in {"analytical_preview", "operator_event", "full_runtime"}
            else runtime_result["epochs"]
        )  # type: ignore[index]
        for event in event_records:
            stream.write(json.dumps(event, sort_keys=True) + "\n")
    return run_dir
