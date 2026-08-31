import json
from pathlib import Path

import pytest

from scripts.summarize_p16_full_task_timeline import (
    EXPECTED_RUNTIME_REQUESTS,
    GPU_OPERATORS,
    HOST_CONTROL_OPERATORS,
    P16QualificationError,
    RUNTIME_MEMORY_OPERATORS,
    summarize,
)


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _gpu_task(task_id: str, operator: str, start: int) -> dict[str, object]:
    return {
        "task_id": task_id,
        "op": operator,
        "resource_id": "gpu0",
        "dependencies": [],
        "duration_fs": 10,
        "timing": {"start_time_fs": start, "completion_time_fs": start + 10},
        "compiled_artifact": {
            "kind": "accel_sim_trace",
            "request_cycle_ready": True,
        },
        "backend_statistics": {
            "cycles": 1,
            "instructions": 1,
            "external_memory_stats": {
                "instances": 1,
                "gpu_parents": 1,
                "gpu_completed": 1,
                "gpu_children": 1,
                "children_sent": 1,
                "children_completed": 1,
                "durable_completed": 1,
                "address_translated": 1,
                "address_unmapped": 0,
                "atlas_parents": 0,
                "atlas_children": 0,
                "atlas_completed": 0,
                "outstanding": 0,
            },
        },
        "input_values": [],
        "validated_input_versions": [],
        "output_values": [{"value_id": f"value.{task_id}", "version": 1}],
    }


def _runtime_memory_task(task_id: str, operator: str, start: int) -> dict[str, object]:
    expected = EXPECTED_RUNTIME_REQUESTS[operator]
    request_count = expected["reads"] + expected["writes"]
    return {
        "task_id": task_id,
        "op": operator,
        "resource_id": "gpu0",
        "dependencies": [],
        "duration_fs": 10,
        "timing": {"start_time_fs": start, "completion_time_fs": start + 10},
        "compiled_artifact": {
            "kind": "runtime_live_ramulator2",
            "request_cycle_ready": True,
            "device_performance_included": True,
            "calibrated": False,
        },
        "fidelity": {
            "memory_fidelity": "cycle_simulated_external_ramulator2",
            "link_fidelity": "cycle_simulated_external_link",
            "performance_eligible": False,
        },
        "input_values": [],
        "validated_input_versions": [],
        "output_values": (
            [
                {"value_id": f"value.{task_id}.query", "version": 1},
                {"value_id": f"value.{task_id}.key", "version": 1},
                {"value_id": f"value.{task_id}.value", "version": 1},
            ]
            if operator == "kv_append"
            else []
        ),
        "backend_statistics": {},
    }


def _control_task(task_id: str, operator: str, start: int) -> dict[str, object]:
    return {
        "task_id": task_id,
        "op": operator,
        "resource_id": "gpu0",
        "dependencies": [],
        "duration_fs": 10,
        "timing": {"start_time_fs": start, "completion_time_fs": start + 10},
        "compiled_artifact": {
            "kind": "host_control_event",
            "causal_timeline_ready": True,
            "request_cycle_ready": False,
            "device_performance_included": False,
            "calibrated": False,
        },
        "fidelity": {
            "compute_fidelity": "host_control_boundary_uncalibrated",
            "device_performance_included": False,
            "memory_fidelity": "not_applicable",
            "performance_eligible": False,
        },
        "backend_statistics": {
            "live_memory_statistics": {
                "host_control_only": True,
                "instances": 0,
                "outstanding": 0,
            }
        },
        "input_values": [],
        "validated_input_versions": [],
        "output_values": [],
    }


def _run(root: Path) -> Path:
    operators = sorted(GPU_OPERATORS) + ["residual_add"]
    operators += sorted(RUNTIME_MEMORY_OPERATORS) + sorted(HOST_CONTROL_OPERATORS)
    tasks: list[dict[str, object]] = []
    commits: list[dict[str, object]] = []
    for index, operator in enumerate(operators):
        task_id = f"task.{index}.{operator}"
        start = index * 20
        if operator in GPU_OPERATORS:
            task = _gpu_task(task_id, operator, start)
        elif operator in RUNTIME_MEMORY_OPERATORS:
            task = _runtime_memory_task(task_id, operator, start)
            expected = EXPECTED_RUNTIME_REQUESTS[operator]
            request_count = expected["reads"] + expected["writes"]
            audit = {
                "duration_fs": 10,
                "contract_cycles": 1,
                "memory_statistics": {
                    "request_count": request_count,
                    "read_request_count": expected["reads"],
                    "write_request_count": expected["writes"],
                    "logical_bytes": expected["logical_bytes"],
                    "instances": 1,
                    "accepted_parent_ids": request_count,
                    "observed_completion_ids": request_count,
                    "completed": request_count,
                    "durable_completed": request_count,
                    "children_sent": request_count,
                    "children_completed": request_count,
                    "outstanding": 0,
                    "rejected": 0,
                    "gpu_cycles": 9,
                    "runtime_fixed_cycles": 1,
                    "initiators": {
                        "gpu0": {
                            "parents": request_count,
                            "children": request_count,
                            "completed": request_count,
                        },
                        "atlas0.compute": {"parents": 0, "children": 0, "completed": 0},
                    },
                },
                "requests": [
                    {"parent_id": parent_id, "size_bytes": 64}
                    for parent_id in range(1, request_count + 1)
                ],
            }
            task["backend_statistics"] = {
                "cycles": 10,
                "contract_cycles": 1,
                "clock_hz": 1,
                "live_memory_statistics": audit["memory_statistics"],
            }
            _write(
                root
                / "backend_runs"
                / "runtime"
                / task_id
                / "runtime_request_audit.json",
                audit,
            )
        else:
            task = _control_task(task_id, operator, start)
        tasks.append(task)
        for output in task["output_values"]:
            commits.append(
                {
                    "task_id": task_id,
                    "value_id": output["value_id"],
                    "version": 1,
                    "commit_time_fs": start + 10,
                    "cause": "backend_completion",
                }
            )

    runtime_bindings = []
    for operator in sorted(RUNTIME_MEMORY_OPERATORS):
        expected = EXPECTED_RUNTIME_REQUESTS[operator]
        runtime_bindings.append(
            {
                "operator": operator,
                "planned_parent_requests": expected["reads"] + expected["writes"],
                "planned_read_bytes": expected["reads"] * 64,
                "planned_write_bytes": expected["writes"] * 64,
                "request_cycle_ready": True,
                "host_control_excluded": False,
            }
        )
    runtime_bindings.extend(
        {
            "operator": operator,
            "planned_parent_requests": 0,
            "planned_read_bytes": 0,
            "planned_write_bytes": 0,
            "request_cycle_ready": False,
            "host_control_excluded": True,
        }
        for operator in sorted(HOST_CONTROL_OPERATORS)
    )
    makespan = int(tasks[-1]["timing"]["completion_time_fs"])
    _write(root / "execution_graph.json", {"tasks": tasks})
    _write(
        root / "online_dispatch.json",
        {
            "backend_dispatch_count": 20,
            "version_checks": 0,
            "version_commits": commits,
            "performance_boundary": {
                "causal_makespan_fs": makespan,
                "device_boundary_start_fs": 0,
                "device_boundary_end_fs": makespan,
                "device_boundary_span_fs": makespan,
                "included_task_count": 18,
                "excluded_task_count": 2,
                "excluded_control_duration_fs": 20,
            },
        },
    )
    _write(
        root / "global_memory_map.json",
        {
            "ranges": [
                {
                    "value_id": "external.shadow",
                    "base_address": 0,
                    "end_address_exclusive": 64,
                }
            ],
            "allocation_count": 1,
            "allocated_bytes": 64,
            "capacity_bytes": 4096,
            "external_input_shadow_count": 1,
            "operator_workspace_count": 0,
            "non_overlapping": True,
            "request_cycle_bindings": [{} for _ in range(15)],
            "runtime_task_bindings": runtime_bindings,
        },
    )
    _write(
        root / "metrics.json",
        {"makespan_fs": makespan, "performance_claim_allowed": False},
    )
    _write(
        root / "provenance.json",
        {"simulation_input_key": "same-key", "simulator_revision": "revision"},
    )
    return root


def test_p16_summarizer_accepts_exact_double_run(tmp_path: Path) -> None:
    summary = summarize(_run(tmp_path / "leg1"), _run(tmp_path / "leg2"))
    assert summary["status"] == "passed"
    assert summary["double_run_deterministic"] is True
    assert summary["runtime_totals"]["requests_per_leg"] == 517
    assert summary["legs"][0]["task_count"] == 20
    assert summary["legs"][0]["version_causality"]["completion_time_commits"] == 18


def test_p16_summarizer_rejects_runtime_request_leak(tmp_path: Path) -> None:
    first = _run(tmp_path / "leg1")
    second = _run(tmp_path / "leg2")
    audit_path = next(
        (second / "backend_runs" / "runtime").glob("*/runtime_request_audit.json")
    )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["memory_statistics"]["outstanding"] = 1
    _write(audit_path, audit)
    with pytest.raises(
        P16QualificationError,
        match="audit and execution stats differ|request conservation",
    ):
        summarize(first, second)
