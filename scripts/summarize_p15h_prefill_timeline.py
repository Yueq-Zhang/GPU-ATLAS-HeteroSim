#!/usr/bin/env python3
"""Validate all twelve request-cycle-ready GPU operators in one Prefill timeline."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path


class P15hQualificationError(RuntimeError):
    """Raised when timeline, address, request, or version causality is broken."""


READY_OPERATORS = {
    "attention_norm",
    "qkv_projection",
    "rope",
    "causal_attention",
    "output_projection",
    "mlp_norm",
    "gate_up_projection",
    "silu_multiply",
    "down_projection",
    "final_norm",
    "lm_head",
    "sampling",
}


def _load(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise P15hQualificationError(f"{path} must contain a JSON object")
    return payload


def _external_stats(task: Mapping[str, object]) -> Mapping[str, object]:
    backend = task.get("backend_statistics")
    if not isinstance(backend, Mapping):
        raise P15hQualificationError(f"{task.get('task_id')} lacks Backend stats")
    external = backend.get("external_memory_stats")
    if not isinstance(external, Mapping):
        raise P15hQualificationError(
            f"{task.get('task_id')} lacks external-memory stats"
        )
    return external


def _timing(task: Mapping[str, object]) -> Mapping[str, object]:
    timing = task.get("timing")
    if not isinstance(timing, Mapping):
        raise P15hQualificationError(f"{task.get('task_id')} lacks timing")
    return timing


def summarize(run_dir: Path) -> dict[str, object]:
    execution = _load(run_dir / "execution_graph.json")
    online = _load(run_dir / "online_dispatch.json")
    memory_map = _load(run_dir / "global_memory_map.json")
    provenance = _load(run_dir / "provenance.json")
    raw_tasks = execution.get("tasks")
    if not isinstance(raw_tasks, list):
        raise P15hQualificationError("execution graph tasks must be an array")
    tasks = {
        str(item["task_id"]): item for item in raw_tasks if isinstance(item, Mapping)
    }
    ready = {
        str(item["op"]): item
        for item in tasks.values()
        if isinstance(item.get("compiled_artifact"), Mapping)
        and item["compiled_artifact"].get("request_cycle_ready") is True
    }
    if set(ready) != READY_OPERATORS:
        raise P15hQualificationError(
            f"expected exactly {sorted(READY_OPERATORS)}, got {sorted(ready)}"
        )

    for task in tasks.values():
        start = int(_timing(task)["start_time_fs"])
        for dependency_id in task.get("dependencies", []):
            dependency = tasks.get(str(dependency_id))
            if dependency is None:
                raise P15hQualificationError(
                    f"{task['task_id']} has unknown dependency {dependency_id}"
                )
            if start < int(_timing(dependency)["completion_time_fs"]):
                raise P15hQualificationError(
                    f"{task['task_id']} starts before {dependency_id} completes"
                )

    gpu_tasks = sorted(
        (item for item in tasks.values() if item.get("resource_id") == "gpu0"),
        key=lambda item: int(_timing(item)["start_time_fs"]),
    )
    for left, right in zip(gpu_tasks, gpu_tasks[1:]):
        if int(_timing(right)["start_time_fs"]) < int(
            _timing(left)["completion_time_fs"]
        ):
            raise P15hQualificationError(
                f"gpu0 overlap: {left['task_id']} and {right['task_id']}"
            )
    if any(item.get("resource_id") != "gpu0" for item in ready.values()):
        raise P15hQualificationError("all twelve ready operators must own gpu0")

    ranges = memory_map.get("ranges")
    if not isinstance(ranges, list):
        raise P15hQualificationError("Global PA ranges must be an array")
    ordered_ranges = sorted(
        (item for item in ranges if isinstance(item, Mapping)),
        key=lambda item: int(item["base_address"]),
    )
    for left, right in zip(ordered_ranges, ordered_ranges[1:]):
        if int(left["end_address_exclusive"]) > int(right["base_address"]):
            raise P15hQualificationError(
                f"Global PA overlap: {left['value_id']} and {right['value_id']}"
            )
    allocation_base = {
        str(item["value_id"]): int(item["base_address"]) for item in ordered_ranges
    }
    binding_records = memory_map.get("request_cycle_bindings")
    if not isinstance(binding_records, list) or len(binding_records) != 12:
        raise P15hQualificationError("Global PA map must contain twelve bindings")
    semantic_count = 0
    for record in binding_records:
        if not isinstance(record, Mapping):
            raise P15hQualificationError("request binding must be an object")
        semantic = record.get("semantic_bindings")
        if not isinstance(semantic, list) or not semantic:
            raise P15hQualificationError("request binding lacks semantic tensors")
        for item in semantic:
            if not isinstance(item, Mapping):
                raise P15hQualificationError("semantic binding must be an object")
            value_id = str(item["value_id"])
            expected = allocation_base.get(value_id)
            actual = int(item["global_pa_base"]) - int(item["value_offset_bytes"])
            if expected is None or actual != expected:
                raise P15hQualificationError(
                    f"{item['tensor_id']} is not based on its graph Value Global PA"
                )
            semantic_count += 1

    request_checks: dict[str, dict[str, int]] = {}
    for operator, task in ready.items():
        stats = _external_stats(task)
        required_zero = (
            "outstanding",
            "address_unmapped",
            "atlas_parents",
            "atlas_children",
            "atlas_completed",
        )
        if any(int(stats.get(key, -1)) != 0 for key in required_zero):
            raise P15hQualificationError(
                f"{operator} has unmapped, ATLAS, or in-flight work"
            )
        parents = int(stats.get("gpu_parents", 0))
        completed = int(stats.get("gpu_completed", -1))
        children = int(stats.get("gpu_children", 0))
        sent = int(stats.get("children_sent", -1))
        child_completed = int(stats.get("children_completed", -1))
        durable = int(stats.get("durable_completed", -1))
        if int(stats.get("instances", 0)) != 1 or parents <= 0 or children <= 0:
            raise P15hQualificationError(f"{operator} lacks one live Ramulator2")
        if completed != parents or sent != children or child_completed != children:
            raise P15hQualificationError(f"{operator} breaks request conservation")
        if durable != parents or int(stats.get("address_translated", 0)) <= 0:
            raise P15hQualificationError(
                f"{operator} lacks durable translated completion"
            )
        request_checks[operator] = {
            "cycles": int(task["backend_statistics"].get("cycles", 0)),
            "parents": parents,
            "children": children,
            "translated": int(stats["address_translated"]),
            "outstanding": int(stats["outstanding"]),
        }

    commits = online.get("version_commits")
    if not isinstance(commits, list):
        raise P15hQualificationError("online dispatch lacks version commits")
    commits_by_task: dict[str, list[Mapping[str, object]]] = {}
    for item in commits:
        if isinstance(item, Mapping):
            commits_by_task.setdefault(str(item["task_id"]), []).append(item)
    for task in ready.values():
        task_id = str(task["task_id"])
        task_commits = commits_by_task.get(task_id, [])
        if not task_commits:
            raise P15hQualificationError(f"{task_id} has no output commit")
        completion = int(_timing(task)["completion_time_fs"])
        if any(
            int(item["commit_time_fs"]) != completion
            or item.get("cause") != "backend_completion"
            for item in task_commits
        ):
            raise P15hQualificationError(
                f"{task_id} commits outside Backend completion"
            )
        expected_inputs = {
            (str(item["value_id"]), int(item["version"]))
            for item in task.get("input_values", [])
            if isinstance(item, Mapping)
        }
        validated = task.get("validated_input_versions")
        if not isinstance(validated, list):
            raise P15hQualificationError(f"{task_id} lacks input version checks")
        actual_inputs = {
            (str(item["value_id"]), int(item["version"]))
            for item in validated
            if isinstance(item, Mapping)
        }
        if actual_inputs != expected_inputs:
            raise P15hQualificationError(
                f"{task_id} validated input versions do not match graph Values"
            )

    ready_timeline = [
        {
            "operator": operator,
            "task_id": str(task["task_id"]),
            "start_time_fs": int(_timing(task)["start_time_fs"]),
            "completion_time_fs": int(_timing(task)["completion_time_fs"]),
        }
        for operator, task in sorted(
            ready.items(), key=lambda pair: int(_timing(pair[1])["start_time_fs"])
        )
    ]
    return {
        "schema_version": "hetero-p15h-prefill-timeline-qualification/v1",
        "status": "passed",
        "performance_eligible": False,
        "claim_boundary": (
            "twelve real request-cycle GPU operators share one logical Prefill "
            "timeline; runtime/control operators remain analytical"
        ),
        "run_dir": str(run_dir.resolve()),
        "simulator_revision": provenance.get("simulator_revision"),
        "ready_operator_count": len(ready),
        "ready_timeline": ready_timeline,
        "dependency_causality": {"all_dependencies_complete_before_start": True},
        "resource_causality": {
            "resource_id": "gpu0",
            "all_gpu_tasks_non_overlapping": True,
        },
        "global_pa_causality": {
            "range_count": len(ordered_ranges),
            "request_binding_count": len(binding_records),
            "semantic_binding_count": semantic_count,
            "all_semantic_bindings_derive_from_graph_value_pa": True,
        },
        "request_completion": request_checks,
        "version_causality": {
            "commit_count": len(commits),
            "ready_operator_commits_at_completion": True,
            "ready_operator_inputs_validated": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = summarize(args.run_dir)
    output = args.output or args.run_dir / "p15h_qualification.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()
