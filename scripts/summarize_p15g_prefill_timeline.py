#!/usr/bin/env python3
"""Validate the two request-cycle-ready GPU operators in one Prefill timeline."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path


class P15gQualificationError(RuntimeError):
    """Raised when dependency, address, request or version causality is broken."""


READY_OPERATORS = {"attention_norm", "qkv_projection"}


def _load(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise P15gQualificationError(f"{path} must contain a JSON object")
    return payload


def _external_stats(task: Mapping[str, object]) -> Mapping[str, object]:
    backend = task.get("backend_statistics")
    if not isinstance(backend, Mapping):
        raise P15gQualificationError(f"{task.get('task_id')} lacks Backend statistics")
    external = backend.get("external_memory_stats")
    if not isinstance(external, Mapping):
        raise P15gQualificationError(
            f"{task.get('task_id')} lacks external-memory stats"
        )
    return external


def summarize(run_dir: Path) -> dict[str, object]:
    execution = _load(run_dir / "execution_graph.json")
    online = _load(run_dir / "online_dispatch.json")
    memory_map = _load(run_dir / "global_memory_map.json")
    provenance = _load(run_dir / "provenance.json")
    raw_tasks = execution.get("tasks")
    if not isinstance(raw_tasks, list):
        raise P15gQualificationError("execution graph tasks must be an array")
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
        raise P15gQualificationError(
            f"expected exactly {sorted(READY_OPERATORS)}, got {sorted(ready)}"
        )
    attention = ready["attention_norm"]
    qkv = ready["qkv_projection"]
    attention_timing = attention.get("timing")
    qkv_timing = qkv.get("timing")
    if not isinstance(attention_timing, Mapping) or not isinstance(qkv_timing, Mapping):
        raise P15gQualificationError("ready tasks lack global timeline timestamps")
    attention_end = int(attention_timing["completion_time_fs"])
    qkv_start = int(qkv_timing["start_time_fs"])
    if str(attention["task_id"]) not in qkv.get("dependencies", []):
        raise P15gQualificationError("QKV does not depend on Attention Norm")
    if qkv_start < attention_end:
        raise P15gQualificationError("QKV starts before Attention Norm completes")
    if attention.get("resource_id") != "gpu0" or qkv.get("resource_id") != "gpu0":
        raise P15gQualificationError("both ready operators must own gpu0")

    binding_records = memory_map.get("request_cycle_bindings")
    if not isinstance(binding_records, list) or len(binding_records) != 2:
        raise P15gQualificationError("Global PA map must contain two request bindings")
    tensor_pa: dict[str, tuple[str, int]] = {}
    for record in binding_records:
        if not isinstance(record, Mapping):
            raise P15gQualificationError("request binding record must be an object")
        semantic = record.get("semantic_bindings")
        if not isinstance(semantic, list):
            raise P15gQualificationError("semantic bindings must be an array")
        for item in semantic:
            if isinstance(item, Mapping):
                tensor_pa[str(item["tensor_id"])] = (
                    str(item["value_id"]),
                    int(item["global_pa_base"]),
                )
    norm_output = tensor_pa.get("tinyllama.layer0.attention_norm.output")
    qkv_input = tensor_pa.get("tinyllama.layer0.qkv_projection.input")
    if norm_output is None or norm_output != qkv_input:
        raise P15gQualificationError(
            "Attention Norm output and QKV input do not share Value/Global PA"
        )

    request_checks: dict[str, dict[str, int]] = {}
    for operator, task in ready.items():
        stats = _external_stats(task)
        required_zero = (
            "outstanding",
            "address_unmapped",
            "atlas_parents",
            "atlas_children",
        )
        if any(int(stats.get(key, -1)) != 0 for key in required_zero):
            raise P15gQualificationError(
                f"{operator} has unmapped, ATLAS or in-flight work"
            )
        parents = int(stats.get("gpu_parents", 0))
        completed = int(stats.get("gpu_completed", -1))
        children = int(stats.get("gpu_children", 0))
        children_sent = int(stats.get("children_sent", -1))
        children_completed = int(stats.get("children_completed", -1))
        durable = int(stats.get("durable_completed", -1))
        if int(stats.get("instances", 0)) != 1 or parents <= 0 or children <= 0:
            raise P15gQualificationError(
                f"{operator} lacks one live non-empty Ramulator2"
            )
        if (
            completed != parents
            or children_sent != children
            or children_completed != children
        ):
            raise P15gQualificationError(f"{operator} breaks parent/child conservation")
        if durable != parents or int(stats.get("address_translated", 0)) <= 0:
            raise P15gQualificationError(
                f"{operator} lacks durable translated completion"
            )
        request_checks[operator] = {
            "parents": parents,
            "children": children,
            "translated": int(stats["address_translated"]),
            "outstanding": int(stats["outstanding"]),
        }

    commits = online.get("version_commits")
    if not isinstance(commits, list):
        raise P15gQualificationError("online dispatch lacks version commit log")
    commits_by_task: dict[str, list[Mapping[str, object]]] = {}
    for item in commits:
        if isinstance(item, Mapping):
            commits_by_task.setdefault(str(item["task_id"]), []).append(item)
    for task in (attention, qkv):
        task_id = str(task["task_id"])
        task_commits = commits_by_task.get(task_id, [])
        if not task_commits:
            raise P15gQualificationError(f"{task_id} has no output version commit")
        completion = int(task["timing"]["completion_time_fs"])
        if any(
            int(item["commit_time_fs"]) != completion
            or item.get("cause") != "backend_completion"
            for item in task_commits
        ):
            raise P15gQualificationError(
                f"{task_id} commits outside Backend completion"
            )
    qkv_validated = qkv.get("validated_input_versions")
    if not isinstance(qkv_validated, list) or not any(
        isinstance(item, Mapping)
        and item.get("value_id") == norm_output[0]
        and int(item.get("version", -1)) == 1
        for item in qkv_validated
    ):
        raise P15gQualificationError("QKV did not validate Attention Norm output v1")

    return {
        "schema_version": "hetero-p15g-prefill-timeline-qualification/v1",
        "status": "passed",
        "performance_eligible": False,
        "claim_boundary": (
            "two request-cycle-ready operators share one logical Prefill timeline; "
            "remaining operators are analytical fallbacks"
        ),
        "run_dir": str(run_dir.resolve()),
        "simulator_revision": provenance.get("simulator_revision"),
        "dependency_causality": {
            "attention_completion_fs": attention_end,
            "qkv_start_fs": qkv_start,
            "qkv_after_attention": True,
        },
        "resource_causality": {
            "resource_id": "gpu0",
            "non_overlapping": qkv_start >= attention_end,
        },
        "global_pa_causality": {
            "value_id": norm_output[0],
            "global_pa_base": norm_output[1],
            "same_output_input_binding": True,
        },
        "request_completion": request_checks,
        "version_causality": {
            "commit_count": len(commits),
            "ready_operator_commits_at_completion": True,
            "qkv_validated_attention_output_v1": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = summarize(args.run_dir)
    output = args.output or args.run_dir / "p15g_qualification.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()
