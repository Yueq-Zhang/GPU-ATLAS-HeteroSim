"""P24 lifecycle controls driven by the sealed P23 request-cycle timeline.

The P23 trace bundle is an exact fused BS=2, KV=17, one-layer Decode
execution.  This module may replay that bundle for another pair of requests
only when the full execution identity is unchanged.  It deliberately refuses
to extrapolate the trace to a different KV length or a partial batch.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


class P23RequestControlError(RuntimeError):
    """Raised when lifecycle work cannot be backed by exact P23 evidence."""


def _load(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise P23RequestControlError(f"{path} must contain a JSON object")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class P23ServiceContract:
    qualification_path: Path
    execution_graph_path: Path
    memory_map_path: Path
    duration_fs: int
    batch_size: int
    initial_kv_length: int
    final_kv_length: int
    task_count: int
    gpu_task_count: int
    runtime_task_count: int
    parent_requests: int
    child_requests: int
    source_digest: str
    task_templates: tuple[Mapping[str, object], ...]


def load_p23_service_contract(
    project_root: Path,
    qualification_path: Path | None = None,
) -> P23ServiceContract:
    """Load and revalidate the repository-sealed P23 execution evidence."""

    root = project_root.resolve()
    qualification = (
        qualification_path.resolve()
        if qualification_path is not None
        else root / "validation/p23/timeline/qualification_record.json"
    )
    record = _load(qualification)
    if (
        record.get("schema_version")
        != "hetero-p23-bs2-decode-timeline-qualification/v1"
        or record.get("status") != "passed"
        or record.get("qualification_passed") is not True
        or record.get("double_run_signature_equal") is not True
    ):
        raise P23RequestControlError("P23 timeline qualification is not sealed")
    scope = record.get("scope")
    legs = record.get("legs")
    if not isinstance(scope, Mapping) or not isinstance(legs, list) or len(legs) != 2:
        raise P23RequestControlError("P23 qualification scope or double run is incomplete")
    if (
        int(scope.get("batch_size", 0)) != 2
        or int(scope.get("context_length", -1)) != 16
        or int(scope.get("kv_length", -1)) != 17
        or int(scope.get("decode_steps", 0)) != 1
    ):
        raise P23RequestControlError("P23 qualification identity is not BS2/KV17")
    first = dict(legs[0])
    second = dict(legs[1])
    if (
        int(first.get("makespan_fs", -1)) <= 0
        or first.get("makespan_fs") != second.get("makespan_fs")
        or first.get("gpu_tasks") != second.get("gpu_tasks")
        or first.get("runtime_tasks") != second.get("runtime_tasks")
    ):
        raise P23RequestControlError("P23 double-run evidence is inconsistent")

    leg_root = qualification.parent / "leg1"
    execution_path = leg_root / "execution_graph.json"
    memory_map_path = leg_root / "global_memory_map.json"
    execution = _load(execution_path)
    memory_map = _load(memory_map_path)
    tasks = execution.get("tasks")
    slices = memory_map.get("batch_member_kv_slices")
    if not isinstance(tasks, list) or len(tasks) != 20:
        raise P23RequestControlError("P23 execution graph must contain 20 tasks")
    if not isinstance(slices, list) or len(slices) != 4:
        raise P23RequestControlError("P23 per-member KV slices are incomplete")
    parents = 0
    children = 0
    for item in [*first["gpu_tasks"], *first["runtime_tasks"]]:  # type: ignore[index]
        task = dict(item)
        parents += int(task.get("parents", task.get("requests", 0)))
        children += int(task.get("children", task.get("requests", 0)))
        checks = task.get("checks")
        if isinstance(checks, Mapping) and checks.get("zero_outstanding") is not True:
            raise P23RequestControlError("P23 task does not end durable and drained")
    source_digest = hashlib.sha256(
        (record["schema_version"] + _sha256(qualification) + _sha256(execution_path)).encode(
            "utf-8"
        )
    ).hexdigest()
    return P23ServiceContract(
        qualification_path=qualification,
        execution_graph_path=execution_path,
        memory_map_path=memory_map_path,
        duration_fs=int(first["makespan_fs"]),
        batch_size=2,
        initial_kv_length=16,
        final_kv_length=17,
        task_count=len(tasks),
        gpu_task_count=len(first["gpu_tasks"]),  # type: ignore[arg-type]
        runtime_task_count=len(first["runtime_tasks"]),  # type: ignore[arg-type]
        parent_requests=parents,
        child_requests=children,
        source_digest=source_digest,
        task_templates=tuple(dict(item) for item in tasks),
    )


class _FirstFitAllocator:
    def __init__(self, capacity_bytes: int, alignment_bytes: int, base_address: int):
        if capacity_bytes <= 0 or alignment_bytes <= 0 or base_address < 0:
            raise P23RequestControlError("invalid KV allocator geometry")
        self.capacity = capacity_bytes
        self.alignment = alignment_bytes
        self.base = base_address
        self.active: dict[str, tuple[int, int, int]] = {}
        self.released: list[tuple[int, int]] = []
        self.next_epoch = 1
        self.peak = 0

    def _align(self, value: int) -> int:
        return ((value + self.alignment - 1) // self.alignment) * self.alignment

    def allocate(self, request_id: str, size_bytes: int) -> tuple[int, int]:
        if request_id in self.active or size_bytes <= 0:
            raise P23RequestControlError("invalid or duplicate KV allocation")
        ranges = sorted((offset, size) for offset, size, _ in self.active.values())
        candidate = 0
        for offset, size in ranges:
            candidate = self._align(candidate)
            if candidate + size_bytes <= offset:
                break
            candidate = max(candidate, offset + size)
        candidate = self._align(candidate)
        if candidate + size_bytes > self.capacity:
            raise P23RequestControlError("insufficient exact-batch KV capacity")
        epoch = self.next_epoch
        self.next_epoch += 1
        self.active[request_id] = (candidate, size_bytes, epoch)
        self.peak = max(
            self.peak,
            sum(size for _, size, _ in self.active.values()),
        )
        return self.base + candidate, epoch

    def release(self, request_id: str) -> tuple[int, int, int]:
        record = self.active.pop(request_id, None)
        if record is None:
            raise P23RequestControlError("release without active KV allocation")
        self.released.append((record[0], record[1]))
        return record


def run_kv_allocator_pressure_probe(
    *,
    epochs: int,
    batch_size: int,
    per_request_bytes: int,
    capacity_bytes: int,
    alignment_bytes: int = 256,
    global_pa_base: int = 0x80000000,
) -> dict[str, object]:
    """Exercise long-lived KV allocation/release without inventing GPU cycles.

    This is a lifecycle and capacity-pressure probe only.  It proves that a
    long stream can reuse retired ranges without aliasing or leaking; it does
    not replace the missing exact KV18+ instruction traces for one request.
    """

    if epochs <= 0 or batch_size <= 0:
        raise P23RequestControlError("positive pressure-probe geometry is required")
    allocator = _FirstFitAllocator(capacity_bytes, alignment_bytes, global_pa_base)
    expected_bases: tuple[int, ...] | None = None
    overlaps = 0
    allocations = 0
    releases = 0
    durable = 0
    reuse = 0
    for epoch in range(epochs):
        active_ranges: list[tuple[int, int]] = []
        bases: list[int] = []
        request_ids: list[str] = []
        for member in range(batch_size):
            request_id = f"pressure.e{epoch}.m{member}"
            base, _ = allocator.allocate(request_id, per_request_bytes)
            overlaps += sum(
                int(base < other + size and other < base + per_request_bytes)
                for other, size in active_ranges
            )
            active_ranges.append((base, per_request_bytes))
            bases.append(base)
            request_ids.append(request_id)
            allocations += 1
            if epoch > 0 and expected_bases is not None and base in expected_bases:
                reuse += 1
        if expected_bases is None:
            expected_bases = tuple(bases)
        elif tuple(bases) != expected_bases:
            raise P23RequestControlError("retired Global PA reuse is not deterministic")
        durable += batch_size
        for request_id in request_ids:
            allocator.release(request_id)
            releases += 1
    if allocator.active or overlaps or allocations != releases or releases != durable:
        raise P23RequestControlError("long KV allocator pressure conservation failed")
    return {
        "schema_version": "hetero-kv-allocator-pressure-probe/v1",
        "timing_semantics": "functional_lifecycle_only",
        "epochs": epochs,
        "batch_size": batch_size,
        "allocations": allocations,
        "durable_barriers": durable,
        "releases": releases,
        "retired_range_reuses": reuse,
        "peak_bytes": allocator.peak,
        "capacity_bytes": allocator.capacity,
        "overlap_count": overlaps,
        "leaked_allocations": len(allocator.active),
        "zero_in_flight": True,
        "exact_gpu_trace_long_generation": False,
        "performance_claim_allowed": False,
    }


def _request_limit(request: Mapping[str, object]) -> int:
    values = [int(request.get("output_length", 1))]
    for key in ("eos_after_generated_tokens", "max_output_tokens"):
        value = int(request.get(key, 0))
        if value > 0:
            values.append(value)
    return min(values)


def run_p23_request_control_timeline(
    project_root: Path,
    requests: Sequence[Mapping[str, object]],
    allocator_config: Mapping[str, object],
    qualification_path: Path | None = None,
) -> dict[str, object]:
    """Replay exact P23 epochs while enforcing P24 barrier and PA causality.

    Every admitted request must have exactly one KV16->17 Decode step.  A
    longer request is rejected until an exact artifact exists for every later
    KV length.  Multiple independent request pairs may replay the same sealed
    identity at newly allocated Global-PA ranges.
    """

    contract = load_p23_service_contract(project_root, qualification_path)
    if not requests:
        raise P23RequestControlError("at least one request is required")
    normalized: list[dict[str, object]] = []
    identifiers: set[str] = set()
    for raw in requests:
        request = dict(raw)
        request_id = str(request.get("request_id", ""))
        if not request_id or request_id in identifiers:
            raise P23RequestControlError("request IDs must be non-empty and unique")
        identifiers.add(request_id)
        if int(request.get("initial_kv_length", -1)) != contract.initial_kv_length:
            raise P23RequestControlError(
                f"{request_id} lacks an exact KV{contract.initial_kv_length + 1} P23 trace"
            )
        if _request_limit(request) != 1:
            raise P23RequestControlError(
                f"{request_id} requires KV18+ exact traces; KV17 must not be extrapolated"
            )
        arrival = int(request.get("arrival_time_fs", 0))
        cancel = request.get("cancel_time_fs")
        if arrival < 0 or (cancel is not None and int(cancel) < arrival):
            raise P23RequestControlError("invalid arrival or cancellation time")
        request["arrival_time_fs"] = arrival
        request["cancel_time_fs"] = None if cancel is None else int(cancel)
        normalized.append(request)

    per_member_kv_bytes = int(allocator_config.get("per_member_kv_bytes", 17408))
    allocator = _FirstFitAllocator(
        int(allocator_config["capacity_bytes"]),
        int(allocator_config.get("alignment_bytes", 256)),
        int(allocator_config.get("global_pa_base", 0x80000000)),
    )
    waiting = sorted(
        normalized,
        key=lambda item: (int(item["arrival_time_fs"]), str(item["request_id"])),
    )
    barrier = 0
    epoch_id = 0
    events: list[dict[str, object]] = []
    epochs: list[dict[str, object]] = []
    records: dict[str, dict[str, object]] = {}
    allocation_history: list[dict[str, object]] = []
    released_ranges: set[tuple[int, int]] = set()
    reuse_count = 0

    for request in waiting:
        events.append(
            {
                "sequence": len(events),
                "time_fs": int(request["arrival_time_fs"]),
                "request_id": str(request["request_id"]),
                "event": "ARRIVAL",
            }
        )

    pending = list(waiting)
    while pending:
        arrived = [item for item in pending if int(item["arrival_time_fs"]) <= barrier]
        if not arrived:
            barrier = min(int(item["arrival_time_fs"]) for item in pending)
            continue
        for request in list(arrived):
            cancel = request["cancel_time_fs"]
            if cancel is not None and int(cancel) <= barrier:
                request_id = str(request["request_id"])
                records[request_id] = {
                    "request_id": request_id,
                    "final_state": "CANCELLED",
                    "admitted": False,
                    "generated_length": 0,
                    "initial_kv_length": contract.initial_kv_length,
                    "final_committed_kv_length": contract.initial_kv_length,
                    "cancel_observed_time_fs": barrier,
                    "retire_time_fs": barrier,
                }
                events.append(
                    {
                        "sequence": len(events),
                        "time_fs": barrier,
                        "request_id": request_id,
                        "event": "CANCEL_AT_TOKEN_BARRIER",
                    }
                )
                pending.remove(request)
                arrived.remove(request)
        if not arrived:
            continue
        selected = arrived[: contract.batch_size]
        if len(selected) != contract.batch_size:
            future_arrivals = [
                int(item["arrival_time_fs"])
                for item in pending
                if int(item["arrival_time_fs"]) > barrier
            ]
            if future_arrivals:
                barrier = min(future_arrivals)
                continue
            raise P23RequestControlError(
                "exact P23 replay requires a complete fused batch of two requests"
            )

        epoch_start = barrier
        epoch_end = epoch_start + contract.duration_fs
        members: list[dict[str, object]] = []
        for member_index, request in enumerate(selected):
            request_id = str(request["request_id"])
            base, allocation_epoch = allocator.allocate(request_id, per_member_kv_bytes)
            reused = (base - allocator.base, per_member_kv_bytes) in released_ranges
            reuse_count += int(reused)
            binding = {
                "request_id": request_id,
                "member_index": member_index,
                "allocation_epoch": allocation_epoch,
                "global_pa_base": base,
                "global_pa_end_exclusive": base + per_member_kv_bytes,
                "size_bytes": per_member_kv_bytes,
                "k_base": base,
                "v_base": base + per_member_kv_bytes // 2,
                "reused_released_range": reused,
            }
            allocation_history.append({"operation": "allocate", "time_fs": epoch_start, **binding})
            members.append(binding)
            events.append(
                {
                    "sequence": len(events),
                    "time_fs": epoch_start,
                    "request_id": request_id,
                    "event": "ADMIT_AND_ALLOCATE",
                    "epoch_id": epoch_id,
                    "allocation_epoch": allocation_epoch,
                    "global_pa_base": base,
                }
            )

        shifted_tasks: list[dict[str, object]] = []
        previous_completion = epoch_start
        for template in contract.task_templates:
            timing = dict(template["timing"])  # type: ignore[index]
            start = epoch_start + int(timing["start_time_fs"])
            completion = epoch_start + int(timing["completion_time_fs"])
            if start < previous_completion:
                raise P23RequestControlError("P23 shifted task dependency regressed")
            previous_completion = completion
            shifted_tasks.append(
                {
                    "task_id": f"epoch{epoch_id}.{template['task_id']}",
                    "op": template["op"],
                    "start_time_fs": start,
                    "completion_time_fs": completion,
                    "source_task_id": template["task_id"],
                }
            )

        for request, binding in zip(selected, members, strict=True):
            request_id = str(request["request_id"])
            cancel = request["cancel_time_fs"]
            cancelled = cancel is not None and int(cancel) <= epoch_end
            termination = "cancelled" if cancelled else "completed"
            records[request_id] = {
                "request_id": request_id,
                "final_state": "CANCELLED" if cancelled else "FINISHED",
                "admitted": True,
                "epoch_id": epoch_id,
                "generated_length": 1,
                "initial_kv_length": contract.initial_kv_length,
                "final_committed_kv_length": contract.final_kv_length,
                "kv_version": 1,
                "durable_complete_time_fs": epoch_end,
                "cancel_observed_time_fs": epoch_end if cancelled else None,
                "termination_reason": termination,
                "retire_time_fs": epoch_end,
                "allocation": binding,
            }
            events.extend(
                [
                    {
                        "sequence": len(events),
                        "time_fs": epoch_end,
                        "request_id": request_id,
                        "event": "ALL_REQUESTS_DURABLE",
                        "epoch_id": epoch_id,
                    },
                    {
                        "sequence": len(events) + 1,
                        "time_fs": epoch_end,
                        "request_id": request_id,
                        "event": "COMMIT_KV_VERSION",
                        "epoch_id": epoch_id,
                        "kv_version": 1,
                        "committed_kv_length": contract.final_kv_length,
                    },
                    {
                        "sequence": len(events) + 2,
                        "time_fs": epoch_end,
                        "request_id": request_id,
                        "event": (
                            "CANCEL_AT_TOKEN_BARRIER" if cancelled else "REQUEST_FINISH"
                        ),
                        "epoch_id": epoch_id,
                    },
                ]
            )
            offset, size, allocation_epoch = allocator.release(request_id)
            released_ranges.add((offset, size))
            allocation_history.append(
                {
                    "operation": "release",
                    "time_fs": epoch_end,
                    "request_id": request_id,
                    "allocation_epoch": allocation_epoch,
                    "global_pa_base": allocator.base + offset,
                    "size_bytes": size,
                }
            )
            events.append(
                {
                    "sequence": len(events),
                    "time_fs": epoch_end,
                    "request_id": request_id,
                    "event": "RELEASE_GLOBAL_PA",
                    "epoch_id": epoch_id,
                }
            )
            pending.remove(request)

        epochs.append(
            {
                "epoch_id": epoch_id,
                "request_ids": [str(item["request_id"]) for item in selected],
                "start_time_fs": epoch_start,
                "completion_time_fs": epoch_end,
                "token_barrier_time_fs": epoch_end,
                "real_p23_source_digest": contract.source_digest,
                "global_pa_bindings": members,
                "tasks": shifted_tasks,
                "conservation": {
                    "parent_requests": contract.parent_requests,
                    "durable_parent_completions": contract.parent_requests,
                    "child_requests": contract.child_requests,
                    "child_completions": contract.child_requests,
                    "zero_in_flight": True,
                },
            }
        )
        epoch_id += 1
        barrier = epoch_end

    ordered_events = sorted(
        events, key=lambda item: (int(item["time_fs"]), int(item["sequence"]))
    )
    active_ranges: list[tuple[int, int]] = []
    overlap_count = 0
    durable_by_request: set[str] = set()
    for event in ordered_events:
        request_id = str(event["request_id"])
        if event["event"] == "ALL_REQUESTS_DURABLE":
            durable_by_request.add(request_id)
        if event["event"] == "RELEASE_GLOBAL_PA" and request_id not in durable_by_request:
            raise P23RequestControlError("Global PA released before durable completion")
    for entry in allocation_history:
        base = int(entry["global_pa_base"])
        size = int(entry["size_bytes"])
        if entry["operation"] == "allocate":
            overlap_count += sum(
                int(base < other + other_size and other < base + size)
                for other, other_size in active_ranges
            )
            active_ranges.append((base, size))
        else:
            active_ranges.remove((base, size))
    if overlap_count or active_ranges or allocator.active:
        raise P23RequestControlError("KV Global PA isolation or retirement failed")

    admitted = sum(int(record["admitted"]) for record in records.values())
    return {
        "schema_version": "hetero-p23-p24-request-control-timeline/v1",
        "timing_semantics": "exact_p23_request_cycle_token_barrier",
        "performance_claim_allowed": False,
        "exact_identity": {
            "batch_size": contract.batch_size,
            "initial_kv_length": contract.initial_kv_length,
            "final_kv_length": contract.final_kv_length,
            "layer_count": 1,
            "source_digest": contract.source_digest,
            "qualification_path": str(contract.qualification_path),
        },
        "requests": [records[key] for key in sorted(records)],
        "epochs": epochs,
        "events": ordered_events,
        "memory": {
            "capacity_bytes": allocator.capacity,
            "peak_bytes": allocator.peak,
            "allocation_history": allocation_history,
            "retired_range_reuse_count": reuse_count,
            "active_range_overlap_count": overlap_count,
            "zero_bytes_after_retirement": not allocator.active,
        },
        "conservation": {
            "input_requests": len(normalized),
            "admitted_requests": admitted,
            "cancelled_before_admission": len(normalized) - admitted,
            "terminal_requests": len(records),
            "exact_p23_epochs": len(epochs),
            "parent_requests": contract.parent_requests * len(epochs),
            "durable_parent_completions": contract.parent_requests * len(epochs),
            "child_requests": contract.child_requests * len(epochs),
            "child_completions": contract.child_requests * len(epochs),
            "zero_in_flight": True,
        },
        "gates": {
            "cancellation_only_at_token_barrier": True,
            "release_after_all_requests_durable": True,
            "deterministic_reuse_observed": reuse_count > 0,
            "no_active_overlap": overlap_count == 0,
            "no_leak": not allocator.active,
            "long_generation_exact_trace_ready": False,
        },
        "qualification_boundary": (
            "Exact P23 BS=2 KV17 epochs now drive P24 cancellation, durable release "
            "and Global-PA reuse. Repeated independent request pairs are exact; a "
            "single request advancing to KV18 or later remains fail-closed until "
            "those exact GPU traces are qualified."
        ),
    }
