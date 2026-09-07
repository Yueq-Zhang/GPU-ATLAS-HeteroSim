"""P24 request termination and KV-pressure functional-cycle contract.

The contract intentionally stops at token-step barriers.  It proves request
state, termination and allocation causality; it is not a calibrated latency or
throughput model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .memory_system import kv_allocation_size
from .runtime_bridge import plan_memory_lifecycle, simulate_token_barrier


class RequestControlRuntimeError(RuntimeError):
    """Raised when P24 scheduler and memory evidence is inconsistent."""


def _effective_natural_limit(request: Mapping[str, object]) -> tuple[int, str]:
    candidates = [(int(request["output_length"]), "completed")]
    eos = int(request.get("eos_after_generated_tokens", 0))
    maximum = int(request.get("max_output_tokens", 0))
    if eos:
        candidates.append((eos, "eos"))
    if maximum:
        candidates.append((maximum, "max_length"))
    # EOS wins an equal-token tie, followed by max length and natural finish.
    rank = {"eos": 0, "max_length": 1, "completed": 2}
    return min(candidates, key=lambda item: (item[0], rank[item[1]]))


def _barrier_at_or_after(time_fs: int, epoch_duration_fs: int) -> int:
    return ((time_fs + epoch_duration_fs - 1) // epoch_duration_fs) * (
        epoch_duration_fs
    )


def _build_memory_lifecycle(
    requests: Sequence[Mapping[str, object]],
    scheduler_result: Mapping[str, object],
    model: Mapping[str, object],
    address: Mapping[str, object],
    memory_space_id: str,
) -> dict[str, object]:
    admissions: dict[str, int] = {}
    for raw_epoch in scheduler_result["epochs"]:  # type: ignore[index]
        epoch = dict(raw_epoch)
        for request_id in epoch.get("admitted_request_ids", []):
            request_id = str(request_id)
            if request_id in admissions:
                raise RequestControlRuntimeError("request admitted more than once")
            admissions[request_id] = int(epoch["boundary_time_fs"])
    finishes = {
        str(item["request_id"]): int(item["finish_time_fs"])
        for item in scheduler_result["requests"]  # type: ignore[index]
    }
    events: list[dict[str, object]] = []
    for request in requests:
        request_id = str(request["request_id"])
        if request_id not in admissions:
            continue
        events.extend(
            [
                {
                    "time_fs": admissions[request_id],
                    "operation": "allocate",
                    "allocation_id": f"{request_id}.kv",
                    "memory_space_id": memory_space_id,
                    "size_bytes": kv_allocation_size(request, model, address),
                    "alignment_bytes": int(
                        address.get("allocation_alignment_bytes", 64)
                    ),
                    "lifetime": "request",
                },
                {
                    "time_fs": finishes[request_id],
                    "operation": "release",
                    "allocation_id": f"{request_id}.kv",
                },
            ]
        )
    events.sort(
        key=lambda item: (
            int(item["time_fs"]),
            0 if item["operation"] == "release" else 1,
            str(item["allocation_id"]),
        )
    )
    return plan_memory_lifecycle(
        [
            {
                "memory_space_id": memory_space_id,
                "capacity_bytes": int(address["kv_capacity_bytes"]),
                "base_alignment_bytes": int(
                    address.get("allocation_alignment_bytes", 64)
                ),
            }
        ],
        events,
    )


def _audit_memory_lifecycle(lifecycle: Mapping[str, object]) -> dict[str, object]:
    active: dict[str, tuple[str, int, int]] = {}
    released: set[tuple[str, int, int]] = set()
    allocation_epochs: set[tuple[str, int]] = set()
    overlap_count = 0
    reuse_count = 0
    for raw_event in lifecycle["events"]:  # type: ignore[index]
        event = dict(raw_event)
        allocation_id = str(event["allocation_id"])
        physical_range = (
            str(event["memory_space_id"]),
            int(event["offset_bytes"]),
            int(event["size_bytes"]),
        )
        if event["operation"] == "release":
            if allocation_id not in active:
                raise RequestControlRuntimeError("release without allocation")
            released.add(active.pop(allocation_id))
            continue
        epoch_key = (physical_range[0], int(event["allocation_epoch"]))
        if epoch_key in allocation_epochs:
            raise RequestControlRuntimeError("duplicate allocation epoch")
        allocation_epochs.add(epoch_key)
        for other_space, other_offset, other_size in active.values():
            if physical_range[0] != other_space:
                continue
            offset, size = physical_range[1:]
            if offset < other_offset + other_size and other_offset < offset + size:
                overlap_count += 1
        if physical_range in released:
            reuse_count += 1
        active[allocation_id] = physical_range
    spaces = [dict(item) for item in lifecycle["memory_spaces"]]  # type: ignore[index]
    return {
        "active_range_overlap_count": overlap_count,
        "allocation_epoch_unique": True,
        "retired_range_reuse_count": reuse_count,
        "active_allocations_after_run": sorted(active),
        "zero_bytes_after_retirement": all(
            int(item["used_bytes"]) == 0 for item in spaces
        ),
        "peak_bytes": max((int(item["peak_bytes"]) for item in spaces), default=0),
    }


def run_request_control_runtime(
    requests: Sequence[Mapping[str, object]],
    scheduling: Mapping[str, object],
    model: Mapping[str, object],
    address: Mapping[str, object],
    memory_space_id: str = "shared0.dram3d",
) -> dict[str, object]:
    """Run and audit the single-layer P24 functional-cycle contract."""

    if int(model["num_layers"]) != 1:
        raise RequestControlRuntimeError("P24 qualification is restricted to one layer")
    if not requests:
        raise RequestControlRuntimeError("requests must not be empty")
    if any(
        str(request.get("execution_scope", "full_request")) != "decode_loop"
        for request in requests
    ):
        raise RequestControlRuntimeError("P24 currently requires decode_loop requests")

    enriched: list[dict[str, object]] = []
    for raw_request in requests:
        request = dict(raw_request)
        request["kv_reservation_bytes"] = kv_allocation_size(
            request, model, address
        )
        enriched.append(request)
    scheduler_contract = dict(scheduling)
    scheduler_contract["kv_capacity_bytes"] = int(address["kv_capacity_bytes"])
    scheduler_result = simulate_token_barrier(enriched, scheduler_contract)
    lifecycle = _build_memory_lifecycle(
        requests, scheduler_result, model, address, memory_space_id
    )
    memory = _audit_memory_lifecycle(lifecycle)
    if memory["active_range_overlap_count"]:
        raise RequestControlRuntimeError("active Global PA ranges overlap")
    if memory["active_allocations_after_run"]:
        raise RequestControlRuntimeError("active allocation remains after retirement")
    if not memory["zero_bytes_after_retirement"]:
        raise RequestControlRuntimeError("KV bytes remain after retirement")
    if int(memory["peak_bytes"]) > int(address["kv_capacity_bytes"]):
        raise RequestControlRuntimeError("KV peak exceeds configured capacity")

    inputs = {str(item["request_id"]): dict(item) for item in requests}
    admissions: dict[str, int] = {}
    retirements: dict[str, int] = {}
    cancellations: dict[str, int] = {}
    for raw_epoch in scheduler_result["epochs"]:  # type: ignore[index]
        epoch = dict(raw_epoch)
        boundary = int(epoch["boundary_time_fs"])
        completion = int(epoch["completion_time_fs"])
        for request_id in epoch.get("admitted_request_ids", []):
            request_id = str(request_id)
            if request_id in admissions:
                raise RequestControlRuntimeError("duplicate admission")
            admissions[request_id] = boundary
        for request_id in epoch.get("cancelled_request_ids", []):
            request_id = str(request_id)
            if request_id in cancellations:
                raise RequestControlRuntimeError("duplicate cancellation")
            cancellations[request_id] = boundary
        for request_id in epoch.get("retired_request_ids", []):
            request_id = str(request_id)
            if request_id in retirements:
                raise RequestControlRuntimeError("duplicate retirement")
            retirements[request_id] = (
                boundary if request_id in cancellations else completion
            )

    epoch_duration = int(scheduling["epoch_duration_fs"])
    records: list[dict[str, object]] = []
    reason_counts = {
        "completed": 0,
        "eos": 0,
        "max_length": 0,
        "cancelled": 0,
    }
    for raw_result in scheduler_result["requests"]:  # type: ignore[index]
        result = dict(raw_result)
        request_id = str(result["request_id"])
        request = inputs[request_id]
        generated = int(result["generated_length"])
        finish = int(result["finish_time_fs"])
        reason = str(result["termination_reason"])
        if reason not in reason_counts:
            raise RequestControlRuntimeError("unknown termination reason")
        reason_counts[reason] += 1
        natural_limit, natural_reason = _effective_natural_limit(request)
        if generated > natural_limit:
            raise RequestControlRuntimeError("request exceeded generation limit")
        if reason != "cancelled" and (generated, reason) != (
            natural_limit,
            natural_reason,
        ):
            raise RequestControlRuntimeError("natural termination contract mismatch")
        cancel_time = request.get("cancel_time_fs")
        if reason == "cancelled":
            if cancel_time is None:
                raise RequestControlRuntimeError("cancelled request lacks cancel time")
            expected_barrier = _barrier_at_or_after(int(cancel_time), epoch_duration)
            if finish != expected_barrier:
                raise RequestControlRuntimeError("cancellation missed token-step barrier")
        ready_times = [int(value) for value in result["token_ready_time_fs"]]
        if len(ready_times) != generated or ready_times != sorted(ready_times):
            raise RequestControlRuntimeError("token commit sequence is inconsistent")
        if any(time > finish for time in ready_times):
            raise RequestControlRuntimeError("token committed after request retirement")
        expected_kv = int(request["initial_kv_length"]) + generated
        if int(result["committed_kv_length"]) != expected_kv:
            raise RequestControlRuntimeError("KV length does not match committed tokens")
        records.append(
            {
                "request_id": request_id,
                "arrival_time_fs": int(request.get("arrival_time_fs", 0)),
                "admission_time_fs": admissions.get(request_id),
                "finish_time_fs": finish,
                "requested_output_length": int(request["output_length"]),
                "effective_natural_limit": natural_limit,
                "generated_length": generated,
                "initial_kv_length": int(request["initial_kv_length"]),
                "final_committed_kv_length": int(result["committed_kv_length"]),
                "termination_reason": reason,
                "cancel_observed_time_fs": cancellations.get(request_id),
                "token_ready_time_fs": ready_times,
                "kv_reservation_bytes": kv_allocation_size(request, model, address),
            }
        )

    allocation_events = [
        dict(item)
        for item in lifecycle["events"]  # type: ignore[index]
        if item["operation"] == "allocate"
    ]
    release_events = [
        dict(item)
        for item in lifecycle["events"]  # type: ignore[index]
        if item["operation"] == "release"
    ]
    delayed_admissions = [
        request_id
        for request_id, admission_time in admissions.items()
        if admission_time
        > _barrier_at_or_after(
            int(inputs[request_id].get("arrival_time_fs", 0)), epoch_duration
        )
    ]
    all_terminal = len(retirements) == len(requests) == len(records)
    return {
        "schema_version": "hetero-p24-request-control-runtime/v1",
        "timing_semantics": "token_step_barrier_functional_cycle",
        "cancellation_granularity": "token_step_barrier",
        "performance_claim_allowed": False,
        "single_layer_qualification": True,
        "requests": sorted(records, key=lambda item: str(item["request_id"])),
        "scheduler_result": scheduler_result,
        "memory_lifecycle": lifecycle,
        "memory": {
            **memory,
            "capacity_bytes": int(address["kv_capacity_bytes"]),
            "capacity_delayed_request_ids": sorted(delayed_admissions),
        },
        "termination": {
            "reason_counts": reason_counts,
            "cancelled_before_admission": sorted(
                request_id
                for request_id in cancellations
                if request_id not in admissions
            ),
        },
        "conservation": {
            "input_requests": len(requests),
            "terminal_requests": len(retirements),
            "admitted_requests": len(admissions),
            "allocated_requests": len(allocation_events),
            "released_requests": len(release_events),
            "all_requests_terminal": all_terminal,
            "admission_allocation_bijective": (
                len(admissions) == len(allocation_events) == len(release_events)
            ),
            "zero_in_flight": all_terminal
            and not memory["active_allocations_after_run"],
        },
        "qualification_boundary": (
            "Functional-cycle request termination, barrier cancellation and KV "
            "capacity/reuse only; no exact batched GPU/ATLAS trace or calibrated "
            "performance claim."
        ),
    }
