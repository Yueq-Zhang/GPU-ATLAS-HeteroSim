"""P22 request lifecycle, causality and shared-resource audit."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence


class MultiBatchRuntimeError(RuntimeError):
    """Raised when scheduler, batch, address or lifecycle evidence disagrees."""


def _request_throughput(output_length: int, latency_fs: int) -> float:
    return output_length * 1.0e15 / latency_fs if latency_fs > 0 else 0.0


def _jain_fairness(values: Sequence[float]) -> float:
    if not values or not any(values):
        return 1.0
    total = sum(values)
    return total * total / (len(values) * sum(value * value for value in values))


def _audit_memory_lifecycle(
    memory_lifecycle: Mapping[str, object] | None,
) -> dict[str, object]:
    if memory_lifecycle is None:
        return {
            "available": False,
            "active_range_overlap_count": 0,
            "allocation_epoch_unique": True,
            "zero_bytes_after_retirement": True,
        }
    active: dict[str, tuple[str, int, int, int]] = {}
    epochs: set[tuple[str, int]] = set()
    overlap_count = 0
    reuse_count = 0
    released_ranges: set[tuple[str, int, int]] = set()
    for raw in memory_lifecycle.get("events", []):
        event = dict(raw)
        allocation_id = str(event["allocation_id"])
        space = str(event["memory_space_id"])
        offset = int(event["offset_bytes"])
        size = int(event["size_bytes"])
        epoch = int(event["allocation_epoch"])
        if event["operation"] == "allocate":
            if (space, epoch) in epochs:
                raise MultiBatchRuntimeError("duplicate allocation epoch")
            epochs.add((space, epoch))
            for other_space, other_offset, other_size, _ in active.values():
                if space != other_space:
                    continue
                if offset < other_offset + other_size and other_offset < offset + size:
                    overlap_count += 1
            if (space, offset, size) in released_ranges:
                reuse_count += 1
            active[allocation_id] = (space, offset, size, epoch)
        elif event["operation"] == "release":
            record = active.pop(allocation_id, None)
            if record is None:
                raise MultiBatchRuntimeError(
                    f"release without active allocation: {allocation_id}"
                )
            released_ranges.add(record[:3])
        else:
            raise MultiBatchRuntimeError("unknown memory lifecycle operation")
    spaces = memory_lifecycle.get("memory_spaces", [])
    zero_bytes = all(int(dict(item).get("used_bytes", -1)) == 0 for item in spaces)
    return {
        "available": True,
        "active_range_overlap_count": overlap_count,
        "allocation_epoch_unique": True,
        "retired_range_reuse_count": reuse_count,
        "zero_bytes_after_retirement": zero_bytes,
        "active_allocations_after_run": sorted(active),
    }


def build_multi_batch_runtime(
    scheduler_result: Mapping[str, object],
    batch_plan: Mapping[str, object],
    requests: Sequence[Mapping[str, object]],
    memory_lifecycle: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Materialize request states and audit the completed barriered timeline."""

    inputs = {str(item["request_id"]): dict(item) for item in requests}
    states: dict[str, dict[str, object]] = {}
    events: list[dict[str, object]] = []
    for request_id, request in sorted(inputs.items()):
        scope = str(request.get("execution_scope", "full_request"))
        states[request_id] = {
            "state": "WAITING",
            "scope": scope,
            "prompt_cursor": 0,
            "generated_length": 0,
            "committed_kv_length": (
                int(request.get("initial_kv_length", 0))
                if scope in {"decode_step", "decode_loop"}
                else 0
            ),
            "kv_version": 0,
            "admission_time_fs": None,
            "finish_time_fs": None,
            "token_ready_time_fs": [],
        }
        events.append(
            {
                "sequence": len(events),
                "time_fs": int(request.get("arrival_time_fs", 0)),
                "request_id": request_id,
                "event": "ARRIVAL",
                "state": "WAITING",
            }
        )

    active: set[str] = set()
    max_active = 0
    version_commits: list[dict[str, object]] = []
    selection_count = 0
    selection_tokens = 0
    last_completion_by_request: dict[str, int] = {}
    for raw_epoch in scheduler_result["epochs"]:  # type: ignore[index]
        epoch = dict(raw_epoch)
        boundary = int(epoch["boundary_time_fs"])
        completion = int(epoch["completion_time_fs"])
        selections = [dict(item) for item in epoch["selections"]]
        admitted = [str(value) for value in epoch.get("admitted_request_ids", [])]
        for request_id in admitted:
            if request_id not in states:
                raise MultiBatchRuntimeError(
                    f"scheduler admitted unknown request: {request_id}"
                )
            state = states[request_id]
            if state["admission_time_fs"] is not None:
                raise MultiBatchRuntimeError("request admitted more than once")
            if boundary < int(inputs[request_id].get("arrival_time_fs", 0)):
                raise MultiBatchRuntimeError("request admitted before arrival")
            state["admission_time_fs"] = boundary
            state["state"] = (
                "DECODE_READY"
                if state["scope"] in {"decode_step", "decode_loop"}
                else "PREFILL_READY"
            )
            active.add(request_id)
            max_active = max(max_active, len(active))
            events.append(
                {
                    "sequence": len(events),
                    "time_fs": boundary,
                    "request_id": request_id,
                    "event": "ADMIT",
                    "state": state["state"],
                }
            )
        for selection in selections:
            request_id = str(selection["request_id"])
            if request_id not in states:
                raise MultiBatchRuntimeError(
                    f"scheduler selected unknown request: {request_id}"
                )
            request = inputs[request_id]
            state = states[request_id]
            arrival = int(request.get("arrival_time_fs", 0))
            if boundary < arrival:
                raise MultiBatchRuntimeError("request selected before arrival")
            if state["admission_time_fs"] is None:
                # Backward-compatible fallback for scheduler-result/v1.
                state["admission_time_fs"] = boundary
                state["state"] = (
                    "DECODE_READY"
                    if state["scope"] in {"decode_step", "decode_loop"}
                    else "PREFILL_READY"
                )
                active.add(request_id)
                max_active = max(max_active, len(active))
                events.append(
                    {
                        "sequence": len(events),
                        "time_fs": boundary,
                        "request_id": request_id,
                        "event": "ADMIT",
                        "state": state["state"],
                    }
                )
            phase = str(selection["phase"])
            expected_ready = "PREFILL_READY" if phase == "prefill" else "DECODE_READY"
            if state["state"] != expected_ready:
                raise MultiBatchRuntimeError(
                    f"{request_id} selected in {phase} from state {state['state']}"
                )
            if request_id in last_completion_by_request and (
                boundary < last_completion_by_request[request_id]
            ):
                raise MultiBatchRuntimeError("next request step starts before commit")
            state["state"] = (
                "PREFILL_RUNNING" if phase == "prefill" else "DECODE_RUNNING"
            )
            events.append(
                {
                    "sequence": len(events),
                    "time_fs": boundary,
                    "request_id": request_id,
                    "event": "START_PHASE",
                    "phase": phase,
                    "state": state["state"],
                    "epoch_id": int(epoch["epoch_id"]),
                }
            )
            selection_count += 1
            token_count = int(selection["token_count"])
            selection_tokens += token_count
            previous_version = int(state["kv_version"])
            if phase == "prefill":
                if int(selection["token_begin"]) != int(state["prompt_cursor"]):
                    raise MultiBatchRuntimeError("prefill cursor discontinuity")
                state["prompt_cursor"] = int(state["prompt_cursor"]) + token_count
                state["committed_kv_length"] = (
                    int(state["committed_kv_length"]) + token_count
                )
                if int(state["prompt_cursor"]) == int(request["prompt_length"]):
                    state["generated_length"] = 1
                    state["token_ready_time_fs"].append(completion)  # type: ignore[union-attr]
                    next_state = (
                        "FINISHED"
                        if int(request["output_length"]) == 1
                        else "DECODE_READY"
                    )
                else:
                    next_state = "PREFILL_READY"
            else:
                if int(selection["token_begin"]) != int(
                    state["committed_kv_length"]
                ):
                    raise MultiBatchRuntimeError("decode KV cursor discontinuity")
                state["generated_length"] = int(state["generated_length"]) + 1
                state["committed_kv_length"] = (
                    int(state["committed_kv_length"]) + token_count
                )
                state["token_ready_time_fs"].append(completion)  # type: ignore[union-attr]
                next_state = (
                    "FINISHED"
                    if int(state["generated_length"]) == int(request["output_length"])
                    else "DECODE_READY"
                )
            state["kv_version"] = previous_version + 1
            version_commits.append(
                {
                    "request_id": request_id,
                    "epoch_id": int(epoch["epoch_id"]),
                    "completion_time_fs": completion,
                    "from_version": previous_version,
                    "to_version": int(state["kv_version"]),
                    "committed_kv_length": int(state["committed_kv_length"]),
                }
            )
            state["state"] = next_state
            last_completion_by_request[request_id] = completion
            events.append(
                {
                    "sequence": len(events),
                    "time_fs": completion,
                    "request_id": request_id,
                    "event": "COMMIT_PHASE",
                    "phase": phase,
                    "state": next_state,
                    "kv_version": int(state["kv_version"]),
                    "committed_kv_length": int(state["committed_kv_length"]),
                    "epoch_id": int(epoch["epoch_id"]),
                }
            )
            if next_state == "FINISHED":
                state["finish_time_fs"] = completion
                active.remove(request_id)
                events.append(
                    {
                        "sequence": len(events),
                        "time_fs": completion,
                        "request_id": request_id,
                        "event": "RETIRE",
                        "state": "FINISHED",
                    }
                )

    result_by_request = {
        str(item["request_id"]): dict(item)
        for item in scheduler_result["requests"]  # type: ignore[index]
    }
    request_records: list[dict[str, object]] = []
    for request_id, state in sorted(states.items()):
        expected = result_by_request[request_id]
        if int(state["generated_length"]) != int(expected["generated_length"]):
            raise MultiBatchRuntimeError("generated-length mismatch")
        if int(state["committed_kv_length"]) != int(
            expected["committed_kv_length"]
        ):
            raise MultiBatchRuntimeError("committed-KV mismatch")
        if list(state["token_ready_time_fs"]) != list(
            expected["token_ready_time_fs"]
        ):
            raise MultiBatchRuntimeError("token-ready timeline mismatch")
        request = inputs[request_id]
        admission = int(state["admission_time_fs"])
        finish = int(state["finish_time_fs"])
        arrival = int(request.get("arrival_time_fs", 0))
        request_records.append(
            {
                "request_id": request_id,
                "final_state": state["state"],
                "arrival_time_fs": arrival,
                "admission_time_fs": admission,
                "queue_wait_fs": admission - arrival,
                "finish_time_fs": finish,
                "e2e_user_fs": finish - arrival,
                "generated_length": int(state["generated_length"]),
                "final_committed_kv_length": int(state["committed_kv_length"]),
                "final_kv_version": int(state["kv_version"]),
                "token_ready_time_fs": list(state["token_ready_time_fs"]),
            }
        )

    subbatches = [dict(item) for item in batch_plan["device_subbatches"]]  # type: ignore[index]
    subbatches_by_epoch: dict[int, list[dict[str, object]]] = defaultdict(list)
    for item in subbatches:
        subbatches_by_epoch[int(item["epoch_id"])].append(item)
    resource_intervals: list[dict[str, object]] = []
    for raw_epoch in batch_plan["epochs"]:  # type: ignore[index]
        epoch = dict(raw_epoch)
        by_device: dict[str, list[str]] = defaultdict(list)
        for item in subbatches_by_epoch[int(epoch["epoch_id"])]:
            by_device[str(item["device_id"])].append(str(item["subbatch_id"]))
        for device_id, identifiers in sorted(by_device.items()):
            resource_intervals.append(
                {
                    "resource_id": device_id,
                    "epoch_id": int(epoch["epoch_id"]),
                    "start_time_fs": int(epoch["boundary_time_fs"]),
                    "completion_time_fs": int(epoch["completion_time_fs"]),
                    "subbatch_ids": identifiers,
                    "timing_source": "scheduling.epoch_duration_fs",
                }
            )
    for device_id in {str(item["resource_id"]) for item in resource_intervals}:
        previous_completion = -1
        intervals = (
            record
            for record in resource_intervals
            if record["resource_id"] == device_id
        )
        for item in sorted(intervals, key=lambda record: int(record["start_time_fs"])):
            if int(item["start_time_fs"]) < previous_completion:
                raise MultiBatchRuntimeError("shared resource intervals overlap")
            previous_completion = int(item["completion_time_fs"])

    memory_audit = _audit_memory_lifecycle(memory_lifecycle)
    if memory_audit["active_range_overlap_count"] != 0:
        raise MultiBatchRuntimeError("concurrent request Global PA ranges overlap")
    if not memory_audit["zero_bytes_after_retirement"]:
        raise MultiBatchRuntimeError("request memory leaked after retirement")
    makespan = max(int(record["finish_time_fs"]) for record in request_records)
    first_arrival = min(int(record["arrival_time_fs"]) for record in request_records)
    generated_tokens = sum(
        int(record["generated_length"]) for record in request_records
    )
    batch_sizes = [len(item["request_ids"]) for item in subbatches]
    throughputs = [
        _request_throughput(
            int(record["generated_length"]), int(record["e2e_user_fs"])
        )
        for record in request_records
    ]
    all_finished = all(
        record["final_state"] == "FINISHED" for record in request_records
    )
    return {
        "schema_version": "hetero-multi-batch-runtime/v1",
        "request_batch_mode": batch_plan["request_batch_mode"],
        "batch_policy": batch_plan["batch_policy"],
        "cycle_mode": batch_plan["cycle_mode"],
        "timing_semantics": "barriered_functional_cycle_composition",
        "performance_claim_allowed": bool(
            batch_plan["performance_claim_allowed"]
        ),
        "requests": request_records,
        "request_events": sorted(
            events,
            key=lambda item: (int(item["time_fs"]), int(item["sequence"])),
        ),
        "version_commits": version_commits,
        "resource_intervals": resource_intervals,
        "memory_isolation": memory_audit,
        "conservation": {
            "input_requests": len(requests),
            "admitted_requests": sum(
                1 for item in events if item["event"] == "ADMIT"
            ),
            "retired_requests": sum(
                1 for item in events if item["event"] == "RETIRE"
            ),
            "scheduler_selections": selection_count,
            "scheduler_tokens": selection_tokens,
            "all_requests_finished": all_finished,
            "all_selected_requests_mapped": batch_plan["conservation"][  # type: ignore[index]
                "all_selected_requests_mapped"
            ],
            "member_fanout_is_bijective_per_subbatch": batch_plan[
                "conservation"
            ]["member_fanout_is_bijective_per_subbatch"],  # type: ignore[index]
            "zero_in_flight": not active,
        },
        "metrics": {
            "makespan_fs": makespan,
            "generated_tokens": generated_tokens,
            "throughput_tokens_per_second": (
                generated_tokens * 1.0e15 / (makespan - first_arrival)
                if makespan > first_arrival
                else 0.0
            ),
            "mean_queue_wait_fs": sum(
                int(record["queue_wait_fs"]) for record in request_records
            )
            / len(request_records),
            "max_queue_wait_fs": max(
                int(record["queue_wait_fs"]) for record in request_records
            ),
            "max_active_requests": max_active,
            "mean_device_subbatch_size": (
                sum(batch_sizes) / len(batch_sizes) if batch_sizes else 0.0
            ),
            "max_device_subbatch_size": max(batch_sizes, default=0),
            "batch_utilization": float(batch_plan["batch_utilization"]),
            "jain_request_throughput_fairness": _jain_fairness(throughputs),
        },
        "qualification_boundary": (
            "performance-qualified exact batched kernels"
            if batch_plan["performance_claim_allowed"]
            else (
                "functional and causal only; exact batched-kernel performance "
                "is closed"
            )
        ),
    }
