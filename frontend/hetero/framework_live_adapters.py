"""Fail-closed normalization for real vLLM and TensorRT-LLM callbacks."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence


class FrameworkLiveAdapterError(RuntimeError):
    """Raised when a claimed live framework event is synthetic or incomplete."""


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise FrameworkLiveAdapterError(f"{name} must be an object")
    return value


def _sequence(value: object, name: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise FrameworkLiveAdapterError(f"{name} must be an array")
    return value


def normalize_vllm_live_observation(
    raw: Mapping[str, object],
    *,
    global_pa_base: int = 1 << 38,
) -> dict[str, object]:
    """Validate in-process vLLM SchedulerOutput and actual block-table deltas."""

    if raw.get("schema_version") != "hetero-vllm-live-observation/v1":
        raise FrameworkLiveAdapterError("unsupported vLLM live observation")
    if raw.get("framework") != "vllm" or not raw.get("framework_version"):
        raise FrameworkLiveAdapterError("vLLM framework identity is incomplete")
    engine = _mapping(raw.get("engine"), "vLLM engine")
    block_size = int(engine.get("block_size_tokens", 0))
    bytes_per_block = int(engine.get("bytes_per_block", 0))
    num_gpu_blocks = int(engine.get("num_gpu_blocks", 0))
    scheduler_class = str(engine.get("scheduler_class", ""))
    if (
        not scheduler_class.startswith("vllm.v1.core.sched.")
        or not scheduler_class.endswith("Scheduler")
        or engine.get("multiprocess_engine_core") is not False
        or block_size <= 0
        or bytes_per_block <= 0
        or num_gpu_blocks <= 0
        or global_pa_base < 0
    ):
        raise FrameworkLiveAdapterError("vLLM engine/block geometry is invalid")

    active: set[str] = set()
    request_aliases: dict[str, str] = {}
    tables: dict[str, list[list[int]]] = {}
    owners: dict[tuple[int, int], str] = {}
    batches: list[list[str]] = []
    normalized_steps: list[dict[str, object]] = []
    allocations: dict[tuple[int, int], dict[str, object]] = {}
    for expected, value in enumerate(_sequence(raw.get("scheduler_steps"), "steps")):
        step = _mapping(value, f"vLLM scheduler step {expected}")
        if int(step.get("sequence", -1)) != expected:
            raise FrameworkLiveAdapterError("vLLM scheduler sequence is not contiguous")
        for value in _sequence(step.get("new_requests", []), "new requests"):
            request = _mapping(value, "new vLLM request")
            raw_request_id = str(request.get("request_id", ""))
            block_groups = _sequence(request.get("block_ids"), "vLLM block groups")
            if (
                not raw_request_id
                or raw_request_id in request_aliases
                or not block_groups
            ):
                raise FrameworkLiveAdapterError("invalid new vLLM request")
            request_id = f"request-{len(request_aliases):04d}"
            request_aliases[raw_request_id] = request_id
            active.add(request_id)
            tables[request_id] = []
            for group_id, group in enumerate(block_groups):
                blocks = [int(item) for item in _sequence(group, "vLLM block row")]
                tables[request_id].append(blocks)
                for block_id in blocks:
                    key = (group_id, block_id)
                    if block_id < 0 or key in owners:
                        raise FrameworkLiveAdapterError("aliased vLLM block table")
                    owners[key] = request_id
                    allocations[key] = {
                        "group_id": group_id,
                        "block_id": block_id,
                        "request_id": request_id,
                        "global_address": global_pa_base
                        + (group_id * num_gpu_blocks + block_id) * bytes_per_block,
                        "size_bytes": bytes_per_block,
                    }
        cached = _mapping(step.get("cached_requests", {}), "cached requests")
        raw_cached_ids = [
            str(item) for item in _sequence(cached.get("request_ids", []), "cached IDs")
        ]
        if any(item not in request_aliases for item in raw_cached_ids):
            raise FrameworkLiveAdapterError("unknown cached vLLM request")
        cached_ids = [request_aliases[item] for item in raw_cached_ids]
        new_rows = _sequence(cached.get("new_block_ids", []), "cached block rows")
        if len(cached_ids) != len(new_rows):
            raise FrameworkLiveAdapterError("vLLM cached block delta is misaligned")
        for request_id, raw_groups in zip(cached_ids, new_rows):
            if request_id not in active:
                raise FrameworkLiveAdapterError("unknown cached vLLM request")
            if raw_groups is None:
                continue
            groups = _sequence(raw_groups, "cached block groups")
            if len(groups) != len(tables[request_id]):
                raise FrameworkLiveAdapterError("vLLM cache-group count changed")
            for group_id, raw_blocks in enumerate(groups):
                for block_id in [int(item) for item in _sequence(raw_blocks, "blocks")]:
                    key = (group_id, block_id)
                    if block_id < 0 or (key in owners and owners[key] != request_id):
                        raise FrameworkLiveAdapterError(
                            "aliased vLLM block-table delta"
                        )
                    if block_id not in tables[request_id][group_id]:
                        tables[request_id][group_id].append(block_id)
                        owners[key] = request_id
                        allocations[key] = {
                            "group_id": group_id,
                            "block_id": block_id,
                            "request_id": request_id,
                            "global_address": global_pa_base
                            + (group_id * num_gpu_blocks + block_id) * bytes_per_block,
                            "size_bytes": bytes_per_block,
                        }
        batch = _mapping(step.get("batch"), "vLLM batch")
        raw_members = [
            str(item) for item in _sequence(batch.get("request_ids"), "batch IDs")
        ]
        if any(item not in request_aliases for item in raw_members):
            raise FrameworkLiveAdapterError("unknown batched vLLM request")
        members = [request_aliases[item] for item in raw_members]
        counts = [
            int(item) for item in _sequence(batch.get("token_counts"), "token counts")
        ]
        raw_step_finished = [
            str(item)
            for item in _sequence(
                step.get("finished_request_ids", []), "step finished IDs"
            )
        ]
        if any(item not in request_aliases for item in raw_step_finished):
            raise FrameworkLiveAdapterError("unknown finished vLLM request")
        step_finished = [request_aliases[item] for item in raw_step_finished]
        lifecycle_only = not members and bool(step_finished)
        if (
            (not members and not lifecycle_only)
            or len(members) != len(counts)
            or len(set(members)) != len(members)
            or any(member not in active for member in members)
            or any(count <= 0 for count in counts)
        ):
            raise FrameworkLiveAdapterError("invalid live vLLM batch")
        if members:
            batches.append(members)
        normalized_steps.append(
            {
                "sequence": expected,
                "new_requests": [
                    {
                        **dict(_mapping(item, "new vLLM request")),
                        "request_id": request_aliases[str(item["request_id"])],
                    }
                    for item in _sequence(step.get("new_requests", []), "new requests")
                ],
                "cached_requests": {
                    **dict(cached),
                    "request_ids": cached_ids,
                },
                "batch": {
                    **dict(batch),
                    "request_ids": members,
                },
                "finished_request_ids": step_finished,
                "new_block_ids_to_zero": list(
                    _sequence(
                        step.get("new_block_ids_to_zero", []),
                        "new block IDs to zero",
                    )
                ),
            }
        )

    raw_finished = [
        str(item) for item in _sequence(raw.get("finished_request_ids"), "finished IDs")
    ]
    if any(item not in request_aliases for item in raw_finished):
        raise FrameworkLiveAdapterError("unknown final vLLM request")
    finished = [request_aliases[item] for item in raw_finished]
    if set(finished) != active:
        raise FrameworkLiveAdapterError("vLLM output/request lifecycle is incomplete")
    released = []
    for request_id in finished:
        for group_id, blocks in enumerate(tables.pop(request_id)):
            for block_id in blocks:
                owners.pop((group_id, block_id))
                released.append(
                    {
                        "request_id": request_id,
                        "group_id": group_id,
                        "block_id": block_id,
                    }
                )
        active.remove(request_id)
    if owners or active or tables:
        raise FrameworkLiveAdapterError("vLLM KV ownership leaked")

    result = _mapping(raw.get("result"), "vLLM result")
    token_ids = _sequence(result.get("token_ids"), "vLLM token IDs")
    if len(token_ids) != len(finished):
        raise FrameworkLiveAdapterError("vLLM result count disagrees with requests")
    identity = {
        "framework_version": raw["framework_version"],
        "model": raw.get("model"),
        "requested_revision": raw.get("requested_revision"),
        "device": raw.get("device"),
        "engine": dict(engine),
        "workload": raw.get("workload"),
        "scheduler_steps": normalized_steps,
        "finished_request_ids": finished,
        "result": dict(result),
    }
    distinct_batches = {tuple(item) for item in batches}
    return {
        "schema_version": "hetero-vllm-live-normalized/v1",
        "identity": identity,
        "identity_sha256": _digest(identity),
        "allocation_ledger": sorted(
            allocations.values(), key=lambda item: (item["group_id"], item["block_id"])
        ),
        "release_ledger": released,
        "scheduler_step_count": len(normalized_steps),
        "real_scheduler_callback_connected": True,
        "real_block_table_observed": bool(allocations),
        "continuous_batching_observed": len(distinct_batches) > 1,
        "ragged_batching_observed": any(
            len({int(item) for item in step["batch"]["token_counts"]}) > 1
            for step in normalized_steps
        ),
        "paged_kv_global_pa_bound": bool(allocations),
        "zero_live_pages": True,
        "feedback_to_framework": False,
        "performance_claim_allowed": False,
    }


def normalize_tensorrt_llm_live_observation(
    raw: Mapping[str, object],
) -> dict[str, object]:
    """Validate real TensorRT-LLM LLM API engine/profile/scheduler events."""

    if raw.get("schema_version") != "hetero-tensorrt-llm-live-observation/v1":
        raise FrameworkLiveAdapterError("unsupported TensorRT-LLM observation")
    if raw.get("framework") != "tensorrt_llm" or not raw.get("framework_version"):
        raise FrameworkLiveAdapterError("TensorRT-LLM identity is incomplete")
    engine = _mapping(raw.get("engine"), "TensorRT-LLM engine")
    profile = _mapping(engine.get("profile"), "TensorRT-LLM profile")
    for field in ("max_batch_size", "max_input_len", "max_seq_len", "max_num_tokens"):
        if int(profile.get(field, 0)) <= 0:
            raise FrameworkLiveAdapterError(f"invalid TensorRT-LLM {field}")
    if (
        engine.get("backend") != "pytorch"
        or not str(engine.get("engine_class", "")).startswith("tensorrt_llm.")
        or not str(engine.get("executor_class", "")).startswith("tensorrt_llm.")
        or not str(engine.get("scheduler_class", "")).startswith("tensorrt_llm.")
    ):
        raise FrameworkLiveAdapterError("TensorRT-LLM live engine classes are invalid")
    events = _sequence(raw.get("events"), "TensorRT-LLM events")
    if not events:
        raise FrameworkLiveAdapterError("TensorRT-LLM event stream is empty")
    kinds = []
    schedule_count = 0
    nonempty_schedule_count = 0
    request_aliases: dict[str, str] = {}
    normalized_events: list[dict[str, object]] = []

    def canonical_request_id(value: object) -> str:
        raw_request_id = str(value)
        if not raw_request_id:
            raise FrameworkLiveAdapterError("empty TensorRT-LLM request ID")
        if raw_request_id not in request_aliases:
            request_aliases[raw_request_id] = f"request-{len(request_aliases):04d}"
        return request_aliases[raw_request_id]

    for expected, value in enumerate(events):
        event = _mapping(value, f"TensorRT-LLM event {expected}")
        if int(event.get("sequence", -1)) != expected:
            raise FrameworkLiveAdapterError(
                "TensorRT-LLM event sequence is not contiguous"
            )
        kind = str(event.get("event", ""))
        kinds.append(kind)
        normalized_event = dict(event)
        for field in (
            "request_ids",
            "context_request_ids",
            "generation_request_ids",
            "paused_request_ids",
        ):
            if field in event:
                normalized_event[field] = [
                    canonical_request_id(item)
                    for item in _sequence(event.get(field), field)
                ]
        if "active_request_states" in event:
            states = _mapping(
                event.get("active_request_states"), "active request states"
            )
            normalized_event["active_request_states"] = {
                canonical_request_id(request_id): state
                for request_id, state in states.items()
            }
        if kind == "scheduler_step":
            schedule_count += 1
            if _sequence(event.get("request_ids"), "scheduled requests"):
                nonempty_schedule_count += 1
        normalized_events.append(normalized_event)
    if (
        "engine_initialized" not in kinds
        or "request_complete" not in kinds
        or schedule_count <= 0
        or nonempty_schedule_count <= 0
        or _mapping(raw.get("result"), "TensorRT-LLM result").get("finished")
        is not True
    ):
        raise FrameworkLiveAdapterError("TensorRT-LLM lifecycle is incomplete")
    identity = {
        "framework_version": raw["framework_version"],
        "model": raw.get("model"),
        "requested_revision": raw.get("requested_revision"),
        "device": raw.get("device"),
        "engine": dict(engine),
        "events": normalized_events,
        "result": dict(_mapping(raw.get("result"), "TensorRT-LLM result")),
    }
    return {
        "schema_version": "hetero-tensorrt-llm-live-normalized/v1",
        "identity": identity,
        "identity_sha256": _digest(identity),
        "real_engine_object_observed": True,
        "real_profile_observed": True,
        "real_scheduler_callback_connected": True,
        "scheduler_callback_count": schedule_count,
        "nonempty_scheduler_callback_count": nonempty_schedule_count,
        "serialized_tensorrt_engine_qualified": False,
        "backend_scope": "tensorrt_llm_llm_api_pytorch_backend",
        "feedback_to_framework": False,
        "performance_claim_allowed": False,
    }


def qualify_live_framework_pair(
    vllm_runs: Sequence[Mapping[str, object]],
    tensorrt_runs: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Require two semantically identical observations for each live adapter."""

    if len(vllm_runs) != 2 or len(tensorrt_runs) != 2:
        raise FrameworkLiveAdapterError("P34 requires two runs per framework")
    normalized_vllm = [normalize_vllm_live_observation(item) for item in vllm_runs]
    normalized_trt = [
        normalize_tensorrt_llm_live_observation(item) for item in tensorrt_runs
    ]
    if normalized_vllm[0]["identity_sha256"] != normalized_vllm[1]["identity_sha256"]:
        raise FrameworkLiveAdapterError("vLLM live double-run identity mismatch")
    if normalized_trt[0]["identity_sha256"] != normalized_trt[1]["identity_sha256"]:
        raise FrameworkLiveAdapterError("TensorRT-LLM double-run identity mismatch")
    payload = {
        "vllm": normalized_vllm[0],
        "tensorrt_llm": normalized_trt[0],
        "qualification": {
            "real_vllm_scheduler_connected": True,
            "real_vllm_block_table_connected": True,
            "real_vllm_paged_kv_global_pa_bound": True,
            "real_tensorrt_llm_engine_profile_connected": True,
            "serialized_tensorrt_engine_qualified": False,
            "online_shadow_input_ready": True,
            "performance_claim_allowed": False,
        },
    }
    return {
        "schema_version": "hetero-p34-live-framework-qualification/v1",
        "status": "passed",
        **payload,
        "record_sha256": _digest(payload),
    }
