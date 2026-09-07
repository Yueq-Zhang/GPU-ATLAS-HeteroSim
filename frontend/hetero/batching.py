"""Deterministic request, phase and device sub-batch planning.

P22 deliberately separates two timing contracts:

``request_cycle_composed``
    Composes request members inside shared resource epochs and records the
    timing source used by the scheduler. It is useful for causality and
    contention studies, but is not fused-batch kernel evidence.

``batched_kernel_cycle``
    Requires an exact batch/shape/device entry in a sealed catalog. No BS=1
    timing is copied or scaled to manufacture a batched-kernel result.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from .ir import ModelNode, NodeKind, Phase
from .placement import place_nodes


class BatchPlanError(ValueError):
    """Raised when a batch violates compatibility or qualification rules."""


@dataclass(frozen=True, slots=True)
class RaggedSequence:
    request_id: str
    token_begin: int
    token_count: int
    packed_begin: int
    causal_kv_length: int


@dataclass(frozen=True, slots=True)
class DeviceSubBatch:
    subbatch_id: str
    epoch_id: int
    phase: str
    layer_id: int | None
    op: str
    operator_group: str | None
    device_id: str
    dtype: str
    request_ids: tuple[str, ...]
    q_lengths: tuple[int, ...]
    kv_lengths: tuple[int, ...]
    effective_tokens: int
    padded_tokens: int
    effective_attention_tokens: int
    padded_attention_tokens: int
    batch_policy: str
    cycle_mode: str
    shape_signature: str
    cu_seqlens: tuple[int, ...]
    input_permutation: tuple[int, ...]
    output_permutation: tuple[int, ...]
    member_mappings: tuple[Mapping[str, object], ...]
    artifact_id: str | None
    functional_cycle_ready: bool
    request_cycle_ready: bool
    performance_eligible: bool


_BATCH_POLICIES = {"homogeneous", "padding_dense", "ragged_split"}
_CYCLE_MODES = {"request_cycle_composed", "batched_kernel_cycle"}


def load_batch_artifact_catalog(
    project_root: Path, catalog_ref: str | None
) -> Mapping[str, object] | None:
    """Load the optional exact batched-kernel catalog."""

    if catalog_ref is None:
        return None
    path = Path(catalog_ref)
    if not path.is_absolute():
        path = project_root / path
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "hetero-batched-kernel-catalog/v1":
        raise BatchPlanError("invalid batched-kernel catalog schema_version")
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise BatchPlanError("batched-kernel catalog entries must be an array")
    return payload


def _representative_node(
    phase: str,
    layer_id: int | None,
    op: str,
    operator_group: str | None,
    q_len: int,
    kv_len: int,
) -> ModelNode:
    return ModelNode(
        node_id=f"runtime.{phase}.l{layer_id}.{op}",
        kind=NodeKind.COMPUTE,
        op=op,
        phase=Phase(phase),
        layer_id=layer_id,
        step_id=0,
        attributes={
            "operator_group": operator_group,
            "q_len": q_len,
            "attention_kv_len": kv_len,
        },
    )


def _shape_signature(
    *,
    phase: str,
    layer_id: int | None,
    op: str,
    device_id: str,
    dtype: str,
    q_lengths: tuple[int, ...],
    kv_lengths: tuple[int, ...],
) -> str:
    layer = "global" if layer_id is None else str(layer_id)
    q_text = ",".join(str(value) for value in q_lengths)
    kv_text = ",".join(str(value) for value in kv_lengths)
    return (
        f"{phase}|l={layer}|op={op}|device={device_id}|dtype={dtype}|"
        f"bs={len(q_lengths)}|q=[{q_text}]|kv=[{kv_text}]"
    )


def _match_exact_artifact(
    catalog: Mapping[str, object] | None,
    *,
    phase: str,
    layer_id: int | None,
    op: str,
    device_id: str,
    dtype: str,
    q_lengths: tuple[int, ...],
    kv_lengths: tuple[int, ...],
) -> Mapping[str, object] | None:
    if catalog is None:
        return None
    raw_entries = catalog.get("entries")
    if not isinstance(raw_entries, list):
        raise BatchPlanError("batched-kernel catalog entries must be an array")
    matches: list[Mapping[str, object]] = []
    for raw in raw_entries:
        if not isinstance(raw, Mapping):
            raise BatchPlanError("batched-kernel catalog entry must be an object")
        if (
            raw.get("phase") == phase
            and raw.get("layer_id") == layer_id
            and raw.get("op") == op
            and raw.get("device_id") == device_id
            and raw.get("dtype") == dtype
            and int(raw.get("batch_size", -1)) == len(q_lengths)
            and tuple(int(value) for value in raw.get("q_lengths", []))
            == q_lengths
            and tuple(int(value) for value in raw.get("kv_lengths", []))
            == kv_lengths
        ):
            matches.append(raw)
    if len(matches) > 1:
        raise BatchPlanError(
            "multiple exact batched-kernel artifacts match one sub-batch"
        )
    return matches[0] if matches else None


def _partition_members(
    members: list[Mapping[str, object]], policy: str
) -> list[list[Mapping[str, object]]]:
    shapes: dict[tuple[int, int], list[Mapping[str, object]]] = defaultdict(list)
    for member in members:
        q_len = int(member["token_count"])
        kv_len = int(member["token_begin"]) + q_len
        shapes[(q_len, kv_len)].append(member)
    if policy == "padding_dense":
        return [members]
    if policy == "homogeneous" and len(shapes) != 1:
        rendered = ", ".join(
            f"q={q_len}/kv={kv_len}" for q_len, kv_len in sorted(shapes)
        )
        raise BatchPlanError(
            "homogeneous batch contains incompatible shapes: " + rendered
        )
    return [shapes[key] for key in sorted(shapes)]


def build_batch_plan(
    scheduler_result: Mapping[str, object],
    model_nodes: Sequence[ModelNode],
    placement: Mapping[str, object],
    scheduling: Mapping[str, object] | None = None,
    *,
    model_dtype: str = "model_default",
    batch_artifact_catalog: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Instantiate compatible batches and split every operator by device."""

    scheduling = scheduling or {}
    policy = str(scheduling.get("batch_policy", "padding_dense"))
    cycle_mode = str(
        scheduling.get("batch_cycle_mode", "request_cycle_composed")
    )
    if policy not in _BATCH_POLICIES:
        raise BatchPlanError(f"unsupported batch policy: {policy}")
    if cycle_mode not in _CYCLE_MODES:
        raise BatchPlanError(f"unsupported batch cycle mode: {cycle_mode}")
    if cycle_mode == "batched_kernel_cycle" and batch_artifact_catalog is None:
        raise BatchPlanError(
            "batched_kernel_cycle requires an exact batch artifact catalog"
        )

    templates_by_phase: dict[
        str, list[tuple[int | None, str, str | None]]
    ] = defaultdict(list)
    seen: set[tuple[str, int | None, str, str | None]] = set()
    for node in model_nodes:
        if node.phase not in {Phase.PREFILL, Phase.DECODE}:
            continue
        phase = str(node.phase.value)
        signature = (
            phase,
            node.layer_id,
            node.op,
            node.attributes.get("operator_group"),
        )
        if signature not in seen:
            seen.add(signature)
            templates_by_phase[phase].append(signature[1:])

    epochs: list[dict[str, object]] = []
    subbatches: list[DeviceSubBatch] = []
    total_scheduler_selections = 0
    total_scheduler_tokens = 0
    for raw_epoch in scheduler_result["epochs"]:  # type: ignore[index]
        epoch = dict(raw_epoch)
        packed_cursor = 0
        ragged: list[RaggedSequence] = []
        phase_groups: dict[str, list[Mapping[str, object]]] = defaultdict(list)
        for raw_selection in epoch["selections"]:
            selection = dict(raw_selection)
            phase = str(selection["phase"])
            token_begin = int(selection["token_begin"])
            token_count = int(selection["token_count"])
            kv_len = token_begin + token_count
            ragged.append(
                RaggedSequence(
                    str(selection["request_id"]),
                    token_begin,
                    token_count,
                    packed_cursor,
                    kv_len,
                )
            )
            packed_cursor += token_count
            total_scheduler_selections += 1
            total_scheduler_tokens += token_count
            phase_groups[phase].append(selection)

        epoch_subbatch_ids: list[str] = []
        for phase, selections in sorted(phase_groups.items()):
            for layer_id, op, operator_group in templates_by_phase.get(phase, []):
                grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
                for selection in selections:
                    q_len = int(selection["token_count"])
                    kv_len = int(selection["token_begin"]) + q_len
                    node = _representative_node(
                        phase, layer_id, op, operator_group, q_len, kv_len
                    )
                    target = place_nodes(
                        [node], placement, active_batch=len(selections)
                    )[0].target_device
                    grouped[target].append(selection)

                for device_id, device_members in sorted(grouped.items()):
                    partitions = _partition_members(device_members, policy)
                    for partition_index, members in enumerate(partitions):
                        q_lengths = tuple(
                            int(item["token_count"]) for item in members
                        )
                        kv_lengths = tuple(
                            int(item["token_begin"]) + int(item["token_count"])
                            for item in members
                        )
                        effective = sum(q_lengths)
                        padded = max(q_lengths) * len(q_lengths)
                        effective_attention = sum(
                            q_len * kv_len
                            for q_len, kv_len in zip(q_lengths, kv_lengths)
                        )
                        padded_attention = (
                            max(q_lengths) * max(kv_lengths) * len(q_lengths)
                        )
                        cu_seqlens = [0]
                        for q_len in q_lengths:
                            cu_seqlens.append(cu_seqlens[-1] + q_len)
                        member_mappings = tuple(
                            {
                                "member_index": index,
                                "request_id": str(item["request_id"]),
                                "token_begin": int(item["token_begin"]),
                                "token_count": int(item["token_count"]),
                                "packed_begin": cu_seqlens[index],
                                "packed_end": cu_seqlens[index + 1],
                            }
                            for index, item in enumerate(members)
                        )
                        signature = _shape_signature(
                            phase=phase,
                            layer_id=layer_id,
                            op=op,
                            device_id=device_id,
                            dtype=model_dtype,
                            q_lengths=q_lengths,
                            kv_lengths=kv_lengths,
                        )
                        exact_artifact = _match_exact_artifact(
                            batch_artifact_catalog,
                            phase=phase,
                            layer_id=layer_id,
                            op=op,
                            device_id=device_id,
                            dtype=model_dtype,
                            q_lengths=q_lengths,
                            kv_lengths=kv_lengths,
                        )
                        if (
                            cycle_mode == "batched_kernel_cycle"
                            and exact_artifact is None
                        ):
                            raise BatchPlanError(
                                "no exact batched-kernel artifact for " + signature
                            )
                        request_cycle_ready = bool(
                            exact_artifact
                            and exact_artifact.get("request_cycle_ready", False)
                        )
                        performance_eligible = bool(
                            request_cycle_ready
                            and exact_artifact
                            and exact_artifact.get("performance_eligible", False)
                        )
                        subbatch_id = (
                            f"e{int(epoch['epoch_id'])}.{phase}.l{layer_id}."
                            f"{op}.{device_id}.p{partition_index}"
                        )
                        record = DeviceSubBatch(
                            subbatch_id=subbatch_id,
                            epoch_id=int(epoch["epoch_id"]),
                            phase=phase,
                            layer_id=layer_id,
                            op=op,
                            operator_group=(
                                str(operator_group)
                                if operator_group is not None
                                else None
                            ),
                            device_id=device_id,
                            dtype=model_dtype,
                            request_ids=tuple(
                                str(item["request_id"]) for item in members
                            ),
                            q_lengths=q_lengths,
                            kv_lengths=kv_lengths,
                            effective_tokens=effective,
                            padded_tokens=padded,
                            effective_attention_tokens=effective_attention,
                            padded_attention_tokens=padded_attention,
                            batch_policy=policy,
                            cycle_mode=cycle_mode,
                            shape_signature=signature,
                            cu_seqlens=tuple(cu_seqlens),
                            input_permutation=tuple(range(len(members))),
                            output_permutation=tuple(range(len(members))),
                            member_mappings=member_mappings,
                            artifact_id=(
                                str(exact_artifact["artifact_id"])
                                if exact_artifact is not None
                                else None
                            ),
                            functional_cycle_ready=True,
                            request_cycle_ready=request_cycle_ready,
                            performance_eligible=performance_eligible,
                        )
                        subbatches.append(record)
                        epoch_subbatch_ids.append(subbatch_id)
        epochs.append(
            {
                "epoch_id": int(epoch["epoch_id"]),
                "boundary_time_fs": int(epoch["boundary_time_fs"]),
                "completion_time_fs": int(epoch["completion_time_fs"]),
                "admitted_request_ids": list(
                    epoch.get("admitted_request_ids", [])
                ),
                "active_request_ids": list(epoch.get("active_request_ids", [])),
                "retired_request_ids": list(
                    epoch.get("retired_request_ids", [])
                ),
                "packed_tokens": packed_cursor,
                "ragged_sequences": [asdict(item) for item in ragged],
                "device_subbatch_ids": epoch_subbatch_ids,
            }
        )

    member_request_ids = {
        str(mapping["request_id"])
        for item in subbatches
        for mapping in item.member_mappings
    }
    selected_request_ids = {
        str(selection["request_id"])
        for epoch in scheduler_result["epochs"]  # type: ignore[index]
        for selection in epoch["selections"]
    }
    effective_tokens = sum(item.effective_tokens for item in subbatches)
    padded_tokens = sum(item.padded_tokens for item in subbatches)
    return {
        "schema_version": "hetero-batch-plan/v2",
        "request_batch_mode": str(scheduling.get("mode", "static_ragged")),
        "batch_policy": policy,
        "cycle_mode": cycle_mode,
        "epochs": epochs,
        "device_subbatches": [asdict(item) for item in subbatches],
        "effective_tokens": effective_tokens,
        "padded_tokens": padded_tokens,
        "effective_attention_tokens": sum(
            item.effective_attention_tokens for item in subbatches
        ),
        "padded_attention_tokens": sum(
            item.padded_attention_tokens for item in subbatches
        ),
        "batch_utilization": (
            effective_tokens / padded_tokens if padded_tokens else 1.0
        ),
        "performance_claim_allowed": bool(subbatches)
        and cycle_mode == "batched_kernel_cycle"
        and all(item.performance_eligible for item in subbatches),
        "conservation": {
            "scheduler_selection_count": total_scheduler_selections,
            "scheduler_token_count": total_scheduler_tokens,
            "selected_request_count": len(selected_request_ids),
            "mapped_request_count": len(member_request_ids),
            "all_selected_requests_mapped": (
                member_request_ids == selected_request_ids
            ),
            "member_fanout_is_bijective_per_subbatch": all(
                tuple(
                    mapping["member_index"] for mapping in item.member_mappings
                )
                == tuple(range(len(item.request_ids)))
                and tuple(
                    mapping["request_id"] for mapping in item.member_mappings
                )
                == item.request_ids
                for item in subbatches
            ),
        },
        "qualification_boundary": (
            "exact_batched_kernel_artifacts"
            if cycle_mode == "batched_kernel_cycle"
            else "scheduler_epoch_request_composition_only"
        ),
    }
