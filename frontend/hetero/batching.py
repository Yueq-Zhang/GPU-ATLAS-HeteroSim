"""Static ragged, continuous and device sub-batch planning."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Mapping, Sequence

from .ir import ModelNode, NodeKind, Phase
from .placement import place_nodes


@dataclass(frozen=True, slots=True)
class RaggedSequence:
    request_id: str
    token_begin: int
    token_count: int
    packed_begin: int
    causal_kv_length: int


@dataclass(frozen=True, slots=True)
class DeviceSubBatch:
    epoch_id: int
    phase: str
    layer_id: int | None
    op: str
    operator_group: str | None
    device_id: str
    request_ids: tuple[str, ...]
    q_lengths: tuple[int, ...]
    kv_lengths: tuple[int, ...]
    effective_tokens: int
    padded_tokens: int


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
        # Placement matching only consumes phase/layer/group/length.
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


def build_batch_plan(
    scheduler_result: Mapping[str, object],
    model_nodes: Sequence[ModelNode],
    placement: Mapping[str, object],
) -> dict[str, object]:
    """Instantiate ragged epochs and split each operator by target device."""

    templates: list[tuple[int | None, str, str | None]] = []
    seen: set[tuple[int | None, str, str | None]] = set()
    for node in model_nodes:
        if node.phase not in {Phase.PREFILL, Phase.DECODE}:
            continue
        signature = (
            node.layer_id,
            node.op,
            node.attributes.get("operator_group"),
        )
        if signature not in seen:
            seen.add(signature)
            templates.append(signature)

    epochs: list[dict[str, object]] = []
    subbatches: list[DeviceSubBatch] = []
    for raw_epoch in scheduler_result["epochs"]:  # type: ignore[index]
        epoch = dict(raw_epoch)
        packed_cursor = 0
        ragged: list[RaggedSequence] = []
        phase_groups: dict[str, list[Mapping[str, object]]] = defaultdict(list)
        for selection in epoch["selections"]:
            selection = dict(selection)
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
            phase_groups[phase].append(selection)

        epoch_subbatch_ids: list[str] = []
        for phase, selections in sorted(phase_groups.items()):
            for layer_id, op, operator_group in templates:
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
                for device_id, members in sorted(grouped.items()):
                    q_lengths = tuple(int(item["token_count"]) for item in members)
                    kv_lengths = tuple(
                        int(item["token_begin"]) + int(item["token_count"])
                        for item in members
                    )
                    effective = sum(q_lengths)
                    padded = max(q_lengths) * len(q_lengths)
                    record = DeviceSubBatch(
                        int(epoch["epoch_id"]),
                        phase,
                        layer_id,
                        op,
                        str(operator_group) if operator_group is not None else None,
                        device_id,
                        tuple(str(item["request_id"]) for item in members),
                        q_lengths,
                        kv_lengths,
                        effective,
                        padded,
                    )
                    subbatches.append(record)
                    epoch_subbatch_ids.append(
                        f"e{record.epoch_id}.{phase}.l{layer_id}.{op}.{device_id}"
                    )
        epochs.append(
            {
                "epoch_id": int(epoch["epoch_id"]),
                "boundary_time_fs": int(epoch["boundary_time_fs"]),
                "completion_time_fs": int(epoch["completion_time_fs"]),
                "packed_tokens": packed_cursor,
                "ragged_sequences": [asdict(item) for item in ragged],
                "device_subbatch_ids": epoch_subbatch_ids,
            }
        )
    return {
        "schema_version": "hetero-batch-plan/v1",
        "epochs": epochs,
        "device_subbatches": [asdict(item) for item in subbatches],
        "effective_tokens": sum(item.effective_tokens for item in subbatches),
        "padded_tokens": sum(item.padded_tokens for item in subbatches),
    }
