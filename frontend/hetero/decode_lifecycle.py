"""Fail-closed KV lifecycle audits for materialized Decode execution."""

from __future__ import annotations

from collections.abc import Mapping

from .global_memory_map import GlobalAllocation
from .ir import ModelGraph
from .model_graph import ModelSpec, RequestSpec


class DecodeLifecycleError(ValueError):
    """Raised when a Decode graph cannot preserve request-scoped KV state."""


def _only(items: list[object], description: str) -> object:
    if len(items) != 1:
        raise DecodeLifecycleError(
            f"expected exactly one {description}, observed {len(items)}"
        )
    return items[0]


def build_decode_kv_lifecycle(
    graph: ModelGraph,
    model: ModelSpec,
    request: RequestSpec,
    allocations: Mapping[str, GlobalAllocation] | None = None,
) -> dict[str, object]:
    """Validate and materialize the initial-KV -> append -> attention contract.

    ``decode_step`` starts with externally initialized KV tokens.  Every layer
    appends exactly one K and one V token, and its attention node must consume
    the new versions before any later layer may run.  Optional Global-PA
    allocations make the append range auditable without pretending that the
    addresses passed through an MMU.
    """

    if request.execution_scope != "decode_step":
        raise DecodeLifecycleError(
            "Decode lifecycle requires execution_scope=decode_step"
        )
    if request.initial_kv_length <= 0:
        raise DecodeLifecycleError("Decode lifecycle requires positive initial KV")
    graph.validate()
    nodes = list(graph.nodes)
    by_id = {node.node_id: index for index, node in enumerate(nodes)}
    final_kv_length = request.initial_kv_length + 1
    token_bytes = model.num_kv_heads * model.head_dim * model.bytes_per_element
    layers: list[dict[str, object]] = []
    used_values: set[str] = set()

    for layer_id in range(model.num_layers):
        append = _only(
            [
                node
                for node in nodes
                if node.layer_id == layer_id
                and node.phase.value == "decode"
                and node.op == "kv_append"
            ],
            f"layer {layer_id} Decode KV append",
        )
        attention = _only(
            [
                node
                for node in nodes
                if node.layer_id == layer_id
                and node.phase.value == "decode"
                and node.op == "causal_attention"
            ],
            f"layer {layer_id} Decode attention",
        )
        assert hasattr(append, "attributes") and hasattr(attention, "attributes")
        expected_attributes = {
            "q_len": 1,
            "past_kv_len": request.initial_kv_length,
            "attention_kv_len": final_kv_length,
        }
        for node in (append, attention):
            actual = {
                key: int(node.attributes.get(key, -1)) for key in expected_attributes
            }
            if actual != expected_attributes:
                raise DecodeLifecycleError(
                    f"{node.node_id} Decode shape mismatch: "
                    f"expected={expected_attributes}, actual={actual}"
                )
        if by_id[append.node_id] >= by_id[attention.node_id]:
            raise DecodeLifecycleError(
                f"layer {layer_id} attention executes before KV append"
            )
        kv_values = tuple(
            value_id
            for value_id in append.write_values
            if value_id.endswith((".k", ".v"))
        )
        if len(kv_values) != 2 or set(kv_values) - set(attention.read_values):
            raise DecodeLifecycleError(
                f"layer {layer_id} attention does not consume appended K/V"
            )
        if used_values.intersection(kv_values):
            raise DecodeLifecycleError("Decode layers alias the same logical KV values")
        used_values.update(kv_values)

        append_offset = request.initial_kv_length * token_bytes
        required_bytes = final_kv_length * token_bytes
        value_records: list[dict[str, object]] = []
        for value_id in sorted(kv_values):
            record: dict[str, object] = {
                "value_id": value_id,
                "initial_valid_range": {
                    "offset_bytes": 0,
                    "size_bytes": append_offset,
                    "token_begin": 0,
                    "token_end_exclusive": request.initial_kv_length,
                },
                "append_range": {
                    "offset_bytes": append_offset,
                    "size_bytes": token_bytes,
                    "token_begin": request.initial_kv_length,
                    "token_end_exclusive": final_kv_length,
                },
                "final_valid_bytes": required_bytes,
            }
            if allocations is not None:
                allocation = allocations.get(value_id)
                if allocation is None:
                    raise DecodeLifecycleError(
                        f"Decode KV value has no Global PA allocation: {value_id}"
                    )
                if allocation.size_bytes < required_bytes:
                    raise DecodeLifecycleError(
                        f"Decode KV allocation is too small: {value_id}"
                    )
                record["global_pa"] = {
                    "base_address": allocation.base_address,
                    "end_address_exclusive": allocation.end_address_exclusive,
                    "append_address": allocation.base_address + append_offset,
                    "allocation_size_bytes": allocation.size_bytes,
                    "address_semantics": "global_pa_identity_untranslated",
                }
            value_records.append(record)
        layers.append(
            {
                "layer_id": layer_id,
                "append_task_id": f"task.{append.node_id}",
                "attention_task_id": f"task.{attention.node_id}",
                "q_len": 1,
                "initial_kv_length": request.initial_kv_length,
                "final_kv_length": final_kv_length,
                "bytes_per_kv_token_per_tensor": token_bytes,
                "values": value_records,
                "append_before_attention": True,
            }
        )

    finish = _only(
        [node for node in nodes if node.op == "request_finish"], "request finish"
    )
    release = _only([node for node in nodes if node.op == "kv_release"], "KV release")
    if by_id[finish.node_id] >= by_id[release.node_id]:
        raise DecodeLifecycleError("KV release must occur after request finish")

    return {
        "schema_version": "hetero-decode-kv-lifecycle/v1",
        "request_id": request.request_id,
        "execution_scope": request.execution_scope,
        "model_spec_name": model.name,
        "layer_count": model.num_layers,
        "batch_size": 1,
        "q_len": 1,
        "initial_kv_length": request.initial_kv_length,
        "appended_tokens": 1,
        "final_kv_length": final_kv_length,
        "bytes_per_kv_token_per_tensor": token_bytes,
        "total_initial_valid_kv_bytes": (
            2 * model.num_layers * request.initial_kv_length * token_bytes
        ),
        "total_appended_kv_bytes": 2 * model.num_layers * token_bytes,
        "kv_release_after_request_finish": True,
        "all_layers_append_before_attention": True,
        "global_pa_bound": allocations is not None,
        "layers": layers,
        "performance_eligible": False,
    }


def build_decode_loop_kv_lifecycle(
    graph: ModelGraph,
    model: ModelSpec,
    request: RequestSpec,
    allocations: Mapping[str, GlobalAllocation] | None = None,
) -> dict[str, object]:
    """Validate a multi-token autoregressive Decode loop.

    The first step consumes an externally supplied token.  Every later step
    must consume the token produced by the preceding sampling task.  Each
    layer appends one K and one V token per step into a stable logical value,
    and attention may consume the new version only after that append.
    """

    if request.execution_scope != "decode_loop":
        raise DecodeLifecycleError(
            "Decode loop lifecycle requires execution_scope=decode_loop"
        )
    if request.initial_kv_length <= 0 or request.output_length <= 0:
        raise DecodeLifecycleError(
            "Decode loop lifecycle requires positive initial KV and output length"
        )
    graph.validate()
    nodes = list(graph.nodes)
    by_id = {node.node_id: index for index, node in enumerate(nodes)}
    token_bytes = model.num_kv_heads * model.head_dim * model.bytes_per_element
    final_kv_length = request.initial_kv_length + request.output_length
    required_bytes = final_kv_length * token_bytes
    step_records: list[dict[str, object]] = []
    layer_values: dict[int, tuple[str, str]] = {}
    all_kv_values: set[str] = set()

    for step_id in range(request.output_length):
        past_kv_length = request.initial_kv_length + step_id
        attention_kv_length = past_kv_length + 1
        embedding = _only(
            [
                node
                for node in nodes
                if node.step_id == step_id
                and node.phase.value == "decode"
                and node.op == "token_embedding"
            ],
            f"Decode step {step_id} embedding",
        )
        sampling = _only(
            [
                node
                for node in nodes
                if node.step_id == step_id
                and node.phase.value == "decode"
                and node.op == "sampling"
            ],
            f"Decode step {step_id} sampling",
        )
        if step_id == 0:
            expected_token = f"{request.request_id}.decode_token_id"
            token_source = "external_decode_token"
        else:
            previous_sampling = _only(
                [
                    node
                    for node in nodes
                    if node.step_id == step_id - 1
                    and node.phase.value == "decode"
                    and node.op == "sampling"
                ],
                f"Decode step {step_id - 1} sampling",
            )
            expected_token = f"{request.request_id}.token.{step_id - 1}"
            token_source = f"task.{previous_sampling.node_id}"
            if by_id[previous_sampling.node_id] >= by_id[embedding.node_id]:
                raise DecodeLifecycleError(
                    f"Decode step {step_id} begins before prior sampling"
                )
        if expected_token not in embedding.read_values:
            raise DecodeLifecycleError(
                f"Decode step {step_id} embedding does not consume {expected_token}"
            )

        current_layers: list[dict[str, object]] = []
        for layer_id in range(model.num_layers):
            append = _only(
                [
                    node
                    for node in nodes
                    if node.layer_id == layer_id
                    and node.step_id == step_id
                    and node.phase.value == "decode"
                    and node.op == "kv_append"
                ],
                f"layer {layer_id} Decode step {step_id} KV append",
            )
            attention = _only(
                [
                    node
                    for node in nodes
                    if node.layer_id == layer_id
                    and node.step_id == step_id
                    and node.phase.value == "decode"
                    and node.op == "causal_attention"
                ],
                f"layer {layer_id} Decode step {step_id} attention",
            )
            expected_attributes = {
                "q_len": 1,
                "past_kv_len": past_kv_length,
                "attention_kv_len": attention_kv_length,
            }
            for node in (append, attention):
                actual = {
                    key: int(node.attributes.get(key, -1))
                    for key in expected_attributes
                }
                if actual != expected_attributes:
                    raise DecodeLifecycleError(
                        f"{node.node_id} Decode shape mismatch: "
                        f"expected={expected_attributes}, actual={actual}"
                    )
            if by_id[append.node_id] >= by_id[attention.node_id]:
                raise DecodeLifecycleError(
                    f"layer {layer_id} step {step_id} attention precedes KV append"
                )
            kv_values = tuple(
                sorted(
                    value_id
                    for value_id in append.write_values
                    if value_id.endswith((".k", ".v"))
                )
            )
            if len(kv_values) != 2 or set(kv_values) - set(attention.read_values):
                raise DecodeLifecycleError(
                    f"layer {layer_id} step {step_id} attention misses appended K/V"
                )
            previous_values = layer_values.get(layer_id)
            if previous_values is None:
                if all_kv_values.intersection(kv_values):
                    raise DecodeLifecycleError(
                        "Decode layers alias the same logical KV values"
                    )
                layer_values[layer_id] = kv_values
                all_kv_values.update(kv_values)
            elif previous_values != kv_values:
                raise DecodeLifecycleError(
                    f"layer {layer_id} changes logical KV values across steps"
                )

            append_offset = past_kv_length * token_bytes
            value_records: list[dict[str, object]] = []
            for value_id in kv_values:
                value_record: dict[str, object] = {
                    "value_id": value_id,
                    "expected_input_version": step_id,
                    "committed_output_version": step_id + 1,
                    "append_range": {
                        "offset_bytes": append_offset,
                        "size_bytes": token_bytes,
                        "token_begin": past_kv_length,
                        "token_end_exclusive": attention_kv_length,
                    },
                    "valid_bytes_after_append": attention_kv_length * token_bytes,
                }
                if allocations is not None:
                    allocation = allocations.get(value_id)
                    if allocation is None:
                        raise DecodeLifecycleError(
                            f"Decode KV value has no Global PA allocation: {value_id}"
                        )
                    if allocation.size_bytes < required_bytes:
                        raise DecodeLifecycleError(
                            f"Decode KV allocation is too small: {value_id}"
                        )
                    value_record["global_pa"] = {
                        "base_address": allocation.base_address,
                        "end_address_exclusive": allocation.end_address_exclusive,
                        "append_address": allocation.base_address + append_offset,
                        "allocation_size_bytes": allocation.size_bytes,
                        "address_semantics": "global_pa_identity_untranslated",
                    }
                value_records.append(value_record)
            current_layers.append(
                {
                    "layer_id": layer_id,
                    "append_task_id": f"task.{append.node_id}",
                    "attention_task_id": f"task.{attention.node_id}",
                    "values": value_records,
                    "append_before_attention": True,
                }
            )
        step_records.append(
            {
                "step_id": step_id,
                "q_len": 1,
                "past_kv_length": past_kv_length,
                "attention_kv_length": attention_kv_length,
                "embedding_task_id": f"task.{embedding.node_id}",
                "sampling_task_id": f"task.{sampling.node_id}",
                "token_input_value_id": expected_token,
                "token_source": token_source,
                "layers": current_layers,
            }
        )

    finish = _only(
        [node for node in nodes if node.op == "request_finish"], "request finish"
    )
    release = _only([node for node in nodes if node.op == "kv_release"], "KV release")
    final_sampling_id = str(step_records[-1]["sampling_task_id"])[5:]
    if by_id[final_sampling_id] >= by_id[finish.node_id]:
        raise DecodeLifecycleError("request finish must follow final sampling")
    if by_id[finish.node_id] >= by_id[release.node_id]:
        raise DecodeLifecycleError("KV release must occur after request finish")

    aggregate_layers: list[dict[str, object]] = []
    for layer_id in range(model.num_layers):
        values: list[dict[str, object]] = []
        for value_id in layer_values[layer_id]:
            value_record: dict[str, object] = {
                "value_id": value_id,
                "initial_valid_range": {
                    "offset_bytes": 0,
                    "size_bytes": request.initial_kv_length * token_bytes,
                    "token_begin": 0,
                    "token_end_exclusive": request.initial_kv_length,
                },
                "final_valid_bytes": required_bytes,
                "final_committed_version": request.output_length,
            }
            if allocations is not None:
                allocation = allocations[value_id]
                value_record["global_pa"] = {
                    "base_address": allocation.base_address,
                    "end_address_exclusive": allocation.end_address_exclusive,
                    "allocation_size_bytes": allocation.size_bytes,
                    "address_semantics": "global_pa_identity_untranslated",
                }
            values.append(value_record)
        aggregate_layers.append({"layer_id": layer_id, "values": values})

    return {
        "schema_version": "hetero-decode-kv-lifecycle/v2",
        "request_id": request.request_id,
        "execution_scope": request.execution_scope,
        "model_spec_name": model.name,
        "layer_count": model.num_layers,
        "batch_size": 1,
        "q_len": 1,
        "generated_tokens": request.output_length,
        "initial_kv_length": request.initial_kv_length,
        "appended_tokens": request.output_length,
        "final_kv_length": final_kv_length,
        "bytes_per_kv_token_per_tensor": token_bytes,
        "total_initial_valid_kv_bytes": (
            2 * model.num_layers * request.initial_kv_length * token_bytes
        ),
        "total_appended_kv_bytes": (
            2 * model.num_layers * request.output_length * token_bytes
        ),
        "kv_release_after_request_finish": True,
        "request_finish_after_final_sampling": True,
        "all_steps_autoregressive": True,
        "all_layers_append_before_attention": True,
        "global_pa_bound": allocations is not None,
        "steps": step_records,
        "layers": aggregate_layers,
        "performance_eligible": False,
    }
