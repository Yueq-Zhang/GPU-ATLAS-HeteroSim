from pathlib import Path

import pytest

from frontend.hetero.ir import ModelNode, NodeKind, Phase
from frontend.hetero.model_graph import ModelSpec
from frontend.hetero.runtime_task_model import (
    RuntimeTaskModelCatalog,
    RuntimeTaskModelError,
)


CATALOG = Path(
    "configs/hetero/runtime_tasks/"
    "tinyllama_prefill_layer0_bs1_ctx16_uncalibrated.json"
)


def _model() -> ModelSpec:
    return ModelSpec(
        "TinyLlama-1.1B",
        2048,
        5632,
        1,
        32,
        4,
        64,
        32000,
        dtype="fp16",
        checkpoint_revision="fe8a4ea1ffedaf415f4da2f062534de366a451e6",
    )


def _node(op: str, *, context: int = 16) -> ModelNode:
    return ModelNode(
        f"n.{op}",
        NodeKind.STATE,
        op,
        Phase.PREFILL if op == "kv_append" else Phase.CONTROL,
        0,
        0,
        attributes={
            "batch_size": 1,
            "context_length": context,
            "q_len": context,
            "attention_kv_len": context,
        },
    )


def test_kv_append_cycle_contract_counts_full_value_bytes() -> None:
    catalog = RuntimeTaskModelCatalog.load(CATALOG)
    result = catalog.estimate(_node("kv_append"), _model())
    assert catalog.calibrated is False
    assert result.memory_read_bytes == 16_384
    assert result.memory_write_bytes == 16_384
    assert result.memory_transactions == 512
    assert result.cycles == 520
    assert result.duration_fs == 346_666_667


def test_control_and_metadata_contracts_are_explicit() -> None:
    catalog = RuntimeTaskModelCatalog.load(CATALOG)
    start = catalog.estimate(_node("request_start"), _model())
    allocate = catalog.estimate(_node("kv_allocate"), _model())
    assert start.cycles == 1
    assert start.memory_transactions == 0
    assert allocate.memory_read_bytes == 64
    assert allocate.memory_write_bytes == 128
    assert allocate.memory_transactions == 3
    assert allocate.cycles == 19


def test_runtime_contract_rejects_context_or_model_revision_change() -> None:
    catalog = RuntimeTaskModelCatalog.load(CATALOG)
    with pytest.raises(RuntimeTaskModelError, match="shape-locked"):
        catalog.estimate(_node("kv_append", context=32), _model())
    changed = _model()
    object.__setattr__(changed, "checkpoint_revision", "different")
    with pytest.raises(RuntimeTaskModelError, match="shape-locked"):
        catalog.estimate(_node("kv_append"), changed)
