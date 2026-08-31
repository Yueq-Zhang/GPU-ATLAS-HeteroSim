"""Auditable, shape-locked cycle contracts for non-SM runtime tasks.

These models close control/state timing plumbing without pretending to be
hardware calibrated.  They are appropriate for request markers, KV allocator
metadata and copy-engine KV writes that do not produce an SM instruction trace.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .ir import ModelNode
from .model_graph import ModelSpec


class RuntimeTaskModelError(ValueError):
    """Raised when a runtime task contract is invalid or shape-incompatible."""


def _positive(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise RuntimeTaskModelError(f"{field} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class RuntimeTaskEstimate:
    cycles: int
    duration_fs: int
    memory_read_bytes: int
    memory_write_bytes: int
    memory_transactions: int
    formula: str


@dataclass(frozen=True, slots=True)
class RuntimeTaskContract:
    operator: str
    model_kind: str
    fixed_cycles: int
    cycles_per_transaction: int
    transaction_bytes: int
    metadata_read_bytes: int
    metadata_write_bytes: int

    @classmethod
    def from_dict(cls, operator: str, raw: Mapping[str, object]) -> "RuntimeTaskContract":
        model_kind = str(raw.get("model_kind", ""))
        if model_kind not in {"fixed_control", "metadata_state", "kv_copy_engine"}:
            raise RuntimeTaskModelError(f"unsupported runtime model for {operator}")
        fixed_cycles = _positive(raw.get("fixed_cycles"), "fixed_cycles")
        transaction_bytes = _positive(
            raw.get("transaction_bytes", 64), "transaction_bytes"
        )
        cycles_per_transaction = _positive(
            raw.get("cycles_per_transaction", 1), "cycles_per_transaction"
        )
        reads = raw.get("metadata_read_bytes", 0)
        writes = raw.get("metadata_write_bytes", 0)
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in (reads, writes)
        ):
            raise RuntimeTaskModelError("metadata byte counts must be unsigned")
        return cls(
            operator,
            model_kind,
            fixed_cycles,
            cycles_per_transaction,
            transaction_bytes,
            int(reads),
            int(writes),
        )

    def estimate(self, node: ModelNode, model: ModelSpec, clock_hz: int) -> RuntimeTaskEstimate:
        read_bytes = self.metadata_read_bytes
        write_bytes = self.metadata_write_bytes
        formula = self.model_kind
        if self.model_kind == "kv_copy_engine":
            batch = int(node.attributes.get("batch_size", 1))
            q_len = int(node.attributes.get("q_len", 1))
            one_kv = (
                batch
                * q_len
                * model.num_kv_heads
                * model.head_dim
                * model.bytes_per_element
            )
            read_bytes += 2 * one_kv
            write_bytes += 2 * one_kv
            formula = "fixed_cycles + ceil((K_read+V_read+K_write+V_write)/transaction_bytes)*cycles_per_transaction"
        total_bytes = read_bytes + write_bytes
        transactions = (
            (total_bytes + self.transaction_bytes - 1) // self.transaction_bytes
            if total_bytes
            else 0
        )
        cycles = self.fixed_cycles + transactions * self.cycles_per_transaction
        duration_fs = (cycles * 1_000_000_000_000_000 + clock_hz - 1) // clock_hz
        return RuntimeTaskEstimate(
            cycles=cycles,
            duration_fs=duration_fs,
            memory_read_bytes=read_bytes,
            memory_write_bytes=write_bytes,
            memory_transactions=transactions,
            formula=formula,
        )


@dataclass(frozen=True, slots=True)
class RuntimeTaskModelCatalog:
    source_path: Path
    catalog_id: str
    clock_hz: int
    parameter_source: str
    calibrated: bool
    model_spec_name: str
    checkpoint_revision: str
    batch_size: int
    context_length: int
    contracts: Mapping[str, RuntimeTaskContract]

    @classmethod
    def load(cls, path: Path) -> "RuntimeTaskModelCatalog":
        path = path.resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "hetero-runtime-task-model/v1":
            raise RuntimeTaskModelError("invalid runtime task model schema_version")
        shape = payload.get("shape_contract")
        models = payload.get("models")
        if not isinstance(shape, Mapping) or not isinstance(models, Mapping) or not models:
            raise RuntimeTaskModelError("shape_contract and models are required")
        calibrated = payload.get("calibrated")
        if not isinstance(calibrated, bool):
            raise RuntimeTaskModelError("calibrated must be boolean")
        return cls(
            source_path=path,
            catalog_id=str(payload.get("catalog_id", "")),
            clock_hz=_positive(payload.get("clock_hz"), "clock_hz"),
            parameter_source=str(payload.get("parameter_source", "")),
            calibrated=calibrated,
            model_spec_name=str(shape.get("model_spec_name", "")),
            checkpoint_revision=str(shape.get("checkpoint_revision", "")),
            batch_size=_positive(shape.get("batch_size"), "batch_size"),
            context_length=_positive(shape.get("context_length"), "context_length"),
            contracts={
                str(operator): RuntimeTaskContract.from_dict(str(operator), raw)
                for operator, raw in models.items()
                if isinstance(raw, Mapping)
            },
        )

    def contract_for(self, node: ModelNode) -> RuntimeTaskContract:
        contract = self.contracts.get(node.op)
        if contract is None:
            raise RuntimeTaskModelError(f"runtime task is not modeled: {node.op}")
        return contract

    def estimate(self, node: ModelNode, model: ModelSpec) -> RuntimeTaskEstimate:
        contract = self.contract_for(node)
        actual_batch = int(node.attributes.get("batch_size", 1))
        actual_context = int(
            node.attributes.get(
                "context_length",
                node.attributes.get("source_q_len", node.attributes.get("q_len", 1)),
            )
        )
        expected = (
            self.model_spec_name,
            self.checkpoint_revision,
            self.batch_size,
            self.context_length,
        )
        actual = (
            model.name,
            model.checkpoint_revision,
            actual_batch,
            actual_context,
        )
        if actual != expected:
            raise RuntimeTaskModelError(
                f"shape-locked runtime task mismatch for {node.node_id}: "
                f"expected={expected}, actual={actual}"
            )
        return contract.estimate(node, model, self.clock_hz)
