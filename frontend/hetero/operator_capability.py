"""Auditable operator modeling and qualification coverage catalog.

The catalog deliberately separates implementation, testing and performance
eligibility.  A captured or modeled operator is not automatically qualified,
and an exact-shape qualification cannot be extrapolated to another model,
batch, context length or tensor shape.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


class OperatorCapabilityError(ValueError):
    """Raised when a capability catalog is malformed or incompatible."""


_BACKENDS = {
    "accel_sim",
    "runtime_cycle",
    "runtime_live_ramulator2",
    "runtime_state",
    "event_marker",
    "none",
}
_IMPLEMENTATION = {"implemented", "partial", "planned"}
_TEST_STATUS = {
    "request_cycle_qualified",
    "runtime_cycle_contract_tested",
    "runtime_semantics_tested",
    "graph_causality_tested",
    "not_tested",
}
_CYCLE_FIDELITY = {
    "request_cycle",
    "runtime_cycle_contract",
    "runtime_state",
    "event_only",
    "unmodeled",
}
_SHAPE_POLICIES = {"exact_only", "not_applicable"}


def _nonempty(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise OperatorCapabilityError(f"{field} must be a non-empty string")
    return value


def _positive(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise OperatorCapabilityError(f"{field} must be a positive integer")
    return value


def _unsigned(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise OperatorCapabilityError(f"{field} must be an unsigned integer")
    return value


@dataclass(frozen=True, slots=True)
class ModelCapabilityContract:
    contract_id: str
    model_spec_name: str
    checkpoint_revision: str
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    vocab_size: int
    dtype: str

    @classmethod
    def from_dict(cls, contract_id: str, raw: Mapping[str, object]) -> "ModelCapabilityContract":
        return cls(
            contract_id=contract_id,
            model_spec_name=_nonempty(raw.get("model_spec_name"), "model_spec_name"),
            checkpoint_revision=_nonempty(
                raw.get("checkpoint_revision"), "checkpoint_revision"
            ),
            hidden_size=_positive(raw.get("hidden_size"), "hidden_size"),
            intermediate_size=_positive(
                raw.get("intermediate_size"), "intermediate_size"
            ),
            num_attention_heads=_positive(
                raw.get("num_attention_heads"), "num_attention_heads"
            ),
            num_kv_heads=_positive(raw.get("num_kv_heads"), "num_kv_heads"),
            head_dim=_positive(raw.get("head_dim"), "head_dim"),
            vocab_size=_positive(raw.get("vocab_size"), "vocab_size"),
            dtype=_nonempty(raw.get("dtype"), "dtype"),
        )

    def assert_matches(self, model: object) -> None:
        actual = {
            "model_spec_name": getattr(model, "name", None),
            "checkpoint_revision": getattr(model, "checkpoint_revision", None),
            "hidden_size": getattr(model, "hidden_size", None),
            "intermediate_size": getattr(model, "intermediate_size", None),
            "num_attention_heads": getattr(model, "num_attention_heads", None),
            "num_kv_heads": getattr(model, "num_kv_heads", None),
            "head_dim": getattr(model, "head_dim", None),
            "vocab_size": getattr(model, "vocab_size", None),
            "dtype": getattr(model, "dtype", None),
        }
        expected = {
            "model_spec_name": self.model_spec_name,
            "checkpoint_revision": self.checkpoint_revision,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_attention_heads": self.num_attention_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "vocab_size": self.vocab_size,
            "dtype": self.dtype,
        }
        mismatches = {
            key: {"expected": expected[key], "actual": actual[key]}
            for key in expected
            if expected[key] != actual[key]
        }
        if mismatches:
            raise OperatorCapabilityError(
                f"model contract {self.contract_id} mismatch: {mismatches}"
            )


@dataclass(frozen=True, slots=True)
class ShapeCapabilityContract:
    contract_id: str
    model_contract_ref: str
    phase: str
    layer_id: int
    batch_size: int
    context_length: int
    q_len: int
    kv_length: int

    @classmethod
    def from_dict(cls, contract_id: str, raw: Mapping[str, object]) -> "ShapeCapabilityContract":
        phase = _nonempty(raw.get("phase"), "phase")
        if phase not in {"prefill", "decode_step", "control"}:
            raise OperatorCapabilityError(f"unsupported phase: {phase}")
        return cls(
            contract_id=contract_id,
            model_contract_ref=_nonempty(
                raw.get("model_contract_ref"), "model_contract_ref"
            ),
            phase=phase,
            layer_id=_unsigned(raw.get("layer_id"), "layer_id"),
            batch_size=_positive(raw.get("batch_size"), "batch_size"),
            context_length=_positive(raw.get("context_length"), "context_length"),
            q_len=_positive(raw.get("q_len"), "q_len"),
            kv_length=_positive(raw.get("kv_length"), "kv_length"),
        )

    def matches(
        self,
        *,
        phase: str,
        layer_id: int,
        batch_size: int,
        context_length: int,
        q_len: int,
        kv_length: int,
    ) -> bool:
        return (
            self.phase == phase
            and self.layer_id == layer_id
            and self.batch_size == batch_size
            and self.context_length == context_length
            and self.q_len == q_len
            and self.kv_length == kv_length
        )


@dataclass(frozen=True, slots=True)
class OperatorCapability:
    operator_type: str
    instances_in_reference_graph: int
    task_kind: str
    backend_kind: str
    implementation_status: str
    test_status: str
    cycle_fidelity: str
    request_cycle_ready: bool
    performance_eligible: bool
    shape_policy: str
    shape_contract_refs: tuple[str, ...]
    artifact_refs: tuple[str, ...]
    notes: str


@dataclass(frozen=True, slots=True)
class OperatorCapabilityCatalog:
    source_path: Path
    model_contracts: Mapping[str, ModelCapabilityContract]
    shape_contracts: Mapping[str, ShapeCapabilityContract]
    operators: Mapping[str, OperatorCapability]

    @classmethod
    def load(cls, path: Path) -> "OperatorCapabilityCatalog":
        path = path.resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "hetero-operator-capability-catalog/v1":
            raise OperatorCapabilityError("invalid capability catalog schema_version")
        raw_models = payload.get("model_contracts")
        raw_shapes = payload.get("shape_contracts")
        raw_operators = payload.get("operator_types")
        if not isinstance(raw_models, Mapping) or not raw_models:
            raise OperatorCapabilityError("model_contracts must be a non-empty object")
        if not isinstance(raw_shapes, Mapping) or not raw_shapes:
            raise OperatorCapabilityError("shape_contracts must be a non-empty object")
        if not isinstance(raw_operators, list) or not raw_operators:
            raise OperatorCapabilityError("operator_types must be a non-empty array")
        models = {
            str(key): ModelCapabilityContract.from_dict(str(key), value)
            for key, value in raw_models.items()
            if isinstance(value, Mapping)
        }
        if len(models) != len(raw_models):
            raise OperatorCapabilityError("model contract entries must be objects")
        shapes = {
            str(key): ShapeCapabilityContract.from_dict(str(key), value)
            for key, value in raw_shapes.items()
            if isinstance(value, Mapping)
        }
        if len(shapes) != len(raw_shapes):
            raise OperatorCapabilityError("shape contract entries must be objects")
        for shape in shapes.values():
            if shape.model_contract_ref not in models:
                raise OperatorCapabilityError(
                    f"unknown model contract: {shape.model_contract_ref}"
                )
        operators: dict[str, OperatorCapability] = {}
        for index, raw in enumerate(raw_operators):
            if not isinstance(raw, Mapping):
                raise OperatorCapabilityError(f"operator_types[{index}] must be an object")
            operator_type = _nonempty(raw.get("operator_type"), "operator_type")
            if operator_type in operators:
                raise OperatorCapabilityError(f"duplicate operator_type: {operator_type}")
            backend = _nonempty(raw.get("backend_kind"), "backend_kind")
            implementation = _nonempty(
                raw.get("implementation_status"), "implementation_status"
            )
            test_status = _nonempty(raw.get("test_status"), "test_status")
            cycle_fidelity = _nonempty(raw.get("cycle_fidelity"), "cycle_fidelity")
            shape_policy = _nonempty(raw.get("shape_policy"), "shape_policy")
            if backend not in _BACKENDS or implementation not in _IMPLEMENTATION:
                raise OperatorCapabilityError(f"invalid capability state for {operator_type}")
            if test_status not in _TEST_STATUS or cycle_fidelity not in _CYCLE_FIDELITY:
                raise OperatorCapabilityError(f"invalid test/fidelity state for {operator_type}")
            if shape_policy not in _SHAPE_POLICIES:
                raise OperatorCapabilityError(f"invalid shape policy for {operator_type}")
            refs = tuple(str(item) for item in raw.get("shape_contract_refs", []))
            if any(ref not in shapes for ref in refs):
                raise OperatorCapabilityError(f"unknown shape contract for {operator_type}")
            request_ready = raw.get("request_cycle_ready")
            performance_eligible = raw.get("performance_eligible")
            if not isinstance(request_ready, bool) or not isinstance(performance_eligible, bool):
                raise OperatorCapabilityError("readiness fields must be boolean")
            if request_ready and (
                cycle_fidelity != "request_cycle"
                or test_status != "request_cycle_qualified"
                or not refs
            ):
                raise OperatorCapabilityError(
                    f"request-cycle readiness is unsupported for {operator_type}"
                )
            operators[operator_type] = OperatorCapability(
                operator_type=operator_type,
                instances_in_reference_graph=_positive(
                    raw.get("instances_in_reference_graph"),
                    "instances_in_reference_graph",
                ),
                task_kind=_nonempty(raw.get("task_kind"), "task_kind"),
                backend_kind=backend,
                implementation_status=implementation,
                test_status=test_status,
                cycle_fidelity=cycle_fidelity,
                request_cycle_ready=request_ready,
                performance_eligible=performance_eligible,
                shape_policy=shape_policy,
                shape_contract_refs=refs,
                artifact_refs=tuple(str(item) for item in raw.get("artifact_refs", [])),
                notes=str(raw.get("notes", "")),
            )
        return cls(path, models, shapes, operators)

    def require_exact_shape(
        self,
        operator_type: str,
        model: object,
        *,
        phase: str,
        layer_id: int,
        batch_size: int,
        context_length: int,
        q_len: int,
        kv_length: int,
        require_request_cycle_ready: bool = False,
    ) -> OperatorCapability:
        capability = self.operators.get(operator_type)
        if capability is None:
            raise OperatorCapabilityError(f"operator type is not cataloged: {operator_type}")
        if require_request_cycle_ready and not capability.request_cycle_ready:
            raise OperatorCapabilityError(
                f"operator is not request-cycle qualified: {operator_type}"
            )
        for ref in capability.shape_contract_refs:
            shape = self.shape_contracts[ref]
            if shape.matches(
                phase=phase,
                layer_id=layer_id,
                batch_size=batch_size,
                context_length=context_length,
                q_len=q_len,
                kv_length=kv_length,
            ):
                self.model_contracts[shape.model_contract_ref].assert_matches(model)
                return capability
        raise OperatorCapabilityError(
            f"no exact tested shape for {operator_type}: "
            f"phase={phase}, layer={layer_id}, batch={batch_size}, "
            f"context={context_length}, q_len={q_len}, kv_length={kv_length}"
        )

    def summary(self) -> dict[str, object]:
        return {
            "schema_version": "hetero-operator-capability-summary/v1",
            "operator_type_count": len(self.operators),
            "reference_graph_instance_count": sum(
                item.instances_in_reference_graph for item in self.operators.values()
            ),
            "implemented_operator_types": sum(
                item.implementation_status == "implemented"
                for item in self.operators.values()
            ),
            "request_cycle_ready_operator_types": sum(
                item.request_cycle_ready for item in self.operators.values()
            ),
            "performance_eligible_operator_types": sum(
                item.performance_eligible for item in self.operators.values()
            ),
            "by_test_status": {
                status: sum(item.test_status == status for item in self.operators.values())
                for status in sorted(_TEST_STATUS)
            },
        }
