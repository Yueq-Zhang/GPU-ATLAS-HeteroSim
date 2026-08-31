"""Deterministic operator-cycle contracts for complete Prefill deployment.

These contracts are neither Roofline estimates nor instruction traces.  They
describe explicit tiled work schedules whose cycles are replayed by the live
request-cycle runtime.  Every result carries an unqualified fidelity label so
that deployment coverage cannot be mistaken for calibrated performance.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

from .ir import ModelNode
from .model_graph import ModelSpec
from .online_operator_runtime import OnlineDispatchSpec
from .operator_artifact import (
    OperatorArtifactCatalog,
    OperatorArtifactError,
    OperatorArtifactManifest,
)


class PrefillCycleArtifactError(ValueError):
    """Raised when an operator-cycle catalog is incomplete or inconsistent."""


def _ceil_div(numerator: int, denominator: int) -> int:
    if numerator < 0 or denominator <= 0:
        raise PrefillCycleArtifactError("cycle formula operands are invalid")
    return (numerator + denominator - 1) // denominator


def _positive(profile: Mapping[str, object], field: str) -> int:
    value = profile.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise PrefillCycleArtifactError(f"profile.{field} must be positive")
    return value


@dataclass(frozen=True, slots=True)
class CycleTaskPlan:
    task_id: str
    backend_id: str
    device_id: str
    device_clock_hz: int
    native_compute_cycles: int
    global_compute_cycles: int
    formula: Mapping[str, object]
    fidelity: Mapping[str, object]
    artifact: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class PrefillCycleCatalog:
    source_path: Path
    payload: Mapping[str, object]
    content_sha256: str

    @classmethod
    def load(cls, path: Path) -> "PrefillCycleCatalog":
        try:
            raw_bytes = path.read_bytes()
            payload = json.loads(raw_bytes.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise PrefillCycleArtifactError(f"failed to load {path}: {error}") from error
        if not isinstance(payload, Mapping):
            raise PrefillCycleArtifactError("cycle catalog root must be an object")
        if payload.get("schema_version") != "hetero-prefill-cycle-catalog/v1":
            raise PrefillCycleArtifactError("invalid cycle catalog schema_version")
        devices = payload.get("devices")
        if not isinstance(devices, Mapping) or not devices:
            raise PrefillCycleArtifactError("cycle catalog devices are required")
        return cls(path.resolve(), dict(payload), hashlib.sha256(raw_bytes).hexdigest())

    def _device(self, device_id: str) -> Mapping[str, object]:
        devices = self.payload["devices"]
        assert isinstance(devices, Mapping)
        device = devices.get(device_id)
        if not isinstance(device, Mapping):
            raise PrefillCycleArtifactError(
                f"catalog has no cycle contract for device {device_id}"
            )
        return device

    def validate_model(self, model: ModelSpec) -> None:
        expected = self.payload.get("model")
        if not isinstance(expected, Mapping):
            raise PrefillCycleArtifactError("cycle catalog model contract is required")
        actual = {
            "name": model.name,
            "hidden_size": model.hidden_size,
            "intermediate_size": model.intermediate_size,
            "num_attention_heads": model.num_attention_heads,
            "num_kv_heads": model.num_kv_heads,
            "head_dim": model.head_dim,
            "vocab_size": model.vocab_size,
            "dtype": model.dtype,
        }
        mismatches = {
            key: (expected.get(key), value)
            for key, value in actual.items()
            if expected.get(key) != value
        }
        if mismatches:
            raise PrefillCycleArtifactError(
                f"cycle catalog model mismatch: {mismatches}"
            )

    def supported_ops(self, device_id: str) -> set[str]:
        profiles = self._device(device_id).get("operator_profiles")
        if not isinstance(profiles, Mapping):
            raise PrefillCycleArtifactError("operator_profiles must be an object")
        return {str(op) for op in profiles}

    def plan(
        self,
        spec: OnlineDispatchSpec,
        global_clock_hz: int,
    ) -> CycleTaskPlan:
        self.validate_model(spec.model)
        if spec.node.phase.value not in {"prefill", "control"}:
            raise PrefillCycleArtifactError(
                f"P11-P14 catalog accepts Prefill/control only: {spec.node.node_id}"
            )
        device = self._device(spec.device_id)
        clock_hz = _positive(device, "clock_hz")
        profiles = device.get("operator_profiles")
        if not isinstance(profiles, Mapping):
            raise PrefillCycleArtifactError("operator_profiles must be an object")
        profile = profiles.get(spec.node.op)
        if not isinstance(profile, Mapping):
            raise PrefillCycleArtifactError(
                f"missing cycle profile for {spec.device_id}.{spec.node.op}"
            )
        native_cycles, formula = _operator_cycles(spec.node, spec.model, profile)
        global_cycles = _ceil_div(native_cycles * global_clock_hz, clock_hz)
        if global_cycles <= 0:
            raise PrefillCycleArtifactError("global compute cycles must be positive")
        backend_id = str(device.get("backend_id", f"{spec.device_id}.cycle_replay"))
        return CycleTaskPlan(
            task_id=spec.task_id,
            backend_id=backend_id,
            device_id=spec.device_id,
            device_clock_hz=clock_hz,
            native_compute_cycles=native_cycles,
            global_compute_cycles=global_cycles,
            formula=formula,
            fidelity={
                "compute_fidelity": "tiled_cycle_contract_unqualified",
                "memory_fidelity": "live_ramulator2_sampled_requests",
                "link_fidelity": (
                    "cycle_modeled_external_link"
                    if spec.device_id == "gpu0"
                    else "internal_hybrid_bond_port"
                ),
                "scheduler_fidelity": "cycle_event",
                "extrapolated_fraction": 1.0,
                "trace_coverage": 0.0,
                "artifact_coverage": 1.0,
                "performance_eligible": False,
            },
            artifact={
                "kind": "prefill_tiled_cycle_contract",
                "catalog": str(self.source_path),
                "catalog_sha256": self.content_sha256,
                "operator": spec.node.op,
                "parameter_source": self.payload.get("parameter_source"),
                "qualification": self.payload.get("qualification"),
            },
        )


def _operator_cycles(
    node: ModelNode,
    model: ModelSpec,
    profile: Mapping[str, object],
) -> tuple[int, dict[str, object]]:
    kind = str(profile.get("kind"))
    launch = int(profile.get("launch_cycles", 0))
    if launch < 0:
        raise PrefillCycleArtifactError("launch_cycles must be unsigned")
    m = int(node.attributes.get("q_len", 1))
    kv = int(node.attributes.get("attention_kv_len", m))
    h = model.hidden_size
    i = model.intermediate_size

    if kind == "fixed":
        work_cycles = _positive(profile, "cycles")
        detail: dict[str, object] = {"kind": kind, "cycles": work_cycles}
    elif kind == "embedding":
        tokens_per_cycle = _positive(profile, "tokens_per_cycle")
        work_cycles = _ceil_div(m, tokens_per_cycle)
        detail = {"kind": kind, "tokens": m, "tokens_per_cycle": tokens_per_cycle}
    elif kind == "elementwise":
        elements_per_cycle = _positive(profile, "elements_per_cycle")
        width_multiplier = int(profile.get("width_multiplier", 1))
        if width_multiplier <= 0:
            raise PrefillCycleArtifactError("width_multiplier must be positive")
        elements = m * h * width_multiplier
        work_cycles = _ceil_div(elements, elements_per_cycle)
        detail = {
            "kind": kind,
            "elements": elements,
            "elements_per_cycle": elements_per_cycle,
        }
    elif kind == "gemm":
        tile_m = _positive(profile, "tile_m")
        tile_n = _positive(profile, "tile_n")
        tile_k = _positive(profile, "tile_k")
        cycles_per_tile = _positive(profile, "cycles_per_tile")
        parallel_tiles = _positive(profile, "parallel_tiles")
        if node.op == "qkv_projection":
            k, n = h, h + 2 * model.num_kv_heads * model.head_dim
        elif node.op == "output_projection":
            k, n = h, h
        elif node.op == "gate_up_projection":
            k, n = h, 2 * i
        elif node.op == "fc1_projection":
            k, n = h, i
        elif node.op == "down_projection":
            k, n = i, h
        elif node.op == "lm_head":
            k, n = h, model.vocab_size
        else:
            raise PrefillCycleArtifactError(f"gemm profile is invalid for {node.op}")
        tiles = _ceil_div(m, tile_m) * _ceil_div(n, tile_n) * _ceil_div(k, tile_k)
        work_cycles = _ceil_div(tiles * cycles_per_tile, parallel_tiles)
        detail = {
            "kind": kind,
            "m": m,
            "n": n,
            "k": k,
            "tile_m": tile_m,
            "tile_n": tile_n,
            "tile_k": tile_k,
            "tiles": tiles,
            "cycles_per_tile": cycles_per_tile,
            "parallel_tiles": parallel_tiles,
        }
    elif kind == "attention":
        tile_q = _positive(profile, "tile_q")
        tile_kv = _positive(profile, "tile_kv")
        cycles_per_tile = _positive(profile, "cycles_per_tile")
        parallel_heads = _positive(profile, "parallel_heads")
        tiles = (
            _ceil_div(m, tile_q)
            * _ceil_div(kv, tile_kv)
            * model.num_attention_heads
        )
        work_cycles = _ceil_div(tiles * cycles_per_tile, parallel_heads)
        detail = {
            "kind": kind,
            "q_tokens": m,
            "kv_tokens": kv,
            "heads": model.num_attention_heads,
            "tiles": tiles,
            "cycles_per_tile": cycles_per_tile,
            "parallel_heads": parallel_heads,
        }
    elif kind == "sampling":
        values_per_cycle = _positive(profile, "values_per_cycle")
        work_cycles = _ceil_div(model.vocab_size, values_per_cycle)
        detail = {
            "kind": kind,
            "values": model.vocab_size,
            "values_per_cycle": values_per_cycle,
        }
    else:
        raise PrefillCycleArtifactError(f"unsupported cycle formula kind {kind!r}")
    cycles = max(1, launch + work_cycles)
    detail.update({"launch_cycles": launch, "native_cycles": cycles})
    return cycles, detail


class PrefillCycleDispatcher:
    """Resolve one explicit cycle contract for every placed Prefill task."""

    def __init__(
        self,
        project_root: Path,
        backends: Mapping[str, object],
        global_clock_hz: int,
    ) -> None:
        self.project_root = project_root
        self.backends = backends
        self.global_clock_hz = global_clock_hz
        self._catalogs: dict[str, PrefillCycleCatalog] = {}
        self._operator_catalogs: dict[str, OperatorArtifactCatalog] = {}
        self._full_traffic_ops: dict[str, set[str]] = {}
        self._traffic_mode: dict[str, str] = {}
        self._registered_artifacts: dict[str, OperatorArtifactManifest] = {}
        self._plans: dict[str, CycleTaskPlan] = {}
        for backend_key in ("gpu", "atlas"):
            backend = backends.get(backend_key)
            if not isinstance(backend, Mapping) or backend.get("kind") == "none":
                continue
            if backend.get("kind") != "cycle_replay":
                raise PrefillCycleArtifactError(
                    f"{backend_key} must use cycle_replay in prefill_cycle mode"
                )
            path = Path(str(backend["cycle_artifact_ref"]))
            if not path.is_absolute():
                path = project_root / path
            self._catalogs[backend_key] = PrefillCycleCatalog.load(path.resolve())
            operator_catalog_ref = backend.get("operator_artifact_catalog_ref")
            if operator_catalog_ref is not None:
                operator_path = Path(str(operator_catalog_ref))
                if not operator_path.is_absolute():
                    operator_path = project_root / operator_path
                self._operator_catalogs[backend_key] = OperatorArtifactCatalog.load(
                    operator_path.resolve()
                )
                full_ops = backend.get("full_traffic_operators", [])
                if not isinstance(full_ops, list):
                    raise PrefillCycleArtifactError(
                        f"{backend_key}.full_traffic_operators must be an array"
                    )
                self._full_traffic_ops[backend_key] = {
                    str(item) for item in full_ops
                }

    def dispatch(self, spec: OnlineDispatchSpec) -> CycleTaskPlan:
        if spec.task_id in self._plans:
            raise PrefillCycleArtifactError(f"duplicate cycle dispatch {spec.task_id}")
        catalog = self._catalogs.get(spec.backend_key)
        if catalog is None:
            raise PrefillCycleArtifactError(
                f"no cycle catalog for backend {spec.backend_key}"
            )
        plan = catalog.plan(spec, self.global_clock_hz)
        traffic_mode = "sampled"
        if spec.node.op in self._full_traffic_ops.get(spec.backend_key, set()):
            operator_catalog = self._operator_catalogs.get(spec.backend_key)
            if operator_catalog is None:
                raise PrefillCycleArtifactError(
                    f"full traffic requested without a catalog: {spec.task_id}"
                )
            backend_kinds = (
                {"accel_sim", "runtime_state"}
                if spec.backend_key == "gpu"
                else {"atlasim", "runtime_state"}
            )
            try:
                registered = operator_catalog.match(
                    model_spec_name=spec.model.name,
                    operator=spec.node.op,
                    phase="prefill",
                    layer_id=int(spec.node.layer_id or 0),
                    batch_size=1,
                    context_length=int(
                        spec.node.attributes.get(
                            "source_q_len", spec.node.attributes.get("q_len", 1)
                        )
                    ),
                    q_len=int(spec.node.attributes.get("q_len", 1)),
                    kv_length=int(
                        spec.node.attributes.get("attention_kv_len", 1)
                    ),
                    dtype=spec.model.dtype,
                    backend_kinds=backend_kinds,
                )
            except OperatorArtifactError as error:
                raise PrefillCycleArtifactError(str(error)) from error
            if registered is None:
                raise PrefillCycleArtifactError(
                    f"no shape-locked artifact permits full traffic for {spec.task_id}"
                )
            traffic_mode = "full"
            self._registered_artifacts[spec.task_id] = registered
            plan = replace(
                plan,
                fidelity={
                    **dict(plan.fidelity),
                    "memory_fidelity": "live_ramulator2_full_value_transactions",
                },
                artifact={
                    **dict(plan.artifact),
                    "operator_artifact_id": registered.artifact_id,
                    "operator_artifact_manifest": str(registered.source_path),
                    "operator_artifact_sha256": registered.content_sha256,
                    "artifact_execution_use": "memory_traffic_lowering_only",
                },
            )
        self._traffic_mode[spec.task_id] = traffic_mode
        self._plans[spec.task_id] = plan
        return plan

    def memory_traffic_mode(self, task_id: str) -> str:
        mode = self._traffic_mode.get(task_id)
        if mode not in {"sampled", "full"}:
            raise PrefillCycleArtifactError(
                f"memory traffic requested before dispatch: {task_id}"
            )
        return mode

    def coverage(self, expected_task_ids: set[str]) -> dict[str, object]:
        actual = set(self._plans)
        if actual != expected_task_ids:
            raise PrefillCycleArtifactError(
                "cycle artifact coverage mismatch: "
                f"missing={sorted(expected_task_ids - actual)}, "
                f"extra={sorted(actual - expected_task_ids)}"
            )
        by_device: dict[str, int] = {}
        by_op: dict[str, int] = {}
        for plan in self._plans.values():
            by_device[plan.device_id] = by_device.get(plan.device_id, 0) + 1
            op = str(plan.artifact["operator"])
            by_op[op] = by_op.get(op, 0) + 1
        full_by_op: dict[str, int] = {}
        for task_id, mode in self._traffic_mode.items():
            if mode != "full":
                continue
            op = str(self._plans[task_id].artifact["operator"])
            full_by_op[op] = full_by_op.get(op, 0) + 1
        return {
            "schema_version": "hetero-prefill-artifact-coverage/v1",
            "expected_tasks": len(expected_task_ids),
            "covered_tasks": len(actual),
            "coverage": 1.0,
            "all_tasks_covered": True,
            "analytical_fallback_tasks": 0,
            "full_traffic_tasks": sum(full_by_op.values()),
            "sampled_traffic_tasks": len(actual) - sum(full_by_op.values()),
            "full_traffic_by_operator": dict(sorted(full_by_op.items())),
            "by_device": dict(sorted(by_device.items())),
            "by_operator": dict(sorted(by_op.items())),
            "catalogs": {
                key: {
                    "path": str(value.source_path),
                    "sha256": value.content_sha256,
                }
                for key, value in sorted(self._catalogs.items())
            },
        }
