"""Operator-event Backend dispatch for the second integration step."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from .analytical import estimate_node_cost
from .backends import (
    AccelSimBackend,
    AccelSimBackendConfig,
    AtlasArtifact,
    AtlasBackend,
    AtlasBackendConfig,
    ResolvedTimingContract,
    TimingOwnershipRegistry,
    resolve_timing_contract,
)
from .global_memory_map import GlobalAllocation
from .ir import ModelNode
from .model_graph import ModelSpec
from .operator_artifact import OperatorArtifactManifest
from .trace_manifest import SimulationBufferBinding, TraceManifest


class OperatorEventError(RuntimeError):
    """Raised when an operator-event Backend cannot produce a valid duration."""


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _reserved_allocation_extents(
    allocations: Mapping[str, GlobalAllocation], capacity_bytes: int
) -> dict[str, int]:
    """Return logical allocations plus their safe alignment padding."""

    ordered = sorted(allocations.values(), key=lambda item: item.base_address)
    reserved: dict[str, int] = {}
    for index, allocation in enumerate(ordered):
        reservation_end = (
            ordered[index + 1].base_address
            if index + 1 < len(ordered)
            else capacity_bytes
        )
        if reservation_end < allocation.end_address_exclusive:
            raise OperatorEventError(
                f"Global PA allocations overlap at {allocation.value_id}"
            )
        reserved[allocation.value_id] = reservation_end - allocation.base_address
    return reserved


@dataclass(frozen=True, slots=True)
class TensorValueBinding:
    tensor_id: str
    source: str
    index: int
    value_offset_bytes: int


@dataclass(frozen=True, slots=True)
class TraceBinding:
    selector: Mapping[str, object]
    manifest: TraceManifest
    compatibility: str
    operator_artifact: OperatorArtifactManifest | None = None
    value_bindings: tuple[TensorValueBinding, ...] = ()
    contract_overrides: Mapping[str, object] = field(default_factory=dict)

    def matches(self, node: ModelNode) -> bool:
        actual = {
            "node_id": node.node_id,
            "phase": node.phase.value,
            "op": node.op,
            "layer_id": node.layer_id,
            "step_id": node.step_id,
            "operator_group": node.attributes.get("operator_group"),
        }
        return all(actual.get(key) == value for key, value in self.selector.items())

    def validate_exact_contract(self, node: ModelNode, model: ModelSpec) -> None:
        if self.compatibility != "exact_operator" or self.operator_artifact is None:
            return
        key = self.operator_artifact.compatibility_key
        actual_phase = (
            "decode_step" if node.phase.value == "decode" else node.phase.value
        )
        actual = {
            "model_spec_name": model.name,
            "operator": node.op,
            "phase": actual_phase,
            "layer_id": node.layer_id,
            "q_len": int(node.attributes.get("q_len", 1)),
            "kv_length": int(node.attributes.get("attention_kv_len", 1)),
            "dtype": model.dtype,
        }
        actual.update(self.contract_overrides)
        expected = key.to_dict()
        mismatches = {
            field: (expected[field], value)
            for field, value in actual.items()
            if expected[field] != value
        }
        if mismatches:
            raise OperatorEventError(
                f"shape-locked operator artifact mismatch for {node.node_id}: "
                f"{mismatches}"
            )


@dataclass(frozen=True, slots=True)
class AtlasArtifactBinding:
    selector: Mapping[str, object]
    chip_config: Path
    artifact: AtlasArtifact
    compatibility: str

    def matches(self, node: ModelNode) -> bool:
        actual = {
            "node_id": node.node_id,
            "phase": node.phase.value,
            "op": node.op,
            "layer_id": node.layer_id,
            "step_id": node.step_id,
            "operator_group": node.attributes.get("operator_group"),
        }
        return all(actual.get(key) == value for key, value in self.selector.items())


@dataclass(frozen=True, slots=True)
class BackendTaskResult:
    backend_id: str
    duration_fs: int
    resource_id: str
    timing_contract: Mapping[str, object]
    fidelity: Mapping[str, object]
    statistics: Mapping[str, object]
    artifact: Mapping[str, object]


def _resolve_path(value: object, project_root: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (project_root / path).resolve()


class OperatorEventDispatcher:
    """Select a concrete or explicit fallback model for every device task."""

    def __init__(
        self,
        project_root: Path,
        output_root: Path,
        backends: Mapping[str, object],
    ) -> None:
        self.project_root = project_root
        self.output_root = output_root
        self.backends = backends
        self.ownership = TimingOwnershipRegistry()
        self._accel_sim: AccelSimBackend | None = None
        self._gpu_contract: ResolvedTimingContract | None = None
        self._trace_bindings: list[TraceBinding] = []
        self._atlas: AtlasBackend | None = None
        self._atlas_contract: ResolvedTimingContract | None = None
        self._atlas_bindings: list[AtlasArtifactBinding] = []
        self._used_traces: dict[str, dict[str, object]] = {}
        self._used_atlas_artifacts: dict[str, dict[str, object]] = {}
        self._require_request_cycle_ready = False
        self._runtime_bindings: dict[str, tuple[SimulationBufferBinding, ...]] = {}
        self._runtime_binding_records: dict[str, dict[str, object]] = {}
        self._prepare_gpu_backend()
        self._prepare_atlas_backend()

    def _backend_config(self, backend_key: str) -> Mapping[str, object]:
        value = self.backends.get(backend_key)
        if not isinstance(value, Mapping):
            raise OperatorEventError(f"backend {backend_key} must be an object")
        return value

    def _prepare_gpu_backend(self) -> None:
        gpu = self._backend_config("gpu")
        if gpu.get("kind") != "accel_sim":
            return
        config_ref = gpu.get("config_ref")
        if not isinstance(config_ref, str) or not config_ref:
            raise OperatorEventError("accel_sim GPU backend requires config_ref")
        backend_config = AccelSimBackendConfig.load(
            _resolve_path(config_ref, self.project_root)
        )
        if backend_config.co_resident_atlas is not None:
            raise OperatorEventError(
                "co_resident_atlas is a duplicate-operator contention stress Backend "
                "and cannot be used by the single-placement operator-event dispatcher"
            )
        self._accel_sim = AccelSimBackend(backend_config)
        self._require_request_cycle_ready = bool(
            gpu.get("require_request_cycle_ready", False)
        )
        raw_bindings = gpu.get("trace_bindings")
        if not isinstance(raw_bindings, list) or not raw_bindings:
            raise OperatorEventError("accel_sim GPU backend requires trace_bindings")
        for index, raw in enumerate(raw_bindings):
            if not isinstance(raw, Mapping):
                raise OperatorEventError(f"trace_bindings[{index}] must be an object")
            allowed = {
                "selector",
                "trace_manifest",
                "operator_artifact",
                "compatibility",
                "value_bindings",
                "contract_overrides",
            }
            unknown = set(raw) - allowed
            if unknown:
                raise OperatorEventError(
                    f"unknown trace_bindings[{index}] fields: {sorted(unknown)}"
                )
            selector = raw.get("selector")
            if not isinstance(selector, Mapping) or not selector:
                raise OperatorEventError(
                    f"trace_bindings[{index}].selector must be a non-empty object"
                )
            allowed_selector = {
                "node_id",
                "phase",
                "op",
                "layer_id",
                "step_id",
                "operator_group",
            }
            if set(selector) - allowed_selector:
                raise OperatorEventError(
                    "unknown trace selector fields: "
                    f"{sorted(set(selector) - allowed_selector)}"
                )
            compatibility = str(raw.get("compatibility", "exact_operator"))
            if compatibility not in {"exact_operator", "surrogate_plumbing_probe"}:
                raise OperatorEventError(
                    f"invalid trace binding compatibility: {compatibility}"
                )
            raw_contract_overrides = raw.get("contract_overrides", {})
            if not isinstance(raw_contract_overrides, Mapping):
                raise OperatorEventError("contract_overrides must be an object")
            allowed_contract_overrides = {"layer_id"}
            unknown_contract_overrides = (
                set(raw_contract_overrides) - allowed_contract_overrides
            )
            if unknown_contract_overrides:
                raise OperatorEventError(
                    "unknown contract override fields: "
                    f"{sorted(unknown_contract_overrides)}"
                )
            if "layer_id" in raw_contract_overrides:
                layer_id_override = raw_contract_overrides["layer_id"]
                if (
                    not isinstance(layer_id_override, int)
                    or isinstance(layer_id_override, bool)
                    or layer_id_override < 0
                ):
                    raise OperatorEventError(
                        "contract_overrides.layer_id must be an unsigned integer"
                    )
            manifest_path = _resolve_path(raw.get("trace_manifest"), self.project_root)
            trace_manifest = TraceManifest.load(manifest_path)
            artifact_value = raw.get("operator_artifact")
            operator_artifact = (
                OperatorArtifactManifest.load(
                    _resolve_path(artifact_value, self.project_root)
                )
                if artifact_value is not None
                else None
            )
            if operator_artifact is not None:
                artifact_backend = operator_artifact.payload["backend"]
                assert isinstance(artifact_backend, Mapping)
                if artifact_backend.get("kind") != "accel_sim":
                    raise OperatorEventError(
                        "GPU trace binding operator_artifact must use accel_sim"
                    )
                if not (
                    operator_artifact.artifact_id == trace_manifest.trace_id
                    or operator_artifact.artifact_id.startswith(
                        trace_manifest.trace_id + "."
                    )
                ):
                    raise OperatorEventError(
                        "operator_artifact artifact_id must equal or be a qualified "
                        "derivative of trace_manifest trace_id"
                    )
                if self._require_request_cycle_ready:
                    execution = operator_artifact.payload["execution_contract"]
                    address = operator_artifact.payload["address_contract"]
                    qualification = operator_artifact.payload["qualification"]
                    assert isinstance(execution, Mapping)
                    assert isinstance(address, Mapping)
                    assert isinstance(qualification, Mapping)
                    if not operator_artifact.request_cycle_ready:
                        raise OperatorEventError(
                            "request-cycle timeline rejects non-ready operator artifact"
                        )
                    if address.get("virtual_memory_mode") != "range_rebase":
                        raise OperatorEventError(
                            "request-cycle timeline requires range_rebase address mode"
                        )
                    if qualification.get("performance_eligible") is not False:
                        raise OperatorEventError(
                            "request-cycle integration remains performance-ineligible"
                        )
            raw_value_bindings = raw.get("value_bindings", [])
            if not isinstance(raw_value_bindings, list):
                raise OperatorEventError("value_bindings must be an array")
            value_bindings: list[TensorValueBinding] = []
            for value_index, value_raw in enumerate(raw_value_bindings):
                if not isinstance(value_raw, Mapping):
                    raise OperatorEventError(
                        f"value_bindings[{value_index}] must be an object"
                    )
                required = {"tensor_id", "source", "index", "value_offset_bytes"}
                if set(value_raw) != required:
                    raise OperatorEventError(
                        f"value_bindings[{value_index}] fields must be "
                        f"{sorted(required)}"
                    )
                source = str(value_raw["source"])
                index_value = value_raw["index"]
                offset_value = value_raw["value_offset_bytes"]
                if source not in {"input", "output"}:
                    raise OperatorEventError(
                        "value binding source must be input or output"
                    )
                if (
                    not isinstance(index_value, int)
                    or isinstance(index_value, bool)
                    or index_value < 0
                    or not isinstance(offset_value, int)
                    or isinstance(offset_value, bool)
                    or offset_value < 0
                ):
                    raise OperatorEventError(
                        "value binding index and offset must be unsigned integers"
                    )
                value_bindings.append(
                    TensorValueBinding(
                        tensor_id=str(value_raw["tensor_id"]),
                        source=source,
                        index=index_value,
                        value_offset_bytes=offset_value,
                    )
                )
            if self._require_request_cycle_ready and not value_bindings:
                raise OperatorEventError(
                    "request-cycle trace binding requires explicit value_bindings"
                )
            if len({item.tensor_id for item in value_bindings}) != len(value_bindings):
                raise OperatorEventError("value_bindings contain duplicate tensor IDs")
            self._trace_bindings.append(
                TraceBinding(
                    selector=dict(selector),
                    manifest=trace_manifest,
                    compatibility=compatibility,
                    operator_artifact=operator_artifact,
                    value_bindings=tuple(value_bindings),
                    contract_overrides=dict(raw_contract_overrides),
                )
            )
        resources = gpu.get("resource_bindings")
        if not isinstance(resources, Mapping) or not resources:
            raise OperatorEventError(
                "accel_sim GPU backend requires explicit resource_bindings"
            )
        requested_mode = str(gpu.get("requested_timing_mode", "total"))
        replay_safe = all(
            binding.manifest.replay_safe for binding in self._trace_bindings
        )
        self._gpu_contract = resolve_timing_contract(
            self._accel_sim.descriptor(),
            requested_mode,
            resources,
            trace_semantics="functional",
            replay_safe=replay_safe,
        )
        self.ownership.register(self._gpu_contract)

    @property
    def requires_global_pa_bindings(self) -> bool:
        return self._require_request_cycle_ready

    def configure_global_pa_bindings(
        self,
        allocations: Mapping[str, GlobalAllocation],
        dispatch_specs: Mapping[str, object],
        global_memory_map: dict[str, object],
        *,
        capacity_bytes: int,
        alignment_bytes: int,
    ) -> None:
        """Bind graph Values and private workspaces before any Backend launch."""

        if not self._require_request_cycle_ready:
            return
        memory_space_id = str(global_memory_map["memory_space_id"])
        cursor = _align_up(int(global_memory_map["allocated_bytes"]), alignment_bytes)
        reserved_extents = _reserved_allocation_extents(allocations, capacity_bytes)
        workspace_ranges: list[dict[str, object]] = []
        binding_records: list[dict[str, object]] = []
        for task_id, spec in sorted(dispatch_specs.items()):
            node = spec.node
            model = spec.model
            binding = self._matching_trace(node, model)
            if binding is None:
                continue
            if not binding.value_bindings:
                raise OperatorEventError(
                    f"request-cycle task {task_id} has no graph Value bindings"
                )
            inputs = tuple(spec.input_values)
            outputs = tuple(spec.output_values)
            extents: dict[str, int] = {}
            alignments: dict[str, int] = {}
            for item in binding.manifest.address_ranges:
                extents[item.tensor_id] = max(
                    extents.get(item.tensor_id, 0),
                    item.tensor_offset_bytes + item.size_bytes,
                )
                alignments[item.tensor_id] = max(
                    alignments.get(item.tensor_id, 1), item.alignment_bytes
                )
            runtime: list[SimulationBufferBinding] = []
            mapped_ids: set[str] = set()
            semantic_records: list[dict[str, object]] = []
            for value_binding in binding.value_bindings:
                if value_binding.tensor_id not in extents:
                    raise OperatorEventError(
                        f"unknown manifest tensor {value_binding.tensor_id} "
                        f"for {task_id}"
                    )
                values = inputs if value_binding.source == "input" else outputs
                if value_binding.index >= len(values):
                    raise OperatorEventError(
                        f"value binding index escapes {value_binding.source}s "
                        f"for {task_id}"
                    )
                value = values[value_binding.index]
                if not isinstance(value, Mapping):
                    raise OperatorEventError("dispatch Value must be an object")
                value_id = str(value["value_id"])
                allocation = allocations.get(value_id)
                extent = extents[value_binding.tensor_id]
                if allocation is None or (
                    value_binding.value_offset_bytes + extent
                    > reserved_extents.get(value_id, 0)
                ):
                    raise OperatorEventError(
                        f"Global PA allocation cannot cover {value_binding.tensor_id}"
                    )
                physical = allocation.base_address + value_binding.value_offset_bytes
                runtime.append(
                    SimulationBufferBinding(
                        tensor_id=value_binding.tensor_id,
                        tensor_offset_bytes=0,
                        size_bytes=extent,
                        memory_space_id=memory_space_id,
                        physical_offset_bytes=physical,
                    )
                )
                mapped_ids.add(value_binding.tensor_id)
                semantic_records.append(
                    {
                        "tensor_id": value_binding.tensor_id,
                        "value_id": value_id,
                        "value_version": int(value["version"]),
                        "source": value_binding.source,
                        "value_index": value_binding.index,
                        "value_offset_bytes": value_binding.value_offset_bytes,
                        "global_pa_base": physical,
                        "size_bytes": extent,
                        "logical_value_size_bytes": allocation.size_bytes,
                    }
                )
            private_records: list[dict[str, object]] = []
            for tensor_id in sorted(set(extents) - mapped_ids):
                alignment = max(alignment_bytes, alignments[tensor_id])
                cursor = _align_up(cursor, alignment)
                extent = extents[tensor_id]
                if cursor + extent > capacity_bytes:
                    raise OperatorEventError(
                        "Global PA workspace capacity exceeded by "
                        f"{task_id}:{tensor_id}"
                    )
                runtime.append(
                    SimulationBufferBinding(
                        tensor_id=tensor_id,
                        tensor_offset_bytes=0,
                        size_bytes=extent,
                        memory_space_id=memory_space_id,
                        physical_offset_bytes=cursor,
                    )
                )
                record = {
                    "value_id": f"workspace:{task_id}:{tensor_id}",
                    "memory_space_id": memory_space_id,
                    "base_address": cursor,
                    "end_address_exclusive": cursor + extent,
                    "size_bytes": extent,
                    "alignment_bytes": alignment,
                    "storage_class": "operator_workspace",
                    "dtype": "opaque",
                }
                workspace_ranges.append(record)
                private_records.append(
                    {
                        "tensor_id": tensor_id,
                        "global_pa_base": cursor,
                        "size_bytes": extent,
                    }
                )
                cursor += extent
            ordered = tuple(
                sorted(runtime, key=lambda item: item.physical_offset_bytes)
            )
            for left, right in zip(ordered, ordered[1:]):
                if left.physical_end > right.physical_offset_bytes:
                    raise OperatorEventError(
                        f"runtime Global PA overlap for {task_id}: "
                        f"{left.tensor_id} and {right.tensor_id}"
                    )
            self._runtime_bindings[task_id] = ordered
            record = {
                "task_id": task_id,
                "trace_id": binding.manifest.trace_id,
                "operator_artifact_id": binding.operator_artifact.artifact_id
                if binding.operator_artifact
                else None,
                "request_cycle_ready": True,
                "contract_overrides": dict(binding.contract_overrides),
                "semantic_bindings": semantic_records,
                "private_bindings": private_records,
            }
            self._runtime_binding_records[task_id] = record
            binding_records.append(record)
        if not self._runtime_bindings:
            raise OperatorEventError(
                "no request-cycle-ready trace was bound to Global PA"
            )
        ranges = global_memory_map.get("ranges")
        if not isinstance(ranges, list):
            raise OperatorEventError("global memory map ranges must be an array")
        ranges.extend(workspace_ranges)
        ranges.sort(key=lambda item: int(item["base_address"]))
        global_memory_map.update(
            {
                "address_semantics": "allocated_global_pa_range_rebased",
                "allocated_bytes": cursor,
                "allocation_count": len(ranges),
                "operator_workspace_count": len(workspace_ranges),
                "request_cycle_bindings": binding_records,
            }
        )

    def _prepare_atlas_backend(self) -> None:
        atlas = self._backend_config("atlas")
        if atlas.get("kind") != "atlasim":
            return
        config_ref = atlas.get("config_ref")
        if not isinstance(config_ref, str) or not config_ref:
            raise OperatorEventError("atlasim backend requires config_ref")
        self._atlas = AtlasBackend(
            AtlasBackendConfig.load(_resolve_path(config_ref, self.project_root))
        )
        raw_bindings = atlas.get("artifact_bindings")
        if not isinstance(raw_bindings, list) or not raw_bindings:
            raise OperatorEventError("atlasim backend requires artifact_bindings")
        for index, raw in enumerate(raw_bindings):
            if not isinstance(raw, Mapping):
                raise OperatorEventError(
                    f"artifact_bindings[{index}] must be an object"
                )
            allowed = {
                "selector",
                "chip_config",
                "operator_list",
                "placement_map",
                "compatibility",
            }
            unknown = set(raw) - allowed
            if unknown:
                raise OperatorEventError(
                    f"unknown artifact_bindings[{index}] fields: {sorted(unknown)}"
                )
            selector = raw.get("selector")
            if not isinstance(selector, Mapping) or not selector:
                raise OperatorEventError(
                    f"artifact_bindings[{index}].selector must be a non-empty object"
                )
            allowed_selector = {
                "node_id",
                "phase",
                "op",
                "layer_id",
                "step_id",
                "operator_group",
            }
            if set(selector) - allowed_selector:
                raise OperatorEventError(
                    f"unknown artifact selector fields: "
                    f"{sorted(set(selector) - allowed_selector)}"
                )
            compatibility = str(raw.get("compatibility", "exact_operator"))
            if compatibility not in {"exact_operator", "surrogate_plumbing_probe"}:
                raise OperatorEventError(
                    f"invalid artifact binding compatibility: {compatibility}"
                )
            self._atlas_bindings.append(
                AtlasArtifactBinding(
                    selector=dict(selector),
                    chip_config=_resolve_path(
                        raw.get("chip_config"), self.project_root
                    ),
                    artifact=AtlasArtifact(
                        operator_list=_resolve_path(
                            raw.get("operator_list"), self.project_root
                        ),
                        placement_map=_resolve_path(
                            raw.get("placement_map"), self.project_root
                        ),
                    ),
                    compatibility=compatibility,
                )
            )
        resources = atlas.get("resource_bindings")
        if not isinstance(resources, Mapping) or not resources:
            raise OperatorEventError(
                "atlasim backend requires explicit resource_bindings"
            )
        self._atlas_contract = resolve_timing_contract(
            self._atlas.descriptor(),
            str(atlas.get("requested_timing_mode", "total")),
            resources,
            trace_semantics="none",
            replay_safe=False,
        )
        self.ownership.register(self._atlas_contract)

    def _matching_trace(self, node: ModelNode, model: ModelSpec) -> TraceBinding | None:
        matches = [binding for binding in self._trace_bindings if binding.matches(node)]
        if len(matches) > 1:
            raise OperatorEventError(
                f"multiple GPU traces match node {node.node_id}; "
                "selectors must be unique"
            )
        if not matches:
            return None
        matches[0].validate_exact_contract(node, model)
        return matches[0]

    def _matching_atlas_artifact(self, node: ModelNode) -> AtlasArtifactBinding | None:
        matches = [binding for binding in self._atlas_bindings if binding.matches(node)]
        if len(matches) > 1:
            raise OperatorEventError(
                f"multiple ATLAS artifacts match node {node.node_id}; "
                "selectors must be unique"
            )
        return matches[0] if matches else None

    def _analytical(
        self,
        backend_key: str,
        backend: Mapping[str, object],
        node: ModelNode,
        model: ModelSpec,
        device_id: str,
        *,
        fallback_reason: str | None = None,
    ) -> BackendTaskResult:
        cost = estimate_node_cost(node, model, backend)
        fidelity = {
            "compute_fidelity": "analytical_fallback"
            if fallback_reason
            else "analytical",
            "memory_fidelity": "analytical",
            "link_fidelity": "not_applicable",
            "scheduler_fidelity": "event_modeled",
            "extrapolated_fraction": 1.0,
            "trace_coverage": 0.0,
        }
        if fallback_reason:
            fidelity["fallback_reason"] = fallback_reason
        contract = {
            "backend_id": f"{backend_key}.{backend.get('kind', 'analytical')}",
            "duration_semantics": "total",
            "owns": [device_id],
            "exports": [],
            "supports_stall_resume": False,
            "trace_semantics": "none",
            "replay_safe": False,
        }
        return BackendTaskResult(
            backend_id=str(contract["backend_id"]),
            duration_fs=cost.duration_fs,
            resource_id=device_id,
            timing_contract=contract,
            fidelity=fidelity,
            statistics={"analytical_cost": cost.to_dict()},
            artifact={
                "kind": "analytical",
                "parameter_source": backend.get("parameter_source"),
            },
        )

    def _run_accel_sim(
        self,
        binding: TraceBinding,
        node: ModelNode,
        device_id: str,
        task_id: str | None = None,
    ) -> BackendTaskResult:
        assert self._accel_sim is not None and self._gpu_contract is not None
        simulation_bindings = (
            self._runtime_bindings.get(task_id) if task_id is not None else None
        )
        if self._require_request_cycle_ready and simulation_bindings is None:
            raise OperatorEventError(
                f"request-cycle task {task_id or node.node_id} lacks runtime Global PA"
            )
        simulation_key = self._accel_sim.simulation_key(
            binding.manifest, simulation_bindings
        )
        output = self.output_root / "gpu" / simulation_key
        stats_path = output / "stats.json"
        cache_hit = stats_path.is_file()
        if cache_hit:
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            cycles = int(stats["cycles"])
            instructions = int(stats["instructions"])
            duration_fs = int(stats["duration_fs"])
            external_memory_stats = stats.get("external_memory_stats")
        else:
            result = self._accel_sim.run(binding.manifest, output, simulation_bindings)
            cycles = result.cycles
            instructions = result.instructions
            duration_fs = result.duration_fs
            external_memory_stats = result.external_memory_stats
        trace_key = binding.manifest.trace_key()
        trace_record = self._used_traces.setdefault(
            trace_key,
            {
                "trace_id": binding.manifest.trace_id,
                "trace_key": trace_key,
                "source_manifest": str(binding.manifest.source_path),
                "kernels_list": str(binding.manifest.kernels_list),
                "trace_semantics": binding.manifest.trace_semantics,
                "replay_safe": binding.manifest.replay_safe,
                "compatibility": binding.compatibility,
                "task_ids": [],
            },
        )
        task_ids = trace_record["task_ids"]
        assert isinstance(task_ids, list)
        task_ids.append(node.node_id)
        exact = binding.compatibility == "exact_operator"
        address_binding_path = output / "online_address_binding.json"
        address_binding = (
            json.loads(address_binding_path.read_text(encoding="utf-8"))
            if address_binding_path.is_file()
            else None
        )
        qualified_performance = exact
        if binding.operator_artifact is not None:
            qualification = binding.operator_artifact.payload["qualification"]
            assert isinstance(qualification, Mapping)
            qualified_performance = bool(
                qualification.get("performance_eligible", False)
            )
        return BackendTaskResult(
            backend_id=self._accel_sim.config.backend_id,
            duration_fs=duration_fs,
            resource_id=device_id,
            timing_contract=self._gpu_contract.to_dict(),
            fidelity={
                "compute_fidelity": "cycle_simulated"
                if exact
                else "cycle_simulated_surrogate",
                "memory_fidelity": "cycle_simulated_external_ramulator2"
                if self._accel_sim.config.external_memory is not None
                else "cycle_simulated_local_dram",
                "link_fidelity": "not_applicable",
                "scheduler_fidelity": "event_modeled",
                "extrapolated_fraction": 0.0 if exact else 1.0,
                "trace_coverage": 1.0,
                "performance_eligible": qualified_performance,
            },
            statistics={
                "cycles": cycles,
                "instructions": instructions,
                "duration_fs": duration_fs,
                "cache_hit": cache_hit,
                "external_memory_stats": external_memory_stats,
                "runtime_global_pa_binding": address_binding,
            },
            artifact={
                "kind": "accel_sim_trace",
                "trace_id": binding.manifest.trace_id,
                "trace_key": trace_key,
                "simulation_key": simulation_key,
                "compatibility": binding.compatibility,
                "output_directory": str(output),
                "operator_artifact_id": binding.operator_artifact.artifact_id
                if binding.operator_artifact
                else None,
                "request_cycle_ready": binding.operator_artifact.request_cycle_ready
                if binding.operator_artifact
                else False,
            },
        )

    def _run_atlas(
        self,
        binding: AtlasArtifactBinding,
        node: ModelNode,
        device_id: str,
    ) -> BackendTaskResult:
        assert self._atlas is not None and self._atlas_contract is not None
        simulation_key = self._atlas.simulation_key(
            binding.chip_config, binding.artifact
        )
        output = self.output_root / "atlas" / simulation_key
        stats_path = output / "stats.json"
        cache_hit = stats_path.is_file()
        if cache_hit:
            normalized = json.loads(stats_path.read_text(encoding="utf-8"))
            cycles = int(normalized["cycles"])
            duration_fs = int(normalized["duration_fs"])
            energy_j = float(normalized["energy_j"])
            native_stats = normalized.get("native_stats", {})
        else:
            result = self._atlas.run(binding.chip_config, binding.artifact, output)
            cycles = result.cycles
            duration_fs = result.duration_fs
            energy_j = result.energy_j
            native_stats = result.stats
        artifact_key = simulation_key
        record = self._used_atlas_artifacts.setdefault(
            artifact_key,
            {
                "artifact_key": artifact_key,
                "chip_config": str(binding.chip_config),
                "operator_list": str(binding.artifact.operator_list),
                "placement_map": str(binding.artifact.placement_map),
                "compatibility": binding.compatibility,
                "task_ids": [],
            },
        )
        task_ids = record["task_ids"]
        assert isinstance(task_ids, list)
        task_ids.append(node.node_id)
        exact = binding.compatibility == "exact_operator"
        e2e_stats = (
            native_stats.get("e2e_stats", {})
            if isinstance(native_stats, Mapping)
            else {}
        )
        return BackendTaskResult(
            backend_id=self._atlas.config.backend_id,
            duration_fs=duration_fs,
            resource_id=device_id,
            timing_contract=self._atlas_contract.to_dict(),
            fidelity={
                "compute_fidelity": "cycle_simulated"
                if exact
                else "cycle_simulated_surrogate",
                "memory_fidelity": "cycle_simulated_internal_3d_dram",
                "link_fidelity": "not_applicable",
                "scheduler_fidelity": "event_modeled",
                "extrapolated_fraction": 0.0 if exact else 1.0,
                "trace_coverage": 0.0,
                "artifact_coverage": 1.0,
                "performance_eligible": exact,
            },
            statistics={
                "cycles": cycles,
                "duration_fs": duration_fs,
                "energy_j": energy_j,
                "cache_hit": cache_hit,
                "component_cycles": {
                    key: value
                    for key, value in e2e_stats.items()
                    if key.endswith("_cycles")
                },
            },
            artifact={
                "kind": "atlas_operator_bundle",
                "artifact_key": artifact_key,
                "compatibility": binding.compatibility,
                "chip_config": str(binding.chip_config),
                "operator_list": str(binding.artifact.operator_list),
                "placement_map": str(binding.artifact.placement_map),
                "output_directory": str(output),
            },
        )

    def dispatch(
        self,
        backend_key: str,
        node: ModelNode,
        model: ModelSpec,
        device_id: str,
        task_id: str | None = None,
    ) -> BackendTaskResult:
        backend = self._backend_config(backend_key)
        kind = str(backend.get("kind"))
        if backend_key == "gpu" and kind == "accel_sim":
            binding = self._matching_trace(node, model)
            if binding is not None:
                return self._run_accel_sim(binding, node, device_id, task_id)
            fallback = backend.get("fallback_kind", "none")
            if fallback != "analytical":
                raise OperatorEventError(
                    f"GPU node {node.node_id} has no matching trace and no "
                    "analytical fallback"
                )
            return self._analytical(
                backend_key,
                backend,
                node,
                model,
                device_id,
                fallback_reason="no_matching_accel_sim_trace",
            )
        if backend_key == "atlas" and kind == "atlasim":
            binding = self._matching_atlas_artifact(node)
            if binding is not None:
                return self._run_atlas(binding, node, device_id)
            fallback = backend.get("fallback_kind", "none")
            if fallback != "analytical":
                raise OperatorEventError(
                    f"ATLAS node {node.node_id} has no matching artifact and no "
                    "analytical fallback"
                )
            return self._analytical(
                backend_key,
                backend,
                node,
                model,
                device_id,
                fallback_reason="no_matching_atlas_artifact",
            )
        if kind == "analytical" or (backend_key == "gpu" and kind == "roofline"):
            return self._analytical(backend_key, backend, node, model, device_id)
        if kind == "none":
            raise OperatorEventError(
                f"node {node.node_id} was placed on disabled backend {backend_key}"
            )
        raise OperatorEventError(
            f"operator-event backend {backend_key}.{kind} is not implemented"
        )

    def trace_bundle(self) -> dict[str, object]:
        return {
            "schema_version": "hetero-run-trace-bundle/v1",
            "captures": list(self._used_traces.values()),
            "atlas_artifacts": list(self._used_atlas_artifacts.values()),
        }

    def provenance(self) -> dict[str, object]:
        return {
            "timing_owners": self.ownership.to_dict(),
            "gpu_timing_contract": self._gpu_contract.to_dict()
            if self._gpu_contract
            else None,
            "atlas_timing_contract": self._atlas_contract.to_dict()
            if self._atlas_contract
            else None,
            "request_cycle_ready_required": self._require_request_cycle_ready,
            "runtime_global_pa_bindings": list(self._runtime_binding_records.values()),
        }
