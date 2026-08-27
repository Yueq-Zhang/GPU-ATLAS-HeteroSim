"""Operator-event Backend dispatch for the second integration step."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

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
from .ir import ModelNode
from .model_graph import ModelSpec
from .trace_manifest import TraceManifest


class OperatorEventError(RuntimeError):
    """Raised when an operator-event Backend cannot produce a valid duration."""


@dataclass(frozen=True, slots=True)
class TraceBinding:
    selector: Mapping[str, object]
    manifest: TraceManifest
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
        self._accel_sim = AccelSimBackend(backend_config)
        raw_bindings = gpu.get("trace_bindings")
        if not isinstance(raw_bindings, list) or not raw_bindings:
            raise OperatorEventError("accel_sim GPU backend requires trace_bindings")
        for index, raw in enumerate(raw_bindings):
            if not isinstance(raw, Mapping):
                raise OperatorEventError(f"trace_bindings[{index}] must be an object")
            allowed = {"selector", "trace_manifest", "compatibility"}
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
                    f"unknown trace selector fields: {sorted(set(selector) - allowed_selector)}"
                )
            compatibility = str(raw.get("compatibility", "exact_operator"))
            if compatibility not in {"exact_operator", "surrogate_plumbing_probe"}:
                raise OperatorEventError(
                    f"invalid trace binding compatibility: {compatibility}"
                )
            manifest_path = _resolve_path(raw.get("trace_manifest"), self.project_root)
            self._trace_bindings.append(
                TraceBinding(
                    selector=dict(selector),
                    manifest=TraceManifest.load(manifest_path),
                    compatibility=compatibility,
                )
            )
        resources = gpu.get("resource_bindings")
        if not isinstance(resources, Mapping) or not resources:
            raise OperatorEventError(
                "accel_sim GPU backend requires explicit resource_bindings"
            )
        requested_mode = str(gpu.get("requested_timing_mode", "total"))
        replay_safe = all(binding.manifest.replay_safe for binding in self._trace_bindings)
        self._gpu_contract = resolve_timing_contract(
            self._accel_sim.descriptor(),
            requested_mode,
            resources,
            trace_semantics="functional",
            replay_safe=replay_safe,
        )
        self.ownership.register(self._gpu_contract)

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
                raise OperatorEventError(f"artifact_bindings[{index}] must be an object")
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
                    chip_config=_resolve_path(raw.get("chip_config"), self.project_root),
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

    def _matching_trace(self, node: ModelNode) -> TraceBinding | None:
        matches = [binding for binding in self._trace_bindings if binding.matches(node)]
        if len(matches) > 1:
            raise OperatorEventError(
                f"multiple GPU traces match node {node.node_id}; selectors must be unique"
            )
        return matches[0] if matches else None

    def _matching_atlas_artifact(
        self, node: ModelNode
    ) -> AtlasArtifactBinding | None:
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
            artifact={"kind": "analytical", "parameter_source": backend.get("parameter_source")},
        )

    def _run_accel_sim(
        self,
        binding: TraceBinding,
        node: ModelNode,
        device_id: str,
    ) -> BackendTaskResult:
        assert self._accel_sim is not None and self._gpu_contract is not None
        simulation_key = self._accel_sim.simulation_key(binding.manifest)
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
            result = self._accel_sim.run(binding.manifest, output)
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
                "performance_eligible": exact,
            },
            statistics={
                "cycles": cycles,
                "instructions": instructions,
                "duration_fs": duration_fs,
                "cache_hit": cache_hit,
                "external_memory_stats": external_memory_stats,
            },
            artifact={
                "kind": "accel_sim_trace",
                "trace_id": binding.manifest.trace_id,
                "trace_key": trace_key,
                "simulation_key": simulation_key,
                "compatibility": binding.compatibility,
                "output_directory": str(output),
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
            result = self._atlas.run(
                binding.chip_config, binding.artifact, output
            )
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
    ) -> BackendTaskResult:
        backend = self._backend_config(backend_key)
        kind = str(backend.get("kind"))
        if backend_key == "gpu" and kind == "accel_sim":
            binding = self._matching_trace(node)
            if binding is not None:
                return self._run_accel_sim(binding, node, device_id)
            fallback = backend.get("fallback_kind", "none")
            if fallback != "analytical":
                raise OperatorEventError(
                    f"GPU node {node.node_id} has no matching trace and no analytical fallback"
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
        }
