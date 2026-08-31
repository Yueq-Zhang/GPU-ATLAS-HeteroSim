"""Versioned, shape-locked operator artifact manifests for P15.

An artifact manifest is deliberately stricter than a selector in an experiment
configuration.  It binds the checkpoint, phase, operator shape, compilation
target, captured files and address semantics.  Loading validates every file
hash so a stale kernels list cannot silently acquire the identity of a prior
capture.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


class OperatorArtifactError(ValueError):
    """Raised when an operator artifact is incomplete or incompatible."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_integer(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise OperatorArtifactError(f"{field} must be a positive integer")
    return value


def _unsigned_integer(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise OperatorArtifactError(f"{field} must be an unsigned integer")
    return value


def _nonempty(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise OperatorArtifactError(f"{field} must be a non-empty string")
    return value


@dataclass(frozen=True, slots=True)
class OperatorCompatibilityKey:
    model_spec_name: str
    checkpoint_revision: str
    operator: str
    phase: str
    layer_id: int
    batch_size: int
    context_length: int
    q_len: int
    kv_length: int
    dtype: str

    @classmethod
    def from_contract(
        cls, contract: Mapping[str, object]
    ) -> "OperatorCompatibilityKey":
        layer_id = contract.get("layer_id")
        if not isinstance(layer_id, int) or isinstance(layer_id, bool) or layer_id < 0:
            raise OperatorArtifactError("source_contract.layer_id must be unsigned")
        phase = _nonempty(contract.get("phase"), "source_contract.phase")
        if phase not in {"prefill", "decode_step"}:
            raise OperatorArtifactError(
                "source_contract.phase must be prefill or decode_step"
            )
        return cls(
            model_spec_name=_nonempty(
                contract.get("model_spec_name"), "source_contract.model_spec_name"
            ),
            checkpoint_revision=_nonempty(
                contract.get("checkpoint_revision"),
                "source_contract.checkpoint_revision",
            ),
            operator=_nonempty(
                contract.get("operator"), "source_contract.operator"
            ),
            phase=phase,
            layer_id=layer_id,
            batch_size=_positive_integer(
                contract.get("batch_size"), "source_contract.batch_size"
            ),
            context_length=_positive_integer(
                contract.get("context_length"), "source_contract.context_length"
            ),
            q_len=_positive_integer(contract.get("q_len"), "source_contract.q_len"),
            kv_length=_positive_integer(
                contract.get("kv_length"), "source_contract.kv_length"
            ),
            dtype=_nonempty(contract.get("dtype"), "source_contract.dtype"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "model_spec_name": self.model_spec_name,
            "checkpoint_revision": self.checkpoint_revision,
            "operator": self.operator,
            "phase": self.phase,
            "layer_id": self.layer_id,
            "batch_size": self.batch_size,
            "context_length": self.context_length,
            "q_len": self.q_len,
            "kv_length": self.kv_length,
            "dtype": self.dtype,
        }


@dataclass(frozen=True, slots=True)
class OperatorArtifactManifest:
    source_path: Path
    artifact_id: str
    compatibility_key: OperatorCompatibilityKey
    payload: Mapping[str, object]
    content_sha256: str

    @classmethod
    def load(cls, path: Path) -> "OperatorArtifactManifest":
        path = path.resolve()
        try:
            raw = path.read_bytes()
            payload = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise OperatorArtifactError(f"cannot read artifact {path}: {error}") from error
        if not isinstance(payload, Mapping):
            raise OperatorArtifactError("operator artifact root must be an object")
        required = {
            "schema_version",
            "artifact_id",
            "source_contract",
            "backend",
            "execution_contract",
            "address_contract",
            "qualification",
            "tensors",
            "files",
        }
        missing = required - payload.keys()
        extra = payload.keys() - required
        if missing or extra:
            raise OperatorArtifactError(
                f"artifact keys mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
            )
        if payload["schema_version"] != "hetero-operator-artifact/v1":
            raise OperatorArtifactError("invalid operator artifact schema_version")
        artifact_id = _nonempty(payload["artifact_id"], "artifact_id")
        source = payload["source_contract"]
        if not isinstance(source, Mapping):
            raise OperatorArtifactError("source_contract must be an object")
        key = OperatorCompatibilityKey.from_contract(source)
        cls._validate_contracts(payload)
        cls._validate_files(path.parent, payload["files"])
        return cls(path, artifact_id, key, dict(payload), hashlib.sha256(raw).hexdigest())

    @staticmethod
    def _validate_contracts(payload: Mapping[str, object]) -> None:
        backend = payload["backend"]
        execution = payload["execution_contract"]
        address = payload["address_contract"]
        qualification = payload["qualification"]
        tensors = payload["tensors"]
        for name, value in (
            ("backend", backend),
            ("execution_contract", execution),
            ("address_contract", address),
            ("qualification", qualification),
        ):
            if not isinstance(value, Mapping):
                raise OperatorArtifactError(f"{name} must be an object")
        if backend.get("kind") not in {"accel_sim", "atlasim", "runtime_state"}:
            raise OperatorArtifactError(
                "backend.kind must be accel_sim, atlasim or runtime_state"
            )
        if execution.get("memory_traffic") not in {
            "not_extracted",
            "sampled",
            "full",
            "full_instruction_trace",
        }:
            raise OperatorArtifactError("invalid execution_contract.memory_traffic")
        stall_resume = execution.get("supports_stall_resume")
        if not isinstance(stall_resume, bool):
            raise OperatorArtifactError(
                "execution_contract.supports_stall_resume must be boolean"
            )
        compute_memory_coupled = execution.get("compute_memory_coupled", False)
        if not isinstance(compute_memory_coupled, bool):
            raise OperatorArtifactError(
                "execution_contract.compute_memory_coupled must be boolean"
            )
        global_pa_binding_ready = execution.get("global_pa_binding_ready", False)
        if not isinstance(global_pa_binding_ready, bool):
            raise OperatorArtifactError(
                "execution_contract.global_pa_binding_ready must be boolean"
            )
        if compute_memory_coupled and (
            execution.get("memory_traffic") != "full_instruction_trace"
            or not stall_resume
        ):
            raise OperatorArtifactError(
                "compute_memory_coupled requires full instruction-trace traffic "
                "and stall/resume"
            )
        if execution.get("request_cycle_ready") is True:
            if (
                execution.get("memory_traffic") != "full_instruction_trace"
                or not stall_resume
                or not compute_memory_coupled
                or not global_pa_binding_ready
            ):
                raise OperatorArtifactError(
                    "request_cycle_ready requires coupled full instruction-trace "
                    "traffic, stall/resume and Global PA binding"
                )
        required_address = {
            "capture_address",
            "normalized_address",
            "global_pa_binding",
            "virtual_memory_mode",
            "dram_mapping",
        }
        optional_address = {"capture_allocator_coverage"}
        if not required_address.issubset(address) or (
            set(address) - required_address - optional_address
        ):
            raise OperatorArtifactError(
                "address_contract must declare all five address boundaries and "
                "only supported optional fields"
            )
        allocator_coverage = address.get("capture_allocator_coverage")
        if allocator_coverage is not None and allocator_coverage not in {
            "semantic_tensors_only",
            "target_window_pytorch_allocator",
            "target_window_pytorch_allocator_plus_tensor_segments",
        }:
            raise OperatorArtifactError("invalid capture_allocator_coverage")
        if address.get("capture_address") not in {
            "trace_address",
            "cuda_allocation_address_no_instruction_trace",
            "atlas_per_core_local_address",
        }:
            raise OperatorArtifactError("invalid capture_address semantics")
        if (
            backend.get("kind") == "runtime_state"
            and address.get("capture_address")
            != "cuda_allocation_address_no_instruction_trace"
        ):
            raise OperatorArtifactError(
                "runtime_state must not claim instruction-trace addresses"
            )
        if (
            backend.get("kind") == "atlasim"
            and address.get("capture_address") != "atlas_per_core_local_address"
        ):
            raise OperatorArtifactError(
                "atlasim must declare its per-core local address boundary"
            )
        if address.get("normalized_address") != "tensor_id_plus_offset":
            raise OperatorArtifactError(
                "normalized_address must be tensor_id_plus_offset"
            )
        if address.get("global_pa_binding") != "required_at_simulation":
            raise OperatorArtifactError(
                "global_pa_binding must be required_at_simulation"
            )
        if address.get("dram_mapping") != "candidate_specific_after_global_pa":
            raise OperatorArtifactError(
                "dram_mapping must be candidate_specific_after_global_pa"
            )
        eligible = qualification.get("performance_eligible")
        if not isinstance(eligible, bool):
            raise OperatorArtifactError(
                "qualification.performance_eligible must be boolean"
            )
        if eligible and execution.get("request_cycle_ready") is not True:
            raise OperatorArtifactError(
                "performance eligibility requires request-cycle-ready execution"
            )
        if not isinstance(tensors, list) or not tensors:
            raise OperatorArtifactError("tensors must be a non-empty array")
        addresses: list[tuple[int, int, str]] = []
        tensor_ids: set[str] = set()
        for index, item in enumerate(tensors):
            if not isinstance(item, Mapping):
                raise OperatorArtifactError(f"tensors[{index}] must be an object")
            tensor_id = _nonempty(item.get("tensor_id"), f"tensors[{index}].tensor_id")
            if tensor_id in tensor_ids:
                raise OperatorArtifactError(f"duplicate tensor_id: {tensor_id}")
            tensor_ids.add(tensor_id)
            base = _unsigned_integer(
                item.get("trace_base"), f"tensors[{index}].trace_base"
            )
            size = _positive_integer(item.get("size_bytes"), f"tensors[{index}].size_bytes")
            alignment = _positive_integer(
                item.get("alignment_bytes"), f"tensors[{index}].alignment_bytes"
            )
            if base % alignment:
                raise OperatorArtifactError(f"unaligned tensor trace_base: {tensor_id}")
            shape = item.get("shape")
            if not isinstance(shape, list) or not shape or any(
                not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0
                for dim in shape
            ):
                raise OperatorArtifactError(f"invalid tensor shape: {tensor_id}")
            addresses.append((base, base + size, tensor_id))
        addresses.sort()
        for left, right in zip(addresses, addresses[1:]):
            if left[1] > right[0]:
                raise OperatorArtifactError(
                    f"trace allocations overlap: {left[2]} and {right[2]}"
                )

    @staticmethod
    def _validate_files(base: Path, files: object) -> None:
        if not isinstance(files, list) or not files:
            raise OperatorArtifactError("files must be a non-empty array")
        seen: set[Path] = set()
        for index, item in enumerate(files):
            if not isinstance(item, Mapping):
                raise OperatorArtifactError(f"files[{index}] must be an object")
            value = _nonempty(item.get("path"), f"files[{index}].path")
            expected = _nonempty(item.get("sha256"), f"files[{index}].sha256")
            path = Path(value)
            if not path.is_absolute():
                path = (base / path).resolve()
            if path in seen:
                raise OperatorArtifactError(f"duplicate artifact file: {path}")
            seen.add(path)
            if not path.is_file():
                raise OperatorArtifactError(f"artifact file does not exist: {path}")
            if _sha256(path) != expected:
                raise OperatorArtifactError(f"artifact file hash mismatch: {path}")

    @property
    def request_cycle_ready(self) -> bool:
        execution = self.payload["execution_contract"]
        assert isinstance(execution, Mapping)
        return execution.get("request_cycle_ready") is True

    @property
    def compute_memory_coupled(self) -> bool:
        execution = self.payload["execution_contract"]
        assert isinstance(execution, Mapping)
        return execution.get("compute_memory_coupled") is True


@dataclass(frozen=True, slots=True)
class OperatorArtifactCatalog:
    source_path: Path
    artifacts: tuple[OperatorArtifactManifest, ...]
    required_operators: tuple[str, ...]
    zero_fallback_required: bool

    @classmethod
    def load(cls, path: Path) -> "OperatorArtifactCatalog":
        path = path.resolve()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise OperatorArtifactError(f"cannot read catalog {path}: {error}") from error
        if not isinstance(payload, Mapping):
            raise OperatorArtifactError("artifact catalog root must be an object")
        if payload.get("schema_version") != "hetero-operator-artifact-catalog/v1":
            raise OperatorArtifactError("invalid artifact catalog schema_version")
        refs = payload.get("artifacts")
        required = payload.get("required_operators")
        zero_fallback = payload.get("zero_fallback_required")
        if not isinstance(refs, list):
            raise OperatorArtifactError("catalog artifacts must be an array")
        if not isinstance(required, list) or any(
            not isinstance(item, str) or not item for item in required
        ):
            raise OperatorArtifactError("required_operators must be a string array")
        if not isinstance(zero_fallback, bool):
            raise OperatorArtifactError("zero_fallback_required must be boolean")
        artifacts = []
        keys: set[tuple[OperatorCompatibilityKey, str]] = set()
        for index, value in enumerate(refs):
            if not isinstance(value, str) or not value:
                raise OperatorArtifactError(f"artifacts[{index}] must be a path")
            artifact_path = Path(value)
            if not artifact_path.is_absolute():
                artifact_path = path.parent / artifact_path
            artifact = OperatorArtifactManifest.load(artifact_path)
            backend = artifact.payload["backend"]
            assert isinstance(backend, Mapping)
            unique_key = (artifact.compatibility_key, str(backend["kind"]))
            if unique_key in keys:
                raise OperatorArtifactError(
                    f"duplicate compatibility/backend key: {unique_key}"
                )
            keys.add(unique_key)
            artifacts.append(artifact)
        return cls(
            path,
            tuple(artifacts),
            tuple(str(item) for item in required),
            zero_fallback,
        )

    def coverage(self) -> dict[str, object]:
        represented = {item.compatibility_key.operator for item in self.artifacts}
        required = set(self.required_operators)
        ready = {
            item.compatibility_key.operator
            for item in self.artifacts
            if item.request_cycle_ready
        }
        by_backend: dict[str, list[str]] = {}
        for item in self.artifacts:
            backend = item.payload["backend"]
            assert isinstance(backend, Mapping)
            kind = str(backend["kind"])
            by_backend.setdefault(kind, []).append(item.compatibility_key.operator)
        return {
            "required_operators": sorted(required),
            "represented_operators": sorted(represented),
            "request_cycle_ready_operators": sorted(ready),
            "missing_operators": sorted(required - represented),
            "not_request_cycle_ready": sorted(required - ready),
            "artifact_count": len(self.artifacts),
            "represented_by_backend": {
                key: sorted(set(value)) for key, value in sorted(by_backend.items())
            },
            "zero_fallback_required": self.zero_fallback_required,
            "registration_complete": required <= represented,
            "request_cycle_coverage_complete": required <= ready,
        }

    def match(
        self,
        *,
        model_spec_name: str,
        operator: str,
        phase: str,
        layer_id: int,
        batch_size: int,
        context_length: int,
        q_len: int,
        kv_length: int,
        dtype: str,
        backend_kinds: set[str],
    ) -> OperatorArtifactManifest | None:
        matches: list[OperatorArtifactManifest] = []
        for artifact in self.artifacts:
            key = artifact.compatibility_key
            backend = artifact.payload["backend"]
            assert isinstance(backend, Mapping)
            if (
                str(backend["kind"]) in backend_kinds
                and key.model_spec_name == model_spec_name
                and key.operator == operator
                and key.phase == phase
                and key.layer_id == layer_id
                and key.batch_size == batch_size
                and key.context_length == context_length
                and key.q_len == q_len
                and key.kv_length == kv_length
                and key.dtype == dtype
            ):
                matches.append(artifact)
        if len(matches) > 1:
            raise OperatorArtifactError(
                f"multiple artifacts match {operator} on {sorted(backend_kinds)}"
            )
        return matches[0] if matches else None
