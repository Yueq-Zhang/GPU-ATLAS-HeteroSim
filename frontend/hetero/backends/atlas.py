"""Independent total-duration adapter for prepared ATLAS operator bundles."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .contracts import BackendDescriptor


class AtlasBackendError(RuntimeError):
    """Raised for ATLAS configuration, invocation, or statistics errors."""


def _resolve(path: object, base: Path) -> Path:
    result = Path(str(path))
    return result if result.is_absolute() else (base / result).resolve()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


_CONFIG_REFERENCE = re.compile(
    r"^\s*(?:config_path|tech_file)\s*[:=]\s*([^\s#;]+)", re.MULTILINE
)


def _config_closure_hashes(root_file: Path, atlas_root: Path) -> dict[str, str]:
    """Hash a chip config and its explicit DRAM/NoC configuration references."""
    pending = [root_file.resolve()]
    visited: set[Path] = set()
    result: dict[str, str] = {}
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        if not path.is_file():
            raise AtlasBackendError(f"referenced ATLAS config does not exist: {path}")
        visited.add(path)
        try:
            name = str(path.relative_to(atlas_root.resolve()))
        except ValueError:
            name = str(path)
        result[name] = _sha256(path)
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in _CONFIG_REFERENCE.finditer(text):
            reference = Path(match.group(1).strip("'\""))
            resolved = (
                reference
                if reference.is_absolute()
                else (atlas_root / reference).resolve()
            )
            pending.append(resolved)
    return dict(sorted(result.items()))


@dataclass(frozen=True)
class AtlasBackendConfig:
    backend_id: str
    python_executable: Path
    atlas_root: Path
    adapter_script: Path
    core_frequency_hz: int
    timeout_seconds: int
    dependency_commits: Mapping[str, str]
    source_path: Path

    @classmethod
    def load(cls, path: Path) -> "AtlasBackendConfig":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise AtlasBackendError(f"cannot read backend config {path}: {error}") from error
        required = {
            "schema_version",
            "backend_id",
            "python_executable",
            "atlas_root",
            "adapter_script",
            "core_frequency_hz",
            "timeout_seconds",
            "dependency_commits",
        }
        if not isinstance(payload, dict) or set(payload) != required:
            actual = set(payload) if isinstance(payload, dict) else set()
            raise AtlasBackendError(
                f"backend config keys mismatch: missing={sorted(required - actual)}, "
                f"extra={sorted(actual - required)}"
            )
        if payload["schema_version"] != "hetero-atlas-backend/v1":
            raise AtlasBackendError("schema_version must be hetero-atlas-backend/v1")
        base = path.resolve().parent
        commits = payload["dependency_commits"]
        if not isinstance(commits, dict) or not commits:
            raise AtlasBackendError("dependency_commits must be a non-empty object")
        result = cls(
            backend_id=str(payload["backend_id"]),
            python_executable=_resolve(payload["python_executable"], base),
            atlas_root=_resolve(payload["atlas_root"], base),
            adapter_script=_resolve(payload["adapter_script"], base),
            core_frequency_hz=int(payload["core_frequency_hz"]),
            timeout_seconds=int(payload["timeout_seconds"]),
            dependency_commits={str(k): str(v) for k, v in commits.items()},
            source_path=path.resolve(),
        )
        if result.core_frequency_hz <= 0 or result.timeout_seconds <= 0:
            raise AtlasBackendError(
                "core_frequency_hz and timeout_seconds must be positive"
            )
        return result

    def validate_files(self) -> None:
        for label, path, kind in (
            ("python_executable", self.python_executable, "file"),
            ("atlas_root", self.atlas_root, "directory"),
            ("adapter_script", self.adapter_script, "file"),
        ):
            valid = path.is_file() if kind == "file" else path.is_dir()
            if not valid:
                raise AtlasBackendError(f"{label} does not exist: {path}")


@dataclass(frozen=True)
class AtlasArtifact:
    operator_list: Path
    placement_map: Path

    def validate_files(self) -> None:
        for label, path in (
            ("operator_list", self.operator_list),
            ("placement_map", self.placement_map),
        ):
            if not path.is_file():
                raise AtlasBackendError(f"{label} does not exist: {path}")


@dataclass(frozen=True)
class AtlasRunResult:
    command: tuple[str, ...]
    duration_fs: int
    cycles: int
    energy_j: float
    stats: Mapping[str, object]
    output_directory: Path


class AtlasBackend:
    """Invoke ATLAS without changing ownership of its internal 3D-DRAM timing."""

    def __init__(self, config: AtlasBackendConfig):
        self.config = config

    def descriptor(self) -> BackendDescriptor:
        return BackendDescriptor(
            backend_id=self.config.backend_id,
            supported_duration_semantics=("total",),
            ownable_resource_kinds=(
                "atlas_core",
                "atlas_sram",
                "atlas_noc",
                "atlas_3d_dram",
            ),
            supported_exports=(),
            supports_stall_resume=False,
            supported_trace_semantics=("none",),
            qualification_records=("adapter_equivalence",),
        )

    def simulation_key(self, chip_path: Path, artifact: AtlasArtifact) -> str:
        chip_path = chip_path.resolve()
        artifact = AtlasArtifact(
            artifact.operator_list.resolve(), artifact.placement_map.resolve()
        )
        payload = {
            "backend_id": self.config.backend_id,
            "dependency_commits": self.config.dependency_commits,
            "adapter_sha256": _sha256(self.config.adapter_script),
            "architecture_config_closure": _config_closure_hashes(
                chip_path, self.config.atlas_root
            ),
            "operator_list_sha256": _sha256(artifact.operator_list),
            "placement_map_sha256": _sha256(artifact.placement_map),
            "core_frequency_hz": self.config.core_frequency_hz,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def run(
        self,
        chip_path: Path,
        artifact: AtlasArtifact,
        output_directory: Path,
    ) -> AtlasRunResult:
        # The subprocess runs from ATLAS's source root so every user-provided
        # artifact path must be made absolute before constructing argv.
        chip_path = chip_path.resolve()
        artifact = AtlasArtifact(
            artifact.operator_list.resolve(), artifact.placement_map.resolve()
        )
        self.config.validate_files()
        artifact.validate_files()
        if not chip_path.is_file():
            raise AtlasBackendError(f"chip config does not exist: {chip_path}")
        output_directory.mkdir(parents=True, exist_ok=True)
        native_stats = output_directory / "native_stats.json"
        command = (
            str(self.config.python_executable),
            str(self.config.adapter_script),
            "--chip",
            str(chip_path),
            "--operators",
            str(artifact.operator_list),
            "--placement",
            str(artifact.placement_map),
            "--output",
            str(native_stats),
        )
        (output_directory / "command.json").write_text(
            json.dumps(
                {
                    "schema_version": "hetero-command/v1",
                    "argv": list(command),
                    "backend_id": self.config.backend_id,
                    "simulation_key": self.simulation_key(chip_path, artifact),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        try:
            process = subprocess.run(
                command,
                cwd=self.config.atlas_root,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise AtlasBackendError(f"ATLAS invocation failed: {error}") from error
        (output_directory / "stdout.log").write_text(process.stdout, encoding="utf-8")
        (output_directory / "stderr.log").write_text(process.stderr, encoding="utf-8")
        if process.returncode != 0 or not native_stats.is_file():
            raise AtlasBackendError(
                f"ATLAS returned {process.returncode}; see {output_directory}"
            )
        stats = json.loads(native_stats.read_text(encoding="utf-8"))
        e2e = stats.get("e2e_stats")
        if not isinstance(e2e, dict) or "e2e_cycles" not in e2e:
            raise AtlasBackendError("ATLAS output lacks e2e_stats.e2e_cycles")
        cycles = int(e2e["e2e_cycles"])
        energy_j = float(e2e.get("e2e_energy", 0.0))
        reported_mhz = float(stats.get("chip_frequency_mhz", 0.0))
        if reported_mhz > 0 and not math.isclose(
            reported_mhz * 1_000_000,
            self.config.core_frequency_hz,
            rel_tol=0.0,
            abs_tol=0.5,
        ):
            raise AtlasBackendError(
                "ATLAS reported chip frequency does not match backend config: "
                f"{reported_mhz} MHz vs {self.config.core_frequency_hz} Hz"
            )
        duration_fs = math.ceil(
            cycles * 1_000_000_000_000_000 / self.config.core_frequency_hz
        )
        normalized = {
            "schema_version": "hetero-atlas-stats/v1",
            "backend_id": self.config.backend_id,
            "cycles": cycles,
            "duration_fs": duration_fs,
            "energy_j": energy_j,
            "core_frequency_hz": self.config.core_frequency_hz,
            "native_stats": stats,
        }
        (output_directory / "stats.json").write_text(
            json.dumps(normalized, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return AtlasRunResult(
            command=command,
            duration_fs=duration_fs,
            cycles=cycles,
            energy_j=energy_j,
            stats=stats,
            output_directory=output_directory,
        )

    def qualify(
        self,
        chip_path: Path,
        artifact: AtlasArtifact,
        output_directory: Path,
    ) -> Path:
        """Require exact deterministic statistics across two adapter invocations."""
        chip_path = chip_path.resolve()
        artifact = AtlasArtifact(
            artifact.operator_list.resolve(), artifact.placement_map.resolve()
        )
        baseline = self.run(chip_path, artifact, output_directory / "native_baseline")
        adapter = self.run(chip_path, artifact, output_directory / "adapter")
        compared = {
            "cycles": [baseline.cycles, adapter.cycles],
            "energy_j": [baseline.energy_j, adapter.energy_j],
            "native_stats": [baseline.stats, adapter.stats],
        }
        passed = all(left == right for left, right in compared.values())
        record = output_directory / "qualification_record.json"
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text(
            json.dumps(
                {
                    "schema_version": "hetero-atlas-qualification/v1",
                    "status": "passed" if passed else "failed",
                    "backend_id": self.config.backend_id,
                    "simulation_key": self.simulation_key(chip_path, artifact),
                    "comparison": compared,
                    "exact_match_required": True,
                    "qualified_scopes": ["adapter_equivalence"],
                    "timing_ownership": {
                        "atlas_core_sram_noc_3d_dram": "atlasim",
                        "external_ramulator2": False,
                        "duration_mode": "total",
                    },
                    "provenance": {
                        "backend_config": str(self.config.source_path),
                        "chip_config": str(chip_path),
                        "operator_list": str(artifact.operator_list),
                        "placement_map": str(artifact.placement_map),
                        "dependency_commits": dict(self.config.dependency_commits),
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        if not passed:
            raise AtlasBackendError(f"qualification mismatch; see {record}")
        return record
