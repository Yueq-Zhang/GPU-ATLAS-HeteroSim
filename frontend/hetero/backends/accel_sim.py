"""Independent, total-duration Accel-Sim trace backend for M5 qualification."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from ..trace_manifest import TraceManifest


class AccelSimBackendError(RuntimeError):
    """Raised for configuration, execution or statistics errors."""


def _resolve(path: object, base: Path) -> Path:
    result = Path(str(path))
    return result if result.is_absolute() else (base / result).resolve()


@dataclass(frozen=True)
class AccelSimBackendConfig:
    backend_id: str
    executable: Path
    gpgpu_config: Path
    trace_config: Path
    target_gpu: str
    target_sm: int
    core_frequency_hz: int
    timeout_seconds: int
    dependency_commits: Mapping[str, str]
    environment: Mapping[str, str]
    source_path: Path

    @classmethod
    def load(cls, path: Path) -> "AccelSimBackendConfig":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise AccelSimBackendError(f"cannot read backend config {path}: {error}") from error
        if not isinstance(payload, dict):
            raise AccelSimBackendError("backend config root must be an object")
        required = {
            "schema_version",
            "backend_id",
            "executable",
            "gpgpu_config",
            "trace_config",
            "target_gpu",
            "target_sm",
            "core_frequency_hz",
            "timeout_seconds",
            "dependency_commits",
            "environment",
        }
        missing = required - payload.keys()
        extra = payload.keys() - required
        if missing or extra:
            raise AccelSimBackendError(
                f"backend config keys mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
            )
        if payload["schema_version"] != "hetero-accel-sim-backend/v1":
            raise AccelSimBackendError(
                "schema_version must be hetero-accel-sim-backend/v1"
            )
        base = path.resolve().parent
        commits = payload["dependency_commits"]
        environment = payload["environment"]
        if not isinstance(commits, dict) or not commits:
            raise AccelSimBackendError("dependency_commits must be a non-empty object")
        if not isinstance(environment, dict):
            raise AccelSimBackendError("environment must be an object")
        result = cls(
            backend_id=str(payload["backend_id"]),
            executable=_resolve(payload["executable"], base),
            gpgpu_config=_resolve(payload["gpgpu_config"], base),
            trace_config=_resolve(payload["trace_config"], base),
            target_gpu=str(payload["target_gpu"]),
            target_sm=int(payload["target_sm"]),
            core_frequency_hz=int(payload["core_frequency_hz"]),
            timeout_seconds=int(payload["timeout_seconds"]),
            dependency_commits={str(k): str(v) for k, v in commits.items()},
            environment={str(k): str(v) for k, v in environment.items()},
            source_path=path.resolve(),
        )
        if result.core_frequency_hz <= 0 or result.timeout_seconds <= 0:
            raise AccelSimBackendError(
                "core_frequency_hz and timeout_seconds must be positive"
            )
        return result

    def validate_files(self) -> None:
        for label, path in {
            "executable": self.executable,
            "gpgpu_config": self.gpgpu_config,
            "trace_config": self.trace_config,
        }.items():
            if not path.is_file():
                raise AccelSimBackendError(f"{label} does not exist: {path}")


@dataclass(frozen=True)
class AccelSimRunResult:
    command: tuple[str, ...]
    return_code: int
    duration_fs: int
    cycles: int
    instructions: int
    stats: Mapping[str, int | float]
    output_directory: Path


_STATISTIC = re.compile(
    r"^\s*([^=]+?)\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$"
)


def parse_accel_sim_stats(text: str) -> dict[str, int | float]:
    stats: dict[str, int | float] = {}
    for line in text.splitlines():
        match = _STATISTIC.match(line)
        if not match:
            continue
        name = match.group(1).strip()
        value = match.group(2)
        stats[name] = float(value) if any(c in value for c in ".eE") else int(value)
    return stats


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class AccelSimBackend:
    """Invoke the official trace executable without changing its timing ownership."""

    def __init__(self, config: AccelSimBackendConfig):
        self.config = config

    def command(self, manifest: TraceManifest) -> tuple[str, ...]:
        if manifest.kernels_list is None:
            raise AccelSimBackendError("manifest does not contain a captured GPU trace")
        return (
            str(self.config.executable),
            "-trace",
            str(manifest.kernels_list),
            "-config",
            str(self.config.gpgpu_config),
            "-config",
            str(self.config.trace_config),
        )

    def simulation_key(self, manifest: TraceManifest) -> str:
        payload = {
            "backend_id": self.config.backend_id,
            "dependency_commits": self.config.dependency_commits,
            "environment": self.config.environment,
            "gpgpu_config_sha256": hashlib.sha256(
                self.config.gpgpu_config.read_bytes()
            ).hexdigest(),
            "trace_config_sha256": hashlib.sha256(
                self.config.trace_config.read_bytes()
            ).hexdigest(),
            "trace_key": manifest.trace_key(),
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def run(self, manifest: TraceManifest, output_directory: Path) -> AccelSimRunResult:
        self.config.validate_files()
        if manifest.kernels_list is None or not manifest.kernels_list.is_file():
            raise AccelSimBackendError(
                f"kernels list does not exist: {manifest.kernels_list}"
            )
        output_directory.mkdir(parents=True, exist_ok=True)
        command = self.command(manifest)
        _write_json(
            output_directory / "command.json",
            {
                "schema_version": "hetero-command/v1",
                "argv": list(command),
                "backend_id": self.config.backend_id,
                "environment_overrides": dict(self.config.environment),
                "simulation_key": self.simulation_key(manifest),
            },
        )
        try:
            environment = os.environ.copy()
            environment.update(self.config.environment)
            process = subprocess.run(
                command,
                cwd=output_directory,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
                check=False,
                env=environment,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise AccelSimBackendError(f"Accel-Sim invocation failed: {error}") from error
        (output_directory / "stdout.log").write_text(process.stdout, encoding="utf-8")
        (output_directory / "stderr.log").write_text(process.stderr, encoding="utf-8")
        if process.returncode != 0:
            raise AccelSimBackendError(
                f"Accel-Sim returned {process.returncode}; see {output_directory}"
            )
        stats = parse_accel_sim_stats(process.stdout + "\n" + process.stderr)
        missing = {"gpu_tot_sim_cycle", "gpu_tot_sim_insn"} - stats.keys()
        if missing:
            raise AccelSimBackendError(
                f"Accel-Sim output lacks required statistics: {sorted(missing)}"
            )
        cycles = int(stats["gpu_tot_sim_cycle"])
        instructions = int(stats["gpu_tot_sim_insn"])
        duration_fs = math.ceil(cycles * 1_000_000_000_000_000 / self.config.core_frequency_hz)
        result = AccelSimRunResult(
            command=command,
            return_code=process.returncode,
            duration_fs=duration_fs,
            cycles=cycles,
            instructions=instructions,
            stats=stats,
            output_directory=output_directory,
        )
        _write_json(
            output_directory / "stats.json",
            {
                "schema_version": "hetero-accel-sim-stats/v1",
                "backend_id": self.config.backend_id,
                "cycles": cycles,
                "instructions": instructions,
                "duration_fs": duration_fs,
                "core_frequency_hz": self.config.core_frequency_hz,
                "raw_stats": stats,
            },
        )
        return result

    def qualify(self, manifest: TraceManifest, output_directory: Path) -> Path:
        """Compare direct baseline and adapter invocations of the same pinned backend."""
        baseline = self.run(manifest, output_directory / "native_baseline")
        adapter = self.run(manifest, output_directory / "adapter")
        compared = {
            "gpu_tot_sim_cycle": [baseline.cycles, adapter.cycles],
            "gpu_tot_sim_insn": [baseline.instructions, adapter.instructions],
        }
        passed = all(left == right for left, right in compared.values())
        record = output_directory / "qualification_record.json"
        _write_json(
            record,
            {
                "schema_version": "hetero-accel-sim-qualification/v1",
                "status": "passed" if passed else "failed",
                "backend_id": self.config.backend_id,
                "target_gpu": self.config.target_gpu,
                "target_sm": self.config.target_sm,
                "trace_id": manifest.trace_id,
                "trace_key": manifest.trace_key(),
                "simulation_key": self.simulation_key(manifest),
                "provenance": {
                    "backend_config": str(self.config.source_path),
                    "trace_manifest": str(manifest.source_path)
                    if manifest.source_path is not None
                    else None,
                    "dependency_commits": dict(self.config.dependency_commits),
                    "environment_overrides": dict(self.config.environment),
                },
                "comparison": compared,
                "exact_match_required": True,
                "timing_ownership": {
                    "gpu_core_cache_noc_local_dram": "accel_sim",
                    "external_ramulator2": False,
                    "duration_mode": "total",
                },
            },
        )
        if not passed:
            raise AccelSimBackendError(f"qualification mismatch; see {record}")
        if manifest.source_path is not None:
            qualified_payload = json.loads(
                manifest.source_path.read_text(encoding="utf-8")
            )
            qualified_payload["replay_safe"] = True
            qualified_payload["qualification_record"] = str(record.resolve())
            qualified_payload["kernels_list"] = str(manifest.kernels_list)
            _write_json(
                output_directory / "qualified_trace_manifest.json",
                qualified_payload,
            )
        return record
