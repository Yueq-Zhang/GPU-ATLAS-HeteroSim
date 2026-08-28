"""Independent, total-duration Accel-Sim trace backend for M5 qualification."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

from ..bandwidth import BandwidthContract, BandwidthContractError
from ..trace_manifest import TraceManifest
from .contracts import BackendDescriptor


class AccelSimBackendError(RuntimeError):
    """Raised for configuration, execution or statistics errors."""


def _resolve(path: object, base: Path) -> Path:
    result = Path(str(path))
    return result if result.is_absolute() else (base / result).resolve()


@dataclass(frozen=True)
class ExternalMemoryConfig:
    kind: str
    config_file: Path
    bridge_library: Path
    timing_owner: str
    expected_instances: int
    require_nonzero_requests: bool
    bandwidth_contract: BandwidthContract

    @classmethod
    def load(cls, payload: object, base: Path) -> "ExternalMemoryConfig":
        if not isinstance(payload, dict):
            raise AccelSimBackendError("external_memory must be an object")
        required = {
            "kind",
            "config_file",
            "bridge_library",
            "timing_owner",
            "expected_instances",
            "require_nonzero_requests",
            "bandwidth_contract",
        }
        missing = required - payload.keys()
        extra = payload.keys() - required
        if missing or extra:
            raise AccelSimBackendError(
                "external_memory keys mismatch: "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        if payload["kind"] != "ramulator2_in_process":
            raise AccelSimBackendError(
                "external_memory.kind must be ramulator2_in_process"
            )
        try:
            bandwidth_contract = BandwidthContract.load(
                payload["bandwidth_contract"]
            )
        except BandwidthContractError as error:
            raise AccelSimBackendError(str(error)) from error
        result = cls(
            kind=str(payload["kind"]),
            config_file=_resolve(payload["config_file"], base),
            bridge_library=_resolve(payload["bridge_library"], base),
            timing_owner=str(payload["timing_owner"]),
            expected_instances=int(payload["expected_instances"]),
            require_nonzero_requests=bool(payload["require_nonzero_requests"]),
            bandwidth_contract=bandwidth_contract,
        )
        if not result.timing_owner or result.expected_instances != 1:
            raise AccelSimBackendError(
                "external memory requires one non-empty timing owner and exactly one instance"
            )
        return result


@dataclass(frozen=True)
class CoResidentAtlasConfig:
    kind: str
    execution_semantics: str
    chip_config: Path
    operator_list: Path
    placement_map: Path
    expected_instances: int
    require_nonzero_requests: bool
    expected_transaction_bytes: int | None

    @classmethod
    def load(cls, payload: object, base: Path) -> "CoResidentAtlasConfig":
        if not isinstance(payload, dict):
            raise AccelSimBackendError("co_resident_atlas must be an object")
        required = {
            "kind",
            "execution_semantics",
            "chip_config",
            "operator_list",
            "placement_map",
            "expected_instances",
            "require_nonzero_requests",
        }
        optional = {"expected_transaction_bytes"}
        missing = required - payload.keys()
        extra = payload.keys() - (required | optional)
        if missing or extra:
            raise AccelSimBackendError(
                "co_resident_atlas keys mismatch: "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        if payload["kind"] != "full_chip_external_dram":
            raise AccelSimBackendError(
                "co_resident_atlas.kind must be full_chip_external_dram"
            )
        if payload["execution_semantics"] != "contention_stress_duplicate_operator":
            raise AccelSimBackendError(
                "co_resident_atlas.execution_semantics must explicitly be "
                "contention_stress_duplicate_operator"
            )
        expected_bytes = payload.get("expected_transaction_bytes")
        result = cls(
            kind=str(payload["kind"]),
            execution_semantics=str(payload["execution_semantics"]),
            chip_config=_resolve(payload["chip_config"], base),
            operator_list=_resolve(payload["operator_list"], base),
            placement_map=_resolve(payload["placement_map"], base),
            expected_instances=int(payload["expected_instances"]),
            require_nonzero_requests=bool(payload["require_nonzero_requests"]),
            expected_transaction_bytes=int(expected_bytes)
            if expected_bytes is not None
            else None,
        )
        if result.expected_instances != 1:
            raise AccelSimBackendError(
                "co-resident ATLAS requires exactly one full-chip runtime"
            )
        if (
            result.expected_transaction_bytes is not None
            and result.expected_transaction_bytes <= 0
        ):
            raise AccelSimBackendError(
                "expected_transaction_bytes must be positive when present"
            )
        return result


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
    external_memory: ExternalMemoryConfig | None
    co_resident_atlas: CoResidentAtlasConfig | None
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
        extra = payload.keys() - (
            required | {"external_memory", "co_resident_atlas"}
        )
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
            external_memory=ExternalMemoryConfig.load(payload["external_memory"], base)
            if "external_memory" in payload
            else None,
            co_resident_atlas=CoResidentAtlasConfig.load(
                payload["co_resident_atlas"], base
            )
            if "co_resident_atlas" in payload
            else None,
            source_path=path.resolve(),
        )
        if result.core_frequency_hz <= 0 or result.timeout_seconds <= 0:
            raise AccelSimBackendError(
                "core_frequency_hz and timeout_seconds must be positive"
            )
        if result.co_resident_atlas is not None and result.external_memory is None:
            raise AccelSimBackendError(
                "co_resident_atlas requires external_memory so both initiators "
                "share one Ramulator2"
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
        if self.external_memory is not None:
            for label, path in {
                "external memory config": self.external_memory.config_file,
                "external memory bridge": self.external_memory.bridge_library,
            }.items():
                if not path.is_file():
                    raise AccelSimBackendError(f"{label} does not exist: {path}")
        if self.co_resident_atlas is not None:
            for label, path in {
                "ATLAS chip config": self.co_resident_atlas.chip_config,
                "ATLAS operator list": self.co_resident_atlas.operator_list,
                "ATLAS placement map": self.co_resident_atlas.placement_map,
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
    external_memory_stats: Mapping[str, int] | None
    atlas_runtime_stats: Mapping[str, int | str] | None
    output_directory: Path


_STATISTIC = re.compile(
    r"^\s*([^=]+?)\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$"
)

_RAMULATOR2_STATISTIC = re.compile(
    r"heterosim_ramulator2(?:_summary)?\s+"
    r"cycles=(\d+)\s+reads=(\d+)\s+writes=(\d+)\s+"
    r"completed=(\d+)\s+rejected=(\d+)\s+outstanding=(\d+)\s+"
    r"instances=(\d+)(?:\s+partitions=(\d+))?"
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


def parse_ramulator2_stats(text: str) -> dict[str, int] | None:
    # ABI v2 emits an extensible final summary.  Preserve every integer field
    # so traffic-volume and per-initiator evidence reaches qualification
    # records instead of being discarded by the legacy fixed-width parser.
    for line in reversed(text.splitlines()):
        marker = "heterosim_ramulator2_summary "
        if marker not in line:
            continue
        fields = {
            name: int(value)
            for name, value in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)=(\d+)", line)
        }
        required = {
            "cycles",
            "reads",
            "writes",
            "completed",
            "rejected",
            "outstanding",
            "instances",
        }
        if required <= fields.keys():
            return fields

    matches = list(_RAMULATOR2_STATISTIC.finditer(text))
    if not matches:
        return None
    match = matches[-1]
    names = (
        "cycles",
        "reads",
        "writes",
        "completed",
        "rejected",
        "outstanding",
        "instances",
        "partitions",
    )
    return {
        name: int(value)
        for name, value in zip(names, match.groups())
        if value is not None
    }


def parse_atlas_full_chip_runtime_stats(
    text: str,
) -> dict[str, int | str] | None:
    marker = "heterosim_atlas_full_chip_runtime_summary "
    for line in reversed(text.splitlines()):
        if marker not in line:
            continue
        fields: dict[str, int | str] = {}
        for name, value in re.findall(
            r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)", line
        ):
            fields[name] = int(value) if value.isdigit() else value
        required = {
            "status",
            "atlas_cycles",
            "atlas_e2e_cycles",
            "finish_gpu_cycle",
            "transaction_bytes",
            "submitted_parents",
            "completed_parents",
            "bridge_atlas_parents",
            "bridge_atlas_completed",
            "runtime_active",
            "instances",
        }
        if required <= fields.keys():
            return fields
    return None


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class AccelSimBackend:
    """Invoke the official trace executable without changing its timing ownership."""

    def __init__(self, config: AccelSimBackendConfig):
        self.config = config

    def descriptor(self) -> BackendDescriptor:
        if self.config.external_memory is not None:
            resource_kinds = [
                "gpu_core",
                "gpu_l1",
                "gpu_l2",
                "gpu_noc",
                "shared_3d_dram",
            ]
            if self.config.co_resident_atlas is not None:
                resource_kinds.extend(("atlas_core", "atlas_sram", "atlas_noc"))
            return BackendDescriptor(
                backend_id=self.config.backend_id,
                supported_duration_semantics=("coupled",),
                ownable_resource_kinds=tuple(resource_kinds),
                supported_exports=(),
                supports_stall_resume=True,
                supported_trace_semantics=("functional",),
                qualification_records=("cycle_coupled_request_response",),
            )
        return BackendDescriptor(
            backend_id=self.config.backend_id,
            supported_duration_semantics=("total",),
            ownable_resource_kinds=(
                "gpu_core",
                "gpu_l1",
                "gpu_l2",
                "gpu_noc",
                "gpu_local_dram",
            ),
            supported_exports=(),
            supports_stall_resume=False,
            supported_trace_semantics=("functional",),
            qualification_records=("adapter_equivalence",),
        )

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
            "external_memory": {
                "kind": self.config.external_memory.kind,
                "timing_owner": self.config.external_memory.timing_owner,
                "config_sha256": hashlib.sha256(
                    self.config.external_memory.config_file.read_bytes()
                ).hexdigest(),
                "bridge_sha256": hashlib.sha256(
                    self.config.external_memory.bridge_library.read_bytes()
                ).hexdigest(),
                "bandwidth_contract": asdict(
                    self.config.external_memory.bandwidth_contract
                ),
            }
            if self.config.external_memory is not None
            else None,
            "co_resident_atlas": {
                "kind": self.config.co_resident_atlas.kind,
                "execution_semantics": self.config.co_resident_atlas.execution_semantics,
                "chip_config_sha256": hashlib.sha256(
                    self.config.co_resident_atlas.chip_config.read_bytes()
                ).hexdigest(),
                "operator_list_sha256": hashlib.sha256(
                    self.config.co_resident_atlas.operator_list.read_bytes()
                ).hexdigest(),
                "placement_map_sha256": hashlib.sha256(
                    self.config.co_resident_atlas.placement_map.read_bytes()
                ).hexdigest(),
                "expected_instances": self.config.co_resident_atlas.expected_instances,
                "require_nonzero_requests": (
                    self.config.co_resident_atlas.require_nonzero_requests
                ),
                "expected_transaction_bytes": (
                    self.config.co_resident_atlas.expected_transaction_bytes
                ),
            }
            if self.config.co_resident_atlas is not None
            else None,
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
            if self.config.external_memory is not None:
                environment["GPGPUSIM_RAMULATOR_CONFIG"] = str(
                    self.config.external_memory.config_file
                )
                contract = self.config.external_memory.bandwidth_contract
                gateway = contract.logic_die_gateway
                environment["HETEROSIM_DRAM_TRANSACTION_BYTES"] = str(
                    contract.internal_dram.transaction_bytes
                )
                environment["HETEROSIM_GPU_CLOCK_HZ"] = str(
                    self.config.core_frequency_hz
                )
                environment["HETEROSIM_GATEWAY_CLOCK_HZ"] = str(
                    gateway.clock_hz
                )
                environment["HETEROSIM_DRAM_CLOCK_HZ"] = str(
                    int(contract.internal_dram.clock_hz)
                )
                environment["HETEROSIM_GATEWAY_INGRESS_QUEUE_DEPTH"] = str(
                    gateway.ingress_queue_depth
                )
                environment["HETEROSIM_GATEWAY_PARENT_TABLE_ENTRIES"] = str(
                    gateway.parent_table_entries
                )
                environment["HETEROSIM_GATEWAY_ISSUE_WIDTH"] = str(
                    gateway.issue_width_per_cycle
                )
                environment["HETEROSIM_GATEWAY_WRITE_ACK_POLICY"] = (
                    gateway.write_ack_policy
                )
                link = contract.external_link
                environment["HETEROSIM_LINK_CLOCK_HZ"] = str(link.clock_hz)
                environment["HETEROSIM_LINK_REQUEST_BANDWIDTH_BPS"] = str(
                    link.request_payload_bandwidth_Bps
                )
                environment["HETEROSIM_LINK_RESPONSE_BANDWIDTH_BPS"] = str(
                    link.response_payload_bandwidth_Bps
                )
                environment["HETEROSIM_LINK_REQUEST_HEADER_BYTES"] = str(
                    link.request_header_bytes
                )
                environment["HETEROSIM_LINK_RESPONSE_HEADER_BYTES"] = str(
                    link.response_header_bytes
                )
                environment["HETEROSIM_LINK_FLIT_BYTES"] = str(link.flit_bytes)
                environment["HETEROSIM_LINK_PROPAGATION_LATENCY_FS"] = str(
                    link.propagation_latency_fs
                )
                environment["HETEROSIM_LINK_QUEUE_DEPTH"] = str(
                    link.queue_depth_transactions
                )
                environment["HETEROSIM_LINK_CREDITS"] = str(link.credits)
                environment["HETEROSIM_LINK_DUPLEX_MODE"] = link.duplex_mode
            if self.config.co_resident_atlas is not None:
                atlas = self.config.co_resident_atlas
                environment["HETEROSIM_ATLAS_CHIP_CONFIG"] = str(
                    atlas.chip_config
                )
                environment["HETEROSIM_ATLAS_OPERATOR_LIST"] = str(
                    atlas.operator_list
                )
                environment["HETEROSIM_ATLAS_PLACEMENT_MAP"] = str(
                    atlas.placement_map
                )
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
        combined_output = process.stdout + "\n" + process.stderr
        stats = parse_accel_sim_stats(combined_output)
        missing = {"gpu_tot_sim_cycle", "gpu_tot_sim_insn"} - stats.keys()
        if missing:
            raise AccelSimBackendError(
                f"Accel-Sim output lacks required statistics: {sorted(missing)}"
            )
        cycles = int(stats["gpu_tot_sim_cycle"])
        instructions = int(stats["gpu_tot_sim_insn"])
        external_memory_stats = parse_ramulator2_stats(combined_output)
        atlas_runtime_stats = parse_atlas_full_chip_runtime_stats(combined_output)
        if self.config.external_memory is not None:
            if external_memory_stats is None:
                raise AccelSimBackendError(
                    "external Ramulator2 is configured but emitted no bridge statistics"
                )
            if external_memory_stats["instances"] != self.config.external_memory.expected_instances:
                raise AccelSimBackendError(
                    "external Ramulator2 instance count violates the single-owner contract"
                )
            if external_memory_stats["outstanding"] != 0:
                raise AccelSimBackendError(
                    "external Ramulator2 exited with outstanding GPU requests"
                )
            accepted = external_memory_stats["reads"] + external_memory_stats["writes"]
            if self.config.external_memory.require_nonzero_requests and accepted == 0:
                raise AccelSimBackendError(
                    "external Ramulator2 accepted zero requests; the trace did not exercise the cycle bridge"
                )
            if external_memory_stats["completed"] != accepted:
                raise AccelSimBackendError(
                    "external Ramulator2 did not complete every accepted parent request"
                )
        if self.config.co_resident_atlas is not None:
            atlas = self.config.co_resident_atlas
            if atlas_runtime_stats is None:
                raise AccelSimBackendError(
                    "co-resident ATLAS is configured but emitted no runtime summary"
                )
            if atlas_runtime_stats["status"] != "passed":
                raise AccelSimBackendError(
                    "co-resident ATLAS runtime did not finish successfully"
                )
            if atlas_runtime_stats["instances"] != atlas.expected_instances:
                raise AccelSimBackendError(
                    "co-resident ATLAS instance count violates the contract"
                )
            submitted = int(atlas_runtime_stats["submitted_parents"])
            completed = int(atlas_runtime_stats["completed_parents"])
            if atlas.require_nonzero_requests and submitted == 0:
                raise AccelSimBackendError(
                    "co-resident ATLAS emitted zero shared-memory requests"
                )
            if submitted != completed:
                raise AccelSimBackendError(
                    "co-resident ATLAS did not complete every submitted request"
                )
            if int(atlas_runtime_stats["bridge_atlas_parents"]) != submitted:
                raise AccelSimBackendError(
                    "ATLAS runtime and shared bridge disagree on submitted parents"
                )
            if int(atlas_runtime_stats["bridge_atlas_completed"]) != completed:
                raise AccelSimBackendError(
                    "ATLAS runtime and shared bridge disagree on completions"
                )
            expected_bytes = atlas.expected_transaction_bytes
            if (
                expected_bytes is not None
                and int(atlas_runtime_stats["transaction_bytes"]) != expected_bytes
            ):
                raise AccelSimBackendError(
                    "co-resident ATLAS transaction-byte count does not match config"
                )
        duration_fs = math.ceil(cycles * 1_000_000_000_000_000 / self.config.core_frequency_hz)
        result = AccelSimRunResult(
            command=command,
            return_code=process.returncode,
            duration_fs=duration_fs,
            cycles=cycles,
            instructions=instructions,
            stats=stats,
            external_memory_stats=external_memory_stats,
            atlas_runtime_stats=atlas_runtime_stats,
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
                "external_memory_stats": external_memory_stats,
                "atlas_runtime_stats": atlas_runtime_stats,
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
        if self.config.external_memory is not None:
            compared["external_memory_stats"] = [
                baseline.external_memory_stats,
                adapter.external_memory_stats,
            ]
        if self.config.co_resident_atlas is not None:
            compared["atlas_runtime_stats"] = [
                baseline.atlas_runtime_stats,
                adapter.atlas_runtime_stats,
            ]
        exact_match = all(left == right for left, right in compared.values())
        overlap_evidence: list[dict[str, int | bool]] = []
        if self.config.co_resident_atlas is not None:
            for result in (baseline, adapter):
                assert result.atlas_runtime_stats is not None
                assert result.external_memory_stats is not None
                finish = int(result.atlas_runtime_stats["finish_gpu_cycle"])
                gpu_parents = int(result.external_memory_stats["gpu_parents"])
                atlas_parents = int(result.external_memory_stats["atlas_parents"])
                overlap_evidence.append(
                    {
                        "gpu_total_cycles": result.cycles,
                        "atlas_finish_gpu_cycle": finish,
                        "gpu_parents": gpu_parents,
                        "atlas_parents": atlas_parents,
                        "both_initiators_active": gpu_parents > 0
                        and atlas_parents > 0,
                        "atlas_finished_before_gpu_run_end": 0
                        < finish
                        < result.cycles,
                    }
                )
        overlap_passed = not overlap_evidence or all(
            bool(item["both_initiators_active"])
            and bool(item["atlas_finished_before_gpu_run_end"])
            for item in overlap_evidence
        )
        passed = exact_match and overlap_passed
        qualified_scopes = [
            "cycle_coupled_request_response"
            if self.config.external_memory is not None
            else "adapter_equivalence"
        ]
        if self.config.co_resident_atlas is not None:
            qualified_scopes.extend(
                (
                    "full_atlas_chip_shared_memory_concurrency",
                    "single_ramulator2_multi_initiator_conservation",
                )
            )
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
                    "co_resident_atlas": {
                        "execution_semantics": (
                            self.config.co_resident_atlas.execution_semantics
                        ),
                        "chip_config": str(
                            self.config.co_resident_atlas.chip_config
                        ),
                        "operator_list": str(
                            self.config.co_resident_atlas.operator_list
                        ),
                        "placement_map": str(
                            self.config.co_resident_atlas.placement_map
                        ),
                    }
                    if self.config.co_resident_atlas is not None
                    else None,
                },
                "comparison": compared,
                "overlap_evidence": overlap_evidence,
                "exact_match_required": True,
                "qualified_scopes": qualified_scopes,
                "replay_safety_qualified": False,
                "timing_ownership": {
                    "gpu_core_cache_noc": "accel_sim",
                    "gpu_local_dram": "accel_sim"
                    if self.config.external_memory is None
                    else None,
                    "external_ramulator2": self.config.external_memory.timing_owner
                    if self.config.external_memory is not None
                    else False,
                    "duration_mode": "coupled"
                    if self.config.external_memory is not None
                    else "total",
                    "co_resident_atlas_core_sram_noc": "atlasim"
                    if self.config.co_resident_atlas is not None
                    else False,
                },
                "claim_boundary": (
                    "The same q_proj shape executes on both backends as a "
                    "shared-memory contention stress case. This does not "
                    "represent a valid single-placement end-to-end schedule."
                    if self.config.co_resident_atlas is not None
                    else None
                ),
            },
        )
        if not passed:
            raise AccelSimBackendError(f"qualification mismatch; see {record}")
        if manifest.source_path is not None:
            qualified_payload = json.loads(
                manifest.source_path.read_text(encoding="utf-8")
            )
            qualified_payload["replay_safe"] = False
            qualified_payload["qualification_record"] = str(record.resolve())
            qualified_payload["kernels_list"] = str(manifest.kernels_list)
            _write_json(
                output_directory / "adapter_qualified_trace_manifest.json",
                qualified_payload,
            )
        return record
