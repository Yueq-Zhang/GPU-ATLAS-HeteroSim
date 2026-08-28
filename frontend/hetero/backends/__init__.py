"""Cycle-simulator backend adapters."""

from .accel_sim import (
    AccelSimBackend,
    AccelSimBackendConfig,
    AccelSimBackendError,
    AccelSimRunResult,
    CoResidentAtlasConfig,
    parse_atlas_full_chip_runtime_stats,
)
from .contracts import (
    BackendDescriptor,
    ResolvedTimingContract,
    TimingContractError,
    TimingOwnershipRegistry,
    resolve_timing_contract,
)
from .memory_bridge import MemoryBridgeError, run_jsonl_bridge
from .ramulator2 import (
    Ramulator2Backend,
    Ramulator2BackendConfig,
    Ramulator2BackendError,
    Ramulator2RunResult,
)
from .atlas import (
    AtlasArtifact,
    AtlasBackend,
    AtlasBackendConfig,
    AtlasBackendError,
    AtlasRunResult,
)

__all__ = [
    "AccelSimBackend",
    "AccelSimBackendConfig",
    "AccelSimBackendError",
    "AccelSimRunResult",
    "CoResidentAtlasConfig",
    "parse_atlas_full_chip_runtime_stats",
    "BackendDescriptor",
    "ResolvedTimingContract",
    "TimingContractError",
    "TimingOwnershipRegistry",
    "MemoryBridgeError",
    "run_jsonl_bridge",
    "Ramulator2Backend",
    "Ramulator2BackendConfig",
    "Ramulator2BackendError",
    "Ramulator2RunResult",
    "resolve_timing_contract",
    "AtlasArtifact",
    "AtlasBackend",
    "AtlasBackendConfig",
    "AtlasBackendError",
    "AtlasRunResult",
]
