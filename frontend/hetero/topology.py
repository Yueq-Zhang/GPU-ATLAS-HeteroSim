"""Frozen four-profile topology roles and dependency lowering decisions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class LoweringKind(str, Enum):
    LOCAL_DEPENDENCY = "local_dependency"
    TRANSFER = "transfer"
    REMOTE_ACCESS = "remote_access"
    MIGRATION = "migration"
    SYNCHRONIZATION = "synchronization"


@dataclass(frozen=True, slots=True)
class LoweringDecision:
    kind: LoweringKind
    route_id: str | None
    source_space: str
    destination_space: str
    actions: tuple[str, ...] = ()


_PRIMARY_3D = {
    "model1_atlas_native": "atlas0.dram3d",
    "model2_host_memory_pcie": "host0.dram3d",
    "model3_gpu_native_3ddram": "shared0.dram3d",
    "model4_cxl_memory_tier": "cxl0.dram3d",
}


def primary_3ddram(profile: str) -> str:
    try:
        return _PRIMARY_3D[profile]
    except KeyError as error:
        raise ValueError(f"unknown system profile: {profile}") from error


def device_memory(profile: str, device_id: str) -> str:
    if device_id == "gpu0":
        if profile == "model3_gpu_native_3ddram":
            return "shared0.dram3d"
        return "gpu0.hbm"
    if device_id == "atlas0.compute":
        return primary_3ddram(profile)
    if device_id == "host0.compute":
        return "host0.dram3d" if profile == "model2_host_memory_pcie" else primary_3ddram(profile)
    raise ValueError(f"unknown compute device: {device_id}")


def lower_cross_device_dependency(
    profile: str,
    source_device: str,
    destination_device: str,
    access_policy: str = "copy",
) -> LoweringDecision:
    source_space = device_memory(profile, source_device)
    destination_space = device_memory(profile, destination_device)
    if source_device == destination_device:
        return LoweringDecision(
            LoweringKind.LOCAL_DEPENDENCY, None, source_space, destination_space
        )
    if profile == "model3_gpu_native_3ddram":
        actions = (
            "writeback",
            "release_fence",
            "invalidate",
            "acquire_fence",
        )
        return LoweringDecision(
            LoweringKind.SYNCHRONIZATION,
            "shared3d.explicit_noncoherent",
            source_space,
            destination_space,
            actions,
        )
    if profile == "model2_host_memory_pcie":
        return LoweringDecision(
            LoweringKind.TRANSFER,
            "pcie0.dma",
            source_space,
            destination_space,
        )
    if profile == "model4_cxl_memory_tier":
        if access_policy == "remote":
            return LoweringDecision(
                LoweringKind.REMOTE_ACCESS,
                "cxl0.remote",
                source_space,
                "cxl0.dram3d",
            )
        if access_policy == "migrate":
            return LoweringDecision(
                LoweringKind.MIGRATION,
                "cxl0.migration",
                source_space,
                destination_space,
            )
        if access_policy != "copy":
            raise ValueError(f"unknown CXL access policy: {access_policy}")
        return LoweringDecision(
            LoweringKind.TRANSFER,
            "cxl0.copy",
            source_space,
            destination_space,
        )
    if profile == "model1_atlas_native":
        return LoweringDecision(
            LoweringKind.TRANSFER,
            "atlas_external_analytical",
            source_space,
            destination_space,
        )
    raise ValueError(f"unknown system profile: {profile}")
