"""Backend capabilities and the resolved timing-ownership contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping


class TimingContractError(ValueError):
    """Raised before execution when a requested timing contract is impossible."""


@dataclass(frozen=True, slots=True)
class BackendDescriptor:
    backend_id: str
    supported_duration_semantics: tuple[str, ...]
    ownable_resource_kinds: tuple[str, ...]
    supported_exports: tuple[str, ...]
    supports_stall_resume: bool
    supported_trace_semantics: tuple[str, ...]
    qualification_records: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.backend_id:
            raise TimingContractError("backend_id must be non-empty")
        if not self.supported_duration_semantics:
            raise TimingContractError("backend must support a duration semantic")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ResolvedTimingContract:
    backend_id: str
    duration_semantics: str
    owns: tuple[str, ...]
    exports: tuple[str, ...]
    supports_stall_resume: bool
    trace_semantics: str
    replay_safe: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class TimingOwnershipRegistry:
    """Reject two Backend contracts claiming the same concrete resource ID."""

    def __init__(self) -> None:
        self._owners: dict[str, str] = {}

    def register(self, contract: ResolvedTimingContract) -> None:
        for resource_id in contract.owns:
            current = self._owners.get(resource_id)
            if current is not None and current != contract.backend_id:
                raise TimingContractError(
                    f"timing resource {resource_id} is owned by both "
                    f"{current} and {contract.backend_id}"
                )
            self._owners[resource_id] = contract.backend_id

    def to_dict(self) -> dict[str, str]:
        return dict(sorted(self._owners.items()))


def resolve_timing_contract(
    descriptor: BackendDescriptor,
    requested_timing_mode: str,
    resource_bindings: Mapping[str, object],
    *,
    requested_exports: tuple[str, ...] = (),
    trace_semantics: str = "none",
    replay_safe: bool = False,
) -> ResolvedTimingContract:
    if requested_timing_mode not in descriptor.supported_duration_semantics:
        raise TimingContractError(
            f"backend {descriptor.backend_id} does not support "
            f"duration semantic {requested_timing_mode}"
        )
    unknown_kinds = set(resource_bindings) - set(descriptor.ownable_resource_kinds)
    if unknown_kinds:
        raise TimingContractError(
            f"backend {descriptor.backend_id} cannot own resource kinds "
            f"{sorted(unknown_kinds)}"
        )
    if set(requested_exports) - set(descriptor.supported_exports):
        raise TimingContractError(
            f"backend {descriptor.backend_id} does not support exports "
            f"{sorted(set(requested_exports) - set(descriptor.supported_exports))}"
        )
    if trace_semantics not in descriptor.supported_trace_semantics:
        raise TimingContractError(
            f"backend {descriptor.backend_id} does not support trace semantic "
            f"{trace_semantics}"
        )
    owns: list[str] = []
    for kind, resource_id in resource_bindings.items():
        if not isinstance(resource_id, str) or not resource_id:
            raise TimingContractError(
                f"resource binding {kind} must be a non-empty resource ID"
            )
        owns.append(resource_id)
    if requested_timing_mode == "total" and requested_exports:
        raise TimingContractError(
            "total duration cannot export timing requests for the same execution"
        )
    if requested_timing_mode == "coupled" and not descriptor.supports_stall_resume:
        raise TimingContractError(
            f"backend {descriptor.backend_id} cannot resolve coupled mode "
            "without stall/resume"
        )
    return ResolvedTimingContract(
        backend_id=descriptor.backend_id,
        duration_semantics=requested_timing_mode,
        owns=tuple(sorted(owns)),
        exports=tuple(sorted(requested_exports)),
        supports_stall_resume=descriptor.supports_stall_resume,
        trace_semantics=trace_semantics,
        replay_safe=replay_safe,
    )
