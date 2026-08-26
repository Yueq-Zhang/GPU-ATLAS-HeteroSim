import pytest

from frontend.hetero.backends.contracts import (
    BackendDescriptor,
    TimingContractError,
    TimingOwnershipRegistry,
    resolve_timing_contract,
)


def _descriptor(backend_id="gpu.accel_sim"):
    return BackendDescriptor(
        backend_id=backend_id,
        supported_duration_semantics=("total",),
        ownable_resource_kinds=("gpu_core", "gpu_local_dram"),
        supported_exports=(),
        supports_stall_resume=False,
        supported_trace_semantics=("functional",),
        qualification_records=("adapter_equivalence",),
    )


def test_total_contract_resolves_explicit_resource_owners() -> None:
    contract = resolve_timing_contract(
        _descriptor(),
        "total",
        {"gpu_core": "gpu0.core", "gpu_local_dram": "gpu0.hbm"},
        trace_semantics="functional",
    )
    assert contract.duration_semantics == "total"
    assert contract.owns == ("gpu0.core", "gpu0.hbm")
    assert contract.exports == ()
    assert contract.supports_stall_resume is False


def test_total_contract_rejects_memory_request_export() -> None:
    with pytest.raises(TimingContractError, match="does not support exports"):
        resolve_timing_contract(
            _descriptor(),
            "total",
            {"gpu_core": "gpu0.core"},
            requested_exports=("memory_requests",),
            trace_semantics="functional",
        )


def test_ownership_registry_rejects_double_counting() -> None:
    first = resolve_timing_contract(
        _descriptor("gpu.first"),
        "total",
        {"gpu_core": "gpu0.core"},
        trace_semantics="functional",
    )
    second = resolve_timing_contract(
        _descriptor("gpu.second"),
        "total",
        {"gpu_core": "gpu0.core"},
        trace_semantics="functional",
    )
    registry = TimingOwnershipRegistry()
    registry.register(first)
    with pytest.raises(TimingContractError, match="owned by both"):
        registry.register(second)
