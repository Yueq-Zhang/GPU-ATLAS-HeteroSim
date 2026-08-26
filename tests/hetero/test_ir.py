import pytest

from frontend.hetero.ir import PhysicalAddress, ValueRef
from frontend.hetero.ir.types import AccessMode


def test_physical_address_is_memory_space_scoped() -> None:
    address = PhysicalAddress("shared0.dram3d", 4096, 2)
    assert address.memory_space_id == "shared0.dram3d"
    assert address.offset_bytes == 4096


def test_value_ref_rejects_empty_ranges() -> None:
    with pytest.raises(ValueError, match="positive"):
        ValueRef("kv.r0.l0.k", 1, 0, 0, AccessMode.READ)

