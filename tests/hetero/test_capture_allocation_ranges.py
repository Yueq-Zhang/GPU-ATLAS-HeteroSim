import pytest

from frontend.hetero.capture_allocation_ranges import (
    CaptureAllocationRangeError,
    allocator_ranges_from_events,
    allocator_segment_ranges_for_addresses,
    merge_address_ranges,
    subtract_address_ranges,
)


def test_allocator_events_merge_reuse_and_ignore_non_alloc_events() -> None:
    events = [
        {"action": "segment_alloc", "addr": 0, "size": 4096},
        {"action": "alloc", "addr": 1024, "size": 256},
        {"action": "free_completed", "addr": 1024, "size": 256},
        {"action": "alloc", "addr": 1024, "size": 512},
        {"action": "alloc", "addr": 1536, "size": 256},
    ]
    assert allocator_ranges_from_events(events) == ((1024, 1792),)


def test_allocator_segments_are_selected_only_for_target_addresses() -> None:
    segments = [
        {"address": 0x1000, "total_size": 0x1000},
        {"address": 0x4000, "total_size": 0x2000},
        {"address": 0x9000, "total_size": 0x1000},
    ]
    assert allocator_segment_ranges_for_addresses(
        segments, (0x1100, 0x1800, 0x5000)
    ) == ((0x1000, 0x2000), (0x4000, 0x6000))
    with pytest.raises(CaptureAllocationRangeError, match="belongs to 0"):
        allocator_segment_ranges_for_addresses(segments, (0x7000,))


def test_semantic_tensors_are_subtracted_from_workspace_ranges() -> None:
    assert subtract_address_ranges(
        ((100, 500),), ((100, 200), (300, 400))
    ) == ((200, 300), (400, 500))


def test_invalid_ranges_fail_closed() -> None:
    with pytest.raises(CaptureAllocationRangeError, match="invalid"):
        merge_address_ranges(((100, 100),))
