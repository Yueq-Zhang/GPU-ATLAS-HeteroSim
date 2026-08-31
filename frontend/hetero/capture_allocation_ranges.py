"""Normalize target-window CUDA allocator events into disjoint trace ranges."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence


class CaptureAllocationRangeError(ValueError):
    """Raised when allocator history cannot form a safe address inventory."""


def merge_address_ranges(
    ranges: Iterable[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    """Merge overlapping or adjacent half-open address ranges."""

    ordered: list[tuple[int, int]] = []
    for begin, end in ranges:
        if begin < 0 or end <= begin:
            raise CaptureAllocationRangeError(
                f"invalid capture allocation range [{begin}, {end})"
            )
        ordered.append((begin, end))
    ordered.sort()
    merged: list[tuple[int, int]] = []
    for begin, end in ordered:
        if not merged or begin > merged[-1][1]:
            merged.append((begin, end))
            continue
        previous_begin, previous_end = merged[-1]
        merged[-1] = (previous_begin, max(previous_end, end))
    return tuple(merged)


def allocator_ranges_from_events(
    events: Sequence[Mapping[str, object]],
) -> tuple[tuple[int, int], ...]:
    """Extract every allocation observed inside one target execution window."""

    ranges: list[tuple[int, int]] = []
    for event in events:
        if event.get("action") != "alloc":
            continue
        address = int(event.get("addr", -1))
        size = int(event.get("size", 0))
        ranges.append((address, address + size))
    return merge_address_ranges(ranges)


def allocator_segment_ranges_for_addresses(
    segments: Sequence[Mapping[str, object]],
    addresses: Iterable[int],
) -> tuple[tuple[int, int], ...]:
    """Return allocator segments backing a selected set of live addresses.

    CUDA kernels may issue transactions into allocator padding immediately
    beyond a tensor's logical extent.  Capturing the backing segment for
    pre-existing inputs and parameters preserves those accesses without
    admitting unrelated allocator segments from the process.
    """

    normalized_segments: list[tuple[int, int]] = []
    for segment in segments:
        begin = int(segment.get("address", -1))
        size = int(segment.get("total_size", 0))
        if begin < 0 or size <= 0:
            raise CaptureAllocationRangeError("invalid allocator segment")
        normalized_segments.append((begin, begin + size))

    selected: list[tuple[int, int]] = []
    for raw_address in addresses:
        address = int(raw_address)
        matches = [
            (begin, end)
            for begin, end in normalized_segments
            if begin <= address < end
        ]
        if len(matches) != 1:
            raise CaptureAllocationRangeError(
                f"address {address} belongs to {len(matches)} allocator segments"
            )
        selected.append(matches[0])
    return merge_address_ranges(selected)


def subtract_address_ranges(
    ranges: Iterable[tuple[int, int]],
    exclusions: Iterable[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    """Remove semantic tensor intervals from captured allocator intervals."""

    result: list[tuple[int, int]] = []
    excluded = merge_address_ranges(exclusions)
    for begin, end in merge_address_ranges(ranges):
        fragments = [(begin, end)]
        for excluded_begin, excluded_end in excluded:
            next_fragments: list[tuple[int, int]] = []
            for fragment_begin, fragment_end in fragments:
                if excluded_end <= fragment_begin or fragment_end <= excluded_begin:
                    next_fragments.append((fragment_begin, fragment_end))
                    continue
                if fragment_begin < excluded_begin:
                    next_fragments.append((fragment_begin, excluded_begin))
                if excluded_end < fragment_end:
                    next_fragments.append((excluded_end, fragment_end))
            fragments = next_fragments
        result.extend(fragments)
    return merge_address_ranges(result)
