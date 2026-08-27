#include "hetero/memory/runtime_memory_planner.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace heterosim::memory {
namespace {

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        throw std::invalid_argument("alignment must be a non-zero power of two");
    }
    const auto mask = alignment - 1;
    if (value > std::numeric_limits<std::uint64_t>::max() - mask) {
        throw std::overflow_error("aligned address exceeds uint64");
    }
    return (value + mask) & ~mask;
}

}  // namespace

RuntimeMemoryPlanner::RuntimeMemoryPlanner(std::vector<MemorySpaceSpec> spaces) {
    if (spaces.empty()) {
        throw std::invalid_argument("at least one memory space is required");
    }
    for (auto& spec : spaces) {
        if (spec.memory_space_id.empty() || spec.capacity_bytes == 0) {
            throw std::invalid_argument("memory space id and capacity are required");
        }
        if (spec.base_alignment_bytes == 0 ||
            (spec.base_alignment_bytes & (spec.base_alignment_bytes - 1)) != 0) {
            throw std::invalid_argument("base alignment must be a power of two");
        }
        SpaceState state;
        state.spec = spec;
        state.free_ranges.push_back(FreeRange{0, spec.capacity_bytes});
        if (!spaces_.emplace(spec.memory_space_id, std::move(state)).second) {
            throw std::invalid_argument("duplicate memory space: " + spec.memory_space_id);
        }
    }
}

AllocationRecord RuntimeMemoryPlanner::allocate(const AllocationRequest& request) {
    if (request.allocation_id.empty() || request.memory_space_id.empty() ||
        request.size_bytes == 0) {
        throw std::invalid_argument("allocation id, memory space and size are required");
    }
    if (allocations_.count(request.allocation_id) != 0) {
        throw std::invalid_argument("duplicate allocation id: " + request.allocation_id);
    }
    auto space_it = spaces_.find(request.memory_space_id);
    if (space_it == spaces_.end()) {
        throw std::invalid_argument("unknown memory space: " + request.memory_space_id);
    }
    auto& space = space_it->second;
    const auto alignment = std::max(
        request.alignment_bytes, space.spec.base_alignment_bytes);
    for (std::size_t index = 0; index < space.free_ranges.size(); ++index) {
        const auto range = space.free_ranges.at(index);
        const auto start = align_up(range.offset_bytes, alignment);
        const auto prefix = start - range.offset_bytes;
        if (prefix > range.size_bytes || request.size_bytes > range.size_bytes - prefix) {
            continue;
        }
        const auto suffix_offset = start + request.size_bytes;
        const auto suffix_size = range.size_bytes - prefix - request.size_bytes;
        space.free_ranges.erase(space.free_ranges.begin() + static_cast<long>(index));
        if (prefix != 0) {
            space.free_ranges.insert(
                space.free_ranges.begin() + static_cast<long>(index),
                FreeRange{range.offset_bytes, prefix});
            ++index;
        }
        if (suffix_size != 0) {
            space.free_ranges.insert(
                space.free_ranges.begin() + static_cast<long>(index),
                FreeRange{suffix_offset, suffix_size});
        }
        if (request.size_bytes >
            std::numeric_limits<std::uint64_t>::max() - space.used_bytes) {
            throw std::overflow_error("memory usage counter overflow");
        }
        space.used_bytes += request.size_bytes;
        space.peak_bytes = std::max(space.peak_bytes, space.used_bytes);
        AllocationRecord record{
            request.allocation_id,
            PhysicalAddress{request.memory_space_id, start, space.next_epoch++},
            request.size_bytes,
            alignment,
            request.lifetime,
            true};
        allocations_.emplace(request.allocation_id, record);
        return record;
    }
    throw std::runtime_error(
        "memory capacity exceeded in space " + request.memory_space_id);
}

AllocationRecord RuntimeMemoryPlanner::release(const std::string& allocation_id) {
    auto allocation_it = allocations_.find(allocation_id);
    if (allocation_it == allocations_.end()) {
        throw std::invalid_argument("unknown allocation: " + allocation_id);
    }
    auto& allocation = allocation_it->second;
    if (!allocation.active) {
        throw std::logic_error("allocation already released: " + allocation_id);
    }
    auto& space = spaces_.at(allocation.physical_address.memory_space_id);
    if (allocation.size_bytes > space.used_bytes) {
        throw std::logic_error("memory usage counter underflow");
    }
    space.used_bytes -= allocation.size_bytes;
    space.free_ranges.push_back(
        FreeRange{allocation.physical_address.offset_bytes, allocation.size_bytes});
    std::sort(
        space.free_ranges.begin(), space.free_ranges.end(),
        [](const FreeRange& lhs, const FreeRange& rhs) {
            return lhs.offset_bytes < rhs.offset_bytes;
        });
    std::vector<FreeRange> merged;
    for (const auto& range : space.free_ranges) {
        if (!merged.empty() &&
            merged.back().offset_bytes + merged.back().size_bytes == range.offset_bytes) {
            merged.back().size_bytes += range.size_bytes;
        } else {
            merged.push_back(range);
        }
    }
    space.free_ranges = std::move(merged);
    allocation.active = false;
    return allocation;
}

const AllocationRecord& RuntimeMemoryPlanner::lookup(
    const std::string& allocation_id) const {
    const auto iterator = allocations_.find(allocation_id);
    if (iterator == allocations_.end()) {
        throw std::invalid_argument("unknown allocation: " + allocation_id);
    }
    return iterator->second;
}

std::uint64_t RuntimeMemoryPlanner::used_bytes(
    const std::string& memory_space_id) const {
    const auto iterator = spaces_.find(memory_space_id);
    if (iterator == spaces_.end()) {
        throw std::invalid_argument("unknown memory space: " + memory_space_id);
    }
    return iterator->second.used_bytes;
}

std::uint64_t RuntimeMemoryPlanner::peak_bytes(
    const std::string& memory_space_id) const {
    const auto iterator = spaces_.find(memory_space_id);
    if (iterator == spaces_.end()) {
        throw std::invalid_argument("unknown memory space: " + memory_space_id);
    }
    return iterator->second.peak_bytes;
}

std::vector<AllocationRecord> RuntimeMemoryPlanner::records() const {
    std::vector<AllocationRecord> result;
    result.reserve(allocations_.size());
    for (const auto& [id, record] : allocations_) {
        static_cast<void>(id);
        result.push_back(record);
    }
    std::sort(result.begin(), result.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.allocation_id < rhs.allocation_id;
    });
    return result;
}

}  // namespace heterosim::memory
