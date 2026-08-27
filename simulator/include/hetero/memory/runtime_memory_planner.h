#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "hetero/types.h"

namespace heterosim::memory {

struct MemorySpaceSpec {
    std::string memory_space_id;
    std::uint64_t capacity_bytes{};
    std::uint64_t base_alignment_bytes{64};
};

struct AllocationRequest {
    std::string allocation_id;
    std::string memory_space_id;
    std::uint64_t size_bytes{};
    std::uint64_t alignment_bytes{1};
    std::string lifetime;
};

struct AllocationRecord {
    std::string allocation_id;
    PhysicalAddress physical_address;
    std::uint64_t size_bytes{};
    std::uint64_t alignment_bytes{};
    std::string lifetime;
    bool active{true};
};

class RuntimeMemoryPlanner {
public:
    explicit RuntimeMemoryPlanner(std::vector<MemorySpaceSpec> spaces);

    AllocationRecord allocate(const AllocationRequest& request);
    AllocationRecord release(const std::string& allocation_id);
    [[nodiscard]] const AllocationRecord& lookup(
        const std::string& allocation_id) const;
    [[nodiscard]] std::uint64_t used_bytes(
        const std::string& memory_space_id) const;
    [[nodiscard]] std::uint64_t peak_bytes(
        const std::string& memory_space_id) const;
    [[nodiscard]] std::vector<AllocationRecord> records() const;

private:
    struct FreeRange {
        std::uint64_t offset_bytes{};
        std::uint64_t size_bytes{};
    };

    struct SpaceState {
        MemorySpaceSpec spec;
        std::vector<FreeRange> free_ranges;
        std::uint64_t used_bytes{};
        std::uint64_t peak_bytes{};
        std::uint64_t next_epoch{1};
    };

    std::unordered_map<std::string, SpaceState> spaces_;
    std::unordered_map<std::string, AllocationRecord> allocations_;
};

}  // namespace heterosim::memory
