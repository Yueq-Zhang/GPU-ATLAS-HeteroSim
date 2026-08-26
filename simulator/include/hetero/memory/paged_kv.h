#pragma once

#include <cstdint>
#include <string>

#include "hetero/types.h"

namespace heterosim::memory {

struct PagedKvRequest {
    std::string request_id;
    std::uint64_t prompt_length{};
    std::uint64_t output_length{};
};

struct PagedKvGeometry {
    std::uint64_t num_layers{};
    std::uint64_t num_kv_heads{};
    std::uint64_t head_dim{};
    std::uint64_t bytes_per_element{};
    std::uint64_t page_tokens{};
};

struct PagedKvAllocation {
    std::string request_id;
    PhysicalAddress physical_address;
    std::uint64_t final_committed_tokens{};
    std::uint64_t allocated_blocks{};
    std::uint64_t bytes_per_block{};
    std::uint64_t logical_bytes{};
    std::uint64_t allocated_bytes{};
};

class PagedKvAllocator {
public:
    PagedKvAllocator(std::string memory_space_id, std::uint64_t capacity_bytes);
    PagedKvAllocation allocate(
        const PagedKvRequest& request,
        const PagedKvGeometry& geometry);
    [[nodiscard]] std::uint64_t used_bytes() const noexcept;

private:
    std::string memory_space_id_;
    std::uint64_t capacity_bytes_{};
    std::uint64_t next_offset_{};
    std::uint64_t next_epoch_{1};
};

}  // namespace heterosim::memory
