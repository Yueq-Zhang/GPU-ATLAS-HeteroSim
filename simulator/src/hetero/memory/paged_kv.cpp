#include "hetero/memory/paged_kv.h"

#include <limits>
#include <stdexcept>

namespace heterosim::memory {
namespace {

std::uint64_t checked_multiply(std::uint64_t lhs, std::uint64_t rhs) {
    if (rhs != 0 && lhs > std::numeric_limits<std::uint64_t>::max() / rhs) {
        throw std::overflow_error("paged KV size overflow");
    }
    return lhs * rhs;
}

}  // namespace

PagedKvAllocator::PagedKvAllocator(
    std::string memory_space_id,
    std::uint64_t capacity_bytes)
    : memory_space_id_(std::move(memory_space_id)), capacity_bytes_(capacity_bytes) {
    if (memory_space_id_.empty() || capacity_bytes_ == 0) {
        throw std::invalid_argument("memory space and capacity are required");
    }
}

PagedKvAllocation PagedKvAllocator::allocate(
    const PagedKvRequest& request,
    const PagedKvGeometry& geometry) {
    if (request.request_id.empty() || request.prompt_length == 0 ||
        request.output_length == 0 || geometry.num_layers == 0 ||
        geometry.num_kv_heads == 0 || geometry.head_dim == 0 ||
        geometry.bytes_per_element == 0 || geometry.page_tokens == 0) {
        throw std::invalid_argument("invalid paged KV request or geometry");
    }
    if (request.prompt_length >
        std::numeric_limits<std::uint64_t>::max() - request.output_length + 1) {
        throw std::overflow_error("paged KV token count overflow");
    }
    const auto final_tokens = request.prompt_length + request.output_length - 1;
    const auto pages_per_layer_kind =
        (final_tokens + geometry.page_tokens - 1) / geometry.page_tokens;
    const auto element_bytes = checked_multiply(
        checked_multiply(geometry.num_kv_heads, geometry.head_dim),
        geometry.bytes_per_element);
    const auto bytes_per_block = checked_multiply(geometry.page_tokens, element_bytes);
    const auto layer_kinds = checked_multiply(geometry.num_layers, 2);
    const auto blocks = checked_multiply(layer_kinds, pages_per_layer_kind);
    const auto allocated = checked_multiply(blocks, bytes_per_block);
    const auto logical = checked_multiply(
        checked_multiply(final_tokens, layer_kinds), element_bytes);
    if (allocated > capacity_bytes_ - next_offset_) {
        throw std::runtime_error("paged KV capacity exceeded");
    }
    PagedKvAllocation allocation{
        request.request_id,
        PhysicalAddress{memory_space_id_, next_offset_, next_epoch_++},
        final_tokens,
        blocks,
        bytes_per_block,
        logical,
        allocated};
    next_offset_ += allocated;
    return allocation;
}

std::uint64_t PagedKvAllocator::used_bytes() const noexcept {
    return next_offset_;
}

}  // namespace heterosim::memory
