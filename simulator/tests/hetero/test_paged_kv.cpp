#include <cassert>

#include "hetero/memory/paged_kv.h"

using heterosim::memory::PagedKvAllocator;
using heterosim::memory::PagedKvGeometry;
using heterosim::memory::PagedKvRequest;

int main() {
    PagedKvAllocator allocator("shared0.dram3d", 1ULL << 30);
    const auto allocation = allocator.allocate(
        PagedKvRequest{"R0", 16, 3},
        PagedKvGeometry{2, 2, 32, 2, 16});
    assert(allocation.final_committed_tokens == 18);
    assert(allocation.allocated_blocks == 8);
    assert(allocation.bytes_per_block == 2048);
    assert(allocation.logical_bytes == 9216);
    assert(allocation.allocated_bytes == 16384);
    assert(allocation.physical_address.memory_space_id == "shared0.dram3d");
    assert(allocation.physical_address.offset_bytes == 0);
    assert(allocation.physical_address.allocation_epoch == 1);
    assert(allocator.used_bytes() == 16384);
    return 0;
}
