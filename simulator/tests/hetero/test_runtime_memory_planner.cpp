#include <cassert>
#include <stdexcept>
#include <vector>

#include "hetero/memory/runtime_memory_planner.h"

using heterosim::memory::AllocationRequest;
using heterosim::memory::MemorySpaceSpec;
using heterosim::memory::RuntimeMemoryPlanner;

int main() {
    RuntimeMemoryPlanner planner({MemorySpaceSpec{"shared0.dram3d", 1024, 64}});
    const auto first = planner.allocate(
        AllocationRequest{"r0.kv", "shared0.dram3d", 256, 64, "request"});
    const auto second = planner.allocate(
        AllocationRequest{"r1.kv", "shared0.dram3d", 128, 64, "request"});
    assert(first.physical_address.offset_bytes == 0);
    assert(second.physical_address.offset_bytes == 256);
    assert(planner.used_bytes("shared0.dram3d") == 384);
    planner.release("r0.kv");
    const auto reused = planner.allocate(
        AllocationRequest{"r2.kv", "shared0.dram3d", 192, 64, "request"});
    assert(reused.physical_address.offset_bytes == 0);
    assert(reused.physical_address.allocation_epoch !=
           first.physical_address.allocation_epoch);
    assert(planner.peak_bytes("shared0.dram3d") == 384);
    bool rejected = false;
    try {
        static_cast<void>(planner.allocate(
            AllocationRequest{"too-large", "shared0.dram3d", 1024, 64, "request"}));
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    assert(rejected);
    return 0;
}
