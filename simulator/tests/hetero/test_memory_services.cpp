#include <cassert>
#include <stdexcept>
#include <string>
#include <vector>

#include "hetero/runtime/timing_ownership.h"
#include "hetero/services/fixed_latency_memory.h"

using heterosim::PhysicalAddress;
using heterosim::runtime::TimingOwnershipRegistry;
using heterosim::services::FixedLatencyMemoryService;
using heterosim::services::MemoryOperation;
using heterosim::services::MemoryRequest;
using heterosim::services::ideal_link_completion_fs;

MemoryRequest request(std::uint64_t id, const std::string& initiator) {
    return MemoryRequest{
        id,
        1,
        initiator,
        PhysicalAddress{"shared0.dram3d", id * 64, 1},
        "value",
        1,
        64,
        MemoryOperation::kRead,
        0,
        0,
        id,
        0};
}

int main() {
    TimingOwnershipRegistry registry;
    registry.claim("shared0.dram3d", "shared3d.memory_service");
    registry.claim("shared0.dram3d", "shared3d.memory_service");
    assert(registry.size() == 1);
    bool conflict = false;
    try {
        registry.claim("shared0.dram3d", "accel_sim.internal_dram");
    } catch (const std::logic_error&) {
        conflict = true;
    }
    assert(conflict);

    FixedLatencyMemoryService memory({"gpu0", "atlas0.compute"}, 100, 10);
    memory.submit(request(0, "gpu0"));
    memory.submit(request(1, "gpu0"));
    memory.submit(request(2, "atlas0.compute"));
    const auto responses = memory.drain();
    assert(responses.size() == 3);
    assert(responses.at(0).request_id == 0);
    assert(responses.at(0).completion_time_fs == 100);
    assert(responses.at(1).request_id == 2);
    assert(responses.at(1).completion_time_fs == 110);
    assert(responses.at(2).request_id == 1);
    assert(responses.at(2).completion_time_fs == 120);

    assert(ideal_link_completion_fs(7, 11, 64, 16, 1000000000000ULL) == 80018);
    return 0;
}
