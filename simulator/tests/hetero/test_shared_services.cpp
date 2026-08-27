#include <cassert>
#include <stdexcept>
#include <string>
#include <vector>

#include "hetero/services/bounded_link.h"
#include "hetero/services/shared_3d_memory.h"

using heterosim::PhysicalAddress;
using heterosim::services::BoundedLinkConfig;
using heterosim::services::BoundedLinkModel;
using heterosim::services::LinkTransaction;
using heterosim::services::MemoryOperation;
using heterosim::services::MemoryRequest;
using heterosim::services::Shared3DMemoryConfig;
using heterosim::services::Shared3DMemoryModel;

int main() {
    const BoundedLinkModel link(BoundedLinkConfig{
        "pcie0.dma", 1000000000000ULL, 10, 1, 1, true});
    const auto link_result = link.run({
        LinkTransaction{0, 0, "pcie0.dma", "host", "gpu", 64, 16, 80, 0},
        LinkTransaction{1, 1, "pcie0.dma", "host", "gpu", 64, 16, 80, 0},
    });
    assert(link_result.completed_transactions == 2);
    assert(link_result.backpressure_events == 1);
    assert(link_result.responses.at(1).completion_time_fs >
           link_result.responses.at(0).completion_time_fs);

    const Shared3DMemoryModel memory(Shared3DMemoryConfig{
        "shared0.dram3d", {"gpu0", "atlas0.compute"}, 2, 4, 64, 4,
        100, 10, 20});
    const std::vector<MemoryRequest> requests{
        MemoryRequest{0, 0, "gpu0", PhysicalAddress{"shared0.dram3d", 0, 1},
                      "a", 0, 128, MemoryOperation::kRead, 0, 0, 0, 0},
        MemoryRequest{1, 1, "atlas0.compute",
                      PhysicalAddress{"shared0.dram3d", 128, 1},
                      "b", 0, 64, MemoryOperation::kWrite, 0, 0, 1, 0},
    };
    const auto memory_result = memory.run(requests);
    assert(memory_result.parent_requests_submitted == 2);
    assert(memory_result.parent_requests_completed == 2);
    assert(memory_result.child_requests_submitted == 3);
    assert(memory_result.child_requests_completed == 3);
    assert(memory_result.submitted_bytes == 192);
    assert(memory_result.completed_bytes == 192);
    assert(memory_result.requests_by_initiator.at("gpu0") == 1);
    assert(memory_result.requests_by_initiator.at("atlas0.compute") == 1);
    auto duplicate_requests = requests;
    duplicate_requests.at(1).request_id = duplicate_requests.at(0).request_id;
    bool duplicate_rejected = false;
    try {
        static_cast<void>(memory.run(duplicate_requests));
    } catch (const std::invalid_argument&) {
        duplicate_rejected = true;
    }
    assert(duplicate_rejected);
    return 0;
}
