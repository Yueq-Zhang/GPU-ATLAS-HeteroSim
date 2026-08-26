#include <cstdint>

#include "hetero/artifact.h"
#include "hetero/services/interfaces.h"

int main() {
    heterosim::PhysicalAddress address{"shared0.dram3d", 4096, 2};
    heterosim::ArtifactRequest artifact;
    artifact.request_id = 7;
    artifact.simulation_buffer_bindings.push_back({"kv.r0.l0.k", address});

    heterosim::services::MemoryRequest request;
    request.request_id = 9;
    request.physical_address = address;
    request.value_id = "kv.r0.l0.k";
    request.value_version = 1;
    request.size_bytes = 64;

    return artifact.simulation_buffer_bindings.front().physical_address.offset_bytes ==
                   request.physical_address.offset_bytes
               ? 0
               : 1;
}

