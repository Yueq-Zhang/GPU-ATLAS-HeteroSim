#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "hetero/types.h"

namespace heterosim {

struct SimulationBufferBinding {
    std::string value_id;
    PhysicalAddress physical_address;
};

struct ArtifactRequest {
    std::uint64_t request_id{};
    std::uint64_t epoch_id{};
    std::string backend_id;
    std::string artifact_kind;
    std::string compile_plan_key;
    std::string task_signature;
    std::string shape_signature;
    std::string placement_result_hash;
    std::vector<SimulationBufferBinding> simulation_buffer_bindings;
    std::string expected_artifact_key;
};

struct ArtifactResponse {
    std::uint64_t request_id{};
    std::string status;
    std::string artifact_ref;
    std::string artifact_hash;
    std::string manifest_schema_version;
    std::string qualification_record_ref;
};

}  // namespace heterosim

