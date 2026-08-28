#include "atlas_full_chip_memory_service.h"

#include <limits>
#include <stdexcept>
#include <utility>

namespace heterosim {

AtlasFullChipMemoryService::AtlasFullChipMemoryService(
    heterosim_ramulator_handle handle, uint32_t completion_partition,
    uint32_t transaction_bytes, uint64_t core_region_bytes,
    double max_bandwidth_gbps_per_core,
    uint32_t issue_width_per_host_cycle)
    : handle_(handle),
      completion_partition_(completion_partition),
      transaction_bytes_(transaction_bytes),
      core_region_bytes_(core_region_bytes),
      max_bandwidth_gbps_per_core_(max_bandwidth_gbps_per_core),
      issue_width_per_host_cycle_(issue_width_per_host_cycle),
      submission_port_(handle, completion_partition, transaction_bytes) {
  if (!handle_ || transaction_bytes_ == 0 || core_region_bytes_ == 0 ||
      max_bandwidth_gbps_per_core_ <= 0 || issue_width_per_host_cycle_ == 0) {
    throw std::invalid_argument("invalid ATLAS full-chip memory service");
  }
  if (core_region_bytes_ % transaction_bytes_ != 0) {
    throw std::invalid_argument(
        "ATLAS per-core Global PA region must be transaction aligned");
  }
}

AtlasFullChipMemoryService::KeyTuple
AtlasFullChipMemoryService::tuple_key(
    const atlasim::ExternalDramRequestKey &key) {
  if (key.core_id < 0 || key.task_id < 0 || key.iteration < 0) {
    throw std::invalid_argument("negative ATLAS external DRAM request key");
  }
  return {key.core_id, key.task_id, key.iteration};
}

uint64_t AtlasFullChipMemoryService::global_base(int core_id) const {
  if (core_id < 0 ||
      static_cast<uint64_t>(core_id) >
          std::numeric_limits<uint64_t>::max() / core_region_bytes_) {
    throw std::overflow_error("ATLAS per-core Global PA projection overflow");
  }
  return static_cast<uint64_t>(core_id) * core_region_bytes_;
}

AtlasHybridBondPort AtlasFullChipMemoryService::make_projection_port(
    int core_id) const {
  return AtlasHybridBondPort(handle_, completion_partition_,
                             transaction_bytes_, global_base(core_id));
}

void AtlasFullChipMemoryService::validate_core_region(
    int core_id, const std::vector<AtlasHbAccess> &accesses) const {
  const uint64_t begin = global_base(core_id);
  if (begin > std::numeric_limits<uint64_t>::max() - core_region_bytes_) {
    throw std::overflow_error("ATLAS per-core Global PA region overflow");
  }
  const uint64_t end = begin + core_region_bytes_;
  for (const auto &access : accesses) {
    if (access.global_address < begin || access.global_address >= end ||
        access.size_bytes > end - access.global_address) {
      throw std::out_of_range(
          "ATLAS local placement exceeds its per-core Global PA region");
    }
  }
}

atlasim::ExternalDramTraffic AtlasFullChipMemoryService::estimate_traffic(
  int core_id, const atlasim::ComponentInput &input) const {
  atlasim::ExternalDramTraffic traffic;
  const auto accesses = make_projection_port(core_id).generate(input);
  validate_core_region(core_id, accesses);
  for (const auto &access : accesses) {
    if (access.operation == HETEROSIM_MEMORY_WRITE) {
      traffic.write_bytes += access.size_bytes;
    } else {
      traffic.read_bytes += access.size_bytes;
    }
  }
  return traffic;
}

double AtlasFullChipMemoryService::max_bandwidth_gbps_per_core() const {
  return max_bandwidth_gbps_per_core_;
}

bool AtlasFullChipMemoryService::submit_iteration(
    const atlasim::ExternalDramRequestKey &key,
    const atlasim::ComponentInput &input) {
  const KeyTuple id = tuple_key(key);
  if (iterations_.count(id)) return true;

  IterationState state;
  const auto accesses = make_projection_port(key.core_id).generate(input);
  validate_core_region(key.core_id, accesses);
  state.accesses.reserve(accesses.size());
  for (const auto &access : accesses) {
    if (next_parent_id_ == std::numeric_limits<uint64_t>::max()) {
      throw std::overflow_error("ATLAS parent ID space exhausted");
    }
    AccessState item;
    item.access = access;
    item.parent_id = next_parent_id_++;
    item.context = std::make_unique<AccessContext>(
        AccessContext{id, state.accesses.size(), item.parent_id});
    logical_bytes_ += access.size_bytes;
    state.accesses.push_back(std::move(item));
  }
  iterations_.emplace(id, std::move(state));
  return true;
}

bool AtlasFullChipMemoryService::iteration_complete(
    const atlasim::ExternalDramRequestKey &key) const {
  const auto found = iterations_.find(tuple_key(key));
  return found != iterations_.end() &&
         found->second.completed == found->second.accesses.size();
}

void AtlasFullChipMemoryService::retire_iteration(
    const atlasim::ExternalDramRequestKey &key) {
  const auto found = iterations_.find(tuple_key(key));
  if (found == iterations_.end() ||
      found->second.completed != found->second.accesses.size()) {
    throw std::logic_error("retiring incomplete ATLAS DRAM iteration");
  }
  iterations_.erase(found);
}

void AtlasFullChipMemoryService::poll_completions() {
  heterosim_parent_completion_v2 completion{};
  while (heterosim_ramulator_pop_completed_for_initiator_v2(
      handle_, HETEROSIM_INITIATOR_ATLAS_LOGIC_DIE, &completion)) {
    if (completion.abi_version != HETEROSIM_RAMULATOR_ABI_VERSION ||
        completion.struct_size != sizeof(completion) || !completion.durable ||
        completion.completed_children != completion.total_children ||
        !completion.payload) {
      throw std::runtime_error("invalid ATLAS full-chip completion");
    }
    auto *context = static_cast<AccessContext *>(completion.payload);
    const auto iteration = iterations_.find(context->key);
    if (iteration == iterations_.end() ||
        context->access_index >= iteration->second.accesses.size()) {
      throw std::runtime_error("ATLAS completion references retired iteration");
    }
    AccessState &access = iteration->second.accesses.at(context->access_index);
    if (access.parent_id != completion.parent_id ||
        context->parent_id != completion.parent_id || !access.issued ||
        access.completed) {
      throw std::runtime_error("ATLAS completion identity mismatch");
    }
    access.completed = true;
    ++iteration->second.completed;
    ++completed_parents_;
  }
}

void AtlasFullChipMemoryService::issue_requests() {
  uint32_t issued = 0;
  for (auto &[key, iteration] : iterations_) {
    while (issued < issue_width_per_host_cycle_ &&
           iteration.next_issue < iteration.accesses.size()) {
      AccessState &access = iteration.accesses.at(iteration.next_issue);
      const uint64_t ordering_domain =
          (static_cast<uint64_t>(std::get<0>(key)) << 32) |
          static_cast<uint32_t>(std::get<1>(key));
      const int status = submission_port_.submit(
          access.access, access.parent_id, ordering_domain,
          static_cast<uint64_t>(std::get<2>(key)), access.context.get());
      if (status == HETEROSIM_SEND_INVALID) {
        throw std::runtime_error("shared gateway rejected ATLAS Chip request");
      }
      if (status == HETEROSIM_SEND_RETRY) return;
      access.issued = true;
      ++iteration.next_issue;
      ++submitted_parents_;
      ++issued;
    }
    if (issued == issue_width_per_host_cycle_) break;
  }
}

void AtlasFullChipMemoryService::advance() {
  poll_completions();
  issue_requests();
}

bool AtlasFullChipMemoryService::idle() const {
  return iterations_.empty() && submitted_parents_ == completed_parents_;
}

}  // namespace heterosim
