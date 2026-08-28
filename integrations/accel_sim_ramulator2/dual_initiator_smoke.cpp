#include "atlas_hb_port.h"
#include "ramulator_bridge.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace {

constexpr std::size_t kGpuParents = 72;
constexpr std::size_t kAtlasReads = 64;
constexpr std::size_t kAtlasWrites = 16;
constexpr std::size_t kAtlasParents = kAtlasReads + kAtlasWrites;

atlasim::ComponentInputItem make_tile(int64_t base_address, int64_t elements,
                                      bool is_write) {
  return {
      {"base_addr", base_address},
      {"element_size", 2},
      {"is_write", is_write ? 1 : 0},
      {"layout_rank", 2},
      {"shape_0", 1},
      {"shape_1", elements},
      {"stride_0", elements},
      {"stride_1", 1},
      {"access_base_0", 0},
      {"access_base_1", 0},
      {"access_extent_0", 1},
      {"access_extent_1", elements},
  };
}

struct CaseResult {
  uint64_t cycles = 0;
  uint64_t gpu_last_cycle = 0;
  uint64_t atlas_last_cycle = 0;
  uint64_t gpu_parents = 0;
  uint64_t atlas_parents = 0;
  uint64_t gpu_completed = 0;
  uint64_t atlas_completed = 0;
  uint64_t gpu_children = 0;
  uint64_t atlas_children = 0;
  uint64_t logical_bytes = 0;
  uint64_t internal_bytes = 0;
  uint64_t outstanding = 0;
};

CaseResult run_case(const char *config, bool enable_gpu, bool enable_atlas) {
  auto handle = heterosim_ramulator_create(config, 0, 1);
  if (!handle) throw std::runtime_error("failed to create shared Ramulator2");

  heterosim::AtlasHybridBondPort atlas_port(handle, 0, 64);
  auto input = std::make_shared<std::vector<atlasim::ComponentInputItem>>();
  input->push_back(make_tile(0, 2048, false));
  input->push_back(make_tile(8192, 512, true));
  const auto atlas_accesses = atlas_port.generate(input);
  if (atlas_accesses.size() != kAtlasParents ||
      std::count_if(atlas_accesses.begin(), atlas_accesses.end(),
                    [](const auto &access) {
                      return access.operation == HETEROSIM_MEMORY_READ;
                    }) != kAtlasReads) {
    throw std::runtime_error("ATLAS ComponentInput translation mismatch");
  }

  std::vector<uint64_t> gpu_payloads(kGpuParents);
  std::vector<uint64_t> atlas_payloads(kAtlasParents);
  std::size_t gpu_sent = 0;
  std::size_t atlas_sent = 0;
  std::size_t gpu_returned = 0;
  std::size_t atlas_returned = 0;
  CaseResult result;
  constexpr uint64_t kCycleLimit = 1000000;
  for (uint64_t host_cycle = 1; host_cycle <= kCycleLimit; ++host_cycle) {
    if (enable_gpu && gpu_sent < kGpuParents) {
      heterosim_parent_request_v2 request{};
      request.abi_version = HETEROSIM_RAMULATOR_ABI_VERSION;
      request.struct_size = sizeof(request);
      request.parent_id = gpu_sent;
      request.global_address = gpu_sent * 128;
      request.size_bytes = 128;
      request.partition_id = 0;
      request.operation = gpu_sent < 64 ? HETEROSIM_MEMORY_READ
                                        : HETEROSIM_MEMORY_WRITE;
      request.flags = HETEROSIM_REQUEST_BYTE_MASK_VALID |
                      HETEROSIM_REQUEST_SECTOR_MASK_VALID;
      request.byte_mask[0] = UINT64_MAX;
      request.byte_mask[1] = UINT64_MAX;
      request.byte_mask_word_count = HETEROSIM_MAX_BYTE_MASK_WORDS;
      request.sector_mask = 0xf;
      request.ordering_domain = 0;
      request.sequence_number = gpu_sent;
      request.payload = &gpu_payloads.at(gpu_sent);
      const int status = heterosim_ramulator_send_v2(handle, &request);
      if (status == HETEROSIM_SEND_INVALID) {
        throw std::runtime_error("GPU parent rejected as invalid");
      }
      if (status == HETEROSIM_SEND_ACCEPTED) ++gpu_sent;
    }
    if (enable_atlas && atlas_sent < atlas_accesses.size()) {
      const int status = atlas_port.submit(
          atlas_accesses.at(atlas_sent), 100000 + atlas_sent, 1, atlas_sent,
          &atlas_payloads.at(atlas_sent));
      if (status == HETEROSIM_SEND_INVALID) {
        throw std::runtime_error("ATLAS parent rejected as invalid");
      }
      if (status == HETEROSIM_SEND_ACCEPTED) ++atlas_sent;
    }

    heterosim_ramulator_tick(handle);
    heterosim_parent_completion_v2 completion{};
    while (heterosim_ramulator_pop_completed_v2(handle, &completion)) {
      if (completion.abi_version != HETEROSIM_RAMULATOR_ABI_VERSION ||
          completion.struct_size != sizeof(completion) ||
          completion.completed_children != completion.total_children ||
          !completion.durable || !completion.payload) {
        throw std::runtime_error("invalid joined parent completion");
      }
      if (completion.initiator == HETEROSIM_INITIATOR_GPU) {
        if (completion.total_children != 2) {
          throw std::runtime_error("GPU parent child count mismatch");
        }
        ++gpu_returned;
        result.gpu_last_cycle = host_cycle;
      } else if (completion.initiator ==
                 HETEROSIM_INITIATOR_ATLAS_LOGIC_DIE) {
        if (completion.total_children != 1) {
          throw std::runtime_error("ATLAS parent child count mismatch");
        }
        ++atlas_returned;
        result.atlas_last_cycle = host_cycle;
      } else {
        throw std::runtime_error("unknown completion initiator");
      }
    }
    const bool gpu_done = !enable_gpu || gpu_returned == kGpuParents;
    const bool atlas_done = !enable_atlas || atlas_returned == kAtlasParents;
    if (gpu_done && atlas_done) break;
    if (host_cycle == kCycleLimit) {
      throw std::runtime_error("dual-initiator qualification timed out");
    }
  }

  result.cycles = heterosim_ramulator_clock(handle);
  result.gpu_parents = heterosim_ramulator_initiator_parents(
      handle, HETEROSIM_INITIATOR_GPU);
  result.atlas_parents = heterosim_ramulator_initiator_parents(
      handle, HETEROSIM_INITIATOR_ATLAS_LOGIC_DIE);
  result.gpu_completed = heterosim_ramulator_initiator_completed(
      handle, HETEROSIM_INITIATOR_GPU);
  result.atlas_completed = heterosim_ramulator_initiator_completed(
      handle, HETEROSIM_INITIATOR_ATLAS_LOGIC_DIE);
  result.gpu_children = heterosim_ramulator_initiator_children(
      handle, HETEROSIM_INITIATOR_GPU);
  result.atlas_children = heterosim_ramulator_initiator_children(
      handle, HETEROSIM_INITIATOR_ATLAS_LOGIC_DIE);
  result.logical_bytes = heterosim_ramulator_logical_bytes(handle);
  result.internal_bytes = heterosim_ramulator_internal_bytes(handle);
  result.outstanding = heterosim_ramulator_outstanding(handle);
  heterosim_ramulator_destroy(handle);
  return result;
}

bool valid_case(const CaseResult &result, bool gpu, bool atlas) {
  const uint64_t gpu_parents = gpu ? kGpuParents : 0;
  const uint64_t atlas_parents = atlas ? kAtlasParents : 0;
  const uint64_t expected_bytes = gpu_parents * 128 + atlas_parents * 64;
  return result.gpu_parents == gpu_parents &&
         result.atlas_parents == atlas_parents &&
         result.gpu_completed == gpu_parents &&
         result.atlas_completed == atlas_parents &&
         result.gpu_children == gpu_parents * 2 &&
         result.atlas_children == atlas_parents &&
         result.logical_bytes == expected_bytes &&
         result.internal_bytes == expected_bytes && result.outstanding == 0;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: dual_initiator_smoke CONFIG.yaml\n";
    return 2;
  }
  try {
    const CaseResult gpu_only = run_case(argv[1], true, false);
    const CaseResult atlas_only = run_case(argv[1], false, true);
    const CaseResult concurrent = run_case(argv[1], true, true);
    const bool contention =
        concurrent.cycles > std::max(gpu_only.cycles, atlas_only.cycles) &&
        (concurrent.gpu_last_cycle > gpu_only.gpu_last_cycle ||
         concurrent.atlas_last_cycle > atlas_only.atlas_last_cycle);
    std::cout << "heterosim_dual_initiator_smoke"
              << " gpu_only_cycles=" << gpu_only.cycles
              << " gpu_only_last=" << gpu_only.gpu_last_cycle
              << " atlas_only_cycles=" << atlas_only.cycles
              << " atlas_only_last=" << atlas_only.atlas_last_cycle
              << " concurrent_cycles=" << concurrent.cycles
              << " concurrent_gpu_last=" << concurrent.gpu_last_cycle
              << " concurrent_atlas_last=" << concurrent.atlas_last_cycle
              << " gpu_parents=" << concurrent.gpu_parents
              << " atlas_parents=" << concurrent.atlas_parents
              << " gpu_children=" << concurrent.gpu_children
              << " atlas_children=" << concurrent.atlas_children
              << " logical_bytes=" << concurrent.logical_bytes
              << " internal_bytes=" << concurrent.internal_bytes
              << " outstanding=" << concurrent.outstanding
              << " contention=" << (contention ? 1 : 0)
              << " instances=1\n";
    return valid_case(gpu_only, true, false) &&
                   valid_case(atlas_only, false, true) &&
                   valid_case(concurrent, true, true) && contention
               ? 0
               : 4;
  } catch (const std::exception &error) {
    std::cerr << "dual-initiator qualification failed: " << error.what()
              << '\n';
    return 3;
  }
}
