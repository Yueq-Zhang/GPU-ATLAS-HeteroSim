#include "atlas_full_chip_memory_service.h"
#include "chip/chip.h"
#include "ramulator_bridge.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

constexpr uint64_t kGpuClockHz = 1200000000ULL;
constexpr uint64_t kAtlasClockHz = 1000000000ULL;
constexpr uint64_t kCoreRegionBytes = 1ULL << 20;
constexpr uint32_t kTransactionBytes = 64;
constexpr uint32_t kAtlasCores = 16;
constexpr uint32_t kSyntheticGpuParents = 4096;

struct CaseResult {
  uint64_t host_gpu_cycles = 0;
  uint64_t atlas_finish_gpu_cycle = 0;
  uint64_t atlas_cycles = 0;
  uint64_t atlas_e2e_cycles = 0;
  uint64_t atlas_memory_bytes = 0;
  uint64_t atlas_parents = 0;
  uint64_t atlas_completed = 0;
  uint64_t gpu_parents = 0;
  uint64_t gpu_completed = 0;
  uint64_t outstanding = 0;
  uint64_t ramulator_cycles = 0;
};

CaseResult run_case(const char *memory_config, const char *chip_config,
                    const char *operator_list, const char *placement_map,
                    bool enable_gpu) {
  auto handle = heterosim_ramulator_create(memory_config, 0, 1);
  if (!handle) throw std::runtime_error("failed to create shared Ramulator2");

  auto service = std::make_shared<heterosim::AtlasFullChipMemoryService>(
      handle, 0, kTransactionBytes, kCoreRegionBytes, 25.6, 16);
  auto chip = std::make_unique<atlasim::Chip>(
      chip_config, operator_list, placement_map, "", service);
  chip->reset_execution_status();

  std::vector<std::unique_ptr<uint64_t>> gpu_payloads;
  gpu_payloads.reserve(kSyntheticGpuParents);
  for (uint32_t index = 0; index < kSyntheticGpuParents; ++index) {
    gpu_payloads.push_back(std::make_unique<uint64_t>(index));
  }

  uint32_t gpu_sent = 0;
  uint32_t gpu_returned = 0;
  uint64_t atlas_phase = 0;
  uint64_t atlas_finish_gpu_cycle = 0;
  constexpr uint64_t kCycleLimit = 20000000ULL;
  uint64_t host_cycle = 0;
  for (host_cycle = 1; host_cycle <= kCycleLimit; ++host_cycle) {
    for (uint32_t issue = 0;
         enable_gpu && issue < 4 && gpu_sent < kSyntheticGpuParents; ++issue) {
      heterosim_parent_request_v2 request{};
      request.abi_version = HETEROSIM_RAMULATOR_ABI_VERSION;
      request.struct_size = sizeof(request);
      request.parent_id = gpu_sent;
      request.global_address = (64ULL << 20) + uint64_t(gpu_sent) * 128;
      request.size_bytes = 128;
      request.partition_id = 0;
      request.operation = HETEROSIM_MEMORY_READ;
      request.flags = HETEROSIM_REQUEST_BYTE_MASK_VALID |
                      HETEROSIM_REQUEST_SECTOR_MASK_VALID;
      request.byte_mask[0] = UINT64_MAX;
      request.byte_mask[1] = UINT64_MAX;
      request.byte_mask_word_count = HETEROSIM_MAX_BYTE_MASK_WORDS;
      request.sector_mask = 0xf;
      request.ordering_domain = 0;
      request.sequence_number = gpu_sent;
      request.payload = gpu_payloads.at(gpu_sent).get();
      const int status = heterosim_ramulator_send_v2(handle, &request);
      if (status == HETEROSIM_SEND_INVALID) {
        throw std::runtime_error("synthetic GPU request rejected as invalid");
      }
      if (status == HETEROSIM_SEND_RETRY) break;
      ++gpu_sent;
    }

    atlas_phase += kAtlasClockHz;
    while (atlas_phase >= kGpuClockHz) {
      if (!chip->is_finished()) chip->tick();
      atlas_phase -= kGpuClockHz;
    }
    service->advance();
    heterosim_ramulator_tick(handle);

    heterosim_parent_completion_v2 completion{};
    while (heterosim_ramulator_pop_completed_for_initiator_v2(
        handle, HETEROSIM_INITIATOR_GPU, &completion)) {
      if (completion.abi_version != HETEROSIM_RAMULATOR_ABI_VERSION ||
          completion.struct_size != sizeof(completion) ||
          completion.initiator != HETEROSIM_INITIATOR_GPU ||
          completion.completed_children != completion.total_children ||
          !completion.durable || !completion.payload) {
        throw std::runtime_error("invalid synthetic GPU completion");
      }
      ++gpu_returned;
    }
    if (chip->is_finished() && atlas_finish_gpu_cycle == 0) {
      atlas_finish_gpu_cycle = host_cycle;
    }

    const bool gpu_done = !enable_gpu ||
                          gpu_returned == kSyntheticGpuParents;
    if (chip->is_finished() && service->idle() && gpu_done &&
        heterosim_ramulator_is_finished(handle)) {
      break;
    }
    if (host_cycle == kCycleLimit) {
      throw std::runtime_error("full-chip scheduler qualification timed out");
    }
  }

  const atlasim::Performance performance = chip->get_performance();
  CaseResult result;
  result.host_gpu_cycles = host_cycle;
  result.atlas_finish_gpu_cycle = atlas_finish_gpu_cycle;
  result.atlas_cycles = chip->get_clock();
  result.atlas_e2e_cycles = performance.e2e_stats.e2e_cycles;
  result.atlas_memory_bytes =
      static_cast<uint64_t>(performance.e2e_stats.memory_access_bytes);
  result.atlas_parents = heterosim_ramulator_initiator_parents(
      handle, HETEROSIM_INITIATOR_ATLAS_LOGIC_DIE);
  result.atlas_completed = heterosim_ramulator_initiator_completed(
      handle, HETEROSIM_INITIATOR_ATLAS_LOGIC_DIE);
  result.gpu_parents = heterosim_ramulator_initiator_parents(
      handle, HETEROSIM_INITIATOR_GPU);
  result.gpu_completed = heterosim_ramulator_initiator_completed(
      handle, HETEROSIM_INITIATOR_GPU);
  result.outstanding = heterosim_ramulator_outstanding(handle);
  result.ramulator_cycles = heterosim_ramulator_clock(handle);

  chip.reset();
  service.reset();
  heterosim_ramulator_destroy(handle);
  return result;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 5) {
    std::cerr << "usage: full_chip_scheduler_smoke MEMORY.yaml CHIP.yaml "
                 "OPERATORS.yaml PLACEMENT.yaml\n";
    return 2;
  }
  try {
    const CaseResult atlas_only =
        run_case(argv[1], argv[2], argv[3], argv[4], false);
    const CaseResult concurrent =
        run_case(argv[1], argv[2], argv[3], argv[4], true);
    const bool conservation =
        atlas_only.atlas_parents > 0 &&
        atlas_only.atlas_parents == atlas_only.atlas_completed &&
        concurrent.atlas_parents == concurrent.atlas_completed &&
        concurrent.gpu_parents == kSyntheticGpuParents &&
        concurrent.gpu_completed == kSyntheticGpuParents &&
        atlas_only.outstanding == 0 && concurrent.outstanding == 0;
    const bool contention = concurrent.atlas_finish_gpu_cycle >
                            atlas_only.atlas_finish_gpu_cycle;
    std::cout << "heterosim_full_chip_scheduler_smoke"
              << " atlas_only_gpu_cycles=" << atlas_only.host_gpu_cycles
              << " atlas_only_finish=" << atlas_only.atlas_finish_gpu_cycle
              << " atlas_only_chip_cycles=" << atlas_only.atlas_cycles
              << " atlas_only_e2e_cycles=" << atlas_only.atlas_e2e_cycles
              << " concurrent_gpu_cycles=" << concurrent.host_gpu_cycles
              << " concurrent_atlas_finish="
              << concurrent.atlas_finish_gpu_cycle
              << " concurrent_chip_cycles=" << concurrent.atlas_cycles
              << " concurrent_e2e_cycles=" << concurrent.atlas_e2e_cycles
              << " atlas_memory_bytes=" << concurrent.atlas_memory_bytes
              << " atlas_parents=" << concurrent.atlas_parents
              << " gpu_parents=" << concurrent.gpu_parents
              << " ramulator_cycles=" << concurrent.ramulator_cycles
              << " outstanding=" << concurrent.outstanding
              << " contention=" << (contention ? 1 : 0)
              << " conservation=" << (conservation ? 1 : 0)
              << " instances=1\n";
    return conservation && contention ? 0 : 4;
  } catch (const std::exception &error) {
    std::cerr << "full-chip scheduler qualification failed: "
              << error.what() << '\n';
    return 3;
  }
}
