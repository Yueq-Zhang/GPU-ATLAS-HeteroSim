#include "atlas_full_chip_memory_service.h"
#include "chip/chip.h"
#include "ramulator_bridge.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

uint64_t positive_env(const char *name, uint64_t fallback) {
  const char *text = std::getenv(name);
  if (!text || !text[0]) return fallback;
  char *end = nullptr;
  const unsigned long long value = std::strtoull(text, &end, 10);
  if (!end || *end || value == 0) {
    throw std::invalid_argument(std::string("invalid positive value: ") + name);
  }
  return static_cast<uint64_t>(value);
}

double positive_double_env(const char *name, double fallback) {
  const char *text = std::getenv(name);
  if (!text || !text[0]) return fallback;
  char *end = nullptr;
  const double value = std::strtod(text, &end);
  if (!end || *end || !std::isfinite(value) || value <= 0) {
    throw std::invalid_argument(std::string("invalid positive value: ") + name);
  }
  return value;
}

const char *required_env(const char *name) {
  const char *value = std::getenv(name);
  if (!value || !value[0]) {
    throw std::invalid_argument(std::string("missing required value: ") + name);
  }
  return value;
}

class AtlasRuntimeState {
 public:
  explicit AtlasRuntimeState(heterosim_ramulator_handle borrowed_handle) {
    const char *chip_config = required_env("HETEROSIM_ATLAS_CHIP_CONFIG");
    const char *operator_list = required_env("HETEROSIM_ATLAS_OPERATOR_LIST");
    const char *placement_map = required_env("HETEROSIM_ATLAS_PLACEMENT_MAP");
    handle_ = heterosim_ramulator_retain(borrowed_handle);
    if (!handle_) throw std::runtime_error("failed to retain shared memory handle");
    try {
      gpu_clock_hz_ = positive_env("HETEROSIM_GPU_CLOCK_HZ", 1200000000ULL);
      service_ = std::make_shared<heterosim::AtlasFullChipMemoryService>(
          handle_, 0,
          static_cast<uint32_t>(positive_env(
              "HETEROSIM_DRAM_TRANSACTION_BYTES", 64)),
          positive_env("HETEROSIM_ATLAS_CORE_REGION_BYTES", 1ULL << 20),
          positive_double_env("HETEROSIM_ATLAS_CORE_BANDWIDTH_GBPS", 25.6),
          static_cast<uint32_t>(positive_env(
              "HETEROSIM_ATLAS_ISSUE_WIDTH", 16)));
      chip_ = std::make_unique<atlasim::Chip>(
          chip_config, operator_list, placement_map, "", service_);
      const double frequency_mhz = chip_->get_arch_config()->frequency;
      atlas_clock_hz_ = static_cast<uint64_t>(
          std::llround(frequency_mhz * 1000000.0));
      if (atlas_clock_hz_ == 0) {
        throw std::runtime_error("ATLAS Chip frequency is zero");
      }
      chip_->reset_execution_status();
    } catch (...) {
      chip_.reset();
      service_.reset();
      heterosim_ramulator_destroy(handle_);
      handle_ = nullptr;
      throw;
    }
    std::cout << "heterosim_atlas_full_chip_runtime_start"
              << " enabled=1 gpu_clock_hz=" << gpu_clock_hz_
              << " atlas_clock_hz=" << atlas_clock_hz_
              << " instances=1\n";
  }

  ~AtlasRuntimeState() { close(); }

  void advance(uint64_t gpu_cycles, uint64_t global_time_fs) {
    if (closed_ || failed_ || !chip_) return;
    try {
      service_->poll_completions();
      atlas_phase_ += atlas_clock_hz_;
      while (atlas_phase_ >= gpu_clock_hz_) {
        if (!chip_->is_finished()) chip_->tick();
        atlas_phase_ -= gpu_clock_hz_;
      }
      service_->issue_requests();
      if (!chip_->is_finished() || finish_gpu_cycle_ != 0) return;
      finish_gpu_cycle_ = gpu_cycles;
      finish_time_fs_ = global_time_fs;
    } catch (const std::exception &error) {
      failed_ = true;
      error_ = error.what();
      std::cerr << "heterosim ATLAS runtime advance failed: " << error.what()
                << '\n';
    }
  }

  bool active() const {
    return !closed_ && !failed_ && chip_ &&
           (!chip_->is_finished() || !service_->idle());
  }

  void close() {
    if (closed_) return;
    closed_ = true;
    if (chip_ && service_) {
      try {
        service_->poll_completions();
        const atlasim::Performance performance = chip_->get_performance();
        const bool complete = chip_->is_finished() && service_->idle();
        std::cout << "heterosim_atlas_full_chip_runtime_summary"
                  << " status="
                  << (failed_ ? "error" : (complete ? "passed" : "incomplete"))
                  << " atlas_cycles=" << chip_->get_clock()
                  << " atlas_e2e_cycles=" << performance.e2e_stats.e2e_cycles
                  << " finish_gpu_cycle=" << finish_gpu_cycle_
                  << " finish_time_fs=" << finish_time_fs_
                  << " memory_bytes="
                  << static_cast<uint64_t>(
                         performance.e2e_stats.memory_access_bytes)
                  << " transaction_bytes=" << service_->logical_bytes()
                  << " submitted_parents=" << service_->submitted_parents()
                  << " completed_parents=" << service_->completed_parents()
                  << " bridge_atlas_parents="
                  << heterosim_ramulator_initiator_parents(
                         handle_, HETEROSIM_INITIATOR_ATLAS_LOGIC_DIE)
                  << " bridge_atlas_completed="
                  << heterosim_ramulator_initiator_completed(
                         handle_, HETEROSIM_INITIATOR_ATLAS_LOGIC_DIE)
                  << " runtime_active=" << (active() ? 1 : 0)
                  << " instances=1\n";
      } catch (const std::exception &error) {
        std::cerr << "heterosim ATLAS runtime finalization failed: "
                  << error.what() << '\n';
      }
    }
    chip_.reset();
    service_.reset();
    if (handle_) {
      heterosim_ramulator_destroy(handle_);
      handle_ = nullptr;
    }
  }

 private:
  heterosim_ramulator_handle handle_ = nullptr;
  std::shared_ptr<heterosim::AtlasFullChipMemoryService> service_;
  std::unique_ptr<atlasim::Chip> chip_;
  uint64_t gpu_clock_hz_ = 0;
  uint64_t atlas_clock_hz_ = 0;
  uint64_t atlas_phase_ = 0;
  uint64_t finish_gpu_cycle_ = 0;
  uint64_t finish_time_fs_ = 0;
  bool failed_ = false;
  bool closed_ = false;
  std::string error_;
};

std::unique_ptr<AtlasRuntimeState> g_runtime;

}  // namespace

extern "C" void heterosim_atlas_runtime_autostart(
    heterosim_ramulator_handle handle) {
  if (g_runtime || !std::getenv("HETEROSIM_ATLAS_CHIP_CONFIG")) return;
  try {
    g_runtime = std::make_unique<AtlasRuntimeState>(handle);
  } catch (const std::exception &error) {
    std::cerr << "heterosim ATLAS runtime autostart failed: " << error.what()
              << '\n';
  }
}

extern "C" void heterosim_atlas_runtime_advance(uint64_t gpu_cycles,
                                                   uint64_t global_time_fs) {
  if (g_runtime) g_runtime->advance(gpu_cycles, global_time_fs);
}

extern "C" int heterosim_atlas_runtime_active() {
  return g_runtime && g_runtime->active() ? 1 : 0;
}

extern "C" void heterosim_atlas_runtime_shutdown() {
  if (!g_runtime) return;
  g_runtime->close();
  g_runtime.reset();
}
