#include "ramulator_bridge.h"

#include <cstdlib>
#include <deque>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "base/config.h"
#include "base/request.h"
#include "frontend/frontend.h"
#include "memory_system/memory_system.h"

namespace {

struct SharedBridge {
  std::shared_ptr<Ramulator::IFrontEnd> frontend;
  std::shared_ptr<Ramulator::IMemorySystem> memory;
  std::vector<std::deque<void *>> completed_payloads;
  std::string config_path;
  uint64_t reads = 0;
  uint64_t writes = 0;
  uint64_t completed = 0;
  uint64_t rejected = 0;
  unsigned partition_count = 0;
  unsigned references = 0;
  bool finalized = false;

  SharedBridge(const char *path, unsigned count)
      : completed_payloads(count), config_path(path), partition_count(count) {
    if (config_path.empty() || partition_count == 0) {
      throw std::invalid_argument("invalid shared Ramulator2 configuration");
    }
    std::vector<std::string> overrides;
    YAML::Node config =
        Ramulator::Config::parse_config_file(config_path, overrides);
    frontend.reset(Ramulator::Factory::create_frontend(config));
    memory.reset(Ramulator::Factory::create_memory_system(config));
    frontend->connect_memory_system(memory.get());
    memory->connect_frontend(frontend.get());
  }

  uint64_t outstanding() const {
    const uint64_t accepted = reads + writes;
    return accepted >= completed ? accepted - completed : 0;
  }

  void finish() {
    if (finalized || !memory) return;
    constexpr uint64_t kDrainLimit = 1000000000ULL;
    uint64_t drain_cycles = 0;
    while (!memory->is_finished() && drain_cycles < kDrainLimit) {
      memory->tick();
      ++drain_cycles;
    }
    if (!memory->is_finished()) {
      throw std::runtime_error("Ramulator2 did not drain before finalization");
    }
    std::cout << "heterosim_ramulator2_summary"
              << " cycles=" << memory->get_clk() << " reads=" << reads
              << " writes=" << writes << " completed=" << completed
              << " rejected=" << rejected << " outstanding=" << outstanding()
              << " instances=1 partitions=" << partition_count << std::endl;
    memory->finalize();
    finalized = true;
  }
};

struct PartitionHandle {
  SharedBridge *shared = nullptr;
  unsigned partition_id = 0;
};

std::unique_ptr<SharedBridge> g_shared_bridge;
bool g_atexit_registered = false;

void finish_shared_bridge_at_exit() {
  if (!g_shared_bridge) return;
  try {
    g_shared_bridge->finish();
  } catch (const std::exception &error) {
    std::cerr << "heterosim_ramulator2 finalization failed: " << error.what()
              << std::endl;
  }
}

PartitionHandle *as_handle(heterosim_ramulator_handle handle) {
  return static_cast<PartitionHandle *>(handle);
}

SharedBridge *shared_from(heterosim_ramulator_handle handle) {
  PartitionHandle *partition = as_handle(handle);
  return partition ? partition->shared : nullptr;
}

}  // namespace

extern "C" heterosim_ramulator_handle heterosim_ramulator_create(
    const char *config_path, unsigned partition_id, unsigned partition_count) {
  try {
    if (!config_path || partition_id >= partition_count) return nullptr;
    if (!g_shared_bridge) {
      g_shared_bridge =
          std::make_unique<SharedBridge>(config_path, partition_count);
      if (!g_atexit_registered) {
        std::atexit(finish_shared_bridge_at_exit);
        g_atexit_registered = true;
      }
    } else if (g_shared_bridge->config_path != config_path ||
               g_shared_bridge->partition_count != partition_count) {
      throw std::runtime_error(
          "all GPU partitions must share one Ramulator2 configuration");
    }
    auto *handle = new PartitionHandle{g_shared_bridge.get(), partition_id};
    ++g_shared_bridge->references;
    return handle;
  } catch (const std::exception &error) {
    std::cerr << "heterosim_ramulator_create failed: " << error.what()
              << std::endl;
    return nullptr;
  }
}

extern "C" void heterosim_ramulator_destroy(
    heterosim_ramulator_handle opaque) {
  PartitionHandle *handle = as_handle(opaque);
  if (!handle) return;
  SharedBridge *shared = handle->shared;
  delete handle;
  if (!shared || shared->references == 0) return;
  --shared->references;
  if (shared->references == 0) {
    try {
      shared->finish();
    } catch (const std::exception &error) {
      std::cerr << "heterosim_ramulator_destroy failed: " << error.what()
                << std::endl;
    }
    g_shared_bridge.reset();
  }
}

extern "C" int heterosim_ramulator_send(heterosim_ramulator_handle opaque,
                                          uint64_t address, int is_write,
                                          void *payload) {
  PartitionHandle *handle = as_handle(opaque);
  SharedBridge *shared = shared_from(opaque);
  if (!handle || !shared || shared->finalized || !payload) return 0;
  const int type = is_write ? Ramulator::Request::Type::Write
                            : Ramulator::Request::Type::Read;
  const unsigned partition_id = handle->partition_id;
  Ramulator::Request request(
      static_cast<Ramulator::Addr_t>(address), type, 0,
      [shared, partition_id, payload](Ramulator::Request &) {
        shared->completed_payloads.at(partition_id).push_back(payload);
        ++shared->completed;
      });
  request.m_payload = payload;
  if (!shared->memory->send(request)) {
    ++shared->rejected;
    return 0;
  }
  if (is_write) {
    // GPU writebacks are posted. Ramulator2 continues to model command timing;
    // the bridge drains the memory system before finalization.
    shared->completed_payloads.at(partition_id).push_back(payload);
    ++shared->writes;
    ++shared->completed;
  } else {
    ++shared->reads;
  }
  return 1;
}

extern "C" void heterosim_ramulator_tick(heterosim_ramulator_handle handle) {
  SharedBridge *shared = shared_from(handle);
  if (shared && !shared->finalized) shared->memory->tick();
}

extern "C" void *heterosim_ramulator_pop_completed(
    heterosim_ramulator_handle opaque) {
  PartitionHandle *handle = as_handle(opaque);
  SharedBridge *shared = shared_from(opaque);
  if (!handle || !shared) return nullptr;
  auto &queue = shared->completed_payloads.at(handle->partition_id);
  if (queue.empty()) return nullptr;
  void *payload = queue.front();
  queue.pop_front();
  return payload;
}

extern "C" uint64_t heterosim_ramulator_clock(
    heterosim_ramulator_handle handle) {
  SharedBridge *shared = shared_from(handle);
  return shared ? static_cast<uint64_t>(shared->memory->get_clk()) : 0;
}

extern "C" uint64_t heterosim_ramulator_reads(
    heterosim_ramulator_handle handle) {
  SharedBridge *shared = shared_from(handle);
  return shared ? shared->reads : 0;
}

extern "C" uint64_t heterosim_ramulator_writes(
    heterosim_ramulator_handle handle) {
  SharedBridge *shared = shared_from(handle);
  return shared ? shared->writes : 0;
}

extern "C" uint64_t heterosim_ramulator_completed(
    heterosim_ramulator_handle handle) {
  SharedBridge *shared = shared_from(handle);
  return shared ? shared->completed : 0;
}

extern "C" uint64_t heterosim_ramulator_rejected(
    heterosim_ramulator_handle handle) {
  SharedBridge *shared = shared_from(handle);
  return shared ? shared->rejected : 0;
}

extern "C" uint64_t heterosim_ramulator_outstanding(
    heterosim_ramulator_handle handle) {
  SharedBridge *shared = shared_from(handle);
  return shared ? shared->outstanding() : 0;
}

extern "C" int heterosim_ramulator_is_finished(
    heterosim_ramulator_handle handle) {
  SharedBridge *shared = shared_from(handle);
  return shared && shared->memory->is_finished();
}

extern "C" void heterosim_ramulator_finish(
    heterosim_ramulator_handle handle) {
  SharedBridge *shared = shared_from(handle);
  if (!shared) return;
  try {
    shared->finish();
  } catch (const std::exception &error) {
    std::cerr << "heterosim_ramulator_finish failed: " << error.what()
              << std::endl;
  }
}
