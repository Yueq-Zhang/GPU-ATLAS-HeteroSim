#include "booksim2_cycle_adapter.h"

#include <climits>
#include <cstdint>
#include <deque>
#include <exception>
#include <fstream>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/packet.h"
#include "include/booksim_config.hpp"
#include "include/credit.hpp"
#include "include/globals.hpp"
#include "include/routefunc.hpp"
#include "include/stats.hpp"
#include "include/trafficmanager.hpp"
#include "noc/noc_wrapper.h"

extern TrafficManager* trafficManager;

int GetSimTime() {
  return trafficManager == nullptr ? 0 : trafficManager->getTime();
}

Stats* GetStats(const std::string& name) {
  return trafficManager == nullptr ? nullptr : trafficManager->getStats(name);
}

namespace {

thread_local std::string g_last_error;
bool g_instance_active = false;

struct FlowRecord {
  uint32_t source;
  uint32_t destination;
  uint32_t flits;
  uint64_t accepted_cycle;
  bool injected;
};

struct Adapter {
  uint32_t nodes;
  uint64_t cycles = 0;
  uint64_t accepted_packets = 0;
  uint64_t injected_packets = 0;
  uint64_t delivered_packets = 0;
  uint64_t accepted_flits = 0;
  uint64_t delivered_flits = 0;
  atlasim::PCNInterfaceSet send_queues;
  atlasim::PCNInterfaceSet received_queues;
  std::unique_ptr<atlasim::NoCWrapper> noc;
  std::unordered_map<uint64_t, FlowRecord> flows;

  Adapter(const char* config_path, uint32_t node_count) : nodes(node_count) {
    if (g_instance_active) {
      throw std::runtime_error("only one BookSim2 adapter instance may be active");
    }
    if (config_path == nullptr || *config_path == '\0' || nodes < 2) {
      throw std::invalid_argument("BookSim2 config and at least two nodes are required");
    }
    std::ifstream config_file(config_path);
    if (!config_file.good()) {
      throw std::invalid_argument("BookSim2 config file does not exist");
    }
    BookSimConfig config;
    config.ParseFile(config_path);
    const int configured_nodes = [&]() {
      const std::string topology = config.GetStr("topology");
      if (topology == "mesh" || topology == "torus") {
        int result = 1;
        for (int dimension = 0; dimension < config.GetInt("n"); ++dimension) {
          result *= config.GetInt("k");
        }
        return result;
      }
      return static_cast<int>(nodes);
    }();
    if (configured_nodes != static_cast<int>(nodes)) {
      throw std::invalid_argument("node_count does not match BookSim2 topology");
    }
    send_queues = std::make_shared<std::vector<atlasim::CNInterface>>();
    received_queues = std::make_shared<std::vector<atlasim::CNInterface>>();
    for (uint32_t node = 0; node < nodes; ++node) {
      send_queues->push_back(std::make_shared<std::deque<atlasim::Packet>>());
      received_queues->push_back(std::make_shared<std::deque<atlasim::Packet>>());
    }
    noc = std::make_unique<atlasim::NoCWrapper>(config, send_queues, received_queues);
    g_instance_active = true;
  }

  ~Adapter() {
    noc.reset();
    g_instance_active = false;
  }

  uint64_t queued_packets() const {
    uint64_t result = 0;
    for (const auto& queue : *send_queues) {
      result += queue->size();
    }
    return result;
  }

  uint64_t unpolled_packets() const {
    uint64_t result = 0;
    for (const auto& queue : *received_queues) {
      result += queue->size();
    }
    return result;
  }
};

template <typename Function>
int guard(Function&& function) {
  try {
    g_last_error.clear();
    function();
    return 1;
  } catch (const std::exception& error) {
    g_last_error = error.what();
    return -1;
  } catch (const char* error) {
    g_last_error = error == nullptr ? "unknown BookSim2 error" : error;
    return -1;
  } catch (...) {
    g_last_error = "unknown BookSim2 exception";
    return -1;
  }
}

Adapter& checked(void* handle) {
  if (handle == nullptr) {
    throw std::invalid_argument("BookSim2 adapter handle is null");
  }
  return *static_cast<Adapter*>(handle);
}

}  // namespace

extern "C" void* heterosim_booksim2_create(
    const char* config_path, uint32_t node_count) {
  try {
    g_last_error.clear();
    return new Adapter(config_path, node_count);
  } catch (const std::exception& error) {
    g_last_error = error.what();
  } catch (const char* error) {
    g_last_error = error == nullptr ? "unknown BookSim2 error" : error;
  } catch (...) {
    g_last_error = "unknown BookSim2 exception";
  }
  return nullptr;
}

extern "C" void heterosim_booksim2_destroy(void* handle) {
  if (handle == nullptr) {
    return;
  }
  try {
    delete static_cast<Adapter*>(handle);
  } catch (...) {
    g_last_error = "BookSim2 adapter destruction failed";
  }
}

extern "C" int heterosim_booksim2_submit(
    void* handle,
    uint64_t flow_id,
    uint32_t source_node,
    uint32_t destination_node,
    uint32_t flit_count) {
  return guard([&]() {
    Adapter& adapter = checked(handle);
    if (flow_id == 0 || flow_id > static_cast<uint64_t>(INT_MAX) ||
        source_node >= adapter.nodes || destination_node >= adapter.nodes ||
        source_node == destination_node || flit_count == 0) {
      throw std::invalid_argument("invalid BookSim2 packet");
    }
    if (adapter.flows.count(flow_id) != 0) {
      throw std::invalid_argument("duplicate BookSim2 flow_id");
    }
    auto path = std::make_shared<atlasim::MCTree>(static_cast<int>(source_node));
    path->add_segment(
        static_cast<int>(source_node),
        static_cast<int>(destination_node),
        nullptr,
        true);
    atlasim::Packet packet;
    packet.type = atlasim::Packet::TransferType::_UNICAST;
    packet.fid = static_cast<int>(flow_id);
    packet.size = static_cast<int>(flit_count);
    packet.path = path;
    (*adapter.send_queues)[source_node]->push_back(packet);
    adapter.flows.emplace(
        flow_id,
        FlowRecord{source_node, destination_node, flit_count, adapter.cycles, false});
    ++adapter.accepted_packets;
    adapter.accepted_flits += flit_count;
  });
}

extern "C" int heterosim_booksim2_step(void* handle) {
  return guard([&]() {
    Adapter& adapter = checked(handle);
    adapter.noc->tick();
    ++adapter.cycles;
    for (auto& [flow_id, flow] : adapter.flows) {
      if (flow.injected) {
        continue;
      }
      bool still_queued = false;
      for (const auto& packet : *(*adapter.send_queues)[flow.source]) {
        if (packet.fid == static_cast<int>(flow_id)) {
          still_queued = true;
          break;
        }
      }
      if (!still_queued) {
        flow.injected = true;
        ++adapter.injected_packets;
      }
    }
  });
}

extern "C" int heterosim_booksim2_pop_completion(
    void* handle,
    uint32_t destination_node,
    heterosim_booksim2_completion_v1* completion) {
  try {
    g_last_error.clear();
    Adapter& adapter = checked(handle);
    if (destination_node >= adapter.nodes || completion == nullptr) {
      throw std::invalid_argument("invalid BookSim2 completion destination");
    }
    auto& queue = *(*adapter.received_queues)[destination_node];
    if (queue.empty()) {
      return 0;
    }
    const atlasim::Packet packet = queue.front();
    queue.pop_front();
    const auto found = adapter.flows.find(static_cast<uint64_t>(packet.fid));
    if (found == adapter.flows.end() || found->second.destination != destination_node) {
      throw std::runtime_error("BookSim2 returned an unknown packet");
    }
    const FlowRecord flow = found->second;
    adapter.flows.erase(found);
    ++adapter.delivered_packets;
    adapter.delivered_flits += flow.flits;
    *completion = heterosim_booksim2_completion_v1{
        HETEROSIM_BOOKSIM2_ABI_VERSION,
        sizeof(heterosim_booksim2_completion_v1),
        static_cast<uint64_t>(packet.fid),
        flow.source,
        flow.destination,
        flow.flits,
        0,
        flow.accepted_cycle,
        adapter.cycles};
    return 1;
  } catch (const std::exception& error) {
    g_last_error = error.what();
    return -1;
  } catch (...) {
    g_last_error = "unknown BookSim2 completion exception";
    return -1;
  }
}

extern "C" int heterosim_booksim2_get_stats(
    void* handle, heterosim_booksim2_stats_v1* stats) {
  return guard([&]() {
    Adapter& adapter = checked(handle);
    if (stats == nullptr) {
      throw std::invalid_argument("BookSim2 stats pointer is null");
    }
    const uint64_t queued = adapter.queued_packets();
    const uint64_t unpolled = adapter.unpolled_packets();
    const uint64_t credits = static_cast<uint64_t>(Credit::OutStanding());
    const bool network_drained = adapter.noc->traffic_drained() && queued == 0;
    const bool zero_in_flight =
        network_drained && unpolled == 0 && adapter.flows.empty() && credits == 0 &&
        adapter.accepted_packets == adapter.delivered_packets &&
        adapter.accepted_flits == adapter.delivered_flits;
    *stats = heterosim_booksim2_stats_v1{
        HETEROSIM_BOOKSIM2_ABI_VERSION,
        sizeof(heterosim_booksim2_stats_v1),
        adapter.cycles,
        adapter.accepted_packets,
        adapter.injected_packets,
        adapter.delivered_packets,
        adapter.accepted_flits,
        adapter.delivered_flits,
        credits,
        queued,
        unpolled,
        network_drained ? 1u : 0u,
        zero_in_flight ? 1u : 0u};
  });
}

extern "C" const char* heterosim_booksim2_last_error(void) {
  return g_last_error.c_str();
}

extern "C" const char* heterosim_booksim2_abi(void) {
  return "gpu-atlas-booksim2-adapter/v1";
}
