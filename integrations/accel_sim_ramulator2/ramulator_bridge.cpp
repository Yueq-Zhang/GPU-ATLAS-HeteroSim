#include "ramulator_bridge.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "base/config.h"
#include "base/request.h"
#include "addr_mapper/impl/linear_mappers.h"
#include "frontend/frontend.h"
#include "memory_system/memory_system.h"

#if defined(__GNUC__)
extern "C" void heterosim_atlas_runtime_autostart(
    heterosim_ramulator_handle handle) __attribute__((weak));
extern "C" void heterosim_atlas_runtime_advance(
    uint64_t gpu_cycles, uint64_t global_time_fs) __attribute__((weak));
extern "C" void heterosim_atlas_runtime_shutdown() __attribute__((weak));
extern "C" int heterosim_atlas_runtime_active() __attribute__((weak));
#endif

namespace {

uint64_t positive_env(const char *name, uint64_t fallback) {
  const char *text = std::getenv(name);
  if (!text || !text[0]) return fallback;
  char *end = nullptr;
  const unsigned long long value = std::strtoull(text, &end, 10);
  if (!end || *end || value == 0) {
    throw std::invalid_argument(std::string("invalid positive environment value: ") +
                                name);
  }
  return static_cast<uint64_t>(value);
}

uint64_t unsigned_env(const char *name, uint64_t fallback) {
  const char *text = std::getenv(name);
  if (!text || !text[0]) return fallback;
  char *end = nullptr;
  const unsigned long long value = std::strtoull(text, &end, 10);
  if (!end || *end) {
    throw std::invalid_argument(std::string("invalid unsigned environment value: ") +
                                name);
  }
  return static_cast<uint64_t>(value);
}

struct ChildDescriptor {
  uint64_t address = 0;
  uint32_t index = 0;
};

struct ParentState {
  heterosim_parent_request_v2 request{};
  std::vector<ChildDescriptor> children;
  std::size_t next_child = 0;
  std::size_t completed_children = 0;
  bool gpu_visible = false;
  uint32_t initiator = HETEROSIM_INITIATOR_GPU;
};

struct LinkTransaction {
  heterosim_parent_request_v2 request{};
  uint64_t remaining_wire_bytes = 0;
  uint64_t ready_cycle = 0;
  uint32_t total_children = 0;
  uint32_t initiator = HETEROSIM_INITIATOR_GPU;
  bool durable = false;
};

struct SharedBridge {
  std::shared_ptr<Ramulator::IFrontEnd> frontend;
  std::shared_ptr<Ramulator::IMemorySystem> memory;
  std::vector<std::deque<heterosim_parent_completion_v2>> completed_payloads;
  std::unordered_map<uint64_t, ParentState> parents;
  std::deque<uint64_t> ingress;
  std::unordered_set<uint64_t> inflight_parent_ids;
  std::deque<LinkTransaction> request_link_queue;
  std::deque<LinkTransaction> request_link_arrivals;
  std::deque<LinkTransaction> response_link_queue;
  std::deque<LinkTransaction> response_link_arrivals;
  LinkTransaction active_request_link;
  LinkTransaction active_response_link;
  bool request_link_busy = false;
  bool response_link_busy = false;
  std::string config_path;
  uint64_t reads = 0;
  uint64_t writes = 0;
  uint64_t completed = 0;
  uint64_t rejected = 0;
  uint64_t durable_completed = 0;
  uint64_t children_sent = 0;
  uint64_t children_completed = 0;
  uint64_t child_retries = 0;
  uint64_t logical_bytes = 0;
  uint64_t internal_bytes = 0;
  uint64_t transaction_bytes = 0;
  uint64_t ingress_queue_depth = 0;
  uint64_t parent_table_entries = 0;
  uint64_t issue_width_per_cycle = 0;
  bool posted_write_ack = false;
  uint64_t link_cycle = 0;
  uint64_t request_link_bytes_per_cycle = 0;
  uint64_t response_link_bytes_per_cycle = 0;
  uint64_t request_header_bytes = 0;
  uint64_t response_header_bytes = 0;
  uint64_t flit_bytes = 0;
  uint64_t propagation_cycles = 0;
  uint64_t external_queue_depth = 0;
  uint64_t external_credits = 0;
  uint64_t credits_in_use = 0;
  bool full_duplex = true;
  bool half_duplex_request_turn = true;
  uint64_t request_payload_bytes = 0;
  uint64_t response_payload_bytes = 0;
  uint64_t request_wire_bytes = 0;
  uint64_t response_wire_bytes = 0;
  uint64_t gpu_clock_hz = 0;
  uint64_t link_clock_hz = 0;
  uint64_t gateway_clock_hz = 0;
  uint64_t dram_clock_hz = 0;
  uint64_t link_phase = 0;
  uint64_t gateway_phase = 0;
  uint64_t dram_phase = 0;
  uint64_t gpu_cycles = 0;
  uint64_t gateway_cycles = 0;
  uint64_t global_time_fs = 0;
  uint64_t global_time_remainder = 0;
  std::array<uint64_t, 2> parents_by_initiator{};
  std::array<uint64_t, 2> completed_by_initiator{};
  std::array<uint64_t, 2> children_by_initiator{};
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
    transaction_bytes = positive_env("HETEROSIM_DRAM_TRANSACTION_BYTES", 64);
    ingress_queue_depth =
        positive_env("HETEROSIM_GATEWAY_INGRESS_QUEUE_DEPTH", 256);
    parent_table_entries =
        positive_env("HETEROSIM_GATEWAY_PARENT_TABLE_ENTRIES", 256);
    issue_width_per_cycle =
        positive_env("HETEROSIM_GATEWAY_ISSUE_WIDTH", 4);
    const char *ack_policy = std::getenv("HETEROSIM_GATEWAY_WRITE_ACK_POLICY");
    const std::string ack = ack_policy ? ack_policy : "durable";
    if (ack != "durable" && ack != "posted") {
      throw std::invalid_argument("invalid gateway write acknowledgement policy");
    }
    posted_write_ack = ack == "posted";
    gpu_clock_hz = positive_env("HETEROSIM_GPU_CLOCK_HZ", 400000000);
    link_clock_hz = positive_env("HETEROSIM_LINK_CLOCK_HZ", 400000000);
    gateway_clock_hz =
        positive_env("HETEROSIM_GATEWAY_CLOCK_HZ", 400000000);
    dram_clock_hz = positive_env("HETEROSIM_DRAM_CLOCK_HZ", 400000000);
    const uint64_t request_bandwidth =
        positive_env("HETEROSIM_LINK_REQUEST_BANDWIDTH_BPS", 12800000000ULL);
    const uint64_t response_bandwidth =
        positive_env("HETEROSIM_LINK_RESPONSE_BANDWIDTH_BPS", 12800000000ULL);
    if (request_bandwidth % link_clock_hz ||
        response_bandwidth % link_clock_hz) {
      throw std::invalid_argument(
          "link bandwidth must be an integer number of bytes per link cycle");
    }
    request_link_bytes_per_cycle = request_bandwidth / link_clock_hz;
    response_link_bytes_per_cycle = response_bandwidth / link_clock_hz;
    request_header_bytes = unsigned_env("HETEROSIM_LINK_REQUEST_HEADER_BYTES", 16);
    response_header_bytes = unsigned_env("HETEROSIM_LINK_RESPONSE_HEADER_BYTES", 16);
    flit_bytes = positive_env("HETEROSIM_LINK_FLIT_BYTES", 32);
    const uint64_t propagation_fs =
        unsigned_env("HETEROSIM_LINK_PROPAGATION_LATENCY_FS", 10000000);
    propagation_cycles =
        (propagation_fs * link_clock_hz + 1000000000000000ULL - 1) /
        1000000000000000ULL;
    external_queue_depth =
        positive_env("HETEROSIM_LINK_QUEUE_DEPTH", 64);
    external_credits = positive_env("HETEROSIM_LINK_CREDITS", 64);
    if (external_credits > external_queue_depth) {
      throw std::invalid_argument("link credits exceed queue depth");
    }
    const char *duplex = std::getenv("HETEROSIM_LINK_DUPLEX_MODE");
    const std::string duplex_mode = duplex ? duplex : "full_duplex";
    if (duplex_mode != "full_duplex" && duplex_mode != "half_duplex") {
      throw std::invalid_argument("invalid external link duplex mode");
    }
    full_duplex = duplex_mode == "full_duplex";
  }

  uint64_t outstanding() const {
    const uint64_t accepted = reads + writes;
    return accepted >= completed ? accepted - completed : 0;
  }

  uint64_t round_to_flit(uint64_t bytes) const {
    if (bytes > std::numeric_limits<uint64_t>::max() - flit_bytes + 1) {
      throw std::overflow_error("external link transaction size overflow");
    }
    return ((bytes + flit_bytes - 1) / flit_bytes) * flit_bytes;
  }

  int compare_atlas_addresses(uint64_t lhs, uint64_t rhs) const {
    auto *mapper = dynamic_cast<Ramulator::LinearMapperBase *>(
        memory->get_addr_mapper());
    if (!mapper || mapper->m_impl->get_name() != "OneLevelInterleave") {
      throw std::runtime_error(
          "ATLAS Hybrid-Bond ordering requires OneLevelInterleave");
    }
    Ramulator::Request lhs_request(static_cast<Ramulator::Addr_t>(lhs),
                                   Ramulator::Request::Type::Read);
    Ramulator::Request rhs_request(static_cast<Ramulator::Addr_t>(rhs),
                                   Ramulator::Request::Type::Read);
    mapper->apply(lhs_request);
    mapper->apply(rhs_request);
    for (int index = mapper->m_row_bits_idx; index >= 1; --index) {
      if (lhs_request.addr_vec[index] != rhs_request.addr_vec[index]) {
        return lhs_request.addr_vec[index] < rhs_request.addr_vec[index] ? -1
                                                                         : 1;
      }
    }
    const int column = mapper->m_col_bits_idx;
    if (lhs_request.addr_vec[column] != rhs_request.addr_vec[column]) {
      return lhs_request.addr_vec[column] < rhs_request.addr_vec[column] ? -1
                                                                         : 1;
    }
    if (lhs_request.addr_vec[0] != rhs_request.addr_vec[0]) {
      return lhs_request.addr_vec[0] < rhs_request.addr_vec[0] ? -1 : 1;
    }
    return 0;
  }

  bool links_empty() const {
    return request_link_queue.empty() && request_link_arrivals.empty() &&
           response_link_queue.empty() && response_link_arrivals.empty() &&
           !request_link_busy && !response_link_busy;
  }

  bool has_completions() const {
    return std::any_of(completed_payloads.begin(), completed_payloads.end(),
                       [](const auto &queue) { return !queue.empty(); });
  }

  uint64_t advance_until_event(uint64_t max_gpu_cycles) {
    if (has_completions() || max_gpu_cycles == 0) return 0;
    uint64_t advanced = 0;
    while (advanced < max_gpu_cycles && !has_completions()) {
      advance_gpu_cycle();
      ++advanced;
    }
    return advanced;
  }

  void enqueue_response(const heterosim_parent_request_v2 &request,
                        uint32_t total_children, bool durable) {
    const uint64_t payload = request.operation == HETEROSIM_MEMORY_READ
                                 ? request.size_bytes
                                 : 0;
    LinkTransaction transaction;
    transaction.request = request;
    transaction.initiator = HETEROSIM_INITIATOR_GPU;
    transaction.total_children = total_children;
    transaction.durable = durable;
    transaction.remaining_wire_bytes =
        round_to_flit(response_header_bytes + payload);
    response_payload_bytes += payload;
    response_wire_bytes += transaction.remaining_wire_bytes;
    response_link_queue.push_back(transaction);
  }

  void accept_request_at_gateway(const heterosim_parent_request_v2 &request,
                                 uint32_t initiator) {
    ParentState state;
    state.request = request;
    state.initiator = initiator;
    state.children = split(request);
    if (state.children.empty()) {
      throw std::runtime_error("accepted parent produced no active child requests");
    }
    parents.emplace(request.parent_id, std::move(state));
    ingress.push_back(request.parent_id);
  }

  void deliver_link_arrivals() {
    while (!request_link_arrivals.empty() &&
           request_link_arrivals.front().ready_cycle <= link_cycle &&
           parents.size() < parent_table_entries &&
           ingress.size() < ingress_queue_depth) {
      accept_request_at_gateway(request_link_arrivals.front().request,
                                HETEROSIM_INITIATOR_GPU);
      request_link_arrivals.pop_front();
    }
    while (!response_link_arrivals.empty() &&
           response_link_arrivals.front().ready_cycle <= link_cycle) {
      const auto transaction = response_link_arrivals.front();
      const auto request = transaction.request;
      heterosim_parent_completion_v2 completion{};
      completion.abi_version = HETEROSIM_RAMULATOR_ABI_VERSION;
      completion.struct_size = sizeof(completion);
      completion.parent_id = request.parent_id;
      completion.partition_id = request.partition_id;
      completion.operation = request.operation;
      completion.total_children = transaction.total_children;
      completion.completed_children = transaction.total_children;
      completion.durable = transaction.durable ? 1U : 0U;
      completion.initiator = transaction.initiator;
      completion.payload = request.payload;
      completed_payloads.at(request.partition_id).push_back(completion);
      ++completed;
      ++completed_by_initiator.at(transaction.initiator);
      if (credits_in_use == 0 || !inflight_parent_ids.erase(request.parent_id)) {
        throw std::runtime_error("external response completed without link credit");
      }
      --credits_in_use;
      response_link_arrivals.pop_front();
    }
  }

  void serialize_direction(std::deque<LinkTransaction> &queue,
                           LinkTransaction &active, bool &busy,
                           std::deque<LinkTransaction> &arrivals,
                           uint64_t bytes_per_cycle) {
    if (!busy && !queue.empty()) {
      active = queue.front();
      queue.pop_front();
      busy = true;
    }
    if (!busy) return;
    const uint64_t transferred =
        std::min(active.remaining_wire_bytes, bytes_per_cycle);
    active.remaining_wire_bytes -= transferred;
    if (active.remaining_wire_bytes == 0) {
      active.ready_cycle = link_cycle + propagation_cycles;
      arrivals.push_back(active);
      busy = false;
    }
  }

  void tick_link() {
    deliver_link_arrivals();
    if (full_duplex) {
      serialize_direction(request_link_queue, active_request_link,
                          request_link_busy, request_link_arrivals,
                          request_link_bytes_per_cycle);
      serialize_direction(response_link_queue, active_response_link,
                          response_link_busy, response_link_arrivals,
                          response_link_bytes_per_cycle);
    } else if (request_link_busy ||
               (!response_link_busy && !request_link_queue.empty() &&
                (response_link_queue.empty() || half_duplex_request_turn))) {
      serialize_direction(request_link_queue, active_request_link,
                          request_link_busy, request_link_arrivals,
                          request_link_bytes_per_cycle);
      if (!request_link_busy) half_duplex_request_turn = false;
    } else {
      serialize_direction(response_link_queue, active_response_link,
                          response_link_busy, response_link_arrivals,
                          response_link_bytes_per_cycle);
      if (!response_link_busy) half_duplex_request_turn = true;
    }
    ++link_cycle;
  }

  void advance_gpu_cycle() {
    ++gpu_cycles;
    const uint64_t time_numerator =
        global_time_remainder + 1000000000000000ULL;
    global_time_fs += time_numerator / gpu_clock_hz;
    global_time_remainder = time_numerator % gpu_clock_hz;

    link_phase += link_clock_hz;
    while (link_phase >= gpu_clock_hz) {
      tick_link();
      link_phase -= gpu_clock_hz;
    }
    gateway_phase += gateway_clock_hz;
    while (gateway_phase >= gpu_clock_hz) {
      issue_children();
      ++gateway_cycles;
      gateway_phase -= gpu_clock_hz;
    }
    dram_phase += dram_clock_hz;
    while (dram_phase >= gpu_clock_hz) {
      memory->tick();
      dram_phase -= gpu_clock_hz;
    }
#if defined(__GNUC__)
    if (heterosim_atlas_runtime_advance) {
      heterosim_atlas_runtime_advance(gpu_cycles, global_time_fs);
    }
#endif
  }

  bool byte_enabled(const heterosim_parent_request_v2 &request,
                    uint64_t relative_byte) const {
    bool enabled = true;
    if (request.flags & HETEROSIM_REQUEST_BYTE_MASK_VALID) {
      if (relative_byte >=
          uint64_t(request.byte_mask_word_count) * uint64_t(64)) {
        return false;
      }
      enabled =
          ((request.byte_mask[relative_byte / 64] >> (relative_byte % 64)) & 1U) !=
          0;
    }
    if (request.flags & HETEROSIM_REQUEST_SECTOR_MASK_VALID) {
      const uint64_t sector = relative_byte / 32;
      if (sector >= 32 || ((request.sector_mask >> sector) & 1U) == 0) {
        enabled = false;
      }
    }
    return enabled;
  }

  std::vector<ChildDescriptor> split(
      const heterosim_parent_request_v2 &request) const {
    if (request.global_address >
        std::numeric_limits<uint64_t>::max() - request.size_bytes) {
      return {};
    }
    const uint64_t end = request.global_address + request.size_bytes;
    const uint64_t first =
        request.global_address - request.global_address % transaction_bytes;
    std::vector<ChildDescriptor> result;
    for (uint64_t child_address = first; child_address < end;
         child_address += transaction_bytes) {
      const uint64_t overlap_begin = std::max(child_address, request.global_address);
      const uint64_t child_end =
          child_address > std::numeric_limits<uint64_t>::max() - transaction_bytes
              ? end
              : child_address + transaction_bytes;
      const uint64_t overlap_end = std::min(child_end, end);
      bool active = false;
      for (uint64_t address = overlap_begin; address < overlap_end; ++address) {
        if (byte_enabled(request, address - request.global_address)) {
          active = true;
          break;
        }
      }
      if (active) {
        result.push_back(
            ChildDescriptor{child_address, static_cast<uint32_t>(result.size())});
      }
      if (child_address >
          std::numeric_limits<uint64_t>::max() - transaction_bytes) {
        break;
      }
    }
    return result;
  }

  void complete_child(uint64_t parent_id) {
    auto found = parents.find(parent_id);
    if (found == parents.end()) {
      throw std::runtime_error("Ramulator2 completed an unknown child request");
    }
    ParentState &parent = found->second;
    ++parent.completed_children;
    ++children_completed;
    if (parent.completed_children != parent.children.size()) return;
    ++durable_completed;
    if (parent.initiator == HETEROSIM_INITIATOR_ATLAS_LOGIC_DIE) {
      heterosim_parent_completion_v2 completion{};
      completion.abi_version = HETEROSIM_RAMULATOR_ABI_VERSION;
      completion.struct_size = sizeof(completion);
      completion.parent_id = parent.request.parent_id;
      completion.partition_id = parent.request.partition_id;
      completion.operation = parent.request.operation;
      completion.total_children =
          static_cast<uint32_t>(parent.children.size());
      completion.completed_children = completion.total_children;
      completion.durable = 1U;
      completion.initiator = HETEROSIM_INITIATOR_ATLAS_LOGIC_DIE;
      completion.payload = parent.request.payload;
      completed_payloads.at(parent.request.partition_id).push_back(completion);
      ++completed;
      ++completed_by_initiator.at(HETEROSIM_INITIATOR_ATLAS_LOGIC_DIE);
      if (!inflight_parent_ids.erase(parent.request.parent_id)) {
        throw std::runtime_error(
            "ATLAS completion did not own an inflight parent ID");
      }
    } else if (!parent.gpu_visible) {
      enqueue_response(parent.request,
                       static_cast<uint32_t>(parent.children.size()), true);
      parent.gpu_visible = true;
    }
    parents.erase(found);
  }

  void issue_children() {
    uint64_t issued = 0;
    while (issued < issue_width_per_cycle && !ingress.empty()) {
      const uint64_t parent_id = ingress.front();
      auto found = parents.find(parent_id);
      if (found == parents.end()) {
        throw std::runtime_error("gateway ingress references an unknown parent");
      }
      ParentState &parent = found->second;
      if (parent.next_child == parent.children.size()) {
        ingress.pop_front();
        continue;
      }
      const ChildDescriptor child = parent.children[parent.next_child];
      const int type = parent.request.operation == HETEROSIM_MEMORY_WRITE
                           ? Ramulator::Request::Type::Write
                           : Ramulator::Request::Type::Read;
      Ramulator::Request dram_request(
          static_cast<Ramulator::Addr_t>(child.address), type, 0,
          [this, parent_id](Ramulator::Request &) { complete_child(parent_id); });
      dram_request.m_payload = parent.request.payload;
      if (!memory->send(dram_request)) {
        ++child_retries;
        break;
      }
      ++parent.next_child;
      ++children_sent;
      ++children_by_initiator.at(parent.initiator);
      internal_bytes += transaction_bytes;
      ++issued;
      if (parent.next_child == parent.children.size()) {
        ingress.pop_front();
        if (posted_write_ack &&
            parent.request.operation == HETEROSIM_MEMORY_WRITE &&
            !parent.gpu_visible) {
          enqueue_response(parent.request,
                           static_cast<uint32_t>(parent.children.size()), false);
          parent.gpu_visible = true;
        }
      }
    }
  }

  void finish() {
    if (finalized || !memory) return;
    constexpr uint64_t kDrainLimit = 1000000000ULL;
    uint64_t drain_cycles = 0;
    while ((!inflight_parent_ids.empty() || !parents.empty() ||
            !links_empty() || !memory->is_finished()) &&
           drain_cycles < kDrainLimit) {
      advance_gpu_cycle();
      ++drain_cycles;
    }
    if (!inflight_parent_ids.empty() || !parents.empty() || !links_empty() ||
        !memory->is_finished()) {
      throw std::runtime_error("Ramulator2 did not drain before finalization");
    }
    std::cout << "heterosim_ramulator2_summary"
              << " cycles=" << memory->get_clk() << " reads=" << reads
              << " writes=" << writes << " completed=" << completed
              << " rejected=" << rejected << " outstanding=" << outstanding()
              << " instances=1 partitions=" << partition_count
              << " durable_completed=" << durable_completed
              << " children_sent=" << children_sent
              << " children_completed=" << children_completed
              << " child_retries=" << child_retries
              << " logical_bytes=" << logical_bytes
              << " internal_bytes=" << internal_bytes
              << " transaction_bytes=" << transaction_bytes
              << " request_payload_bytes=" << request_payload_bytes
              << " response_payload_bytes=" << response_payload_bytes
              << " request_wire_bytes=" << request_wire_bytes
              << " response_wire_bytes=" << response_wire_bytes
              << " gpu_cycles=" << gpu_cycles
              << " link_cycles=" << link_cycle
              << " gateway_cycles=" << gateway_cycles
              << " global_time_fs=" << global_time_fs
              << " gpu_parents="
              << parents_by_initiator.at(HETEROSIM_INITIATOR_GPU)
              << " atlas_parents="
              << parents_by_initiator.at(HETEROSIM_INITIATOR_ATLAS_LOGIC_DIE)
              << " gpu_completed="
              << completed_by_initiator.at(HETEROSIM_INITIATOR_GPU)
              << " atlas_completed="
              << completed_by_initiator.at(HETEROSIM_INITIATOR_ATLAS_LOGIC_DIE)
              << " gpu_children="
              << children_by_initiator.at(HETEROSIM_INITIATOR_GPU)
              << " atlas_children="
              << children_by_initiator.at(HETEROSIM_INITIATOR_ATLAS_LOGIC_DIE)
              << std::endl;
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
#if defined(__GNUC__)
  if (heterosim_atlas_runtime_shutdown) {
    heterosim_atlas_runtime_shutdown();
  }
#endif
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

bool valid_parent_request(PartitionHandle *handle, SharedBridge *shared,
                          const heterosim_parent_request_v2 *parent) {
  if (!handle || !shared || shared->finalized || !parent ||
      parent->abi_version != HETEROSIM_RAMULATOR_ABI_VERSION ||
      parent->struct_size != sizeof(heterosim_parent_request_v2) ||
      parent->partition_id != handle->partition_id ||
      parent->size_bytes == 0 || !parent->payload ||
      parent->byte_mask_word_count > HETEROSIM_MAX_BYTE_MASK_WORDS ||
      parent->operation > HETEROSIM_MEMORY_WRITE) {
    return false;
  }
  if ((parent->flags & HETEROSIM_REQUEST_BYTE_MASK_VALID) &&
      (parent->byte_mask_word_count == 0 || parent->size_bytes > 128)) {
    return false;
  }
  if ((parent->flags & HETEROSIM_REQUEST_SECTOR_MASK_VALID) &&
      parent->size_bytes > 1024) {
    return false;
  }
  return !shared->inflight_parent_ids.count(parent->parent_id) &&
         !shared->split(*parent).empty();
}

void record_parent_acceptance(SharedBridge *shared,
                              const heterosim_parent_request_v2 &parent,
                              uint32_t initiator) {
  shared->logical_bytes += parent.size_bytes;
  if (parent.operation == HETEROSIM_MEMORY_WRITE) {
    ++shared->writes;
  } else {
    ++shared->reads;
  }
  ++shared->parents_by_initiator.at(initiator);
  shared->inflight_parent_ids.insert(parent.parent_id);
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
#if defined(__GNUC__)
    if (partition_id == 0 && heterosim_atlas_runtime_autostart) {
      heterosim_atlas_runtime_autostart(handle);
    }
#endif
    return handle;
  } catch (const std::exception &error) {
    std::cerr << "heterosim_ramulator_create failed: " << error.what()
              << std::endl;
    return nullptr;
  }
}

extern "C" heterosim_ramulator_handle heterosim_ramulator_retain(
    heterosim_ramulator_handle opaque) {
  PartitionHandle *handle = as_handle(opaque);
  SharedBridge *shared = shared_from(opaque);
  if (!handle || !shared || shared->finalized) return nullptr;
  ++shared->references;
  return new PartitionHandle{shared, handle->partition_id};
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
  heterosim_parent_request_v2 request{};
  request.abi_version = HETEROSIM_RAMULATOR_ABI_VERSION;
  request.struct_size = sizeof(request);
  request.parent_id = reinterpret_cast<uintptr_t>(payload);
  request.global_address = address;
  request.size_bytes = 1;
  request.partition_id = as_handle(opaque) ? as_handle(opaque)->partition_id : 0;
  request.operation =
      is_write ? HETEROSIM_MEMORY_WRITE : HETEROSIM_MEMORY_READ;
  request.payload = payload;
  return heterosim_ramulator_send_v2(opaque, &request);
}

extern "C" int heterosim_ramulator_send_v2(
    heterosim_ramulator_handle opaque,
    const heterosim_parent_request_v2 *parent) {
  PartitionHandle *handle = as_handle(opaque);
  SharedBridge *shared = shared_from(opaque);
  if (!valid_parent_request(handle, shared, parent))
    return HETEROSIM_SEND_INVALID;
  if (shared->credits_in_use >= shared->external_credits ||
      shared->request_link_queue.size() +
              (shared->request_link_busy ? 1U : 0U) >=
          shared->external_queue_depth) {
    ++shared->rejected;
    return HETEROSIM_SEND_RETRY;
  }
  const uint64_t payload = parent->operation == HETEROSIM_MEMORY_WRITE
                               ? parent->size_bytes
                               : 0;
  LinkTransaction transaction;
  transaction.request = *parent;
  transaction.remaining_wire_bytes =
      shared->round_to_flit(shared->request_header_bytes + payload);
  shared->request_payload_bytes += payload;
  shared->request_wire_bytes += transaction.remaining_wire_bytes;
  record_parent_acceptance(shared, *parent, HETEROSIM_INITIATOR_GPU);
  shared->request_link_queue.push_back(transaction);
  ++shared->credits_in_use;
  return HETEROSIM_SEND_ACCEPTED;
}

extern "C" int heterosim_ramulator_send_internal_v2(
    heterosim_ramulator_handle opaque,
    const heterosim_parent_request_v2 *parent) {
  PartitionHandle *handle = as_handle(opaque);
  SharedBridge *shared = shared_from(opaque);
  if (!valid_parent_request(handle, shared, parent))
    return HETEROSIM_SEND_INVALID;
  if (shared->parents.size() >= shared->parent_table_entries ||
      shared->ingress.size() >= shared->ingress_queue_depth) {
    ++shared->rejected;
    return HETEROSIM_SEND_RETRY;
  }
  record_parent_acceptance(shared, *parent,
                           HETEROSIM_INITIATOR_ATLAS_LOGIC_DIE);
  shared->accept_request_at_gateway(*parent,
                                    HETEROSIM_INITIATOR_ATLAS_LOGIC_DIE);
  return HETEROSIM_SEND_ACCEPTED;
}

extern "C" void heterosim_ramulator_tick(heterosim_ramulator_handle handle) {
  SharedBridge *shared = shared_from(handle);
  if (shared && !shared->finalized) shared->advance_gpu_cycle();
}

extern "C" uint64_t heterosim_ramulator_advance_until_event(
    heterosim_ramulator_handle handle, uint64_t max_gpu_cycles) {
  SharedBridge *shared = shared_from(handle);
  if (!shared) return 0;
  try {
    return shared->advance_until_event(max_gpu_cycles);
  } catch (const std::exception &error) {
    std::cerr << "heterosim_ramulator_advance_until_event failed: "
              << error.what() << std::endl;
    return 0;
  }
}

extern "C" void heterosim_ramulator_advance_gpu_cycle() {
  if (g_shared_bridge && !g_shared_bridge->finalized) {
    g_shared_bridge->advance_gpu_cycle();
  }
}

extern "C" int heterosim_ramulator_external_runtime_active() {
#if defined(__GNUC__)
  return heterosim_atlas_runtime_active
             ? heterosim_atlas_runtime_active()
             : 0;
#else
  return 0;
#endif
}

extern "C" void *heterosim_ramulator_pop_completed(
    heterosim_ramulator_handle opaque) {
  heterosim_parent_completion_v2 completion{};
  return heterosim_ramulator_pop_completed_for_initiator_v2(
             opaque, HETEROSIM_INITIATOR_GPU, &completion)
             ? completion.payload
             : nullptr;
}

extern "C" int heterosim_ramulator_pop_completed_v2(
    heterosim_ramulator_handle opaque,
    heterosim_parent_completion_v2 *completion) {
  PartitionHandle *handle = as_handle(opaque);
  SharedBridge *shared = shared_from(opaque);
  if (!handle || !shared || !completion) return 0;
  auto &queue = shared->completed_payloads.at(handle->partition_id);
  if (queue.empty()) return 0;
  *completion = queue.front();
  queue.pop_front();
  return 1;
}

extern "C" int heterosim_ramulator_pop_completed_for_initiator_v2(
    heterosim_ramulator_handle opaque, uint32_t initiator,
    heterosim_parent_completion_v2 *completion) {
  PartitionHandle *handle = as_handle(opaque);
  SharedBridge *shared = shared_from(opaque);
  if (!handle || !shared || !completion ||
      initiator > HETEROSIM_INITIATOR_ATLAS_LOGIC_DIE) {
    return 0;
  }
  auto &queue = shared->completed_payloads.at(handle->partition_id);
  const auto found = std::find_if(
      queue.begin(), queue.end(), [initiator](const auto &candidate) {
        return candidate.initiator == initiator;
      });
  if (found == queue.end()) return 0;
  *completion = *found;
  queue.erase(found);
  return 1;
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

extern "C" uint64_t heterosim_ramulator_durable_completed(
    heterosim_ramulator_handle handle) {
  SharedBridge *shared = shared_from(handle);
  return shared ? shared->durable_completed : 0;
}

extern "C" uint64_t heterosim_ramulator_children_sent(
    heterosim_ramulator_handle handle) {
  SharedBridge *shared = shared_from(handle);
  return shared ? shared->children_sent : 0;
}

extern "C" uint64_t heterosim_ramulator_children_completed(
    heterosim_ramulator_handle handle) {
  SharedBridge *shared = shared_from(handle);
  return shared ? shared->children_completed : 0;
}

extern "C" uint64_t heterosim_ramulator_logical_bytes(
    heterosim_ramulator_handle handle) {
  SharedBridge *shared = shared_from(handle);
  return shared ? shared->logical_bytes : 0;
}

extern "C" uint64_t heterosim_ramulator_internal_bytes(
    heterosim_ramulator_handle handle) {
  SharedBridge *shared = shared_from(handle);
  return shared ? shared->internal_bytes : 0;
}

extern "C" uint64_t heterosim_ramulator_request_payload_bytes(
    heterosim_ramulator_handle handle) {
  SharedBridge *shared = shared_from(handle);
  return shared ? shared->request_payload_bytes : 0;
}

extern "C" uint64_t heterosim_ramulator_response_payload_bytes(
    heterosim_ramulator_handle handle) {
  SharedBridge *shared = shared_from(handle);
  return shared ? shared->response_payload_bytes : 0;
}

extern "C" uint64_t heterosim_ramulator_request_wire_bytes(
    heterosim_ramulator_handle handle) {
  SharedBridge *shared = shared_from(handle);
  return shared ? shared->request_wire_bytes : 0;
}

extern "C" uint64_t heterosim_ramulator_response_wire_bytes(
    heterosim_ramulator_handle handle) {
  SharedBridge *shared = shared_from(handle);
  return shared ? shared->response_wire_bytes : 0;
}

extern "C" uint64_t heterosim_ramulator_link_cycles(
    heterosim_ramulator_handle handle) {
  SharedBridge *shared = shared_from(handle);
  return shared ? shared->link_cycle : 0;
}

extern "C" uint64_t heterosim_ramulator_gpu_cycles(
    heterosim_ramulator_handle handle) {
  SharedBridge *shared = shared_from(handle);
  return shared ? shared->gpu_cycles : 0;
}

extern "C" uint64_t heterosim_ramulator_gateway_cycles(
    heterosim_ramulator_handle handle) {
  SharedBridge *shared = shared_from(handle);
  return shared ? shared->gateway_cycles : 0;
}

extern "C" uint64_t heterosim_ramulator_global_time_fs(
    heterosim_ramulator_handle handle) {
  SharedBridge *shared = shared_from(handle);
  return shared ? shared->global_time_fs : 0;
}

extern "C" uint64_t heterosim_ramulator_initiator_parents(
    heterosim_ramulator_handle handle, uint32_t initiator) {
  SharedBridge *shared = shared_from(handle);
  return shared && initiator < shared->parents_by_initiator.size()
             ? shared->parents_by_initiator.at(initiator)
             : 0;
}

extern "C" uint64_t heterosim_ramulator_initiator_completed(
    heterosim_ramulator_handle handle, uint32_t initiator) {
  SharedBridge *shared = shared_from(handle);
  return shared && initiator < shared->completed_by_initiator.size()
             ? shared->completed_by_initiator.at(initiator)
             : 0;
}

extern "C" uint64_t heterosim_ramulator_initiator_children(
    heterosim_ramulator_handle handle, uint32_t initiator) {
  SharedBridge *shared = shared_from(handle);
  return shared && initiator < shared->children_by_initiator.size()
             ? shared->children_by_initiator.at(initiator)
             : 0;
}

extern "C" int heterosim_ramulator_compare_atlas_addresses(
    heterosim_ramulator_handle handle, uint64_t lhs, uint64_t rhs) {
  SharedBridge *shared = shared_from(handle);
  if (!shared) return 2;
  try {
    return shared->compare_atlas_addresses(lhs, rhs);
  } catch (const std::exception &error) {
    std::cerr << "heterosim_ramulator_compare_atlas_addresses failed: "
              << error.what() << std::endl;
    return 2;
  }
}

extern "C" int heterosim_ramulator_is_finished(
    heterosim_ramulator_handle handle) {
  SharedBridge *shared = shared_from(handle);
  return shared && shared->inflight_parent_ids.empty() &&
         shared->parents.empty() && shared->ingress.empty() &&
         shared->links_empty() && shared->memory->is_finished();
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
