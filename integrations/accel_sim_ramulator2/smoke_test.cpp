#include "ramulator_bridge.h"

#include <cstdint>
#include <iostream>
#include <vector>

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: ramulator_bridge_smoke CONFIG.yaml\n";
    return 2;
  }
  constexpr unsigned kPartitions = 4;
  constexpr std::size_t kReads = 64;
  constexpr std::size_t kWrites = 8;
  constexpr std::size_t kRequests = kReads + kWrites;
  std::vector<heterosim_ramulator_handle> handles;
  for (unsigned partition = 0; partition < kPartitions; ++partition) {
    auto handle =
        heterosim_ramulator_create(argv[1], partition, kPartitions);
    if (!handle) return 3;
    handles.push_back(handle);
  }

  std::vector<uint64_t> payloads(kRequests);
  std::size_t sent = 0;
  std::size_t returned = 0;
  heterosim_parent_request_v2 invalid{};
  invalid.abi_version = 1;
  invalid.struct_size = sizeof(invalid);
  invalid.parent_id = 9999;
  invalid.global_address = 0;
  invalid.size_bytes = 64;
  invalid.partition_id = 0;
  invalid.payload = &payloads[0];
  if (heterosim_ramulator_send_v2(handles[0], &invalid) !=
      HETEROSIM_SEND_INVALID) {
    return 5;
  }
  for (uint64_t cycle = 0; cycle < 1000000 && returned < kRequests; ++cycle) {
    if (sent < kRequests) {
      const unsigned partition = static_cast<unsigned>(sent % kPartitions);
      heterosim_parent_request_v2 request{};
      request.abi_version = HETEROSIM_RAMULATOR_ABI_VERSION;
      request.struct_size = sizeof(request);
      request.parent_id = sent;
      request.global_address = sent * 128;
      request.size_bytes = 128;
      request.partition_id = partition;
      request.operation =
          sent < kReads ? HETEROSIM_MEMORY_READ : HETEROSIM_MEMORY_WRITE;
      request.byte_mask[0] = UINT64_MAX;
      request.byte_mask[1] = UINT64_MAX;
      request.byte_mask_word_count = HETEROSIM_MAX_BYTE_MASK_WORDS;
      request.sector_mask = 0xf;
      request.flags = HETEROSIM_REQUEST_BYTE_MASK_VALID |
                      HETEROSIM_REQUEST_SECTOR_MASK_VALID;
      request.ordering_domain = partition;
      request.sequence_number = sent;
      request.qos_class = 0;
      request.payload = &payloads[sent];
      if (heterosim_ramulator_send_v2(handles[partition], &request)) {
        ++sent;
      }
    }
    heterosim_ramulator_tick(handles.back());
    for (auto handle : handles) {
      heterosim_parent_completion_v2 completion{};
      while (heterosim_ramulator_pop_completed_v2(handle, &completion)) {
        if (completion.abi_version != HETEROSIM_RAMULATOR_ABI_VERSION ||
            completion.struct_size != sizeof(completion) ||
            completion.completed_children != completion.total_children ||
            completion.total_children != 2 || !completion.durable ||
            !completion.payload) {
          return 6;
        }
        ++returned;
      }
    }
  }

  const uint64_t cycles = heterosim_ramulator_clock(handles.front());
  const uint64_t reads = heterosim_ramulator_reads(handles.front());
  const uint64_t writes = heterosim_ramulator_writes(handles.front());
  const uint64_t completed =
      heterosim_ramulator_completed(handles.front());
  const uint64_t durable_completed =
      heterosim_ramulator_durable_completed(handles.front());
  const uint64_t children_sent =
      heterosim_ramulator_children_sent(handles.front());
  const uint64_t children_completed =
      heterosim_ramulator_children_completed(handles.front());
  const uint64_t logical_bytes =
      heterosim_ramulator_logical_bytes(handles.front());
  const uint64_t internal_bytes =
      heterosim_ramulator_internal_bytes(handles.front());
  const uint64_t request_payload_bytes =
      heterosim_ramulator_request_payload_bytes(handles.front());
  const uint64_t response_payload_bytes =
      heterosim_ramulator_response_payload_bytes(handles.front());
  const uint64_t request_wire_bytes =
      heterosim_ramulator_request_wire_bytes(handles.front());
  const uint64_t response_wire_bytes =
      heterosim_ramulator_response_wire_bytes(handles.front());
  const uint64_t link_cycles =
      heterosim_ramulator_link_cycles(handles.front());
  const uint64_t gpu_cycles =
      heterosim_ramulator_gpu_cycles(handles.front());
  const uint64_t gateway_cycles =
      heterosim_ramulator_gateway_cycles(handles.front());
  const uint64_t global_time_fs =
      heterosim_ramulator_global_time_fs(handles.front());
  const uint64_t outstanding =
      heterosim_ramulator_outstanding(handles.front());
  std::cout << "heterosim_ramulator2_smoke"
            << " sent=" << sent << " returned=" << returned
            << " cycles=" << cycles << " reads=" << reads
            << " writes=" << writes << " completed=" << completed
            << " durable_completed=" << durable_completed
            << " children_sent=" << children_sent
            << " children_completed=" << children_completed
            << " logical_bytes=" << logical_bytes
            << " internal_bytes=" << internal_bytes
            << " request_payload_bytes=" << request_payload_bytes
            << " response_payload_bytes=" << response_payload_bytes
            << " request_wire_bytes=" << request_wire_bytes
            << " response_wire_bytes=" << response_wire_bytes
            << " gpu_cycles=" << gpu_cycles
            << " link_cycles=" << link_cycles
            << " gateway_cycles=" << gateway_cycles
            << " global_time_fs=" << global_time_fs
            << " outstanding=" << outstanding << " instances=1\n";
  for (auto it = handles.rbegin(); it != handles.rend(); ++it) {
    heterosim_ramulator_destroy(*it);
  }
  return sent == kRequests && returned == kRequests && reads == kReads &&
                 writes == kWrites && completed == kRequests &&
                 durable_completed == kRequests &&
                 children_sent == kRequests * 2 &&
                 children_completed == kRequests * 2 &&
                 logical_bytes == kRequests * 128 &&
                 internal_bytes == kRequests * 128 &&
                 request_payload_bytes == kWrites * 128 &&
                 response_payload_bytes == kReads * 128 &&
                 request_wire_bytes == kReads * 32 + kWrites * 160 &&
                 response_wire_bytes == kReads * 160 + kWrites * 32 &&
                 link_cycles > 0 && gpu_cycles == link_cycles * 3 &&
                 gateway_cycles == link_cycles && cycles == link_cycles &&
                 global_time_fs == link_cycles * 2500000 && outstanding == 0
             ? 0
             : 4;
}
