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
  constexpr std::size_t kRequests = 64;
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
  for (uint64_t cycle = 0; cycle < 1000000 && returned < kRequests; ++cycle) {
    if (sent < kRequests) {
      const unsigned partition = static_cast<unsigned>(sent % kPartitions);
      if (heterosim_ramulator_send(handles[partition], sent * 4096, 0,
                                    &payloads[sent])) {
        ++sent;
      }
    }
    heterosim_ramulator_tick(handles.back());
    for (auto handle : handles) {
      while (heterosim_ramulator_pop_completed(handle)) ++returned;
    }
  }

  const uint64_t cycles = heterosim_ramulator_clock(handles.front());
  const uint64_t reads = heterosim_ramulator_reads(handles.front());
  const uint64_t completed =
      heterosim_ramulator_completed(handles.front());
  const uint64_t outstanding =
      heterosim_ramulator_outstanding(handles.front());
  std::cout << "heterosim_ramulator2_smoke"
            << " sent=" << sent << " returned=" << returned
            << " cycles=" << cycles << " reads=" << reads
            << " completed=" << completed
            << " outstanding=" << outstanding << " instances=1\n";
  for (auto it = handles.rbegin(); it != handles.rend(); ++it) {
    heterosim_ramulator_destroy(*it);
  }
  return sent == kRequests && returned == kRequests && reads == kRequests &&
                 completed == kRequests && outstanding == 0
             ? 0
             : 4;
}
