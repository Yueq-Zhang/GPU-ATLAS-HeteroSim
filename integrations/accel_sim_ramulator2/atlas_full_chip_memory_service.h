#ifndef HETEROSIM_ATLAS_FULL_CHIP_MEMORY_SERVICE_H
#define HETEROSIM_ATLAS_FULL_CHIP_MEMORY_SERVICE_H

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "atlas_hb_port.h"
#include "dram/external_dram_service.h"

namespace heterosim {

class AtlasFullChipMemoryService final
    : public atlasim::IExternalDramService {
 public:
  AtlasFullChipMemoryService(heterosim_ramulator_handle handle,
                             uint32_t completion_partition,
                             uint32_t transaction_bytes,
                             uint64_t core_region_bytes,
                             double max_bandwidth_gbps_per_core,
                             uint32_t issue_width_per_host_cycle = 16);

  atlasim::ExternalDramTraffic estimate_traffic(
      int core_id, const atlasim::ComponentInput &input) const override;
  double max_bandwidth_gbps_per_core() const override;
  bool submit_iteration(const atlasim::ExternalDramRequestKey &key,
                        const atlasim::ComponentInput &input) override;
  bool iteration_complete(
      const atlasim::ExternalDramRequestKey &key) const override;
  void retire_iteration(
      const atlasim::ExternalDramRequestKey &key) override;

  void poll_completions();
  void issue_requests();
  void advance();
  bool idle() const;
  uint64_t submitted_parents() const { return submitted_parents_; }
  uint64_t completed_parents() const { return completed_parents_; }
  uint64_t logical_bytes() const { return logical_bytes_; }

 private:
  using KeyTuple = std::tuple<int, int, int>;

  struct AccessContext {
    KeyTuple key;
    std::size_t access_index = 0;
    uint64_t parent_id = 0;
  };

  struct AccessState {
    AtlasHbAccess access;
    uint64_t parent_id = 0;
    std::unique_ptr<AccessContext> context;
    bool issued = false;
    bool completed = false;
  };

  struct IterationState {
    std::vector<AccessState> accesses;
    std::size_t next_issue = 0;
    std::size_t completed = 0;
  };

  static KeyTuple tuple_key(const atlasim::ExternalDramRequestKey &key);
  uint64_t global_base(int core_id) const;
  AtlasHybridBondPort make_projection_port(int core_id) const;
  void validate_core_region(int core_id,
                            const std::vector<AtlasHbAccess> &accesses) const;
  heterosim_ramulator_handle handle_;
  uint32_t completion_partition_;
  uint32_t transaction_bytes_;
  uint64_t core_region_bytes_;
  double max_bandwidth_gbps_per_core_;
  uint32_t issue_width_per_host_cycle_;
  AtlasHybridBondPort submission_port_;
  std::map<KeyTuple, IterationState> iterations_;
  uint64_t next_parent_id_ = uint64_t{1} << 63;
  uint64_t submitted_parents_ = 0;
  uint64_t completed_parents_ = 0;
  uint64_t logical_bytes_ = 0;
};

}  // namespace heterosim

#endif
