#ifndef HETEROSIM_ATLAS_HYBRID_BOND_PORT_H
#define HETEROSIM_ATLAS_HYBRID_BOND_PORT_H

#include <cstdint>
#include <vector>

#include "core/component.h"
#include "ramulator_bridge.h"

namespace heterosim {

struct AtlasHbAccess {
  uint64_t global_address = 0;
  uint32_t size_bytes = 0;
  uint32_t operation = HETEROSIM_MEMORY_READ;
};

/*
 * Converts the exact ComponentInput records emitted by ATLAS Core into the
 * aligned request sequence used by ATLAS HBFrontend. Address ordering is
 * delegated to the shared Ramulator2 mapper, so this adapter neither owns nor
 * creates a second memory-timing model.
 */
class AtlasHybridBondPort {
 public:
  AtlasHybridBondPort(heterosim_ramulator_handle handle,
                      uint32_t partition_id, uint32_t transaction_bytes);

  std::vector<AtlasHbAccess> generate(
      const atlasim::ComponentInput &input) const;

  int submit(const AtlasHbAccess &access, uint64_t parent_id,
             uint64_t ordering_domain, uint64_t sequence_number,
             void *payload) const;

 private:
  heterosim_ramulator_handle handle_;
  uint32_t partition_id_;
  uint32_t transaction_bytes_;
};

}  // namespace heterosim

#endif
