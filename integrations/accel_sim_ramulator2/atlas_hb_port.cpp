#include "atlas_hb_port.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace heterosim {

AtlasHybridBondPort::AtlasHybridBondPort(heterosim_ramulator_handle handle,
                                         uint32_t partition_id,
                                         uint32_t transaction_bytes,
                                         uint64_t global_address_base)
    : handle_(handle),
      partition_id_(partition_id),
      transaction_bytes_(transaction_bytes),
      global_address_base_(global_address_base) {
  if (!handle_ || transaction_bytes_ == 0) {
    throw std::invalid_argument("invalid ATLAS Hybrid-Bond port contract");
  }
}

std::vector<AtlasHbAccess> AtlasHybridBondPort::generate(
    const atlasim::ComponentInput &input) const {
  if (!input) throw std::invalid_argument("null ATLAS ComponentInput");
  std::vector<uint64_t> reads;
  std::vector<uint64_t> writes;

  // This traversal intentionally mirrors ATLAS HBFrontend::send. The adapter
  // consumes the native Core-produced fields instead of inventing a second
  // trace format for Logic-Die traffic.
  for (const auto &tile : *input) {
    auto &addresses = tile.at("is_write") ? writes : reads;
    const int64_t rank = tile.at("layout_rank");
    if (rank <= 0) throw std::invalid_argument("ATLAS layout_rank must be positive");
    std::vector<int64_t> strides(rank, 1);
    std::vector<int64_t> access_base(rank, 0);
    std::vector<int64_t> access_extent(rank, 1);
    for (int64_t dim = 0; dim < rank; ++dim) {
      (void)tile.at("shape_" + std::to_string(dim));
      strides.at(dim) = tile.at("stride_" + std::to_string(dim));
      access_base.at(dim) = tile.at("access_base_" + std::to_string(dim));
      access_extent.at(dim) = tile.at("access_extent_" + std::to_string(dim));
      if (access_extent.at(dim) <= 0) {
        throw std::invalid_argument("ATLAS access_extent must be positive");
      }
    }
    const int64_t base_address = tile.at("base_addr");
    const int64_t element_size = tile.at("element_size");
    if (base_address < 0 || element_size <= 0) {
      throw std::invalid_argument("invalid ATLAS address or element size");
    }

    std::unordered_set<uint64_t> tile_addresses;
    auto emit_contiguous_line = [&](int64_t element_offset,
                                    int64_t element_count) {
      const uint64_t local_address =
          static_cast<uint64_t>(base_address + element_offset * element_size);
      if (local_address >
          std::numeric_limits<uint64_t>::max() - global_address_base_) {
        throw std::overflow_error("ATLAS Global PA projection overflow");
      }
      const uint64_t byte_address = global_address_base_ + local_address;
      const uint64_t byte_count =
          static_cast<uint64_t>(element_count * element_size);
      uint64_t aligned = byte_address;
      uint64_t count = (byte_count + transaction_bytes_ - 1) /
                       transaction_bytes_;
      if (byte_address % transaction_bytes_ != 0) {
        aligned -= byte_address % transaction_bytes_;
        count = 1 + (byte_count + transaction_bytes_ - 1) /
                        transaction_bytes_;
      }
      for (uint64_t index = 0; index < count; ++index) {
        tile_addresses.insert(aligned + index * transaction_bytes_);
      }
    };

    int64_t contiguous_dim = -1;
    for (int64_t dim = rank - 1; dim >= 0; --dim) {
      if (strides.at(dim) == 1) {
        contiguous_dim = dim;
        break;
      }
    }
    if (contiguous_dim >= 0) {
      std::function<void(int64_t, int64_t)> emit =
          [&](int64_t dim, int64_t element_offset) {
            if (dim == rank) {
              emit_contiguous_line(element_offset,
                                   access_extent.at(contiguous_dim));
              return;
            }
            if (dim == contiguous_dim) {
              emit(dim + 1, element_offset +
                                    access_base.at(dim) * strides.at(dim));
              return;
            }
            for (int64_t index = 0; index < access_extent.at(dim); ++index) {
              emit(dim + 1,
                   element_offset +
                       (access_base.at(dim) + index) * strides.at(dim));
            }
          };
      emit(0, 0);
    } else {
      std::function<void(int64_t, int64_t)> emit =
          [&](int64_t dim, int64_t element_offset) {
            if (dim == rank) {
              const uint64_t local_address = static_cast<uint64_t>(
                  base_address + element_offset * element_size);
              if (local_address >
                  std::numeric_limits<uint64_t>::max() -
                      global_address_base_) {
                throw std::overflow_error(
                    "ATLAS Global PA projection overflow");
              }
              const uint64_t byte_address =
                  global_address_base_ + local_address;
              const uint64_t aligned =
                  byte_address - byte_address % transaction_bytes_;
              const uint64_t covered =
                  (byte_address - aligned) + static_cast<uint64_t>(element_size);
              const uint64_t count =
                  (covered + transaction_bytes_ - 1) / transaction_bytes_;
              for (uint64_t index = 0; index < count; ++index) {
                tile_addresses.insert(aligned + index * transaction_bytes_);
              }
              return;
            }
            for (int64_t index = 0; index < access_extent.at(dim); ++index) {
              emit(dim + 1,
                   element_offset +
                       (access_base.at(dim) + index) * strides.at(dim));
            }
          };
      emit(0, 0);
    }
    addresses.insert(addresses.end(), tile_addresses.begin(),
                     tile_addresses.end());
  }

  auto mapped_less = [this](uint64_t lhs, uint64_t rhs) {
    const int order =
        heterosim_ramulator_compare_atlas_addresses(handle_, lhs, rhs);
    if (order == 2) {
      throw std::runtime_error("shared Ramulator2 rejected ATLAS address ordering");
    }
    return order < 0;
  };
  auto sort_and_deduplicate = [&](std::vector<uint64_t> &addresses) {
    std::sort(addresses.begin(), addresses.end(), mapped_less);
    addresses.erase(std::unique(addresses.begin(), addresses.end()),
                    addresses.end());
  };
  sort_and_deduplicate(reads);
  sort_and_deduplicate(writes);

  std::vector<AtlasHbAccess> result;
  result.reserve(reads.size() + writes.size());
  for (uint64_t address : reads) {
    result.push_back(
        AtlasHbAccess{address, transaction_bytes_, HETEROSIM_MEMORY_READ});
  }
  for (uint64_t address : writes) {
    result.push_back(
        AtlasHbAccess{address, transaction_bytes_, HETEROSIM_MEMORY_WRITE});
  }
  return result;
}

int AtlasHybridBondPort::submit(const AtlasHbAccess &access,
                                uint64_t parent_id,
                                uint64_t ordering_domain,
                                uint64_t sequence_number,
                                void *payload) const {
  heterosim_parent_request_v2 request{};
  request.abi_version = HETEROSIM_RAMULATOR_ABI_VERSION;
  request.struct_size = sizeof(request);
  request.parent_id = parent_id;
  request.global_address = access.global_address;
  request.size_bytes = access.size_bytes;
  request.partition_id = partition_id_;
  request.operation = access.operation;
  request.ordering_domain = ordering_domain;
  request.sequence_number = sequence_number;
  request.payload = payload;
  return heterosim_ramulator_send_internal_v2(handle_, &request);
}

}  // namespace heterosim
