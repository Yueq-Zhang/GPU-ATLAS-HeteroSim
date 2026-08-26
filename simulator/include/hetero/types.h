#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace heterosim {

using TimeFs = std::uint64_t;

struct PhysicalAddress {
    std::string memory_space_id;
    std::uint64_t offset_bytes{};
    std::uint64_t allocation_epoch{};
};

inline TimeFs cycle_to_fs(std::uint64_t cycle, std::uint64_t frequency_hz) {
    if (frequency_hz == 0) {
        throw std::invalid_argument("frequency_hz must be positive");
    }
    constexpr unsigned __int128 kFsPerSecond = 1000000000000000ULL;
    const unsigned __int128 numerator =
        static_cast<unsigned __int128>(cycle) * kFsPerSecond;
    const unsigned __int128 result =
        (numerator + static_cast<unsigned __int128>(frequency_hz) - 1) /
        static_cast<unsigned __int128>(frequency_hz);
    if (result > std::numeric_limits<TimeFs>::max()) {
        throw std::overflow_error("cycle_to_fs exceeds TimeFs");
    }
    return static_cast<TimeFs>(result);
}

}  // namespace heterosim
