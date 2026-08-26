#include "hetero/runtime/timing_ownership.h"

#include <stdexcept>

namespace heterosim::runtime {

void TimingOwnershipRegistry::claim(
    const std::string& resource_id,
    const std::string& owner_id) {
    if (resource_id.empty() || owner_id.empty()) {
        throw std::invalid_argument("timing resource and owner are required");
    }
    const auto [iterator, inserted] = owners_.emplace(resource_id, owner_id);
    if (!inserted && iterator->second != owner_id) {
        throw std::logic_error(
            "timing ownership conflict for " + resource_id + ": " +
            iterator->second + " vs " + owner_id);
    }
}

const std::string& TimingOwnershipRegistry::owner(
    const std::string& resource_id) const {
    const auto iterator = owners_.find(resource_id);
    if (iterator == owners_.end()) {
        throw std::out_of_range("unowned timing resource: " + resource_id);
    }
    return iterator->second;
}

std::size_t TimingOwnershipRegistry::size() const noexcept {
    return owners_.size();
}

}  // namespace heterosim::runtime
