#pragma once

#include <string>
#include <unordered_map>

namespace heterosim::runtime {

class TimingOwnershipRegistry {
public:
    void claim(const std::string& resource_id, const std::string& owner_id);
    [[nodiscard]] const std::string& owner(const std::string& resource_id) const;
    [[nodiscard]] std::size_t size() const noexcept;

private:
    std::unordered_map<std::string, std::string> owners_;
};

}  // namespace heterosim::runtime
