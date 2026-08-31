#include "ramulator_bridge.h"

#include <cstdint>
#include <iostream>

int main() {
  uint64_t address = 0;
  if (heterosim_translate_gpu_address(0x1000, 64, &address) !=
          HETEROSIM_ADDRESS_TRANSLATED ||
      address != 0x200) {
    std::cerr << "capture address did not rebase to Global PA" << std::endl;
    return 1;
  }
  if (heterosim_translate_gpu_address(0x220, 32, &address) !=
          HETEROSIM_ADDRESS_IDENTITY ||
      address != 0x220) {
    std::cerr << "already-global request was not idempotent" << std::endl;
    return 2;
  }
  if (heterosim_translate_gpu_address(0x5000, 64, &address) !=
      HETEROSIM_ADDRESS_INVALID) {
    std::cerr << "unmapped request did not fail closed" << std::endl;
    return 3;
  }
  if (heterosim_address_translated_requests() != 1 ||
      heterosim_address_already_global_requests() != 1 ||
      heterosim_address_unmapped_requests() != 1 ||
      heterosim_address_binding_ranges() != 1) {
    std::cerr << "address translation counter conservation failed" << std::endl;
    return 4;
  }
  std::cout << "online address translation smoke passed" << std::endl;
  return 0;
}
