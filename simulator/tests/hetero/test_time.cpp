#include <iostream>

#include "hetero/types.h"

namespace {

bool expect_equal(
    heterosim::TimeFs actual,
    heterosim::TimeFs expected,
    const char* name) {
    if (actual == expected) {
        return true;
    }
    std::cerr << name << ": expected " << expected << ", got " << actual << '\n';
    return false;
}

}  // namespace

int main() {
    bool ok = true;
    ok &= expect_equal(heterosim::cycle_to_fs(1, 1000000000ULL), 1000000ULL, "1GHz");
    ok &= expect_equal(heterosim::cycle_to_fs(3, 2000000000ULL), 1500000ULL, "2GHz");
    ok &= expect_equal(heterosim::cycle_to_fs(1, 3000000000ULL), 333334ULL, "ceil");
    return ok ? 0 : 1;
}

