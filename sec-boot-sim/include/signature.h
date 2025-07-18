#pragma once

#include <cstdint>
#include <vector>

constexpr std::uint32_t MAX_SIGLEN = 1 << 16;
using Signature = std::vector<uint8_t>;
