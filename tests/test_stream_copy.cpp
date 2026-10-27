// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: BbCopy::Copy (gpu/shim/bbport_memcpy.h) against std::memcpy.
//
// The streaming path (non-temporal 32-byte stores) writes memory the GPU reads over PCIe. It
// has an alignment head, a 128-byte body, a 32-byte tail and a byte-wise remainder, so every
// size and every destination alignment is checked here, with the destination placed inside
// guard bytes that must stay untouched.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "bbport_memcpy.h"

namespace {

int failures = 0;

void Check(bool ok, const char* what, int line, std::size_t size, std::size_t offset) {
    if (!ok) {
        std::printf("FAIL line %d: %s (size %zu, destination offset %zu)\n", line, what, size, offset);
        ++failures;
    }
}

#define CHECK(condition) Check((condition), #condition, __LINE__, size, offset)

void One(std::size_t size, std::size_t offset, BbCopy::CopyPolicy policy) {
    constexpr std::size_t Guard = 64;
    std::vector<unsigned char> source(size + Guard * 2), destination(size + Guard * 4);
    // Distinct, non-zero data on both sides: a copy that misses a byte is visible.
    for (std::size_t i = 0; i < source.size(); ++i) {
        source[i] = static_cast<unsigned char>(i * 31 + 7);
    }
    std::memset(destination.data(), 0xa5, destination.size());
    BbCopy::Copy(destination.data() + Guard + offset, source.data() + Guard, size, policy);
    CHECK(std::memcmp(destination.data() + Guard + offset, source.data() + Guard, size) == 0);
    for (std::size_t i = 0; i < Guard + offset; ++i) {
        CHECK(destination[i] == 0xa5);
    }
    for (std::size_t i = Guard + offset + size; i < destination.size(); ++i) {
        CHECK(destination[i] == 0xa5);
    }
}

} // namespace

int main() {
    const char* env = std::getenv("BB_STREAM_COPY");
    std::printf("streaming copy: %s, BB_STREAM_COPY=%s\n",
                BbCopy::HaveStreamingCopy() ? "AVX" : "not compiled in", env ? env : "(unset)");
    // Around every internal boundary (32-byte stores, 128-byte body) and the auto threshold.
    const std::size_t sizes[] = {
        0, 1, 2, 3, 7, 8, 15, 16, 31, 32, 33, 63, 64, 65, 96, 127, 128, 129, 191, 192, 255, 256,
        257, 511, 512, 1000, 1023, 1024, 4095, 4096, 8191, 8192,
        BbCopy::AutoStreamBytes - 1, BbCopy::AutoStreamBytes, BbCopy::AutoStreamBytes + 1,
        BbCopy::AutoStreamBytes + 63, 2 * BbCopy::AutoStreamBytes + 17,
    };
    for (std::size_t size : sizes) {
        for (std::size_t offset = 0; offset < 40; ++offset) {
            One(size, offset, BbCopy::CopyPolicy::Cached);
            One(size, offset, BbCopy::CopyPolicy::Stream);
            One(size, offset, BbCopy::CopyPolicy::Auto);
        }
    }
    if (failures != 0) {
        std::printf("stream copy: %d failures\n", failures);
        return 1;
    }
    std::printf("stream copy: %zu sizes x 40 alignments x 3 policies copied correctly\n",
                sizeof(sizes) / sizeof(*sizes));
    return 0;
}
