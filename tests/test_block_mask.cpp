// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: AmdGpu::RegBlockMask against a scalar reference.
//
// The mask says which 32-word blocks of the register file a submission wrote. It is built from
// (word, count) pairs of the packet decoder -- fields, whole packets, and ClearState resets -- and
// then walked with ForEachSet to copy exactly those blocks. A bit set in the wrong place costs a
// worker a stale register (a wrong draw), a bit missed in the wrong place costs a register that
// never reaches the worker: the ranges are checked here against a plain bool array, at every
// boundary of the 64-bit words, for block counts that are not a multiple of 64 as well.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "video_core/amdgpu/block_mask.h"

namespace {

int failures = 0;

void Check(bool ok, const char* what, int line, std::uint32_t first, std::uint32_t last) {
    if (!ok) {
        std::printf("FAIL line %d: %s (range [%u, %u])\n", line, what, first, last);
        ++failures;
    }
}
#define CHECK(condition) Check((condition), #condition, __LINE__, first, last)

template <std::uint32_t NumBlocks>
void Exercise(std::uint32_t seed) {
    using Mask = AmdGpu::RegBlockMask<NumBlocks>;
    std::mt19937 rng(seed);
    std::vector<bool> reference(NumBlocks, false);
    Mask mask;

    const auto matches = [&](std::uint32_t first, std::uint32_t last) {
        for (std::uint32_t block = 0; block < NumBlocks; ++block) {
            CHECK(mask.Test(block) == reference[block]);
        }
        // The walk must visit every set block once, in increasing order.
        std::vector<std::uint32_t> walked;
        mask.ForEachSet([&](std::uint32_t block) { walked.push_back(block); });
        std::vector<std::uint32_t> expected;
        for (std::uint32_t block = 0; block < NumBlocks; ++block) {
            if (reference[block]) {
                expected.push_back(block);
            }
        }
        CHECK(walked == expected);
        CHECK(mask.Any() == !expected.empty());
    };

    // Every single block, then every range that starts or ends on a word boundary.
    for (std::uint32_t first = 0; first < NumBlocks; ++first) {
        const std::uint32_t last = first;
        mask.Set(first);
        reference[first] = true;
        matches(first, last);
    }
    mask.Reset();
    std::fill(reference.begin(), reference.end(), false);
    matches(0, NumBlocks - 1);

    for (std::uint32_t first = 0; first < NumBlocks; first += 61) {
        const std::uint32_t last = std::min<std::uint32_t>(NumBlocks - 1, first + 200);
        mask.Reset();
        std::fill(reference.begin(), reference.end(), false);
        // A run that starts and ends at arbitrary offsets, then one aligned to the words.
        mask.SetRange(first, last);
        for (std::uint32_t block = first; block <= last; ++block) {
            reference[block] = true;
        }
        matches(first, last);
        mask.SetRange(NumBlocks - 1 - (last - first), NumBlocks - 1);
        for (std::uint32_t block = NumBlocks - 1 - (last - first); block < NumBlocks; ++block) {
            reference[block] = true;
        }
        matches(first, NumBlocks - 1);
    }

    // The whole array in one range, and a single block at each end.
    mask.Reset();
    std::fill(reference.begin(), reference.end(), false);
    mask.SetRange(0, NumBlocks - 1);
    std::fill(reference.begin(), reference.end(), true);
    matches(0, NumBlocks - 1);
    mask.Reset();
    std::fill(reference.begin(), reference.end(), false);
    mask.Set(0);
    mask.Set(NumBlocks - 1);
    reference[0] = reference[NumBlocks - 1] = true;
    matches(0, NumBlocks - 1);
    Check(mask.Test(0) && mask.Test(NumBlocks - 1), "both ends set", __LINE__, 0, NumBlocks - 1);

    // Random ranges on top of each other, the way a packet stream marks registers.
    mask.Reset();
    std::fill(reference.begin(), reference.end(), false);
    for (int round = 0; round < 400; ++round) {
        const std::uint32_t first = rng() % NumBlocks;
        const std::uint32_t last = std::min<std::uint32_t>(NumBlocks - 1, first + rng() % 40);
        mask.SetRange(first, last);
        for (std::uint32_t block = first; block <= last; ++block) {
            reference[block] = true;
        }
        for (std::uint32_t block = 0; block < NumBlocks; ++block) {
            if (mask.Test(block) != reference[block]) {
                Check(false, "random ranges", __LINE__, first, last);
                return;
            }
        }
    }
    mask.Reset();
    std::fill(reference.begin(), reference.end(), false);
    matches(0, 0);
}

} // namespace

int main() {
    // 1664 blocks: the register file's 0xD000 words in 32-word blocks (a multiple of 64).
    // 6656 and 1700: the same register file in 8-word blocks, and a count that is not a multiple
    // of 64 at all, which is where a tail word would be written past the end.
    Exercise<1664>(1);
    Exercise<1664>(2);
    Exercise<6656>(3);
    Exercise<1700>(4);
    Exercise<64>(5);
    Exercise<1>(6);
    if (failures != 0) {
        std::printf("block mask: %d failures\n", failures);
        return 1;
    }
    std::printf("block mask: all checks passed (1664, 6656, 1700, 64 and 1 blocks)\n");
    return 0;
}
