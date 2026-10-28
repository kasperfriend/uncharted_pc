// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: a bitmap of register blocks, walked by its set bits.
//
// The draw-preparation scanner records, per submission, which 32-word blocks of the register file
// the submission wrote (AmdGpu::RegDirty) and the draw pipe records the same set for its packets.
// Both then walk only the set bits: std::bitset has no portable way to do that, and testing all
// 1664 blocks costs more than copying the blocks that matter.
//
// Deliberately free of the rest of the video core (no fmt, no logging, no Types.h) so
// tests/test_block_mask.cpp can check it on its own, on any host.
#pragma once

#include <array>
#include <bit>
#include <cstdint>

namespace AmdGpu {

/// `NumBlocks` bits, in 64-bit words. SetRange is inclusive on both ends.
template <std::uint32_t NumBlocks>
struct RegBlockMask {
    static constexpr std::uint32_t Words = (NumBlocks + 63) / 64;
    std::array<std::uint64_t, Words> words{};

    void Set(std::uint32_t block) {
        words[block >> 6] |= std::uint64_t{1} << (block & 63);
    }
    /// Marks the closed range [first, last] with a handful of word writes instead of one per bit.
    void SetRange(std::uint32_t first, std::uint32_t last) {
        std::uint32_t word = first >> 6;
        const std::uint32_t last_word = last >> 6;
        const std::uint64_t head = ~std::uint64_t{0} << (first & 63);
        const std::uint64_t tail = ~std::uint64_t{0} >> (63 - (last & 63));
        if (word == last_word) {
            words[word] |= head & tail;
            return;
        }
        words[word++] |= head;
        for (; word < last_word; ++word) {
            words[word] = ~std::uint64_t{0};
        }
        words[last_word] |= tail;
    }
    bool Test(std::uint32_t block) const {
        return (words[block >> 6] >> (block & 63)) & 1;
    }
    bool Any() const {
        for (const std::uint64_t word : words) {
            if (word) {
                return true;
            }
        }
        return false;
    }
    void Reset() {
        words.fill(0);
    }
    /// Calls fn(block) for every set block, in increasing order, skipping empty words.
    template <typename F>
    void ForEachSet(F&& fn) const {
        for (std::uint32_t index = 0; index < Words; ++index) {
            std::uint64_t word = words[index];
            while (word) {
                fn(index * 64 + std::uint32_t(std::countr_zero(word)));
                word &= word - 1;
            }
        }
    }
};

} // namespace AmdGpu
