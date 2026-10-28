// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: the checksum fold over a graphics-register packet (Liverpool::HashRegisterPacket).
//
// The GPU command thread and the draw-preparation workers both fold every register-writing packet
// into a running checksum, and two draws whose checksums are equal have equal register files --
// that is what lets a draw reuse the key, the pipeline lookup and the dynamic state of the draw
// before it (PipelineCache::TryPreparedPipeline, Rasterizer::Draw). Every register set a draw
// depends on is re-sent this way, so the fold is on the command thread's path for each of them.
//
// A single chain of `(c ^ word) * K` costs the multiplier's latency (3-4 cycles) per word, because
// each word depends on the one before it. Four chains that only meet at the end cost their
// throughput instead: 1.8x fewer cycles for an 8-word packet, 2.1x at 12, 2.8x at 24
// (tests/bench_reg_packets.cpp). The chains start from different constants derived from the
// incoming checksum and are combined with rotated products, so the result still changes when any
// single word changes (tests/test_reg_fold.cpp), and it is still a pure function of (checksum,
// words) -- the property the caller relies on: the same packets folded from the same starting
// checksum give the same result, on whichever thread runs them. Packets shorter than the eight
// words in which the lanes pay for their seeding stay on one chain.
#pragma once

#include <bit>
#include <cstdint>

namespace AmdGpu {

/// The multiply of the fold: an odd constant, so the map is invertible and no information of the
/// packet is thrown away by the multiply alone.
inline constexpr std::uint64_t FoldMultiplier = 0x9E3779B97F4A7C15ull;

/// Folds `count` words into `checksum`. Deterministic: the same starting checksum and the same
/// words always give the same result.
[[nodiscard]] inline std::uint64_t FoldRegisterWords(std::uint64_t checksum, const std::uint32_t* words,
                                                     std::uint32_t count) {
    constexpr std::uint64_t k = FoldMultiplier;
    if (count < 8) {
        for (std::uint32_t i = 0; i < count; ++i) {
            checksum = (checksum ^ words[i]) * k;
        }
        return checksum;
    }
    // Four chains, each starting from its own mix of the incoming checksum.
    std::uint64_t a = checksum;
    std::uint64_t b = checksum * k + 1;
    std::uint64_t c = checksum * (k | 1) + 2;
    std::uint64_t d = checksum * 3 + 3;
    std::uint32_t i = 0;
    for (; i + 4 <= count; i += 4) {
        a = (a ^ words[i + 0]) * k;
        b = (b ^ words[i + 1]) * k;
        c = (c ^ words[i + 2]) * k;
        d = (d ^ words[i + 3]) * k;
    }
    for (; i < count; ++i) {
        a = (a ^ words[i]) * k;
    }
    // One more multiply per lane before they meet: a lane's state must not enter the result as
    // itself, or two lanes could cancel out (a packet of zeros would then fold to the checksum it
    // started from).
    return a ^ std::rotl(b * k, 17) ^ std::rotl(c * k, 33) ^ std::rotl(d * k, 49);
}

} // namespace AmdGpu
