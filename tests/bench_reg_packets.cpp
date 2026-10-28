// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: what the register-packet checksum fold costs on the GPU command thread
// (video_core/amdgpu/reg_checksum.h, Liverpool::HashRegisterPacket):
//
//  - the fold itself, packet sizes 8 to 24 words, against the single multiply-xor chain it
//    replaces: a chain costs the multiplier's latency per word (each word depends on the last),
//    the four-lane fold costs its throughput. Packets under 8 words have nothing to overlap and
//    stay on the single chain;
//  - a stream of register packets as a pass sends them -- write into the register file, mark the
//    blocks for the draw-preparation workers, fold the packet into the running checksum -- with
//    both folds, to show what the change is worth per packet.
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <random>
#include <vector>

#include "video_core/amdgpu/reg_checksum.h"
#include "video_core/amdgpu/regs.h"

namespace {

double NowNs() {
    timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return static_cast<double>(t.tv_sec) * 1e9 + static_cast<double>(t.tv_nsec);
}

std::uint64_t sink = 0;

/// The fold as it was before the lanes: one chain over the words.
__attribute__((noinline)) std::uint64_t FoldSingleChain(std::uint64_t checksum,
                                                        const std::uint32_t* words,
                                                        std::uint32_t count) {
    for (std::uint32_t i = 0; i < count; ++i) {
        checksum = (checksum ^ words[i]) * AmdGpu::FoldMultiplier;
    }
    return checksum;
}

/// ApplyGraphicsRegisterPacket: the write, the block marking and the fold.
__attribute__((noinline)) void ApplyPacket(AmdGpu::Regs& regs, std::uint32_t word,
                                           const std::uint32_t* payload, std::uint32_t count,
                                           AmdGpu::RegDirty& dirty, std::uint64_t& checksum,
                                           bool lanes) {
    std::memcpy(regs.reg_array.data() + word, payload, count * sizeof(std::uint32_t));
    dirty.Mark(word, count);
    checksum = lanes ? AmdGpu::FoldRegisterWords(checksum, payload, count)
                     : FoldSingleChain(checksum, payload, count);
}

} // namespace

int main() {
    // 1. The fold alone, per packet size.
    {
        std::uint32_t words[24];
        std::mt19937 rng{0x5EED};
        for (std::uint32_t& word : words) {
            word = rng();
        }
        constexpr int Reps = 300000;
        // Packets shorter than the lane threshold (8 words) stay on the single chain, unchanged;
        // the sizes below are the ones a pass actually sends.
        for (std::uint32_t count : {8u, 12u, 16u, 24u}) {
            double single = 1e18, lanes = 1e18;
            std::uint64_t checksum = 0;
            for (int round = 0; round < 5; ++round) {
                checksum = 0;
                const double begin = NowNs();
                for (int i = 0; i < Reps; ++i) {
                    checksum = FoldSingleChain(checksum, words, count);
                }
                single = std::min(single, NowNs() - begin);
                sink ^= checksum;
                checksum = 0;
                const double begin2 = NowNs();
                for (int i = 0; i < Reps; ++i) {
                    checksum = AmdGpu::FoldRegisterWords(checksum, words, count);
                }
                lanes = std::min(lanes, NowNs() - begin2);
                sink ^= checksum;
            }
            const double single_ns = single / Reps;
            const double lane_ns = lanes / Reps;
            std::printf("checksum fold, %2u words: single chain %6.2f ns, four lanes %5.2f ns "
                        "(%.2fx)\n",
                        count, single_ns, lane_ns, single_ns / lane_ns);
        }
    }

    // 2. A stream of register packets, as a pass sends them: the write into the register file, the
    //    block marking for the draw-preparation workers and the fold, at three packet sizes.
    for (std::uint32_t count : {6u, 12u, 24u}) {
        constexpr unsigned NumPackets = 20000;
        std::vector<std::array<std::uint32_t, 24>> payloads(NumPackets);
        std::vector<std::uint32_t> offsets(NumPackets);
        std::mt19937 rng{0xC0FFEE};
        for (unsigned i = 0; i < NumPackets; ++i) {
            offsets[i] = AmdGpu::Regs::ShRegWordOffset + 1 + (rng() % 400) * 8;
            for (std::uint32_t w = 0; w < count; ++w) {
                payloads[i][w] = rng();
            }
        }
        constexpr int Rounds = 40;
        double single = 1e18, lanes = 1e18;
        for (int round = 0; round < Rounds; ++round) {
            for (const bool use_lanes : {false, true}) {
                AmdGpu::Regs regs{};
                AmdGpu::RegDirty dirty;
                std::uint64_t checksum = 0;
                const double begin = NowNs();
                for (unsigned i = 0; i < NumPackets; ++i) {
                    ApplyPacket(regs, offsets[i], payloads[i].data(), count, dirty, checksum,
                                use_lanes);
                }
                const double elapsed = NowNs() - begin;
                (use_lanes ? lanes : single) = std::min(use_lanes ? lanes : single, elapsed);
                sink ^= checksum;
            }
        }
        std::printf("register packet, %2u words (write, mark, fold): single chain %5.2f ns, "
                    "four lanes %5.2f ns (%+.1f%%)\n",
                    count, single / NumPackets, lanes / NumPackets,
                    100.0 * (lanes - single) / single);
    }
    return 0;
}
