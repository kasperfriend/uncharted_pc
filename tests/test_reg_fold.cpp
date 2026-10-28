// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: the register-packet checksum fold (video_core/amdgpu/reg_checksum.h).
//
// Liverpool::ApplyGraphicsRegisterPacket folds every register-writing packet into a running
// checksum, on the GPU command thread and again on the draw-preparation worker that replays the
// same commands. Equal checksums at a draw mean equal register files, and that is what lets the
// draw reuse the pipeline lookup and the dynamic state of the draw before it -- so the fold must
// be a pure function of the words it is given, and two different register states must not fold to
// the same value. Both are checked here.
#include <cstdint>
#include <cstdio>
#include <array>
#include <random>
#include <vector>

#include "video_core/amdgpu/reg_checksum.h"

namespace {

int failures = 0;

void Expect(bool condition, const char* what, std::uint32_t a, std::uint32_t b) {
    if (!condition) {
        ++failures;
        std::printf("FAIL: %s (%u, %u)\n", what, a, b);
    }
}

/// Folds `count` words of a packet.
std::uint64_t Fold(std::uint64_t checksum, const std::uint32_t* words, std::uint32_t count) {
    return AmdGpu::FoldRegisterWords(checksum, words, count);
}

} // namespace

int main() {
    std::mt19937_64 rng{0x9E3779B97F4A7C15ull};
    std::uint32_t words[32];
    for (std::uint32_t& word : words) {
        word = static_cast<std::uint32_t>(rng());
    }

    // 1. An empty packet folds to the checksum it was given, and the fold is deterministic.
    Expect(Fold(0x123456789ABCull, words, 0) == 0x123456789ABCull, "an empty packet folds to its "
           "input", 0, 0);

    // 2. Every packet length either side of the lane threshold: folding twice gives the same
    //    value (the two threads must agree), and changing any single word changes it (two
    //    different register files must not fold alike).
    std::uint32_t mutations = 0, collisions = 0;
    for (std::uint32_t count = 1; count <= 32; ++count) {
        const std::uint64_t once = Fold(0xDEADBEEFull, words, count);
        Expect(Fold(0xDEADBEEFull, words, count) == once, "the fold is deterministic", count, 0);
        for (std::uint32_t at = 0; at < count; ++at) {
            const std::uint32_t saved = words[at];
            // A change in the low bits and one in the high bits: a lane that dropped a word would
            // show up in at least one of them.
            for (const std::uint32_t delta : {1u, 0x10000000u}) {
                words[at] = saved ^ delta;
                const std::uint64_t other = Fold(0xDEADBEEFull, words, count);
                ++mutations;
                if (other == once) {
                    ++collisions;
                    Expect(false, "a changed word changes the checksum", count, at);
                }
            }
            words[at] = saved;
        }
    }

    // 3. Register states that differ only in how many words they carry must not fold alike: a
    //    packet that appends a zero word is a different write.
    for (std::uint32_t count = 1; count < 32; ++count) {
        words[count] = 0;
        const std::uint64_t shorter = Fold(7, words, count);
        const std::uint64_t longer = Fold(7, words, count + 1);
        Expect(shorter != longer, "an appended word changes the checksum", count, 0);
    }
    for (std::uint32_t& word : words) {
        word = static_cast<std::uint32_t>(rng());
    }

    // 4. Replaying a stream of packets the way the worker does -- from the same starting checksum,
    //    over the same packets -- reaches the checksum the GPU thread reached.
    std::vector<std::array<std::uint32_t, 32>> packets;
    std::vector<std::uint32_t> counts;
    for (std::uint32_t packet = 0; packet < 4000; ++packet) {
        std::array<std::uint32_t, 32> payload{};
        const std::uint32_t count = 2 + static_cast<std::uint32_t>(rng() % 20);
        for (std::uint32_t i = 0; i < count; ++i) {
            payload[i] = static_cast<std::uint32_t>(rng());
        }
        packets.push_back(payload);
        counts.push_back(count);
    }
    std::uint64_t reference = 0x1122334455667788ull;
    for (std::size_t i = 0; i < packets.size(); ++i) {
        reference = Fold(reference, packets[i].data(), counts[i]);
    }
    std::uint64_t replay = 0x1122334455667788ull;
    for (std::size_t i = 0; i < packets.size(); ++i) {
        replay = Fold(replay, packets[i].data(), counts[i]);
    }
    Expect(replay == reference, "a replayed stream reaches the same checksum", 0, 0);

    std::printf("register packet fold: %u single-word changes, %u collisions, %u lengths\n",
                mutations, collisions, 32u);
    if (failures) {
        std::printf("%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("register packet fold: every check passed\n");
    return 0;
}
