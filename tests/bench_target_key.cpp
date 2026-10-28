// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: the cost of one render-target memo lookup, before and after the key change.
//
// Every color and depth attachment of every draw asks the memo the same question: "are these
// target registers and this extent the ones I resolved last time?". The version this bench was
// written against zero-filled a 256-byte buffer for the key on *every* lookup, kept 256 bytes of
// it in the last-hit and memo entries, and hashed the key with a word-at-a-time FNV -- for keys
// of 56 to 68 bytes. The draw path now stores exactly the bytes of the call's parts
// (common/byte_key.h) and hashes those with XXH3_64bits. This measures both shapes on the same
// inputs and the same hit/miss pattern.
//
// The parts live in a working set the size of a frame's attachment set: the draw path reads its
// registers from a cache-hot struct, and the memo only has to be cheaper than the image lookup
// behind it (a hash-map probe plus a texture-cache view search).
#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "common/byte_key.h"

#if __has_include(<xxhash.h>)
#include <xxhash.h>
#define BENCH_HAS_XXH3 1
#else
#define BENCH_HAS_XXH3 0
#endif

namespace {

using Clock = std::chrono::steady_clock;
using Parts = std::array<u32, 17>; // a tag, the 60-byte color buffer register, the extent hint

constexpr std::size_t Lookups = 8'000'000;
constexpr std::size_t KeyBytes = sizeof(Parts);
constexpr std::size_t WorkingSet = 24; // the attachments a frame draws into
constexpr std::size_t MemoEntries = 64;

/// The hash the draw path hashes a key with (XXH3_64bits) when the dependency is at hand; a
/// word-at-a-time fold otherwise, so the bench still runs in a checkout without it.
u64 FastHash(const u8* data, std::size_t size) {
#if BENCH_HAS_XXH3
    return XXH3_64bits(data, size);
#else
    u64 hash = 0x9E3779B97F4A7C15ull ^ size;
    std::size_t at = 0;
    for (; at + 8 <= size; at += 8) {
        u64 word;
        std::memcpy(&word, data + at, 8);
        hash = (hash ^ word) * 0x9E3779B97F4A7C15ull;
        hash ^= hash >> 29;
    }
    for (; at < size; ++at) {
        hash = (hash ^ data[at]) * 0x100000001B3ull;
    }
    return hash;
#endif
}

/// The word-at-a-time FNV the old key was hashed with, over the key's own bytes.
u64 OldHash(const u8* data, std::size_t size) {
    u64 hash = 0xCBF29CE484222325ull;
    for (std::size_t i = 0; i + 8 <= size; i += 8) {
        u64 word;
        std::memcpy(&word, data + i, 8);
        hash = (hash ^ word) * 0x100000001B3ull;
        hash ^= hash >> 29;
    }
    return hash;
}

struct OldMemo {
    std::array<u8, 256> bytes{};
    u32 size = ~0u;
    u64 image_uid = 0;
};

struct NewMemo {
    Common::ByteKey<72> key;
    u64 image_uid = 0;
};

} // namespace

int main() {
    // The frame's attachment registers, and the order the frame asks for them in: mostly the
    // same target again (consecutive draws of a pass), sometimes the next one.
    std::mt19937 rng{12345};
    std::array<Parts, WorkingSet> targets{};
    for (auto& parts : targets) {
        for (auto& word : parts) {
            word = static_cast<u32>(rng());
        }
        parts[16] = 1920 | (1080u << 16); // the extent hint of the draw
    }
    std::vector<u32> order(Lookups);
    u32 current = 0;
    for (auto& entry : order) {
        if ((rng() & 31) == 0) {
            current = rng() % WorkingSet;
        }
        entry = current;
    }
    std::vector<u32> switching(Lookups);
    for (auto& entry : switching) {
        entry = rng() % WorkingSet;
    }

    std::array<OldMemo, MemoEntries> old_memo{};
    std::array<NewMemo, MemoEntries> new_memo{};
    OldMemo old_last;
    Common::ByteKey<72> new_last;
    u64 old_hits = 0, new_hits = 0;

    // Consecutive draws into the same target: the lookup is a compare against the last hit.
    const auto old_begin = Clock::now();
    for (const u32 index : order) {
        std::array<u8, 256> key{}; // what the old code built per lookup
        std::memcpy(key.data(), &targets[index], KeyBytes);
        if (old_last.size == KeyBytes && std::memcmp(old_last.bytes.data(), key.data(), KeyBytes) == 0) {
            ++old_hits;
            continue;
        }
        const u64 hash = OldHash(key.data(), KeyBytes);
        auto& memo = old_memo[hash % old_memo.size()];
        if (memo.size == KeyBytes && std::memcmp(memo.bytes.data(), key.data(), KeyBytes) == 0) {
            ++old_hits;
        }
        memo.bytes = key;
        memo.size = KeyBytes;
        old_last.bytes = key;
        old_last.size = KeyBytes;
    }
    const auto old_end = Clock::now();

    const auto new_begin = Clock::now();
    for (const u32 index : order) {
        Common::ByteKey<72> key;
        key.Set(targets[index]);
        if (new_last.Equals(key)) {
            ++new_hits;
            continue;
        }
        auto& memo = new_memo[FastHash(key.bytes, key.size) % new_memo.size()];
        if (memo.key.Equals(key)) {
            ++new_hits;
        }
        memo.key = key;
        new_last = key;
    }
    const auto new_end = Clock::now();

    // A pass that alternates targets: every lookup builds, hashes and stores a key. The results
    // are accumulated and printed, so neither side can be optimised away.
    u64 old_sink = 0;
    const auto old2_begin = Clock::now();
    for (const u32 index : switching) {
        std::array<u8, 256> key{};
        std::memcpy(key.data(), &targets[index], KeyBytes);
        const u64 hash = OldHash(key.data(), KeyBytes);
        auto& memo = old_memo[hash % old_memo.size()];
        old_sink += hash ^ u64(std::memcmp(memo.bytes.data(), key.data(), KeyBytes));
        memo.bytes = key;
        memo.size = KeyBytes;
    }
    const auto old2_end = Clock::now();
    u64 new_sink = 0;
    const auto new2_begin = Clock::now();
    for (const u32 index : switching) {
        Common::ByteKey<72> key;
        key.Set(targets[index]);
        const u64 hash = FastHash(key.bytes, key.size);
        auto& memo = new_memo[hash % new_memo.size()];
        new_sink += hash ^ u64(memo.key.Equals(key));
        memo.key = key;
    }
    const auto new2_end = Clock::now();

    // The validation a memo hit does now instead of comparing the registry generation: the image
    // in the remembered slot must still carry the remembered uid.
    std::array<u64, WorkingSet> uids{};
    std::array<u32, WorkingSet> ids{};
    for (u32 i = 0; i < WorkingSet; ++i) {
        uids[i] = 0x1000ull + i;
        ids[i] = i;
    }
    for (u32 i = 0; i < WorkingSet; ++i) {
        new_memo[i].image_uid = uids[i];
    }
    u64 uid_ok = 0;
    const auto uid_begin = Clock::now();
    for (std::size_t i = 0; i < Lookups; ++i) {
        const u32 slot = order[i] % WorkingSet;
        if (new_memo[ids[slot]].image_uid == uids[slot]) {
            ++uid_ok;
        }
    }
    const auto uid_end = Clock::now();

    const auto ns_per = [](Clock::duration d) {
        return double(std::chrono::duration_cast<std::chrono::nanoseconds>(d).count()) /
               double(Lookups);
    };
    const double old_ns = ns_per(old_end - old_begin);
    const double new_ns = ns_per(new_end - new_begin);
    const double old2_ns = ns_per(old2_end - old2_begin);
    const double new2_ns = ns_per(new2_end - new2_begin);
    std::printf("target memo, same target as the previous draw: %.2f -> %.2f ns/lookup (%.2fx)\n",
                old_ns, new_ns, old_ns / new_ns);
    std::printf("target memo, target changes every draw:         %.2f -> %.2f ns/lookup (%.2fx)\n",
                old2_ns, new2_ns, old2_ns / new2_ns);
    std::printf("  %zu-byte key in a %zu-entry working set; %llu of %zu lookups repeat\n", KeyBytes,
                WorkingSet, static_cast<unsigned long long>(old_hits), Lookups);
    std::printf("  hits %llu (old) / %llu (new); uid validation %.2f ns (%llx %llx %llx)\n",
                static_cast<unsigned long long>(old_hits),
                static_cast<unsigned long long>(new_hits), ns_per(uid_end - uid_begin),
                static_cast<unsigned long long>(old_sink),
                static_cast<unsigned long long>(new_sink),
                static_cast<unsigned long long>(uid_ok));
#if !BENCH_HAS_XXH3
    std::printf("  (xxhash.h not found: the new key was hashed with a local fold, not XXH3_64bits)\n");
#endif
    return 0;
}
