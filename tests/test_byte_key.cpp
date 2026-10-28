// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: Common::ByteKey, the memo key of the render-target and render-state lookups.
//
// The draw path keys those memos on the raw bytes of a few guest registers, and a false match
// there binds the wrong render target (a corrupted frame, not a slow frame), so the key has to
// be exactly the bytes it was given: same parts give the same key, a different part at any
// offset gives a different key, a shorter key with the same prefix is not the same key, and a
// key that would not fit is left empty rather than truncated.
#include <cstdio>
#include <cstring>
#include <vector>

#include "common/byte_key.h"

namespace {

int failures = 0;

void Check(bool ok, const char* what) {
    if (!ok) {
        ++failures;
        if (failures < 16) {
            std::printf("FAIL %s\n", what);
        }
    }
}

struct Register {
    u32 first;
    u32 second;
};

struct Big {
    u8 bytes[56];
};

} // namespace

int main() {
    // The storage the draw path uses for a target key (vk_rasterizer.h: TargetKeyBytes).
    using Key = Common::ByteKey<72>;

    const Register reg{0x1234'5678u, 0x9ABC'DEF0u};
    const u32 tag = 7;
    const u32 hint = 0x0438'0780u;

    // Same parts, same key; and the key is exactly those bytes.
    Key a;
    a.Set(tag, reg, hint);
    Key b;
    b.Set(tag, reg, hint);
    Check(a.Equals(b), "identical parts are equal");
    Check(a.size == sizeof(tag) + sizeof(reg) + sizeof(hint), "size is the sum of the parts");
    u8 expected[sizeof(tag) + sizeof(reg) + sizeof(hint)];
    std::memcpy(expected, &tag, sizeof(tag));
    std::memcpy(expected + sizeof(tag), &reg, sizeof(reg));
    std::memcpy(expected + sizeof(tag) + sizeof(reg), &hint, sizeof(hint));
    Check(a.Matches(expected, sizeof(expected)), "the key is the parts' bytes in order");

    // A single changed bit in any part makes the key different.
    for (std::size_t byte = 0; byte < sizeof(expected); ++byte) {
        for (int bit = 0; bit < 8; ++bit) {
            Register other = reg;
            u32 other_tag = tag;
            u32 other_hint = hint;
            auto* base = byte < sizeof(tag)       ? reinterpret_cast<u8*>(&other_tag)
                         : byte < sizeof(tag) + sizeof(reg) ? reinterpret_cast<u8*>(&other)
                                                          : reinterpret_cast<u8*>(&other_hint);
            const std::size_t offset =
                byte < sizeof(tag) ? byte
                : byte < sizeof(tag) + sizeof(reg) ? byte - sizeof(tag)
                                                    : byte - sizeof(tag) - sizeof(reg);
            base[offset] ^= static_cast<u8>(1u << bit);
            Key diff;
            diff.Set(other_tag, other, other_hint);
            if (diff.Equals(a)) {
                std::printf("FAIL a change at byte %zu bit %d still matches\n", byte, bit);
                ++failures;
            }
        }
    }

    // The same bytes split differently are the same key: the key is bytes, not parts.
    Key split;
    split.Set(expected[0], expected[1]);
    Check(!split.Equals(a), "a shorter key with the same prefix is not equal");
    Key joined;
    joined.Append(expected, sizeof(expected));
    Check(joined.Equals(a), "the same bytes either way");

    // Set() replaces: it does not append to what the key held before.
    a.Set(hint);
    Check(a.size == sizeof(hint), "Set replaces the previous key");
    Key only_hint;
    only_hint.Set(hint);
    Check(a.Equals(only_hint), "Set(hint) after a longer key equals a fresh Set(hint)");

    // Empty keys compare equal and match only nothing.
    Key empty1;
    Key empty2;
    Check(empty1.Equals(empty2), "two fresh keys are equal");
    Check(!empty1.Matches(expected, sizeof(expected)), "an empty key does not match bytes");
    Check(empty1.Matches(nullptr, 0), "an empty key matches no bytes");

    // A part that does not fit leaves the key as it was: a memo never matches on a truncated
    // key, and nothing is written past the storage.
    Key full;
    full.Set(tag);
    Big big{};
    for (u32 i = 0; i < sizeof(big.bytes); ++i) {
        big.bytes[i] = static_cast<u8>(i);
    }
    full.Append(big);
    Check(full.size == sizeof(tag) + sizeof(big.bytes), "a part that fits is appended");
    const u32 before = full.size;
    full.Append(big);
    Check(full.size == before, "an oversized part is dropped, not truncated");
    full.Append(hint);
    Check(full.size == before + sizeof(hint), "the key keeps working after a dropped part");

    // The real key sizes of the draw path still fit and stay distinguishable.
    // A color target key: tag, the color buffer register (60 bytes) and the extent hint.
    Check(sizeof(tag) + 60 + sizeof(hint) <= Key::Capacity, "a color target key fits");
    Key color0, color1;
    Register buf0{1, 2};
    Register buf1{1, 3};
    color0.Set(u32(1), buf0, hint);
    color1.Set(u32(1), buf1, hint);
    Check(!color0.Equals(color1), "two color targets differ");

    Key depth0, depth1;
    const u64 htile = 0x1000'0000ull;
    depth0.Set(u32(2), reg, u32(0), u32(0x1F), htile, hint); // 56 bytes, like the draw path
    depth1.Set(u32(2), reg, u32(0), u32(0x1F), htile + 8, hint);
    Check(!depth0.Equals(depth1), "two depth targets differ");

    if (failures == 0) {
        std::printf("byte key: all checks passed\n");
        return 0;
    }
    std::printf("byte key: %d failures\n", failures);
    return 1;
}
