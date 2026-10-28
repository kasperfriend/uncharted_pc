// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstddef>
#include <cstring>
#include "common/types.h"

namespace Common {

/// bbport: a memo key that is exactly the bytes of the values it was given, in order.
///
/// The render-target memos key on a handful of guest registers (a C# or Z# register, its extent
/// hint, the depth view: 56 to 68 bytes). They used to build the key in a zero-filled 256-byte
/// array -- a 256-byte store per attachment per draw, and the same 256 bytes copied into the
/// memo entries on a miss. This stores only the bytes that were appended and compares only
/// those. `bytes` is never read past `size`, so it needs no zero fill; a fresh key is empty and
/// matches nothing.
template <std::size_t MaxBytes>
struct ByteKey {
    static constexpr std::size_t Capacity = MaxBytes;

    u8 bytes[MaxBytes];
    u32 size = 0;

    void Clear() noexcept {
        size = 0;
    }

    void Append(const void* data, std::size_t length) noexcept {
        if (size + length > Capacity) {
            return; // a key larger than the storage: the caller's static_assert prevents it
        }
        std::memcpy(bytes + size, data, length);
        size += static_cast<u32>(length);
    }

    template <typename T>
    void Append(const T& part) noexcept {
        Append(&part, sizeof(T));
    }

    /// Replaces the key with the exact bytes of `parts`, in order.
    template <typename... Parts>
    void Set(const Parts&... parts) noexcept {
        size = 0;
        (Append(parts), ...);
    }

    [[nodiscard]] bool Equals(const ByteKey& other) const noexcept {
        if (size != other.size) {
            return false;
        }
        // memcmp with a length of zero and a null pointer is undefined, and an empty key is the
        // state every fresh key starts in.
        return size == 0 || std::memcmp(bytes, other.bytes, size) == 0;
    }

    [[nodiscard]] bool Matches(const void* data, std::size_t length) const noexcept {
        if (size != length) {
            return false;
        }
        return length == 0 || std::memcmp(bytes, data, length) == 0;
    }
};

} // namespace Common
