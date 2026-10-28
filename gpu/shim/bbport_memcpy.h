// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: copying guest memory into the buffers the GPU reads.
//
// Streaming a new area copies 50-400 MB per frame out of guest memory into staging buffers and
// mapped buffer ranges. A normal `memcpy` writes through the cache: every 64-byte line is first
// read from memory (read-for-ownership) and then held in L2/L3 although nothing on the CPU
// reads it again -- the GPU does, over PCIe, or the tiling shaders do. Non-temporal stores skip
// the read and write straight to memory, which halves the memory traffic of a large copy and
// leaves the caches to the game's own threads.
//
// The stores are `vmovntdq`, which is AVX (the port's floor), not AVX2.
#pragma once

#include <cstddef>
#include <cstring>

namespace BbCopy {

enum class CopyPolicy {
    /// Streaming stores when the copy is large enough to pay off (see AutoStreamBytes).
    Auto,
    /// Always streaming stores: a chunk of a large upload that only the GPU reads.
    Stream,
    /// Always plain stores: small copies the CPU reads back straight away.
    Cached,
};

/// Copies this large and up are streamed with Auto: below it the data is still in cache when
/// the CPU (or the tiling pass) looks at it, and a memory round trip would be slower.
inline constexpr std::size_t AutoStreamBytes = 4 * 1024 * 1024;

/// Copies `size` bytes from `source` to `destination` (overlap-free). With streaming stores the
/// copy ends with an `sfence`, so a thread or GPU told about the copy sees it complete.
void Copy(void* destination, const void* source, std::size_t size,
          CopyPolicy policy = CopyPolicy::Auto) noexcept;

/// True when the build has AVX (the streaming path is compiled in).
bool HaveStreamingCopy() noexcept;

} // namespace BbCopy
