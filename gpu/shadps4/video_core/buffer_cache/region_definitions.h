// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>

#include "common/bit_array.h"
#include "common/types.h"

namespace VideoCore {

constexpr u64 TRACKER_PAGE_BITS = 12; // 4K pages
constexpr u64 TRACKER_BYTES_PER_PAGE = 1ULL << TRACKER_PAGE_BITS;

constexpr u64 TRACKER_HIGHER_PAGE_BITS = 22; // each region is 4MB
constexpr u64 TRACKER_HIGHER_PAGE_SIZE = 1ULL << TRACKER_HIGHER_PAGE_BITS;
constexpr u64 TRACKER_HIGHER_PAGE_MASK = TRACKER_HIGHER_PAGE_SIZE - 1ULL;
constexpr u64 NUM_PAGES_PER_REGION = TRACKER_HIGHER_PAGE_SIZE / TRACKER_BYTES_PER_PAGE;

enum class Type {
    CPU,
    GPU,
};

using RegionBits = Common::BitArray<NUM_PAGES_PER_REGION>;

/// bbport: the pages whose protection has to follow a change of the tracked bits
/// (RegionManager::UpdateProtection). The read mask is the complement of the difference
/// (~(gpu ^ readable)); the write mask is the *plain* difference (cpu ^ writeable) and must not
/// be complemented: complementing it marks every page that did not change, the page manager's
/// watcher counts then drift and an unprotect finds a page without a watcher ("Not enough
/// watchers" in page_manager.cpp). Kept here as a named function so tests/test_bit_array.cpp can
/// check it against a scalar reference.
inline RegionBits ProtectionChangeMask(bool is_read, const RegionBits& gpu,
                                       const RegionBits& readable, const RegionBits& cpu,
                                       const RegionBits& writeable) {
    return is_read ? RegionBits::NotXor(gpu, readable) : cpu ^ writeable;
}

/// bbport: pages of a write-fault window that can be unprotected at once: writable by the CPU
/// already and holding no GPU-written data (RegionManager::ExtendWriteFault).
inline RegionBits WriteFaultWindowMask(const RegionBits& gpu, const RegionBits& writeable) {
    return RegionBits::NotOr(gpu, writeable);
}

/// bbport: guest flips so far (Rasterizer::OnGuestFlip); the age of GPU data in pages.
inline std::atomic<u32> g_guest_frames{0};

/// bbport: a scheduler tick the GPU reaches only after the work being recorded now (a GPU write
/// marked now completes by then; BufferCache sets it).
inline u64 (*g_gpu_write_done_tick)() = nullptr;

} // namespace VideoCore
