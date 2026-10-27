// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstddef>
#include "common/types.h"

// bbport: stamping the per-page frame table the buffer cache keeps for the GPU heap window
// (NoteGpuUse). A buffer is bound by several draws in a row and a pass reuses its buffers from
// draw to draw, so most pages of a range already carry the frame being written: storing it again
// only dirties cache lines -- over the 8 MB table that is most of the cost. Ranges of 16 pages
// (64 KiB) and up are therefore compared 8 pages at a time with two 128-bit integer compares and
// only stored where they differ, which measured 1.06-1.09x faster at 16 pages, 1.22-1.25x at 32,
// 1.41-1.45x at 64, 1.31-1.41x at 256 and 1.73-1.82x at 1024 (tests/bench_page_stamp.cpp; a
// 4 MiB bind stamps in 200 ns instead of 360). Shorter ranges are plain stores, where the
// compare does not pay for itself. AVX1 builds take the same 128-bit path -- no AVX2 needed.
// Kept here, without a Vulkan or logging dependency, so tests/test_page_stamp.cpp can check it
// against the plain loop.
#if defined(__SSE2__) || defined(_M_X64)
#include <emmintrin.h>
#define BBPORT_PAGE_STAMP 1
#endif

namespace VideoCore::PageStamp {

/// Shorter ranges are stamped with plain stores (see above).
inline constexpr std::size_t CompareMinPages = 16;
/// Pages compared and stored per step on the vector path.
inline constexpr std::size_t PagesPerStep = 8;

/// Marks the pages [page, last] of `table` as used in `frame`; returns how many were marked.
/// `compare` false stamps every page unconditionally (the toggle that turns the fast path off).
inline std::size_t Mark(u32* table, std::size_t page, std::size_t last, u32 frame,
                        bool compare = true) noexcept {
    if (last < page) {
        return 0;
    }
    const std::size_t pages = last - page + 1;
#ifdef BBPORT_PAGE_STAMP
    if (compare && pages >= CompareMinPages) {
        const __m128i stamp = _mm_set1_epi32(static_cast<int>(frame));
        for (; page + PagesPerStep <= last + 1; page += PagesPerStep) {
            u32* const at = table + page;
            const __m128i first = _mm_loadu_si128(reinterpret_cast<const __m128i*>(at));
            const __m128i second = _mm_loadu_si128(reinterpret_cast<const __m128i*>(at + 4));
            if (_mm_movemask_epi8(_mm_cmpeq_epi32(stamp, first)) != 0xFFFF) {
                _mm_storeu_si128(reinterpret_cast<__m128i*>(at), stamp);
            }
            if (_mm_movemask_epi8(_mm_cmpeq_epi32(stamp, second)) != 0xFFFF) {
                _mm_storeu_si128(reinterpret_cast<__m128i*>(at + 4), stamp);
            }
        }
    }
#else
    (void)compare;
#endif
    for (; page <= last; ++page) {
        table[page] = frame;
    }
    return pages;
}

} // namespace VideoCore::PageStamp
