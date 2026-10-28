// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: cost of stamping the buffer cache's page-use table, plain stores against the
// compare-and-skip path of VideoCore::PageStamp (NoteGpuUse runs this for every buffer binding).
//
// The table is the port's (8 MB, one 4 KiB page each over the GPU heap window); the ranges are
// the binds of a frame -- the same pages are bound again by the next draws, which is what the
// compare path skips. Times are per call on this machine; the ratio is what carries over.
#include <algorithm>
#include <cstdint>
#include <tuple>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <vector>

#include "video_core/buffer_cache/page_stamp.h"

namespace {

double NowNs() {
    timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return static_cast<double>(t.tv_sec) * 1e9 + static_cast<double>(t.tv_nsec);
}

} // namespace

int main() {
    constexpr std::size_t Pages = 2u << 20; // 8 MB of u32: the port's page-use table
    std::unique_ptr<u32[]> table{new u32[Pages]{}};
    std::printf("%-10s %12s %12s %10s\n", "pages", "plain ns", "compare ns", "ratio");
    for (const std::size_t pages : {1, 2, 4, 8, 16, 32, 64, 256, 1024}) {
        const std::size_t binds = std::max<std::size_t>(4096, 400000 / pages);
        double plain = 0;
        double compare = 0;
        for (int rep = 0; rep < 5; ++rep) {
            u32 frame = 1;
            const double t0 = NowNs();
            for (std::size_t i = 0; i < binds; ++i) {
                const std::size_t first = (i * 7919) % (Pages - pages);
                if ((i % 512) == 0) {
                    ++frame; // a new frame: the pages still hold the old one
                }
                VideoCore::PageStamp::Mark(table.get(), first, first + pages - 1, frame, false);
            }
            const double t1 = NowNs();
            for (std::size_t i = 0; i < binds; ++i) {
                const std::size_t first = (i * 7919) % (Pages - pages);
                if ((i % 512) == 0) {
                    ++frame;
                }
                VideoCore::PageStamp::Mark(table.get(), first, first + pages - 1, frame, true);
            }
            const double t2 = NowNs();
            plain += (t1 - t0) / static_cast<double>(binds);
            compare += (t2 - t1) / static_cast<double>(binds);
        }
        std::printf("%-10zu %12.1f %12.1f %9.2fx\n", pages, plain / 5, compare / 5,
                    plain / compare);
    }
    constexpr std::size_t MemoSlots = 1024;       // NoteGpuUse's StampedSlots
    // A frame binds the same buffers over and over: the same ranges, drawn from again and
    // again. Stamp each bind, then the same binds with the range memo the buffer cache keeps per
    // thread (NoteGpuUse) -- a repeat only compares, and does not touch the table at all.
    //
    // Two orders: a frame that binds a batch's buffers again and again (the memo's hits), and
    // one that walks more distinct ranges than the memo has slots (its misses, plus the stamp
    // it would have done anyway, so it can only be a little worse there).
    for (const auto& [label, ranges_count, locality] :
         {std::tuple{"memo batch", std::size_t(64), true},
          std::tuple{"memo mixed", std::size_t(512), false}}) {
        constexpr std::size_t Slots = MemoSlots; // the memo's slots
        constexpr std::size_t Binds = 400000;
        struct Range {
            std::size_t begin, end;
        };
        std::vector<Range> ranges(ranges_count);
        for (std::size_t i = 0; i < ranges_count; ++i) {
            const std::size_t first = (i * 7919) % (Pages - 64);
            ranges[i] = {first, first + 1 + (i % 24)}; // 1 to 24 pages
        }
        std::vector<std::size_t> order(Binds);
        std::size_t current = 0;
        std::uint32_t seed = 12345;
        for (auto& entry : order) {
            seed = seed * 1103515245u + 12345u;
            if (!locality || (seed >> 16) % 16 == 0) {
                current = (seed >> 8) % ranges_count;
            }
            entry = current;
        }
        double plain = 0, memo = 0;
        for (int rep = 0; rep < 5; ++rep) {
            struct Stamped {
                std::size_t begin, end;
                u32 frame;
            };
            Stamped slots[Slots]{};
            u32 frame = 1;
            const double t0 = NowNs();
            for (std::size_t i = 0; i < Binds; ++i) {
                if ((i % 2048) == 0) {
                    ++frame;
                }
                const auto& r = ranges[order[i]];
                VideoCore::PageStamp::Mark(table.get(), r.begin, r.end - 1, frame, true);
            }
            const double t1 = NowNs();
            for (std::size_t i = 0; i < Binds; ++i) {
                if ((i % 2048) == 0) {
                    ++frame;
                }
                const auto& r = ranges[order[i]];
                auto& slot = slots[(r.begin * 0x9E3779B97F4A7C15ull >> 54) & (Slots - 1)];
                if (slot.frame == frame && slot.begin == r.begin && slot.end >= r.end) {
                    continue;
                }
                VideoCore::PageStamp::Mark(table.get(), r.begin, r.end - 1, frame, true);
                if (slot.frame == frame && slot.begin == r.begin) {
                    slot.end = std::max(slot.end, r.end);
                } else {
                    slot = {r.begin, r.end, frame};
                }
            }
            const double t2 = NowNs();
            plain += (t1 - t0) / static_cast<double>(Binds);
            memo += (t2 - t1) / static_cast<double>(Binds);
        }
        std::printf("%-10s %12.1f %12.1f %9.2fx\n", label, plain / 5, memo / 5, plain / memo);
    }
    // The fast path must leave the same table the plain loop does: every page of a range holds
    // its frame. (tests/test_page_stamp.cpp checks this in full; here it keeps the work alive.)
    std::size_t stamped = 0;
    for (std::size_t page = 0; page < Pages; ++page) {
        stamped += table[page] != 0;
    }
    std::printf("%zu of %zu pages stamped\n", stamped, Pages);
    return 0;
}
