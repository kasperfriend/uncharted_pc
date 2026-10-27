// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: cost of stamping the buffer cache's page-use table, plain stores against the
// compare-and-skip path of VideoCore::PageStamp (NoteGpuUse runs this for every buffer binding).
//
// The table is the port's (8 MB, one 4 KiB page each over the GPU heap window); the ranges are
// the binds of a frame -- the same pages are bound again by the next draws, which is what the
// compare path skips. Times are per call on this machine; the ratio is what carries over.
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
    // The fast path must leave the same table the plain loop does: every page of a range holds
    // its frame. (tests/test_page_stamp.cpp checks this in full; here it keeps the work alive.)
    std::size_t stamped = 0;
    for (std::size_t page = 0; page < Pages; ++page) {
        stamped += table[page] != 0;
    }
    std::printf("%zu of %zu pages stamped\n", stamped, Pages);
    return 0;
}
