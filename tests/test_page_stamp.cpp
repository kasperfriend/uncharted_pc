// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: VideoCore::PageStamp against the plain store loop it replaces.
//
// The buffer cache stamps the frame table of the GPU heap window for every buffer binding
// (NoteGpuUse). The fast path compares 8 pages at a time and only stores where the frame
// differs, so its answer has to be identical to writing every page: same pages, same values,
// nothing outside the range touched. Checked here over every range length up to three steps past
// the compare threshold, every starting offset that straddles a step, both toggle states and
// random frames -- including frames already present, which are the runs the fast path skips.
#include <cstdio>
#include <random>
#include <vector>

#include "video_core/buffer_cache/page_stamp.h"

namespace {

using VideoCore::PageStamp::CompareMinPages;
using VideoCore::PageStamp::PagesPerStep;

bool Check(int& failures, const char* what, std::size_t first, std::size_t last, u32 frame,
           bool compare) {
    constexpr std::size_t Extra = 4;
    const std::size_t pages = 4096;
    // A table pre-filled with a different frame, so every page of the range needs the store on
    // the first call and none on the second.
    std::vector<u32> got(pages, 0xDEADBEEF);
    std::vector<u32> want = got;
    for (std::size_t page = first; page <= last; ++page) {
        want[page] = frame;
    }
    const std::size_t marked = VideoCore::PageStamp::Mark(got.data(), first, last, frame, compare);
    if (marked != last - first + 1) {
        std::printf("FAIL %s: [%zu, %zu] marked %zu pages\n", what, first, last, marked);
        return false;
    }
    for (std::size_t page = 0; page < pages; ++page) {
        if (got[page] != want[page]) {
            if (++failures < 8) {
                std::printf("FAIL %s: [%zu, %zu] frame %u compare %d: page %zu is %u, want %u\n",
                            what, first, last, frame, compare, page, got[page], want[page]);
            }
            return false;
        }
    }
    (void)Extra;
    return true;
}

int Run() {
    int failures = 0;
    int checks = 0;
    // Empty and inverted ranges do nothing.
    {
        std::vector<u32> table(64, 7);
        if (VideoCore::PageStamp::Mark(table.data(), 10, 9, 3) != 0) {
            std::printf("FAIL: an empty range marked pages\n");
            ++failures;
        }
        ++checks;
        if (table[9] != 7 || table[10] != 7) {
            std::printf("FAIL: an empty range wrote outside the range\n");
            ++failures;
        }
    }
    // Every length around the compare threshold, from every offset that straddles a step, so
    // the 8-page groups, their tails and the short-range path are all covered.
    for (std::size_t length = 1; length <= CompareMinPages + 3 * PagesPerStep; ++length) {
        for (std::size_t offset = 0; offset < PagesPerStep + 2; ++offset) {
            for (const bool compare : {false, true}) {
                const std::size_t first = 64 + offset;
                ++checks;
                failures += !Check(failures, "sweep", first, first + length - 1,
                                   static_cast<u32>(length * 7 + 1), compare);
            }
        }
    }
    // Random ranges over random frames, half of them already stamped (the skip path).
    std::mt19937 rng{20261028};
    for (int i = 0; i < 2000; ++i) {
        const std::size_t first = rng() % 3000;
        const std::size_t length = 1 + rng() % 700;
        const u32 frame = static_cast<u32>(1 + rng() % 100000);
        const bool compare = (rng() & 1) != 0;
        std::vector<u32> table(4096, frame); // already stamped: nothing must change
        if (VideoCore::PageStamp::Mark(table.data(), first, first + length - 1, frame, compare) !=
            length) {
            std::printf("FAIL random: [%zu, +%zu] marked the wrong page count\n", first, length);
            ++failures;
        }
        for (std::size_t page = 0; page < table.size(); ++page) {
            if (table[page] != frame) {
                std::printf("FAIL random: [%zu, +%zu] page %zu left at %u\n", first, length, page,
                            table[page]);
                ++failures;
                break;
            }
        }
        ++checks;
        failures += !Check(failures, "random", first, first + length - 1, frame, compare);
    }
    if (failures != 0) {
        std::printf("%d of %d page-stamp checks failed\n", failures, checks);
        return 1;
    }
    std::printf("page stamp: %d checks passed (compare threshold %zu, %zu pages per step)\n",
                checks, CompareMinPages, PagesPerStep);
    return 0;
}

} // namespace

int main() {
    return Run();
}
