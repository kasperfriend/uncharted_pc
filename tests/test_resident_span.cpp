// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: ResidentSpan against the interval list it caches answers for.
//
// BufferCache::EnsureResident asks the interval list of resident block ranges whether a buffer's
// blocks are resident, for every buffer binding. The span a search found gap-free is kept and
// widened over the next binding, so two properties have to hold and are checked here over
// thousands of queries against a list with gaps in it:
//   1. a span never covers a block the list does not: a hit must never skip a needed bind, or the
//      GPU reads memory nothing was made resident for;
//   2. a widened search reports the same gaps inside the query as the plain search, and a widened
//      span published afterwards is gap-free for real.
#include <cstdio>
#include <random>
#include <vector>

#include "common/interval_set.h"

namespace {

/// The blocks of [first, last] that are not resident.
std::vector<std::pair<u64, u64>> Gaps(const IntervalList<Interval>& list, u64 first, u64 last) {
    std::vector<std::pair<u64, u64>> gaps;
    list.ForEachGap(first, last + 1, [&](u64 a, u64 b) { gaps.emplace_back(a, b); });
    return gaps;
}

} // namespace

int main() {
    int failures = 0;
    int checks = 0;
    int hits = 0;
    int widened = 0;
    std::mt19937_64 rng{20261028};

    // A list with gaps: runs of resident blocks separated by holes, the way the buffer cache's
    // list looks while a level is being streamed in.
    IntervalList<Interval> list;
    std::vector<u64> run_starts;
    u64 at = 0;
    for (int i = 0; i < 200; ++i) {
        const u64 length = 4 + (rng() % 96);
        list.Add({at, at + length});
        run_starts.push_back(at);
        at += length + 1 + (rng() % 40);
    }
    const u64 span_end = at;
    ResidentSpan span;
    u64 next_first = run_starts.front();
    u64 previous_first = 0;
    u64 previous_length = 0;
    bool previous_valid = false;
    constexpr u64 CacheId = 0x5151;
    constexpr u64 OtherCacheId = 0x6262;

    for (int i = 0; i < 20000; ++i) {
        // The way bindings arrive: a pass allocates its buffers one after another in the arena,
        // so a query mostly follows the previous one (often touching it, which is what widens the
        // span), sometimes repeats it, and now and then jumps to another point of the list or
        // straddles a hole.
        u64 first;
        u64 length;
        const u64 dice = rng() % 100;
        if (dice < 55 && next_first < span_end) {
            first = next_first;
            length = 1 + (rng() % 16);
            next_first = first + length + (rng() % 3);
        } else if (dice < 70 && previous_valid) {
            first = previous_first;
            length = previous_length;
        } else if (dice < 95) {
            first = run_starts[rng() % run_starts.size()] + (rng() % 8);
            length = 1 + (rng() % 16);
            next_first = first + length + (rng() % 3);
        } else {
            first = rng() % span_end;
            length = 1 + (rng() % 64);
        }
        previous_first = first;
        previous_length = length;
        previous_valid = true;
        const u64 last = first + length - 1;
        const auto reference = Gaps(list, first, last);

        if (ResidentSpanCovers(span, CacheId, first, last)) {
            ++hits;
            ++checks;
            if (!reference.empty()) {
                // A span published for another cache must never be used, and a real hit must be
                // gap-free. Distinguish the two to keep the message useful.
                std::printf("FAIL: span [%llu, %llu] covers [%llu, %llu] with %zu gap(s) in it\n",
                            static_cast<unsigned long long>(span.first),
                            static_cast<unsigned long long>(span.last),
                            static_cast<unsigned long long>(first),
                            static_cast<unsigned long long>(last), reference.size());
                if (++failures > 8) {
                    return 1;
                }
            }
            continue;
        }

        u64 search_first = first;
        u64 search_last = last;
        ResidentSpanWiden(span, CacheId, search_first, search_last);
        if (search_first != first || search_last != last) {
            ++widened;
        }
        ++checks;
        const auto gaps = Gaps(list, search_first, search_last);
        // Property 2a: the widened range adds no gap outside the query.
        for (const auto& [gap_first, gap_last] : gaps) {
            if (gap_first < first || gap_last > last + 1) {
                std::printf("FAIL: widened [%llu, %llu] found gap [%llu, %llu] outside the query\n",
                            static_cast<unsigned long long>(search_first),
                            static_cast<unsigned long long>(search_last),
                            static_cast<unsigned long long>(gap_first),
                            static_cast<unsigned long long>(gap_last));
                if (++failures > 8) {
                    return 1;
                }
            }
        }
        // Property 2b: the gaps inside the query are exactly the plain search's.
        std::vector<std::pair<u64, u64>> inside;
        for (const auto& gap : gaps) {
            inside.emplace_back(std::max<u64>(gap.first, first), std::min<u64>(gap.second, last + 1));
        }
        if (inside != reference) {
            std::printf("FAIL: query [%llu, %llu] search differs from the plain one\n",
                        static_cast<unsigned long long>(first),
                        static_cast<unsigned long long>(last));
            if (++failures > 8) {
                return 1;
            }
        }
        if (gaps.empty()) {
            ResidentSpanPublish(span, CacheId, search_first, search_last);
            if (!Gaps(list, span.first, span.last).empty()) {
                std::printf("FAIL: published [%llu, %llu] has a gap in it\n",
                            static_cast<unsigned long long>(span.first),
                            static_cast<unsigned long long>(span.last));
                if (++failures > 8) {
                    return 1;
                }
            }
        }
        // A span published for another cache is ignored, in both directions.
        u64 other_first = first;
        u64 other_last = last;
        ResidentSpanWiden(span, OtherCacheId, other_first, other_last);
        if (other_first != first || other_last != last) {
            std::printf("FAIL: a span was used for another cache id\n");
            ++failures;
        }
    }

    // Newly resident blocks never invalidate a span: adding an overlapping run keeps it gap-free.
    for (int i = 0; i < 2000; ++i) {
        const u64 start = rng() % span_end;
        list.Add({start, start + 1 + (rng() % 64)});
    }
    if (!Gaps(list, span.first, span.last).empty() && span.first <= span.last) {
        std::printf("FAIL: a published span stopped being gap-free after more ranges were added\n");
        ++failures;
    }
    if (failures != 0) {
        std::printf("%d of %d resident-span checks failed\n", failures, checks);
        return 1;
    }
    std::printf("resident span: %d checks passed (%d hits, %d widened)\n", checks, hits, widened);
    return 0;
}
