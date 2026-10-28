// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: the residency search BufferCache::EnsureResident runs for every buffer binding, against
// the per-thread span it now keeps (BufferCache::EnsureResident).
//
// The search bisects the port's resident ranges -- the buffer cache adds one whenever a block
// range is made resident, and nothing ever removes them, so after a few minutes of play there are
// hundreds spread over dozens of KiB. The span a successful search proved resident is kept, and a
// later binding inside it (the next draw's bind of the same buffer, or one of its neighbours in
// the arena, since the span is widened over them) is a compare.
#include <cstdio>
#include <ctime>
#include <random>
#include <vector>

#include "common/interval_set.h"

namespace {

double NowNs() {
    timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return static_cast<double>(t.tv_sec) * 1e9 + static_cast<double>(t.tv_nsec);
}

} // namespace

int main() {
    std::mt19937_64 rng{1234};
    std::printf("%-8s %12s %12s %12s %10s\n", "ranges", "search warm", "search cold", "span hit",
                "ratio");
    for (const std::size_t ranges : {64, 256, 1024, 4096, 65536}) {
        IntervalList<Interval> resident;
        std::vector<u64> run_firsts;
        u64 at = 0;
        for (std::size_t i = 0; i < ranges; ++i) {
            const u64 length = 1 + (rng() % 64);
            resident.Add({at, at + length});
            run_firsts.push_back(at);
            at += length + 1 + (rng() % 32);
        }
        // The buffers a pass binds live in one made-resident stretch (an arena chunk); that is
        // the span a successful search hands to the hint.
        const u64 span_first = at + 1;
        const u64 span_blocks = 65536;
        resident.Add({span_first, span_first + span_blocks});
        const u64 span_last = span_first + span_blocks;
        constexpr std::size_t Queries = 200000;
        std::vector<u64> warm;
        std::vector<u64> cold;
        warm.reserve(Queries);
        cold.reserve(Queries);
        for (std::size_t i = 0; i < Queries; ++i) {
            // Warm: a binding of the pass, inside its residents.
            warm.push_back(span_first + rng() % (span_blocks - 16));
            // Cold: the same kind of binding somewhere else in the list (another pass), so the
            // search lands on ranges the last one did not touch.
            cold.push_back(run_firsts[rng() % run_firsts.size()] + rng() % 8);
        }
        u64 gaps = 0;
        const auto search = [&](const std::vector<u64>& queries) {
            gaps = 0;
            const double t0 = NowNs();
            for (const u64 first : queries) {
                resident.ForEachGap(first, first + 16, [&](u64 a, u64 b) { gaps += b - a; });
            }
            return (NowNs() - t0) / Queries;
        };
        const double warm_ns = search(warm);
        const double cold_ns = search(cold);
        u64 hits = 0;
        const double t2 = NowNs();
        for (const u64 first : warm) {
            hits += first >= span_first && first + 16 <= span_last;
        }
        const double hint_ns = (NowNs() - t2) / Queries;
        std::printf("%-8zu %12.2f %12.2f %12.2f %9.1fx\n", ranges, warm_ns, cold_ns, hint_ns,
                    cold_ns / hint_ns);
        if (hits != Queries) {
            std::printf("  (unexpected: %llu of %zu warm bindings covered by the span)\n",
                        static_cast<unsigned long long>(hits), Queries);
            return 1;
        }
    }
    return 0;
}
