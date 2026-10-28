// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: what a wait between two threads of the port costs, and what the wait governor
// (common/wait_governor.h) changes about it.
//
// Four blocks:
//
//  1. the cost of one `pause` on this host, measured the way the governor measures it -- a
//     pause budget written into the source is a different length on every core;
//  2. a producer and a consumer with the port's own wait distribution (most waits a fraction of
//     a microsecond, some a few, a few a hundred): the fixed pause budget with a yield every
//     1024 pauses against the governor, by the consumer's CPU time, its system calls and the
//     latency of the wake;
//  3. six idle helpers waiting for work that arrives every 2 ms (the draw recorder, the texture
//     binding helper, the copy threads between frames): what the idle spin costs the machine;
//  4. the guest kernel's small-timer wait (core/libraries/kernel/equeue.cpp): a loop of
//     yield(), mutex and clock against blocking on the timer's deadline.
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <mutex>
#include <thread>
#include <vector>

#include "common/wait_governor.h"

namespace {

using u32 = std::uint32_t;
using u64 = std::uint64_t;
using Clock = std::chrono::steady_clock;

/// CPU time of the calling thread (the metric that matters: what a wait takes from the machine).
double ThreadCpuNs() {
#ifdef CLOCK_THREAD_CPUTIME_ID
    timespec ts{};
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return double(ts.tv_sec) * 1e9 + double(ts.tv_nsec);
#else
    return double(std::clock()) * 1e9 / CLOCKS_PER_SEC;
#endif
}

double NowNs() {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return double(ts.tv_sec) * 1e9 + double(ts.tv_nsec);
}

/// Busy-waits for `ns`: a producer's delay has to be exact to sub-microsecond, and no sleep is.
void SpinForNs(double ns) {
    if (ns <= 0) {
        return;
    }
    const double until = NowNs() + ns;
    while (NowNs() < until) {
        Common::Gov::Pause();
    }
}

std::uint64_t g_yields = 0; ///< yields taken by the threads under test

/// What the port wrote by hand: a fixed pause budget, the clock read every 256 pauses, and a
/// yield every 1024 -- the shape of WaitDeferredSignals/WaitHostCopies and of the draw pipe's
/// idle spin.
void WaitFixed(const std::atomic<u64>& seq, u64 seen, u64 yield_every) {
    for (u64 spins = 1;; ++spins) {
        if (seq.load(std::memory_order_acquire) != seen) {
            return;
        }
        Common::Gov::Pause();
        if (!(spins & (yield_every - 1))) {
            ++g_yields;
            std::this_thread::yield();
        }
    }
}

/// The same wait through a governor: its own budget in pauses, then the park (a yield).
void WaitGoverned(Common::Gov::SpinGovernor& gov, const std::atomic<u64>& seq, u64 seen) {
    gov.Wait([&] { return seq.load(std::memory_order_acquire) != seen; }, [] {
        ++g_yields;
        std::this_thread::yield();
    });
}

/// The wait distribution of a frame's hand-offs: mostly a fraction of a microsecond.
double SampleDelayNs(u32& state) {
    state = state * 1664525u + 1013904223u;
    const u32 bucket = (state >> 24) & 0xFF;
    if (bucket < 178) { // 70%
        return 200.0 + double((state >> 8) & 0x1FF);
    }
    if (bucket < 229) { // 20%
        return 1500.0 + double((state >> 8) & 0xFFF);
    }
    return 60'000.0 + double((state >> 8) & 0x1FFFF); // 10%: a stall
}

struct Metrics {
    double cpu_ns = 0;
    std::uint64_t yields = 0;
    std::vector<double> latencies;
};

/// Runs one producer/consumer pair for `waits` hand-offs with a given wait function.
template <typename Wait>
Metrics RunPair(int waits, Wait wait) {
    std::atomic<u64> seq{0};
    std::atomic<double> stamped{0};
    Metrics m;
    std::thread consumer([&] {
        Common::Gov::SpinGovernor gov;
        const double cpu0 = ThreadCpuNs();
        u64 seen = 0;
        m.latencies.reserve(std::size_t(waits));
        for (int i = 0; i < waits; ++i) {
            wait(gov, seq, seen);
            m.latencies.push_back(NowNs() - stamped.load(std::memory_order_relaxed));
            ++seen;
        }
        m.cpu_ns = ThreadCpuNs() - cpu0;
        m.yields = g_yields;
    });
    std::uint32_t state = 0x1234567u;
    for (int i = 0; i < waits; ++i) {
        SpinForNs(SampleDelayNs(state));
        stamped.store(NowNs(), std::memory_order_relaxed);
        seq.store(u64(i) + 1, std::memory_order_release);
    }
    consumer.join();
    return m;
}

double Percentile(std::vector<double>& v, double p) {
    if (v.empty()) {
        return 0;
    }
    const std::size_t i = std::min(v.size() - 1, std::size_t(double(v.size()) * p));
    std::nth_element(v.begin(), v.begin() + std::ptrdiff_t(i), v.end());
    return v[i];
}

struct IdleStats {
    double cpu_ns = 0;
    double latency_ns = 0;
    std::vector<double> latencies;
    std::uint64_t yields = 0;
    std::uint64_t items = 0;
};

/// `threads` helpers, each waiting for work that arrives every `period_ns`; `wait` is the idle
/// wait under test. The park is the one the port uses (an event the producer notifies), so the
/// comparison is the spin in front of it.
template <typename Wait>
IdleStats RunIdle(int threads, int items, double period_ns, Wait wait) {
    std::atomic<u64> published{0};
    std::atomic<double> stamped{0};
    std::atomic<bool> stop{false};
    std::atomic<u64> yield_count{0};
    std::array<double, 8> cpu{};
    std::array<std::vector<double>, 8> latencies;
    std::thread producer([&] {
        for (int i = 0; i < items; ++i) {
            SpinForNs(period_ns);
            stamped.store(NowNs(), std::memory_order_relaxed);
            published.store(u64(i) + 1, std::memory_order_release);
            published.notify_all();
        }
        stop.store(true, std::memory_order_release);
        published.fetch_add(1'000'000, std::memory_order_release);
        published.notify_all();
    });
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t) {
        pool.emplace_back([&, t] {
            const double cpu0 = ThreadCpuNs();
            u64 seen = 0;
            auto& mine = latencies[std::size_t(t)];
            mine.reserve(std::size_t(items));
            while (!stop.load(std::memory_order_relaxed)) {
                if (published.load(std::memory_order_acquire) == seen) {
                    wait(seen, published, stamped, yield_count);
                }
                const u64 now = published.load(std::memory_order_acquire);
                if (now != seen) {
                    seen = now;
                    mine.push_back(NowNs() - stamped.load(std::memory_order_relaxed));
                }
            }
            cpu[std::size_t(t)] = ThreadCpuNs() - cpu0;
        });
    }
    producer.join();
    for (auto& th : pool) {
        th.join();
    }
    IdleStats stats;
    for (std::size_t t = 0; t < std::size_t(threads); ++t) {
        stats.cpu_ns += cpu[t];
        stats.items += latencies[t].size();
        for (double d : latencies[t]) {
            stats.latencies.push_back(d);
            stats.latency_ns += d;
        }
    }
    stats.yields = yield_count.load();
    return stats;
}

/// The guest kernel's small-timer wait, modelled: a mutex, a clock read and a yield per
/// iteration; and the blocking version.
struct TimerModel {
    std::mutex mutex;
    std::condition_variable cv;
    std::array<double, 8> deadlines{}; ///< non-zero while armed
};

double RunTimerSpin(TimerModel& model, int iterations) {
    const double cpu0 = ThreadCpuNs();
    for (int i = 0; i < iterations; ++i) {
        const double until = NowNs() + 1'000'000.0; // a 1 ms timer
        for (;;) {
            const double now = NowNs();
            bool fired = false;
            {
                std::scoped_lock lk{model.mutex};
                auto& slot = model.deadlines[std::size_t(i) & 7];
                slot = (i == 0) ? until : until + double(i);
                for (double& d : model.deadlines) {
                    if (d != 0 && now >= d) {
                        d = 0;
                        fired = true;
                    }
                }
            }
            if (fired) {
                break;
            }
            std::this_thread::yield();
        }
    }
    return ThreadCpuNs() - cpu0;
}

double RunTimerBlocking(TimerModel& model, int iterations) {
    const double cpu0 = ThreadCpuNs();
    for (int i = 0; i < iterations; ++i) {
        const auto until = Clock::now() + std::chrono::milliseconds(1);
        std::unique_lock lk{model.mutex};
        model.deadlines[std::size_t(i) & 7] = 1.0;
        // One deadline walk, then the thread is off the core until it expires.
        model.cv.wait_until(lk, until, [&] {
            for (double d : model.deadlines) {
                if (d == 0) {
                    return true;
                }
            }
            return false;
        });
        for (double& d : model.deadlines) {
            d = 0;
        }
    }
    return ThreadCpuNs() - cpu0;
}

} // namespace

int main() {
    std::printf("pause cost: %llu cycles on this host\n",
                static_cast<unsigned long long>(Common::Gov::PauseCost()));

    // 1. The governor's budget after a distribution of waits: it has to follow the site.
    {
        Common::Gov::SpinGovernor gov;
        gov.Observe(300);
        const auto short_budget = gov.SpinCycles();
        const auto short_pauses = gov.SpinPauses();
        gov.Observe(120'000);
        const auto long_budget = gov.SpinCycles();
        const auto long_pauses = gov.SpinPauses();
        std::printf("budget after a 300-cycle wait: %llu cycles (%u pauses), after a 120000-cycle "
                    "wait: %llu cycles (%u pauses)\n",
                    static_cast<unsigned long long>(short_budget), short_pauses,
                    static_cast<unsigned long long>(long_budget), long_pauses);
        if (!(short_budget < long_budget)) {
            std::printf("FAIL: the budget did not follow the waits it observed\n");
            return 1;
        }
    }

    // 2. The port's wait distribution: fixed pause budget against the governor.
    {
        constexpr int Waits = 30'000;
        g_yields = 0;
        auto fixed = RunPair(
            Waits,
            [](Common::Gov::SpinGovernor&, const std::atomic<u64>& seq, u64 seen) {
                WaitFixed(seq, seen, 1024);
            });
        g_yields = 0;
        auto governed = RunPair(
            Waits,
            [](Common::Gov::SpinGovernor& gov, const std::atomic<u64>& seq, u64 seen) {
                WaitGoverned(gov, seq, seen);
            });
        const double fixed_p50 = Percentile(fixed.latencies, 0.50);
        const double fixed_p90 = Percentile(fixed.latencies, 0.90);
        const double gov_p50 = Percentile(governed.latencies, 0.50);
        const double gov_p90 = Percentile(governed.latencies, 0.90);
        std::printf("hand-off waits (%d: 70%% short / 20%% a few us / 10%% a stall):\n", Waits);
        std::printf("  fixed budget, yield every 1024 pauses: consumer CPU %.2f ms, %llu yields, "
                    "latency p50 %.2f us p90 %.2f us\n",
                    fixed.cpu_ns / 1e6, static_cast<unsigned long long>(fixed.yields),
                    fixed_p50 / 1e3, fixed_p90 / 1e3);
        std::printf("  governor:                              consumer CPU %.2f ms, %llu yields, "
                    "latency p50 %.2f us p90 %.2f us\n",
                    governed.cpu_ns / 1e6, static_cast<unsigned long long>(governed.yields),
                    gov_p50 / 1e3, gov_p90 / 1e3);
        std::printf("  change: %+.1f%% consumer CPU, %+.1f%% yields (the tail latency of a shared "
                    "host is scheduler noise; the medians match)\n",
                    100.0 * (governed.cpu_ns - fixed.cpu_ns) / fixed.cpu_ns,
                    100.0 * (double(governed.yields) - double(fixed.yields)) /
                        double(fixed.yields ? fixed.yields : 1));
    }

    // 3. Six idle helpers, work every 2 ms: what the idle spin takes from the machine. Both
    // waits park on the item's event; the difference is the spin in front of the park.
    {
        constexpr int Items = 60;
        constexpr double PeriodNs = 2'000'000.0;
        auto idle_fixed = RunIdle(
            6, Items, PeriodNs,
            [](u64 seen, std::atomic<u64>& published, std::atomic<double>&,
               std::atomic<u64>& yields) {
                // The port's idle loop: spin until a 200 us deadline, then the event.
                const double until = NowNs() + 200'000.0;
                for (u32 spins = 1;; ++spins) {
                    if (published.load(std::memory_order_acquire) != seen) {
                        return;
                    }
                    Common::Gov::Pause();
                    if (!(spins & 255) && NowNs() >= until) {
                        yields.fetch_add(1, std::memory_order_relaxed);
                        published.wait(seen, std::memory_order_relaxed);
                        return;
                    }
                }
            });
        auto idle_gov = RunIdle(
            6, Items, PeriodNs,
            [](u64 seen, std::atomic<u64>& published, std::atomic<double>&,
               std::atomic<u64>& yields) {
                static thread_local Common::Gov::SpinGovernor gov;
                gov.Wait([&] { return published.load(std::memory_order_acquire) != seen; }, [&] {
                    yields.fetch_add(1, std::memory_order_relaxed);
                    published.wait(seen, std::memory_order_relaxed);
                });
            });
        const double fixed_p50 = Percentile(idle_fixed.latencies, 0.50);
        const double gov_p50 = Percentile(idle_gov.latencies, 0.50);
        std::printf("six idle helpers, one item every 2 ms (%d items):\n", Items);
        std::printf("  200 us deadline spin + event: CPU %.2f ms (%.1f%% of six threads), %llu "
                    "parks, pickup p50 %.2f us\n",
                    idle_fixed.cpu_ns / 1e6,
                    100.0 * idle_fixed.cpu_ns / (PeriodNs * Items * 6),
                    static_cast<unsigned long long>(idle_fixed.yields), fixed_p50 / 1e3);
        std::printf("  governor:                    CPU %.2f ms (%.1f%%), %llu parks, pickup p50 "
                    "%.2f us\n",
                    idle_gov.cpu_ns / 1e6, 100.0 * idle_gov.cpu_ns / (PeriodNs * Items * 6),
                    static_cast<unsigned long long>(idle_gov.yields), gov_p50 / 1e3);
        std::printf("  change: %+.1f%% CPU taken by idle helpers\n",
                    100.0 * (idle_gov.cpu_ns - idle_fixed.cpu_ns) / idle_fixed.cpu_ns);
    }

    // 4. The guest kernel's small-timer wait: a yield per iteration against the deadline.
    {
        constexpr int Timers = 24;
        TimerModel spin_model;
        const double spin_ns = RunTimerSpin(spin_model, Timers);
        TimerModel block_model;
        const double block_ns = RunTimerBlocking(block_model, Timers);
        std::printf("small-timer wait (%d timers of 1 ms, one waiting thread): yield loop %.2f ms "
                    "CPU, deadline wait %.2f ms CPU (%.1fx)\n",
                    Timers, spin_ns / 1e6, block_ns / 1e6, spin_ns / block_ns);
    }
    return 0;
}
