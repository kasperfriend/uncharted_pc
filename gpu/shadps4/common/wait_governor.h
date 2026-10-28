// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: the wait governor (docs/PERFORMANCE.md, "Waiting between the port's threads").
//
// The port's threads hand work to each other many times per frame and wait for it for
// microseconds: the GPU thread for the draw recorder, the recorder for its next packet, the
// copy pool's callers for their copies. A wait that ends inside a spin costs a handful of
// cycles; one that ends in a system call costs a kernel transition and, on the GPU thread, a
// reschedule and a wait to be scheduled in again.
//
// How long to spin is not a constant:
//
//  - `pause` costs about five cycles on some cores and about a hundred and forty on others, so
//    a pause budget written into the source spins thirty times longer on one machine than on
//    another (4096 pauses are 7 microseconds or 190). The governor measures the cost once.
//  - the waits of one site are microseconds (a draw packet) and of another milliseconds (the
//    next frame), and both move with the scene, so a fixed budget either parks on every wait or
//    spins through the idle time. The governor measures the site's own waits and spins for
//    about three of them.
//  - yields are the expensive way out of a spin: every one of them is a system call, and a
//    site that yields every 1024 pauses (1.7 microseconds here) asks the scheduler hundreds of
//    times for a wait that outlives it by two orders of magnitude.
//
// So a governor spins on `pause` for its own budget -- no clock read, no system call -- and only
// a wait that outlives that budget reaches its park (a yield, an event, a sleep). The budget is
// re-derived from every wait the site completes, inside a clamp that keeps the governor from
// spinning through a frame's idle time (200 microseconds) or from parking on a wait that a spin
// would have covered (5 microseconds).

#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <thread>

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#include <immintrin.h>
#endif

namespace Common::Gov {

/// One processor hint: `pause` on x86 (the name the rest of the port uses), a yield elsewhere.
inline void Pause() noexcept {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    _mm_pause();
#elif (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
    __builtin_ia32_pause();
#else
    std::this_thread::yield();
#endif
}

/// The governor's clock: TSC cycles where there is one (a register read, no call), nanoseconds
/// where there is not. Only differences are used, so the epoch and the frequency do not matter.
inline std::uint64_t Ticks() noexcept {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    return __rdtsc();
#elif defined(__x86_64__) || defined(__i386__)
    return __builtin_ia32_rdtsc();
#else
    return static_cast<std::uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
#endif
}

/// Cycles one `pause` costs on this host, measured once on first use: a loop of pauses against
/// an empty loop of the same length, the best of three runs (a preemption only makes a run
/// longer, so the minimum is the measurement).
inline std::uint64_t PauseCost() noexcept {
    static const std::uint64_t cost = []() -> std::uint64_t {
        constexpr unsigned Count = 2048;
        for (unsigned i = 0; i < Count; ++i) {
            Pause();
        }
        std::uint64_t best = ~std::uint64_t{0};
        for (int rep = 0; rep < 3; ++rep) {
            const auto start = Ticks();
            for (unsigned i = 0; i < Count; ++i) {
                Pause();
            }
            const auto done = Ticks();
            best = std::min(best, (done - start) / Count);
        }
        return std::max<std::uint64_t>(1, best);
    }();
    return cost;
}

/// The clamp the budget lives in, in cycles: a spin of about 5 microseconds at the bottom (below
/// that a park costs more than the wait it shortens) and about 200 at the top (above that the
/// core is better spent on the thread the wait is for, or on the game's own threads).
inline constexpr std::uint64_t MinSpinCycles = 16'000;
inline constexpr std::uint64_t MaxSpinCycles = 600'000;

/// One wait site's spin budget, learned from the waits that site completes. Belongs to a thread
/// (the sites are single-threaded waits); a template's function-local static, or a member.
class SpinGovernor {
public:
    /// seed_cycles: the budget until the site has completed its first wait.
    explicit SpinGovernor(std::uint64_t seed_cycles = 32'000) noexcept {
        SetBudget(seed_cycles);
    }

    /// Cycles this site spins before it parks.
    [[nodiscard]] std::uint64_t SpinCycles() const noexcept {
        return spin_cycles;
    }

    /// Pauses this site spins before it parks (the budget in pause instructions).
    [[nodiscard]] std::uint32_t SpinPauses() const noexcept {
        return budget_pauses;
    }

    /// True when the site is idle (its waits keep reaching their park).
    [[nodiscard]] bool Idle() const noexcept {
        return idle;
    }

    /// Waits this site has completed, and how many of them reached their park.
    [[nodiscard]] std::uint64_t Waits() const noexcept {
        return waits;
    }
    [[nodiscard]] std::uint64_t Parks() const noexcept {
        return parks;
    }

    /// Waits until done() is true: spins for about three of this site's waits, then calls
    /// park() (a yield, an event wait, a sleep -- whatever the site has) and spins again for a
    /// quarter of that, so a wait that is much longer than the site's own does not cost a
    /// system call per budget. The completed wait re-derives the budget.
    template <typename Done, typename Park>
    void Wait(Done done, Park park) {
        const std::uint64_t start = Ticks();
        std::uint32_t spent = 0;
        std::uint32_t limit = budget_pauses;
        while (!done()) {
            // The predicate is checked after every pause while a wait is short, and every 32
            // while it lasts: the check is a load of a shared cache line.
            const std::uint32_t stride = spent < 32 ? 1 : (spent < 256 ? 8 : 32);
            for (std::uint32_t i = 0; i < stride; ++i) {
                Pause();
            }
            spent += stride;
            if (done()) {
                break;
            }
            if (spent >= limit) {
                ++parks;
                parked_this_wait = true;
                park();
                spent = 0;
                limit = post_park_pauses;
            }
        }
        Observe(Ticks() - start);
    }

    /// Like Wait(), for a site whose park decides whether the spin is over (an idle loop with
    /// its own deadline): `park` returns true to end the wait -- the site then blocks on its
    /// event itself -- and false to keep spinning. The clock is therefore read once per
    /// budget, not once per iteration.
    template <typename Done, typename Park>
    void WaitUntilParked(Done done, Park park) {
        const std::uint64_t start = Ticks();
        std::uint32_t spent = 0;
        std::uint32_t limit = budget_pauses;
        while (!done()) {
            const std::uint32_t stride = spent < 32 ? 1 : (spent < 256 ? 8 : 32);
            for (std::uint32_t i = 0; i < stride; ++i) {
                Pause();
            }
            spent += stride;
            if (done()) {
                break;
            }
            if (spent >= limit) {
                ++parks;
                parked_this_wait = true;
                if (park()) {
                    break;
                }
                spent = 0;
                limit = post_park_pauses;
            }
        }
        Observe(Ticks() - start);
    }

    /// Spins only, for at most `max_pauses`: true when done() became true inside them. For a
    /// wait that must not yield (a lock, a hand-off the waiter itself makes progress on).
    template <typename Done>
    bool Spin(Done done, std::uint32_t max_pauses = 0) {
        const std::uint64_t start = Ticks();
        if (max_pauses == 0) {
            max_pauses = budget_pauses;
        }
        for (std::uint32_t spent = 0;;) {
            if (done()) {
                Observe(Ticks() - start);
                return true;
            }
            if (spent >= max_pauses) {
                break;
            }
            const std::uint32_t stride = spent < 32 ? 1 : (spent < 256 ? 8 : 32);
            for (std::uint32_t i = 0; i < stride; ++i) {
                Pause();
            }
            spent += stride;
        }
        Observe(Ticks() - start);
        return done();
    }

    /// Records one completed wait and re-derives the budget: three of the site's own waits,
    /// inside the clamp, in pauses of this host's measured cost. A site whose waits keep
    /// reaching their park is an idle one -- the draw recorder between frames, the binding
    /// helper between draws -- and spins only a fraction of its waits' length, because spinning
    /// through idle time takes a core from the game's own threads and shortens nothing.
    void Observe(std::uint64_t cycles) noexcept {
        ++waits;
        park_ratio -= park_ratio / 8;
        if (parked_this_wait) {
            park_ratio += 65536 / 8;
        }
        parked_this_wait = false;
        // Grow fast (a park that should have been a spin shows up on the next wait), shrink
        // slowly (one frame's outlier must not park the next frame's short waits).
        const std::uint64_t sample = std::min(cycles, MaxSpinCycles * 4);
        if (sample > ewma_cycles) {
            ewma_cycles += (sample - ewma_cycles + 1) / 2;
        } else {
            ewma_cycles -= (ewma_cycles - sample) / 8;
        }
        idle = waits >= 16 && park_ratio > 58'982; // 90% of the recent waits parked
        SetBudget(ewma_cycles * 3);
    }

    /// Pauses that fit in `cycles` on this host, clamped to the budget range.
    static std::uint32_t PausesFor(std::uint64_t cycles) noexcept {
        const std::uint64_t pauses = std::clamp(cycles, MinSpinCycles, MaxSpinCycles) / PauseCost();
        return static_cast<std::uint32_t>(std::clamp<std::uint64_t>(pauses, 8, 1u << 20));
    }

private:
    void SetBudget(std::uint64_t cycles) noexcept {
        spin_cycles = std::clamp(cycles, MinSpinCycles, MaxSpinCycles);
        if (idle) {
            spin_cycles = MinSpinCycles;
        }
        budget_pauses = PausesFor(spin_cycles);
        // After a park the state the wait was for is usually about to change: a short spin
        // catches the wake, and a wait that is still longer goes back to the event quickly.
        post_park_pauses = std::max<std::uint32_t>(8, budget_pauses / 4);
    }

    std::uint64_t ewma_cycles = 32'000;
    std::uint64_t spin_cycles = 32'000;
    std::uint32_t budget_pauses = 0;
    std::uint32_t post_park_pauses = 0;
    std::uint32_t park_ratio = 0; ///< 0..65536: how many of the recent waits parked
    std::uint64_t waits = 0;
    std::uint64_t parks = 0;
    bool parked_this_wait = false;
    bool idle = false;
};

/// The governor of a wait loop written as a template: one per call site and thread.
template <std::uint64_t SeedCycles = 32'000>
inline SpinGovernor& SiteGovernor() {
    static thread_local SpinGovernor governor{SeedCycles};
    return governor;
}

/// Spins and then yields: the park the sites used to hand-roll (a yield every 1024 pauses).
inline void ParkYield() {
    std::this_thread::yield();
}

/// Sleeps for a wait that has already outlived its spin many times over (idle loops): the
/// caller's budget is microseconds and the event is a frame away. Never sleeps longer than
/// 1 ms, so a wake that comes early costs at most that much latency.
class PatientPark {
public:
    template <typename... Args>
    void operator()(Args&&...) {
        ++rounds;
        const unsigned shift = std::min(rounds, 5u);
        std::this_thread::sleep_for(std::chrono::microseconds(25u << shift));
    }

private:
    unsigned rounds = 0;
};

} // namespace Common::Gov
