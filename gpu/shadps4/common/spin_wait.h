// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: waiting for another thread of this process without a system call.
//
// The GPU command thread, the draw-pipeline threads and the copy pool hand work to each other
// many times per frame and wait for it for microseconds. Asking the scheduler for another
// thread (`std::this_thread::yield`, `SwitchToThread`, `sched_yield`) for such a wait costs a
// kernel transition of a few hundred cycles and, worse, gives the core away to a thread that
// may run for milliseconds -- the woken thread itself has to be scheduled back in. Spinning on
// the hardware's own hint (`pause`) keeps the core, lets the sibling hyperthread run, and the
// wait ends as soon as the other thread's store lands in the cache.
#pragma once

#include <thread>
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#include <immintrin.h>
#endif

namespace Common {

/// One processor hint: `pause` on x86 (a handful of cycles on the cores the port targets,
/// exactly what the kernel's own spin locks use), a plain yield elsewhere.
inline void CpuRelax() {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    _mm_pause();
#elif (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
    __builtin_ia32_pause();
#else
    std::this_thread::yield();
#endif
}

/// Waits until `done()` is true: spins with `pause`, and only asks the scheduler for another
/// thread every `yield_every` iterations, so a short wait costs no system call at all and a
/// long one still leaves the core to the game. `done()` is polled only after a pause.
template <typename Predicate>
inline void SpinWait(Predicate done, unsigned yield_every = 1024) {
    unsigned iterations = 0;
    while (!done()) {
        CpuRelax();
        if (++iterations >= yield_every) {
            iterations = 0;
            std::this_thread::yield();
        }
    }
}

/// Same, for a wait that is expected to be very short (a few hundred cycles): pure spinning
/// bounded by `max_pauses`, then a real yield. Returns false when the wait outlived the spins.
template <typename Predicate>
inline bool SpinWaitShort(Predicate done, unsigned max_pauses = 4096) {
    for (unsigned i = 0; i < max_pauses; ++i) {
        if (done()) {
            return true;
        }
        CpuRelax();
    }
    return done();
}

} // namespace Common
