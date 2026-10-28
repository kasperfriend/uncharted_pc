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
//
// How long to spin is what common/wait_governor.h answers: the cost of a `pause` differs by a
// factor of thirty between cores and so does the length of the waits a site sees, so the
// governors here learn both and hand the wait to the scheduler only when spinning has stopped
// paying (see the header for the measurements).

#pragma once

#include <thread>

#include "common/wait_governor.h"

namespace Common {

/// One processor hint: `pause` on x86 (a handful of cycles on the cores the port targets,
/// exactly what the kernel's own spin locks use), a plain yield elsewhere.
inline void CpuRelax() {
    Gov::Pause();
}

/// Waits until `done()` is true: spins for the budget this call site has learned, then yields.
/// A short wait costs no system call at all; a long one leaves the core to the game after a
/// fraction of the yields the hand-rolled loops spent.
template <typename Predicate>
inline void SpinWait(Predicate done) {
    // One governor per call site and thread: the predicate is the site's own type, so the
    // static below is that site's -- two sites with different waits do not share a budget.
    static thread_local Gov::SpinGovernor governor{32'000};
    governor.Wait(done, [] { std::this_thread::yield(); });
}

/// Same, for a wait that is expected to be very short (a few hundred cycles to a few
/// microseconds): a bounded spin with no park at all. Returns false when the wait outlived it.
template <typename Predicate>
inline bool SpinWaitShort(Predicate done) {
    static thread_local Gov::SpinGovernor governor{8'000};
    return governor.Spin(done);
}

} // namespace Common
