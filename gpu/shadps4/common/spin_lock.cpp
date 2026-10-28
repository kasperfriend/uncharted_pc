// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: test-and-test-and-set with a pause backoff, see lock() below.

#include "common/spin_lock.h"

#include <algorithm>
#include <thread>

#if _MSC_VER
#include <intrin.h>
#if _M_AMD64
#define __x86_64__ 1
#endif
#if _M_ARM64
#define __aarch64__ 1
#endif
#else
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif
#endif

namespace {

void ThreadPause() {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif __aarch64__ && _MSC_VER
    __yield();
#elif __aarch64__
    asm("yield");
#else
    std::this_thread::yield();
#endif
}

/// Pauses between two attempts at the flag: long enough that the holder's critical section
/// (a few hundred cycles in the region manager and the page manager) finishes inside it, short
/// enough not to add latency when the lock is contended. The backoff doubles per round: a lock
/// that is held through a preemption stops costing the waiting core a read-modify-write of the
/// line per few hundred cycles.
constexpr unsigned BackoffPauses = 32;
constexpr unsigned MaxBackoffPauses = 512;
/// Rounds after which the waiting thread lets the scheduler run the holder (the lock is never
/// held for long, so this only happens when the holder was preempted).
constexpr unsigned SpinsBeforeYield = 2;

#if defined(_MSC_VER)
#define BB_NOINLINE __declspec(noinline)
#else
#define BB_NOINLINE __attribute__((noinline))
#endif

/// Everything a waiting thread does. Out of line on purpose: lock() stays the one locked
/// read-modify-write of the uncontended case, with no loop and no branch around it.
BB_NOINLINE void LockSlow(std::atomic_flag& lck) {
    unsigned pauses = BackoffPauses;
    for (unsigned round = 0;;) {
        while (lck.test(std::memory_order_relaxed)) {
            for (unsigned pause = 0; pause < pauses; ++pause) {
                ThreadPause();
            }
            if (++round >= SpinsBeforeYield) {
                round = 0;
                pauses = std::min(pauses * 2, MaxBackoffPauses);
                std::this_thread::yield();
            }
        }
        if (!lck.test_and_set(std::memory_order_acquire)) {
            return;
        }
    }
}

} // Anonymous namespace

namespace Common {

void SpinLock::lock() {
    // One locked read-modify-write for the uncontended case, exactly as before: a lock taken by
    // one thread at a time (the buffer cache's region manager, the page manager) costs one
    // `lock bts` and nothing else.
    if (!lck.test_and_set(std::memory_order_acquire)) {
        return;
    }
    // Contended: test and test-and-set (LockSlow). Reading the flag with a plain load until it
    // looks free keeps the lock's cache line where it is (the holder's, or shared); claiming it
    // without looking would make every waiting core pull the line across, on the model this
    // port targets a trip through the shared cache, and hand it back again on release.
    LockSlow(lck);
}

void SpinLock::unlock() {
    lck.clear(std::memory_order_release);
}

bool SpinLock::try_lock() {
    if (lck.test_and_set(std::memory_order_acquire)) {
        return false;
    }
    return true;
}

} // namespace Common
