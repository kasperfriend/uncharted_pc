// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: a second thread for the GPU command thread's per-draw work (docs/parallel_gpu.md,
// "Texture binding on a helper thread"). The GPU thread forks one task per draw and joins it
// before anything that depends on the task's results; the helper spins for the next task
// while draws flow and sleeps after a short idle period.

#pragma once

#include <atomic>
#include <chrono>
#include <thread>
#include <x86intrin.h>

#include "common/thread.h"
#include "common/types.h"
#include "common/wait_governor.h"

namespace Vulkan {

class BindHelper {
public:
    using Task = void (*)(void*);

    bool running = false;

public:
    explicit BindHelper(bool enabled) {
        if (enabled) {
            thread = std::jthread([this](std::stop_token stop) { Run(stop); });
            running = true;
        }
    }

    ~BindHelper() {
        if (thread.joinable()) {
            thread.request_stop();
            posted.fetch_add(1, std::memory_order_seq_cst);
            posted.notify_one();
        }
    }

    [[nodiscard]] bool Available() const noexcept {
        return running; // bbport: not thread.joinable(), two system calls on Windows
    }

    /// True on the helper thread (its callees must not join it).
    [[nodiscard]] static bool OnHelper() noexcept {
        return on_helper;
    }

    /// True between Fork() and Join() (GPU thread only).
    [[nodiscard]] bool Active() const noexcept {
        return active;
    }

    /// Starts `task(context)` on the helper.
    void Fork(Task task_, void* context_) {
        task = task_;
        context = context_;
        active = true;
        posted.fetch_add(1, std::memory_order_seq_cst);
        if (sleeping.load(std::memory_order_seq_cst)) {
            posted.notify_one();
        }
    }

    /// Waits for the forked task; no-op when none is running.
    void Join() {
        if (!active) {
            return;
        }
        const u64 target = posted.load(std::memory_order_relaxed);
        if (done.load(std::memory_order_acquire) != target) {
            const u64 start = __rdtsc();
            // The task is one draw's texture binding: microseconds. The wait is governed by
            // what the joins have measured, and a helper that lost its core to the game's own
            // threads costs a yield per budget instead of spinning the GPU thread at 100%.
            join_governor.Wait([&] { return done.load(std::memory_order_acquire) == target; },
                               [] { std::this_thread::yield(); });
            wait_cycles += __rdtsc() - start;
        }
        active = false;
    }

    /// Cycles the GPU thread waited in Join() and the helper spent in tasks (BB_FRAME_STATS).
    u64 wait_cycles = 0;
    std::atomic<u64> task_cycles{0};

private:
    void Run(std::stop_token stop) {
        Common::SetCurrentThreadName("bb:TexBind");
        on_helper = true;
        u64 seen = 0;
        while (!stop.stop_requested()) {
            // Draws arrive every few microseconds while a frame is recorded, so the wait
            // spins; between frames it parks. The governor learns which of the two this is
            // (an idle site stops spinning through the gap) instead of spinning to a fixed
            // 100 us deadline between every frame.
            u64 now = posted.load(std::memory_order_acquire);
            if (now == seen) {
                idle_governor.Wait(
                    [&] {
                        return posted.load(std::memory_order_acquire) != seen ||
                               stop.stop_requested();
                    },
                    [&] {
                        sleeping.store(true, std::memory_order_seq_cst);
                        posted.wait(seen, std::memory_order_seq_cst);
                        sleeping.store(false, std::memory_order_relaxed);
                    });
                now = posted.load(std::memory_order_acquire);
            }
            if (stop.stop_requested()) {
                break;
            }
            seen = now;
            const u64 start = __rdtsc();
            task(context);
            task_cycles.fetch_add(__rdtsc() - start, std::memory_order_relaxed);
            done.store(seen, std::memory_order_release);
        }
    }

    static inline thread_local bool on_helper = false;
    Task task = nullptr;
    void* context = nullptr;
    bool active = false;
    Common::Gov::SpinGovernor join_governor{16'000}; ///< GPU thread, per join
    Common::Gov::SpinGovernor idle_governor{64'000}; ///< helper, between tasks
    alignas(64) std::atomic<u64> posted{0};
    alignas(64) std::atomic<u64> done{0};
    alignas(64) std::atomic<bool> sleeping{false};
    std::jthread thread;
};

} // namespace Vulkan
