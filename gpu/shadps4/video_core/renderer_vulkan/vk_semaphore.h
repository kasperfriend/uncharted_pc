// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <condition_variable>
#include <thread>
#include <queue>
#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {

class Instance;
class Scheduler;

/// bbport: prints the GPU crash report (driver fault info, checkpoints, arenas) and ends the game.
[[noreturn]] void DeviceLost(vk::Device device);

class Semaphore {
public:
    explicit Semaphore(const Instance& instance_);
    ~Semaphore();

    [[nodiscard]] u64 CurrentTick() const noexcept {
        return current_tick.load(std::memory_order_acquire);
    }

    [[nodiscard]] u64 KnownGpuTick() const noexcept {
        return gpu_tick.load(std::memory_order_acquire);
    }

    [[nodiscard]] bool IsFree(u64 tick) const noexcept {
        return KnownGpuTick() >= tick;
    }

    [[nodiscard]] u64 NextTick() noexcept {
        return current_tick.fetch_add(1, std::memory_order_release);
    }

    [[nodiscard]] vk::Semaphore Handle() const noexcept {
        return semaphore.get();
    }

    /// bbport: what this timeline belongs to ("draw", "present", "arena binds"): the GPU stall
    /// report names it, so a wait that never completes says whose timeline it is waiting for.
    void SetName(const char* timeline_name) noexcept {
        name.store(timeline_name, std::memory_order_release);
    }

    [[nodiscard]] const char* GetName() const noexcept {
        return name.load(std::memory_order_acquire);
    }

    /// bbport: the timeline value the last submission of this timeline signals, where
    /// CurrentTick() is the value the *next* one will: a stall report that shows them apart says
    /// whether a tick was allocated and never submitted (the queue never took it) or submitted and
    /// never executed (the GPU is behind, or waiting for another timeline).
    [[nodiscard]] u64 LastSubmitted() const noexcept {
        return last_submitted.load(std::memory_order_acquire);
    }

    /// bbport: the tick of the submission this timeline is inside vkQueueSubmit with, 0 if none.
    [[nodiscard]] u64 Submitting() const noexcept {
        return submitting.load(std::memory_order_acquire);
    }

    void NoteSubmitting(u64 tick) noexcept {
        submitting.store(tick, std::memory_order_release);
    }

    void NoteSubmitted(u64 tick) noexcept {
        last_submitted.store(tick, std::memory_order_release);
        submitting.store(0, std::memory_order_release);
    }

    /// Refresh the known GPU tick
    void Refresh();

    /// Waits for a tick to be hit on the GPU
    void Wait(u64 tick);

protected:
    const Instance& instance;
    vk::UniqueSemaphore semaphore;    ///< Timeline semaphore.
    std::atomic<u64> gpu_tick{0};     ///< Current known GPU tick.
    std::atomic<u64> current_tick{1}; ///< Current logical tick.
    std::atomic<u64> last_submitted{0};
    std::atomic<u64> submitting{0};
    std::atomic<const char*> name{"unnamed"};
};

/// bbport: prints one line about a timeline the port owns ("draw": GPU 2120, submitted 2121,
/// next 2123), or says it is not one of them (a swapchain semaphore, another library's).
void DescribeTimeline(vk::Semaphore handle);

} // namespace Vulkan
