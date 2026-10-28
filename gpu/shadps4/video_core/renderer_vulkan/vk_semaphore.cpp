// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <limits>
#include <mutex>
#include <vector>
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_platform.h"
#include "video_core/renderer_vulkan/vk_semaphore.h"

#include "common/assert.h"

namespace Vulkan {

// bbport: called once by a GPU wait that has blocked for 5 seconds (the scheduler prints what
// the queue is waiting for).
void (*g_gpu_stall_report)() = nullptr;
// bbport: more context for a GPU crash (the buffer cache's address ranges).
void (*g_gpu_crash_context)() = nullptr;
// bbport: the addresses the driver named for the fault (READ/WRITE/EXECUTE), so the crash
// report can say where they are: an arena's guest or device range, a resident page or neither.
std::array<u64, 4> g_gpu_crash_addresses{};
u32 g_gpu_crash_address_count = 0;
// bbport: the draws and dispatches the GPU last reached (BB_GPU_CHECKPOINTS=1).
void (*g_gpu_checkpoint_report)() = nullptr;
// bbport: subsystem records for a device lost (the tiling dispatches and the buffers they were
// given, video_core/texture_cache/tile_manager.cpp).
void (*g_gpu_crash_detail)() = nullptr;
extern bool g_device_fault_ext;
} // namespace Vulkan
namespace VideoCore {
void DumpBuffersNear(u64 address);
}
namespace Vulkan {

// bbport: the GPU crashed (device lost). The driver's description of the fault is printed and
// the game ends: every later GPU wait would fail the same way (it used to wait forever).
// Also called where a submission or present reports the loss (vk_semaphore.h).
[[noreturn]] void DeviceLost(vk::Device device) {
    std::printf("GPU crash: the graphics device was lost\n");
    if (g_device_fault_ext && VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceFaultInfoEXT) {
        const auto get = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceFaultInfoEXT;
        VkDeviceFaultCountsEXT counts{.sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_COUNTS_EXT};
        get(device, &counts, nullptr);
        std::vector<VkDeviceFaultAddressInfoEXT> addresses(counts.addressInfoCount);
        std::vector<VkDeviceFaultVendorInfoEXT> vendor(counts.vendorInfoCount);
        counts.vendorBinarySize = 0;
        VkDeviceFaultInfoEXT info{.sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_INFO_EXT,
                                  .pAddressInfos = addresses.data(),
                                  .pVendorInfos = vendor.data()};
        get(device, &counts, &info);
        std::printf("GPU crash: %s\n", info.description);
        static constexpr const char* kinds[] = {"none", "read of an invalid address",
                                                "write to an invalid address",
                                                "execution of an invalid address",
                                                "instruction pointer (unknown)",
                                                "instruction pointer (invalid)",
                                                "instruction pointer (fault)"};
        for (const auto& a : addresses) {
            std::printf("GPU crash: %s at 0x%llx (+- 0x%llx)\n",
                        a.addressType < std::size(kinds) ? kinds[a.addressType] : "?",
                        static_cast<unsigned long long>(a.reportedAddress),
                        static_cast<unsigned long long>(a.addressPrecision));
        }
        g_gpu_crash_address_count = 0;
        for (const auto& a : addresses) {
            if (a.addressType >= VK_DEVICE_FAULT_ADDRESS_TYPE_READ_INVALID_EXT &&
                a.addressType <= VK_DEVICE_FAULT_ADDRESS_TYPE_EXECUTE_INVALID_EXT) {
                if (g_gpu_crash_address_count < g_gpu_crash_addresses.size()) {
                    g_gpu_crash_addresses[g_gpu_crash_address_count++] = a.reportedAddress;
                }
                VideoCore::DumpBuffersNear(a.reportedAddress);
                if (AddressReportEnabled()) {
                    DumpAddressBindings(a.reportedAddress);
                }
            }
        }
        for (const auto& v : vendor) {
            std::printf("GPU crash: vendor 0x%llx/0x%llx %s\n",
                        static_cast<unsigned long long>(v.vendorFaultCode),
                        static_cast<unsigned long long>(v.vendorFaultData), v.description);
        }
    } else {
        std::printf("GPU crash: the driver gives no details (VK_EXT_device_fault unavailable)\n");
    }
    if (g_gpu_checkpoint_report) {
        g_gpu_checkpoint_report();
    }
    if (g_gpu_crash_context) {
        g_gpu_crash_context();
    }
    if (g_gpu_crash_detail) {
        g_gpu_crash_detail();
    }
    if (g_gpu_stall_report) {
        g_gpu_stall_report();
    }
    std::fflush(stdout);
    std::_Exit(5);
}

namespace {
// bbport: every timeline the port creates, so the stall report can name the one a submission
// waits for and say how far it got. Guarded (submissions are made from several threads) and
// deliberately leaked: a timeline destroyed after this vector was destroyed at exit would
// otherwise write into freed memory.
std::mutex& TimelinesMutex() {
    static auto* mutex = new std::mutex;
    return *mutex;
}

std::vector<const Semaphore*>& Timelines() {
    static auto* timelines = new std::vector<const Semaphore*>;
    return *timelines;
}
} // Anonymous namespace

void DescribeTimeline(vk::Semaphore handle) {
    std::scoped_lock lk{TimelinesMutex()};
    for (const Semaphore* timeline : Timelines()) {
        if (timeline->Handle() != handle) {
            continue;
        }
        std::printf(" ['%s': GPU %llu, submitted %llu, next %llu%s]", timeline->GetName(),
                    static_cast<unsigned long long>(timeline->KnownGpuTick()),
                    static_cast<unsigned long long>(timeline->LastSubmitted()),
                    static_cast<unsigned long long>(timeline->CurrentTick()),
                    timeline->Submitting() ? " (inside a queue submit)" : "");
        return;
    }
    std::printf(" [not a timeline of the port]");
}

Semaphore::Semaphore(const Instance& instance_) : instance{instance_} {
    const vk::StructureChain semaphore_chain = {
        vk::SemaphoreCreateInfo{},
        vk::SemaphoreTypeCreateInfo{
            .semaphoreType = vk::SemaphoreType::eTimeline,
            .initialValue = 0,
        },
    };
    auto [semaphore_result, sem] =
        instance.GetDevice().createSemaphoreUnique(semaphore_chain.get());
    ASSERT_MSG(semaphore_result == vk::Result::eSuccess, "Failed to create master semaphore: {}",
               vk::to_string(semaphore_result));
    semaphore = std::move(sem);
    {
        std::scoped_lock lk{TimelinesMutex()};
        Timelines().push_back(this);
    }
}

Semaphore::~Semaphore() {
    std::scoped_lock lk{TimelinesMutex()};
    std::erase(Timelines(), this);
}

void Semaphore::Refresh() {
    u64 this_tick{};
    u64 counter{};
    do {
        this_tick = gpu_tick.load(std::memory_order_acquire);
        auto [counter_result, cntr] = instance.GetDevice().getSemaphoreCounterValue(*semaphore);
        ASSERT_MSG(counter_result == vk::Result::eSuccess,
                   "Failed to get master semaphore value: {}", vk::to_string(counter_result));
        counter = cntr;
        if (counter < this_tick) {
            return;
        }
    } while (!gpu_tick.compare_exchange_weak(this_tick, counter, std::memory_order_release,
                                             std::memory_order_relaxed));
}

void Semaphore::Wait(u64 tick) {
    // No need to wait if the GPU is ahead of the tick
    if (IsFree(tick)) {
        return;
    }
    // Update the GPU tick and try again
    Refresh();
    if (IsFree(tick)) {
        return;
    }

    // If none of the above is hit, fallback to a regular wait
    const vk::SemaphoreWaitInfo wait_info = {
        .semaphoreCount = 1,
        .pSemaphores = &semaphore.get(),
        .pValues = &tick,
    };

    constexpr u64 StallReport = 5'000'000'000ull;
    u32 reports = 0;
    vk::Result result;
    // bbport: the first timeout prints what the queue is waiting for (the schedulers' timelines
    // and the last submissions, see Scheduler::ReportStall). The wait used to go on with no
    // timeout at all, so a device that never advances hung the game with nothing more in the log;
    // it is re-asked every 5 s now and the report is repeated (12 times), so a log ends with
    // what was stuck instead of ending nowhere.
    while ((result = instance.GetDevice().waitSemaphores(&wait_info, StallReport)) !=
           vk::Result::eSuccess) {
        if (result == vk::Result::eErrorDeviceLost) {
            DeviceLost(instance.GetDevice());
        }
        if (result != vk::Result::eTimeout) {
            // Any other failure leaves a timeline that will not advance: report and end, the game
            // cannot continue with a GPU that is gone (this used to loop forever).
            std::printf("GPU stall: waiting for tick %llu of timeline '%s' failed: %s\n",
                        static_cast<unsigned long long>(tick), GetName(),
                        vk::to_string(result).c_str());
            DeviceLost(instance.GetDevice());
        }
        if (g_gpu_stall_report && reports < 12) {
            ++reports;
            std::printf("GPU stall: waiting %u s for tick %llu of timeline '%s' (%p)\n",
                        reports * 5u, static_cast<unsigned long long>(tick), GetName(),
                        reinterpret_cast<void*>(static_cast<VkSemaphore>(Handle())));
            g_gpu_stall_report();
        }
    }
    Refresh();
}

} // namespace Vulkan
