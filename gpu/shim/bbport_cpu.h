// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: the process's share of the CPU.
//
// The port runs a GPU command thread, a recording thread, draw preparation workers and copy
// threads next to the game's own threads, and the frame rate is set by how quickly the command
// thread gets through a frame's draws. Two things take that time away from it that no amount of
// code tuning can win back:
//
//  * Windows 11's EcoQoS clocks a thread down (and parks it on the efficiency cores of a hybrid
//    CPU) once it decides the thread is background work. The port's helper threads are
//    deliberately below normal priority, so they are exactly what Windows considers background
//    -- including the recording thread the command thread blocks on. The process opts out here.
//  * A busy guest thread at normal priority preempts the command thread mid-draw. The command
//    thread therefore runs one step above the process.
//
//  * a thread the command thread waits on inside a frame -- the recorder it kicks with a chunk,
//    the landing thread a guest-visible label waits behind, the copy pool a host copy runs on --
//    is a helper by every other measure (it is idle most of the time, and it was left below
//    normal priority on purpose), so Windows may clock it down as background work. The command
//    thread then waits on a thread running at half speed. Those threads are raised here.
//
// Everything here is a hint to the scheduler: nothing is pinned, so a machine that needs its
// cores for something else still gets them. All settings are read once, from
// bbgpu_init(), and can be overridden with BB_CPU_PRIORITY (normal, above, high, realtime).
// On platforms other than Windows none of this applies: the helper policy is in
// bbport_threads.h, and there is no per-process throttle to opt out of.

#pragma once

#include <cstdlib>
#include <cstring>

#ifdef _WIN32
#include <windows.h>
#endif

namespace BbCpu {

#ifdef _WIN32

namespace Detail {
/// PROCESS_POWER_THROTTLING_STATE / THREAD_POWER_THROTTLING_STATE: the same layout, and the
/// same two masks, for both information classes.
struct ThrottlingState {
    ULONG Version;
    ULONG ControlMask;
    ULONG StateMask;
};
constexpr ULONG ThrottlingVersion = 1;   ///< PROCESS_POWER_THROTTLING_CURRENT_VERSION
constexpr ULONG ExecutionSpeed = 0x0001; ///< PROCESS_POWER_THROTTLING_EXECUTION_SPEED
constexpr int ProcessPowerThrottling = 4; ///< ProcessInformationClass
constexpr int ThreadPowerThrottling = 1;  ///< ThreadInformationClass

/// Turns the execution-speed throttle off for a process or a thread. Both entry points exist
/// since Windows 10 1709 and are looked up dynamically: a build against older headers still
/// links, and on an older system this is simply a no-op.
inline void AllowExecutionSpeed(HANDLE handle, bool process) {
    static const auto set_information = [] {
        struct Entry {
            FARPROC process{};
            FARPROC thread{};
        } entry;
        if (const HMODULE kernel = ::GetModuleHandleW(L"kernel32.dll")) {
            entry.process = ::GetProcAddress(kernel, "SetProcessInformation");
            entry.thread = ::GetProcAddress(kernel, "SetThreadInformation");
        }
        return entry;
    }();
    using SetInformationFn = BOOL(WINAPI*)(HANDLE, int, LPVOID, DWORD);
    const FARPROC fn = process ? set_information.process : set_information.thread;
    if (!fn) {
        return;
    }
    ThrottlingState state{ThrottlingVersion, ExecutionSpeed, 0};
    reinterpret_cast<SetInformationFn>(fn)(handle, process ? ProcessPowerThrottling
                                                         : ThreadPowerThrottling,
                                           &state, sizeof(state));
}

inline DWORD PriorityClassFromSetting() {
    if (const char* setting = std::getenv("BB_CPU_PRIORITY"); setting && setting[0]) {
        if (std::strcmp(setting, "normal") == 0) {
            return NORMAL_PRIORITY_CLASS;
        }
        if (std::strcmp(setting, "high") == 0) {
            return HIGH_PRIORITY_CLASS;
        }
        if (std::strcmp(setting, "realtime") == 0) {
            return REALTIME_PRIORITY_CLASS;
        }
    }
    // Above normal by default: the game keeps its cores, other applications wait a few
    // milliseconds longer for theirs.
    return ABOVE_NORMAL_PRIORITY_CLASS;
}
} // namespace Detail

/// Called once, from bbgpu_init().
inline void InitProcess() {
    Detail::AllowExecutionSpeed(::GetCurrentProcess(), true);
    ::SetPriorityClass(::GetCurrentProcess(), Detail::PriorityClassFromSetting());
}

/// Called on the thread that records and submits the game's draws (bbgpu_init runs there).
/// Cheap to call more than once: the work is done per thread, on the first call.
inline void InitCommandThread() {
    static thread_local const bool done = [] {
        Detail::AllowExecutionSpeed(::GetCurrentThread(), false);
        ::SetThreadPriority(::GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
        return true;
    }();
    (void)done;
}

/// Called on the threads the command thread waits on within a frame (the recorder, the landing
/// thread, the copy pool): they are idle most of the time, so a scheduler that watches their
/// duty cycle may treat them as background work, and the command thread then spends its frame
/// waiting on a thread that runs slower than the ones it preempted. Cheap after the first call.
inline void InitSyncThread() {
    static thread_local const bool done = [] {
        Detail::AllowExecutionSpeed(::GetCurrentThread(), false);
        ::SetThreadPriority(::GetCurrentThread(), THREAD_PRIORITY_NORMAL);
        return true;
    }();
    (void)done;
}

#else

/// No per-process throttle to opt out of; the helper policy is in bbport_threads.h.
inline void InitProcess() {}

/// Nothing to raise without privileges: a thread that is not the game's own keeps the
/// scheduler's default share.
inline void InitCommandThread() {}

/// The same: a thread created by the port starts at the process's own share, and raising it
/// above the guest's threads would only trade one wait for another.
inline void InitSyncThread() {}

#endif

} // namespace BbCpu
