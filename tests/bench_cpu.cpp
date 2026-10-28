// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: what a draw's bookkeeping costs when it is not doing anything.
//
// Two costs of the per-draw path that do not show up in a profiler as a hot function but are
// paid 2000 times a frame:
//
//  * the diagnostic counters. Each one is a locked read-modify-write on a cache line that other
//    threads also touch; BbStats::Count() turns them into a branch on a constant when
//    BB_FRAME_STATS is off (this program runs with it off, like a normal game session).
//  * Rasterizer::Draw() asked FilterDrawPasses() -- a dozen register loads and compares -- three
//    times per draw and only ever needed the answer once. The model below does the same loads
//    and compares on the same fields so the cost of the two extra asks can be measured.
//
// Build: see the "cpu" stage of tools/ci/linux-tests.sh.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <span>

#include "bbport_toggles.h"

using u32 = std::uint32_t;
using u64 = std::uint64_t;

namespace {

constexpr u32 DrawsPerFrame = 2000;
constexpr u32 Rounds = 400;

template <typename Fn>
double NsPerCall(Fn&& fn, u32 iterations) {
    const auto start = std::chrono::steady_clock::now();
    fn();
    const auto end = std::chrono::steady_clock::now();
    return double(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()) /
           iterations;
}

/// The counters the draw path bumps today: one per draw, one per bound buffer's tracker fault,
/// the fault handler's two, the reduced-scene bookkeeping.
std::atomic<u64> c_draws{0}, c_dispatches{0}, c_reduced{0}, c_scene{0},
    c_tracker_faults{0};
std::atomic<std::int64_t> c_hot_pages{0};

[[gnu::noinline]] void CountDrawUnchecked() {
    c_draws.fetch_add(1, std::memory_order_relaxed);
    c_dispatches.fetch_add(1, std::memory_order_relaxed);
    c_reduced.fetch_add(1, std::memory_order_relaxed);
    c_scene.fetch_add(1, std::memory_order_relaxed);
    c_tracker_faults.fetch_add(1, std::memory_order_relaxed);
    c_hot_pages.fetch_add(1, std::memory_order_relaxed);
}

[[gnu::noinline]] void CountDrawGated() {
    BbStats::Count(BbStats::draws);
    BbStats::Count(BbStats::dispatches);
    BbStats::Count(BbStats::reduced_draws);
    BbStats::Count(BbStats::scene_draws);
    BbStats::Count(BbStats::tracker_faults);
    BbStats::Count(BbStats::hot_pages);
}

/// The register fields Rasterizer::FilterDrawPasses() reads, with the same kinds of tests.
struct PassRegs {
    u32 color_control_mode;
    u32 primitive_type;
    u32 force_z_dirty, force_z_valid, depth_valid, depth_write_valid;
    u64 depth_address, depth_write_address;
    u32 force_stencil_dirty, force_stencil_valid, stencil_valid, stencil_write_valid;
    u64 stencil_address, stencil_write_address;
    std::uint8_t pad[128]; ///< the rest of the register file, so the reads are not all in one line
};

/// Read through a global, like the rasterizer reads the live register file: with an argument
/// the compiler proves the reads pure and fires the three calls of the draw path only once.
volatile PassRegs pass_regs{};
volatile u32 sink_keep = 0;

[[gnu::noinline]] bool FilterDrawPasses() {
    const volatile PassRegs& regs = pass_regs;
    constexpr u32 EliminateFastClear = 1, FmaskDecompress = 2, Resolve = 3;
    const u32 mode = regs.color_control_mode;
    if (mode == EliminateFastClear || mode == FmaskDecompress || mode == Resolve ||
        regs.primitive_type == 0) {
        return false;
    }
    const bool depth_copy = regs.force_z_dirty && regs.force_z_valid && regs.depth_valid &&
                            regs.depth_write_valid &&
                            regs.depth_address != regs.depth_write_address;
    const bool stencil_copy = regs.force_stencil_dirty && regs.force_stencil_valid &&
                              regs.stencil_valid && regs.stencil_write_valid &&
                              regs.stencil_address != regs.stencil_write_address;
    return !(mode == 0 && (depth_copy || stencil_copy));
}

void BenchCounters() {
    const auto unchecked = NsPerCall(
        [] {
            for (u32 round = 0; round < Rounds; ++round) {
                for (u32 draw = 0; draw < DrawsPerFrame; ++draw) {
                    CountDrawUnchecked();
                }
            }
        },
        Rounds * DrawsPerFrame);
    const auto gated = NsPerCall(
        [] {
            for (u32 round = 0; round < Rounds; ++round) {
                for (u32 draw = 0; draw < DrawsPerFrame; ++draw) {
                    CountDrawGated();
                }
            }
        },
        Rounds * DrawsPerFrame);
    std::printf("counters: 6 per draw unconditional %.2f ns/draw, gated %.2f ns/draw "
                "(%.1f us per %u-draw frame)\n",
                unchecked, gated, (unchecked - gated) * DrawsPerFrame / 1000.0, DrawsPerFrame);
    std::printf("  counters are collected: %s\n", BbStats::enabled ? "yes" : "no");
}

void BenchFilterDrawPasses() {
    pass_regs.color_control_mode = 0; // Disable: the draw passes the filter
    pass_regs.primitive_type = 4;
    pass_regs.depth_valid = 1;
    pass_regs.depth_write_valid = 1;
    pass_regs.depth_address = 0x1000;
    pass_regs.depth_write_address = 0x2000;
    u32 sink = 0;
    // The clock this runs on may only tick every few milliseconds: the loop has to take long
    // enough that a call's share of it is not lost in the rounding.
    constexpr u32 Iterations = 64 * Rounds * DrawsPerFrame;
    const auto once = NsPerCall(
        [&] {
            for (u32 draw = 0; draw < Iterations; ++draw) {
                // The registers move between draws (the target images' addresses do), so the
                // call cannot be hoisted out of the loop.
                pass_regs.depth_address = 0x1000 + (draw & 0xff);
                sink += FilterDrawPasses();
            }
        },
        Iterations);
    const auto thrice = NsPerCall(
        [&] {
            for (u32 draw = 0; draw < Iterations; ++draw) {
                pass_regs.depth_address = 0x1000 + (draw & 0xff);
                sink += FilterDrawPasses();
                sink += FilterDrawPasses();
                sink += FilterDrawPasses();
            }
        },
        Iterations);
    // Kept observable, or the loop below is dead code.
    sink_keep = sink;
    std::printf("filter passes: %.2f ns per call, 3 asks/draw %.2f ns vs 1 ask %.2f ns "
                "(%.1f us per %u-draw frame)\n",
                once, thrice, once, (thrice - once) * DrawsPerFrame / 1000.0, DrawsPerFrame);
}

} // namespace

int main() {
    BenchCounters();
    BenchFilterDrawPasses();
    return 0;
}
