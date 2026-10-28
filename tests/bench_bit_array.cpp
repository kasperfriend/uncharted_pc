// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: how fast the BitArray<1024> loops the buffer cache's region manager runs are, on this
// machine and with this build's vector path. It prints the path next to the timings, so building
// it twice shows what the vector code is worth on the CPU at hand:
//
//   g++ -std=c++20 -O2 -mavx -mno-avx2 -Igpu/shadps4 tests/bench_bit_array.cpp -o out/bench-avx
//   g++ -std=c++20 -O2 -mno-avx   -Igpu/shadps4 tests/bench_bit_array.cpp -o out/bench-sse2
//
// The scans are the point: AnyInRange answers "did the CPU touch these pages of this buffer"
// for every buffer binding, None() answers "did anything change" after every protection update.
#include <chrono>
#include <cstdio>

#include "common/bit_array.h"

namespace {

using Bits = Common::BitArray<1024>;

/// Nanoseconds per call of f (the array is marked as memory so the call is not hoisted).
template <typename F>
double PerCall(F&& f, int iterations) {
    unsigned sink = 0;
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
        sink += f();
    }
    asm volatile("" : "+r"(sink) : : "memory");  // the loop must not be optimised away
    const double total = std::chrono::duration<double, std::nano>(
                             std::chrono::steady_clock::now() - start).count();
    return total / iterations + 0.0 * sink;
}

} // namespace

int main() {
    constexpr int Spin = 1'000'000;
    Bits empty{}, full{};
    full.Fill();
    Bits sparse{};
    for (int i = 0; i < 40; ++i) {
        sparse.Set(static_cast<std::size_t>(i) * 25);
    }

    // A range that has to walk most of the array: no bit set in it.
    const double scan = PerCall(
        [&] {
            asm volatile("" : "+m"(empty) : : "memory");
            return static_cast<unsigned>(empty.AnyInRange(0, 1000));
        },
        Spin);
    // The same walk without the early exit.
    const double none = PerCall(
        [&] {
            asm volatile("" : "+m"(empty) : : "memory");
            return static_cast<unsigned>(empty.None());
        },
        Spin);
    // RegionBits::NotXor: the mask of pages whose protection changes on every update.
    const double not_xor = PerCall(
        [&] {
            asm volatile("" : "+m"(sparse) : : "memory");
            return static_cast<unsigned>(Bits::NotXor(sparse, full).Get(0));
        },
        Spin / 4);
    const double not_or = PerCall(
        [&] {
            asm volatile("" : "+m"(sparse) : : "memory");
            return static_cast<unsigned>(Bits::NotOr(sparse, full).Get(0));
        },
        Spin / 4);
    const double equal = PerCall(
        [&] {
            asm volatile("" : "+m"(sparse) : : "memory");
            return static_cast<unsigned>(sparse == full);
        },
        Spin);

    std::printf("bit array (%s): AnyInRange %.2f ns, None %.2f ns, NotXor %.2f ns, "
                "NotOr %.2f ns, == %.2f ns\n",
                BIT_ARRAY_PATH, scan, none, not_xor, not_or, equal);
    return 0;
}
