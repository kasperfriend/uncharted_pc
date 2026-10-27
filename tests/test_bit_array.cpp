// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: Common::BitArray against a scalar reference implementation.
//
// The buffer cache asks the bit array for the ranges of the pages a buffer covers
// (FirstRangeFrom/LastRangeFrom) for every upload, and those scans have a vector path: 4 words
// per instruction with AVX2, 2 words with SSE2 (the cores this port targets have no AVX2).
// Both are checked here against the plain word loops, over random patterns, every range
// boundary and every starting position -- the same code must give the same answer at -O0 and
// -O3, with and without AVX2.
#include <algorithm>
#include <cstdio>
#include <random>
#include <utility>
#include <vector>

#include "common/bit_array.h"
#include "video_core/buffer_cache/region_definitions.h"

namespace {

int failures = 0;

void Check(bool ok, const char* what, int line) {
    if (!ok) {
        std::printf("FAIL line %d: %s\n", line, what);
        ++failures;
    }
}
#define CHECK(condition) Check((condition), #condition, __LINE__)

/// The bit array's answer, computed the slow way.
class Reference {
public:
    explicit Reference(std::size_t size) : bits(size, false) {}

    void Set(std::size_t at) {
        bits[at] = true;
    }
    void SetRange(std::size_t first, std::size_t last) {
        for (std::size_t i = first; i < last; ++i) {
            bits[i] = true;
        }
    }
    void UnsetRange(std::size_t first, std::size_t last) {
        for (std::size_t i = first; i < last; ++i) {
            bits[i] = false;
        }
    }
    void Clear() {
        std::fill(bits.begin(), bits.end(), false);
    }
    void Fill() {
        std::fill(bits.begin(), bits.end(), true);
    }
    bool Get(std::size_t at) const {
        return bits[at];
    }
    bool Any(std::size_t first, std::size_t last) const {
        for (std::size_t i = first; i < last; ++i) {
            if (bits[i]) {
                return true;
            }
        }
        return false;
    }
    bool None() const {
        return !Any(0, bits.size());
    }
    std::size_t Ones() const {
        return static_cast<std::size_t>(std::count(bits.begin(), bits.end(), true));
    }
    std::pair<std::size_t, std::size_t> First(std::size_t from) const {
        std::size_t at = from;
        while (at < bits.size() && !bits[at]) {
            ++at;
        }
        if (at == bits.size()) {
            return {bits.size(), bits.size()};
        }
        std::size_t end = at;
        while (end < bits.size() && bits[end]) {
            ++end;
        }
        return {at, end};
    }
    std::pair<std::size_t, std::size_t> Last(std::size_t end) const {
        if (end > bits.size()) {
            end = bits.size();
        }
        if (end == 0) {
            return {0, 0};
        }
        std::ptrdiff_t at = static_cast<std::ptrdiff_t>(end) - 1;
        while (at >= 0 && !bits[at]) {
            --at;
        }
        if (at < 0) {
            return {0, 0};
        }
        std::ptrdiff_t first = at;
        while (first >= 0 && bits[first]) {
            --first;
        }
        return {static_cast<std::size_t>(first + 1), static_cast<std::size_t>(at + 1)};
    }

private:
    std::vector<bool> bits;
};

/// A random mix of operations, then every query against the reference.
template <std::size_t N>
void Exercise(unsigned seed) {
    std::mt19937 rng(seed);
    Common::BitArray<N> array;
    Reference reference(N);
    for (int round = 0; round < 60; ++round) {
        const std::size_t first = rng() % N;
        const std::size_t last = std::min<std::size_t>(N, first + 1 + rng() % (N / 2));
        switch (rng() % 5) {
        case 0:
            array.SetRange(first, last);
            reference.SetRange(first, last);
            break;
        case 1:
            array.UnsetRange(first, last);
            reference.UnsetRange(first, last);
            break;
        case 2:
            array.Clear();
            reference.Clear();
            for (int k = 0; k < 8; ++k) {
                const std::size_t a = rng() % N;
                const std::size_t b = std::min<std::size_t>(N, a + 1 + rng() % 96);
                array.SetRange(a, b);
                reference.SetRange(a, b);
            }
            break;
        case 3:
            array.Fill();
            reference.Fill();
            break;
        default: {
            const std::size_t at = rng() % N;
            array.Set(at);
            reference.Set(at);
        } break;
        }
        for (std::size_t i = 0; i < N; i += 1 + rng() % 5) {
            CHECK(array.Get(i) == reference.Get(i));
        }
        CHECK(array.Any() == !reference.None());
        CHECK(array.None() == reference.None());
        for (int k = 0; k < 4; ++k) {
            const std::size_t a = rng() % N;
            const std::size_t b = std::min<std::size_t>(N, a + 1 + rng() % (N / 2));
            CHECK(array.AnyInRange(a, b) == reference.Any(a, b));
        }
        for (int k = 0; k < 4; ++k) {
            const std::size_t from = rng() % (N + 1);
            CHECK(array.FirstRangeFrom(from) == reference.First(from));
        }
        for (int k = 0; k < 4; ++k) {
            const std::size_t end = rng() % (N + 1);
            CHECK(array.LastRangeFrom(end) == reference.Last(end));
        }
        CHECK(array.FirstRange() == reference.First(0));
        CHECK(array.LastRange() == reference.Last(N));
        // The iterator must visit exactly the set bits, in order.
        std::size_t covered = 0, previous = 0;
        for (const auto& range : array) {
            CHECK(range.first >= previous && range.second > range.first);
            CHECK(reference.Any(range.first, range.second));
            covered += range.second - range.first;
            previous = range.second;
        }
        CHECK(covered == reference.Ones());
        // Copies, masking and the bitwise operators.
        Common::BitArray<N> part(array, first, last);
        for (std::size_t i = 0; i < N; i += 1 + rng() % 3) {
            CHECK(part.Get(i) == (i >= first && i < last && reference.Get(i)));
        }
        Common::BitArray<N> whole(array, 0, N);
        CHECK(whole == array);
        CHECK((~array).Any() == (reference.Ones() != N));
        Common::BitArray<N> xored = array;
        xored ^= array;
        CHECK(xored.None());
        // The fused forms the region manager uses must equal the operators they replace, and
        // must not be confused with each other (~(a|b) clears where both are set, ~(a^b) does
        // not).
        Common::BitArray<N> other_array;
        Reference other_reference(N);
        for (int k = 0; k < 12; ++k) {
            const std::size_t a = rng() % N;
            const std::size_t b = std::min<std::size_t>(N, a + 1 + rng() % 128);
            if (rng() & 1) {
                other_array.SetRange(a, b);
                other_reference.SetRange(a, b);
            } else {
                other_array.UnsetRange(a, b);
                other_reference.UnsetRange(a, b);
            }
        }
        CHECK(Common::BitArray<N>::NotXor(array, other_array) == (~array ^ other_array));
        CHECK(Common::BitArray<N>::NotOr(array, other_array) == ~(array | other_array));
        for (std::size_t i = 0; i < N; i += 1 + rng() % 3) {
            const bool a = reference.Get(i), b = other_reference.Get(i);
            CHECK(Common::BitArray<N>::NotXor(array, other_array).Get(i) == !(a ^ b));
            CHECK(Common::BitArray<N>::NotOr(array, other_array).Get(i) == !(a || b));
        }
        CHECK(Common::BitArray<N>::NotXor(array, array).None() == false);
    }
}

/// The page masks the buffer cache's region manager builds (region_definitions.h), against the
/// arithmetic they have to mean. The write mask is a plain XOR of "CPU wrote" and "CPU may write";
/// complementing it instead (the tempting symmetry with the read mask) marks every page that did
/// not change, which drifts the page manager's watcher counts until an unprotect finds a page with
/// no watcher and the GPU library stops with "Not enough watchers". Nothing else in the port
/// exercised that formula, so it is pinned here.
void CheckRegionManagerMasks(unsigned seed) {
    using Bits = VideoCore::RegionBits;
    static_assert(Bits{}.Size() == 1024, "RegionBits is a 4 MiB region of 4 KiB pages");
    constexpr std::size_t N = 1024;
    std::mt19937 rng(seed);
    Bits gpu, readable, cpu, writeable;
    Reference gpu_ref(N), readable_ref(N), cpu_ref(N), writeable_ref(N);
    for (int k = 0; k < 24; ++k) {
        const std::size_t first = rng() % N;
        const std::size_t last = std::min<std::size_t>(N, first + rng() % 200);
        const int target = rng() % 4;
        Bits* bits = target == 0 ? &gpu : target == 1 ? &readable : target == 2 ? &cpu : &writeable;
        Reference* ref = target == 0   ? &gpu_ref
                         : target == 1 ? &readable_ref
                         : target == 2 ? &cpu_ref
                                       : &writeable_ref;
        if (rng() & 1) {
            bits->SetRange(first, last);
            ref->SetRange(first, last);
        } else {
            bits->UnsetRange(first, last);
            ref->UnsetRange(first, last);
        }
    }
    const Bits read_mask = VideoCore::ProtectionChangeMask(true, gpu, readable, cpu, writeable);
    const Bits write_mask = VideoCore::ProtectionChangeMask(false, gpu, readable, cpu, writeable);
    const Bits window_mask = VideoCore::WriteFaultWindowMask(gpu, writeable);
    for (std::size_t i = 0; i < N; i += 1 + rng() % 3) {
        const bool g = gpu_ref.Get(i), r = readable_ref.Get(i);
        const bool c = cpu_ref.Get(i), w = writeable_ref.Get(i);
        CHECK(read_mask.Get(i) == !(g ^ r));   // ~(gpu ^ readable)
        CHECK(write_mask.Get(i) == (c ^ w));   // cpu ^ writeable, NOT ~(cpu ^ writeable)
        CHECK(window_mask.Get(i) == !(g || w)); // ~(gpu | writeable)
    }
    // The whole arrays, not only the sampled bits.
    CHECK(write_mask == (cpu ^ writeable));
    CHECK(!(write_mask == Bits::NotXor(cpu, writeable)));
    CHECK(read_mask == (~gpu ^ readable));
    CHECK(window_mask == ~(gpu | writeable));
    // "Nothing changed" must come back empty: that early exit is on the hot path.
    CHECK(VideoCore::ProtectionChangeMask(false, gpu, readable, cpu, cpu).None());
    CHECK(VideoCore::ProtectionChangeMask(true, gpu, readable, cpu, writeable).None() ==
          (gpu == readable));
}

} // namespace

int main() {
#if defined(BIT_ARRAY_USE_AVX)
    // AVX1 is enough for this path (vptest, vmovdqu, vandps/vorps/vxorps on 256 bits).
    std::printf("bit array: AVX path, 4 words per instruction\n");
#elif defined(BIT_ARRAY_USE_SSE)
    std::printf("bit array: SSE2 path, 2 words per instruction\n");
#else
    std::printf("bit array: no vector path (scalar words)\n");
#endif
    for (unsigned seed = 1; seed <= 3; ++seed) {
        Exercise<64>(seed);    // one word
        Exercise<128>(seed);   // two
        Exercise<1024>(seed);  // NUM_PAGES_PER_REGION: the buffer cache's RegionBits
    }
    for (unsigned seed = 1; seed <= 8; ++seed) {
        CheckRegionManagerMasks(seed);
    }
    if (failures != 0) {
        std::printf("%d failures\n", failures);
        return 1;
    }
    std::printf("bit array: all checks passed\n");
    return 0;
}
