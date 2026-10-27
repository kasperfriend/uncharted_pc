// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <bit>
#include <cstddef>
#include <type_traits>
#include <utility>
#include "common/types.h"

// bbport: the port's floor is AVX without AVX2, so the 256-bit path must not need AVX2. The
// scans below (FirstRangeFrom, LastRangeFrom, AnyInRange, None, the set operators -- the buffer
// cache's region manager runs them on every buffer binding) only ever load, store, AND, OR,
// XOR and *test* 256 bits, and every one of those exists in AVX1: vptest replaces the AVX2
// vpcmpeqq+vmovmskpd pair with a single instruction, which is where the gain is on a core that
// executes a 256-bit operation as two 128-bit halves. (AVX2 builds take the same path.)
#if defined(__AVX__)
#define BIT_ARRAY_USE_AVX
#include <immintrin.h>
#elif defined(__SSE2__)
// No AVX: the SSE unit still handles two 64-bit words per instruction.
#define BIT_ARRAY_USE_SSE
#include <emmintrin.h>
#endif
#if defined(BIT_ARRAY_USE_AVX) || defined(BIT_ARRAY_USE_SSE)
#define BIT_ARRAY_USE_VECTOR
#endif
#if defined(BIT_ARRAY_USE_AVX)
#define BIT_ARRAY_PATH "AVX 256-bit"
#elif defined(BIT_ARRAY_USE_SSE)
#define BIT_ARRAY_PATH "SSE2 128-bit"
#else
#define BIT_ARRAY_PATH "scalar"
#endif

#ifdef BIT_ARRAY_USE_VECTOR
namespace Common::BitArrayDetail {
#if defined(BIT_ARRAY_USE_AVX)
inline constexpr size_t WordsPerVector = 4;
using Vector = __m256i;
inline Vector Load(const u64* p) {
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
}
inline void Store(u64* p, Vector v) {
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), v);
}
inline Vector Ones() {
    return _mm256_set1_epi64x(-1);
}
inline Vector Zeros() {
    return _mm256_setzero_si256();
}
/// True when every bit is set: CF from vptest is (Ones & ~v) == 0.
inline bool AllOnes(Vector v) {
    return _mm256_testc_si256(v, Ones());
}
/// True when any bit is set: ZF from vptest is (v & v) == 0.
inline bool AnySet(Vector v) {
    return !_mm256_testz_si256(v, v);
}
/// True when the two vectors are equal: ZF from vptest on their XOR.
inline bool Equal(Vector a, Vector b) {
    const Vector difference = _mm256_castps_si256(
        _mm256_xor_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b)));
    return _mm256_testz_si256(difference, difference);
}
// AVX1 has no 256-bit *integer* bitwise instructions (those are AVX2); the single-precision
// versions are bitwise operations on the same 256 bits -- no arithmetic, no exceptions, no NaN
// or denormal handling -- so they are exactly what the words need. On AVX2 builds clang and GCC
// select the integer forms for these anyway.
inline Vector And(Vector a, Vector b) {
    return _mm256_castps_si256(
        _mm256_and_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b)));
}
inline Vector Or(Vector a, Vector b) {
    return _mm256_castps_si256(_mm256_or_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b)));
}
inline Vector Xor(Vector a, Vector b) {
    return _mm256_castps_si256(
        _mm256_xor_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b)));
}
inline Vector Not(Vector v) {
    return _mm256_castps_si256(_mm256_xor_ps(_mm256_castsi256_ps(v), _mm256_castsi256_ps(Ones())));
}
#else
inline constexpr size_t WordsPerVector = 2;
using Vector = __m128i;
inline Vector Load(const u64* p) {
    return _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
}
inline void Store(u64* p, Vector v) {
    _mm_storeu_si128(reinterpret_cast<__m128i*>(p), v);
}
inline Vector Ones() {
    return _mm_set1_epi64x(-1);
}
inline Vector Zeros() {
    return _mm_setzero_si128();
}
inline bool AllOnes(Vector v) {
    return _mm_movemask_epi8(_mm_cmpeq_epi8(v, _mm_set1_epi8(-1))) == 0xFFFF;
}
inline bool AnySet(Vector v) {
    return _mm_movemask_epi8(_mm_cmpeq_epi8(v, Zeros())) != 0xFFFF;
}
inline bool Equal(Vector a, Vector b) {
    return _mm_movemask_epi8(_mm_cmpeq_epi8(a, b)) == 0xFFFF;
}
inline Vector And(Vector a, Vector b) {
    return _mm_and_si128(a, b);
}
inline Vector Or(Vector a, Vector b) {
    return _mm_or_si128(a, b);
}
inline Vector Xor(Vector a, Vector b) {
    return _mm_xor_si128(a, b);
}
inline Vector Not(Vector v) {
    return _mm_xor_si128(v, Ones());
}
#endif
} // namespace Common::BitArrayDetail
#endif

namespace Common {

template <size_t N>
class BitArray {
    static_assert(N % 64 == 0, "BitArray size must be a multiple of 64 bits.");

    static constexpr size_t BITS_PER_WORD = 64;
    static constexpr size_t WORD_COUNT = N / BITS_PER_WORD;
#ifdef BIT_ARRAY_USE_VECTOR
    static constexpr size_t WORDS_PER_VECTOR = BitArrayDetail::WordsPerVector;
#endif

public:
    using Range = std::pair<size_t, size_t>;

    class Iterator {
    public:
        explicit Iterator(const BitArray& bit_array_, u64 start) : bit_array(bit_array_) {
            range = bit_array.FirstRangeFrom(start);
        }

        Iterator& operator++() {
            range = bit_array.FirstRangeFrom(range.second);
            return *this;
        }

        bool operator==(const Iterator& other) const {
            return range == other.range;
        }

        bool operator!=(const Iterator& other) const {
            return !(*this == other);
        }

        const Range& operator*() const {
            return range;
        }

        const Range* operator->() const {
            return &range;
        }

    private:
        const BitArray& bit_array;
        Range range;
    };

    using const_iterator = Iterator;
    using iterator_category = std::forward_iterator_tag;
    using value_type = Range;
    using difference_type = std::ptrdiff_t;
    using pointer = const Range*;
    using reference = const Range&;

    BitArray() = default;
    BitArray(const BitArray& other) = default;
    BitArray& operator=(const BitArray& other) = default;
    BitArray(BitArray&& other) noexcept = default;
    BitArray& operator=(BitArray&& other) noexcept = default;
    ~BitArray() = default;

    BitArray(const BitArray& other, size_t start, size_t end) {
        if (start >= end || end > N) {
            return;
        }
        const size_t first_word = start / BITS_PER_WORD;
        const size_t last_word = (end - 1) / BITS_PER_WORD;
        const size_t start_bit = start % BITS_PER_WORD;
        const size_t end_bit = (end - 1) % BITS_PER_WORD;
        const u64 start_mask = ~((1ULL << start_bit) - 1);
        const u64 end_mask = end_bit == BITS_PER_WORD - 1 ? ~0ULL : (1ULL << (end_bit + 1)) - 1;
        if (first_word == last_word) {
            data[first_word] = other.data[first_word] & (start_mask & end_mask);
        } else {
            data[first_word] = other.data[first_word] & start_mask;
            size_t i = first_word + 1;
#ifdef BIT_ARRAY_USE_VECTOR
            for (; i + WORDS_PER_VECTOR <= last_word; i += WORDS_PER_VECTOR) {
                BitArrayDetail::Store(&data[i], BitArrayDetail::Load(&other.data[i]));
            }
#endif
            for (; i < last_word; ++i) {
                data[i] = other.data[i];
            }
            data[last_word] = other.data[last_word] & end_mask;
        }
    }

    BitArray(const BitArray& other, const Range& range)
        : BitArray(other, range.first, range.second) {}

    const_iterator begin() const {
        return Iterator(*this, 0);
    }
    const_iterator end() const {
        return Iterator(*this, N);
    }

    inline constexpr void Set(size_t idx) {
        data[idx / BITS_PER_WORD] |= (1ULL << (idx % BITS_PER_WORD));
    }

    inline constexpr void Unset(size_t idx) {
        data[idx / BITS_PER_WORD] &= ~(1ULL << (idx % BITS_PER_WORD));
    }

    inline constexpr bool Get(size_t idx) const {
        return (data[idx / BITS_PER_WORD] & (1ULL << (idx % BITS_PER_WORD))) != 0;
    }

    inline void SetRange(size_t start, size_t end) {
        if (start >= end || end > N) {
            return;
        }
        const size_t first_word = start / BITS_PER_WORD;
        const size_t last_word = (end - 1) / BITS_PER_WORD;
        const size_t start_bit = start % BITS_PER_WORD;
        const size_t end_bit = (end - 1) % BITS_PER_WORD;
        const u64 start_mask = ~((1ULL << start_bit) - 1);
        const u64 end_mask = end_bit == BITS_PER_WORD - 1 ? ~0ULL : (1ULL << (end_bit + 1)) - 1;
        if (first_word == last_word) {
            data[first_word] |= start_mask & end_mask;
        } else {
            data[first_word] |= start_mask;
            size_t i = first_word + 1;
#ifdef BIT_ARRAY_USE_VECTOR
            const auto value = BitArrayDetail::Ones();
            for (; i + WORDS_PER_VECTOR <= last_word; i += WORDS_PER_VECTOR) {
                BitArrayDetail::Store(&data[i], value);
            }
#endif
            for (; i < last_word; ++i) {
                data[i] = ~0ULL;
            }
            data[last_word] |= end_mask;
        }
    }

    inline void UnsetRange(size_t start, size_t end) {
        if (start >= end || end > N) {
            return;
        }
        size_t first_word = start / BITS_PER_WORD;
        const size_t last_word = (end - 1) / BITS_PER_WORD;
        const size_t start_bit = start % BITS_PER_WORD;
        const size_t end_bit = (end - 1) % BITS_PER_WORD;
        const u64 start_mask = (1ULL << start_bit) - 1;
        const u64 end_mask = end_bit == BITS_PER_WORD - 1 ? 0ULL : ~((1ULL << (end_bit + 1)) - 1);
        if (first_word == last_word) {
            data[first_word] &= start_mask | end_mask;
        } else {
            data[first_word] &= start_mask;
            size_t i = first_word + 1;
#ifdef BIT_ARRAY_USE_VECTOR
            const auto value = BitArrayDetail::Zeros();
            for (; i + WORDS_PER_VECTOR <= last_word; i += WORDS_PER_VECTOR) {
                BitArrayDetail::Store(&data[i], value);
            }
#endif
            for (; i < last_word; ++i) {
                data[i] = 0ULL;
            }
            data[last_word] &= end_mask;
        }
    }

    inline constexpr void SetRange(const Range& range) {
        SetRange(range.first, range.second);
    }

    inline constexpr void UnsetRange(const Range& range) {
        UnsetRange(range.first, range.second);
    }

    inline constexpr void Clear() {
        data.fill(0);
    }

    inline constexpr void Fill() {
        data.fill(~0ULL);
    }

    inline constexpr bool None() const {
        size_t i = 0;
#ifdef BIT_ARRAY_USE_VECTOR
        // Whole vectors at a time, combined into one accumulator and tested once at the end: no
        // branch per vector (the region manager asks this after every change to the tracked
        // pages, and an answer of "no bits" is the common one).
        if (!std::is_constant_evaluated()) {
            if (WORD_COUNT >= WORDS_PER_VECTOR) {
                BitArrayDetail::Vector any = BitArrayDetail::Load(&data[0]);
                for (i = WORDS_PER_VECTOR; i + WORDS_PER_VECTOR <= WORD_COUNT;
                     i += WORDS_PER_VECTOR) {
                    any = BitArrayDetail::Or(any, BitArrayDetail::Load(&data[i]));
                }
                if (BitArrayDetail::AnySet(any)) {
                    return false;
                }
            }
        }
#endif
        u64 result = 0;
        for (; i < WORD_COUNT; ++i) {
            result |= data[i];
        }
        return result == 0;
    }

    inline constexpr bool Any() const {
        return !None();
    }

    inline constexpr bool AnyInRange(size_t start, size_t end) const {
        const size_t first_word = start / BITS_PER_WORD;
        const size_t last_word = (end - 1) / BITS_PER_WORD;
        const size_t start_bit = start % BITS_PER_WORD;
        const size_t end_bit = (end - 1) % BITS_PER_WORD;
        const u64 start_mask = ~((1ULL << start_bit) - 1);
        const u64 end_mask = end_bit == BITS_PER_WORD - 1 ? ~0ULL : (1ULL << (end_bit + 1)) - 1;
        if (first_word == last_word) {
            return data[first_word] & (start_mask & end_mask);
        }
        if (data[first_word] & start_mask) {
            return true;
        }
        size_t i = first_word + 1;
#ifdef BIT_ARRAY_USE_VECTOR
        // The words strictly inside the range, four at a time: this is the check the region
        // manager makes for every buffer binding (IsRegionModified / ForEachModifiedRange).
        if (!std::is_constant_evaluated()) {
            for (; i + WORDS_PER_VECTOR <= last_word; i += WORDS_PER_VECTOR) {
                if (BitArrayDetail::AnySet(BitArrayDetail::Load(&data[i]))) {
                    return true;
                }
            }
        }
#endif
        for (; i < last_word; ++i) {
            if (data[i]) {
                return true;
            }
        }
        if (data[last_word] & end_mask) {
            return true;
        }
        return false;
    }

    Range FirstRangeFrom(size_t start) const {
        if (start >= N) {
            return {N, N};
        }
        const auto find_end_bit = [&](size_t word) {
#ifdef BIT_ARRAY_USE_VECTOR
            for (; word + WORDS_PER_VECTOR <= WORD_COUNT; word += WORDS_PER_VECTOR) {
                if (!BitArrayDetail::AllOnes(BitArrayDetail::Load(&data[word]))) {
                    break;
                }
            }
#endif
            for (; word < WORD_COUNT; ++word) {
                if (data[word] != ~0ULL) {
                    return (word * BITS_PER_WORD) + std::countr_one(data[word]);
                }
            }
            return N;
        };

        const auto word_bits = [&](size_t index, u64 word) {
            const size_t empty_bits = std::countr_zero(word);
            const size_t ones_count = std::countr_one(word >> empty_bits);
            const size_t start_bit = index * BITS_PER_WORD + empty_bits;
            if (ones_count + empty_bits < BITS_PER_WORD) {
                return Range{start_bit, start_bit + ones_count};
            }
            return Range{start_bit, find_end_bit(index + 1)};
        };

        const size_t start_word = start / BITS_PER_WORD;
        const size_t start_bit = start % BITS_PER_WORD;
        const u64 masked_first = data[start_word] & (~((1ULL << start_bit) - 1));
        if (masked_first) {
            return word_bits(start_word, masked_first);
        }

        size_t word = start_word + 1;
#ifdef BIT_ARRAY_USE_VECTOR
        for (; word + WORDS_PER_VECTOR <= WORD_COUNT; word += WORDS_PER_VECTOR) {
            if (BitArrayDetail::AnySet(BitArrayDetail::Load(&data[word]))) {
                break;
            }
        }
#endif
        for (; word < WORD_COUNT; ++word) {
            if (data[word] != 0) {
                return word_bits(word, data[word]);
            }
        }
        return {N, N};
    }

    inline constexpr Range FirstRange() const {
        return FirstRangeFrom(0);
    }

    Range LastRangeFrom(size_t end) const {
        if (end == 0) {
            return {0, 0};
        }
        if (end > N) {
            end = N;
        }
        const auto find_start_bit = [&](size_t word) {
#ifdef BIT_ARRAY_USE_VECTOR
            // Backwards over runs of all-ones words: the run ends at the first word that is not
            // all ones. (shadPS4's AVX2 version of this loop compared against zero instead, so
            // it ran to word 0 whenever the words below the run were empty and the range then
            // started at bit 0 -- an off-by-a-run bug this port would have hit with AVX2.)
            for (; word >= WORDS_PER_VECTOR; word -= WORDS_PER_VECTOR) {
                if (!BitArrayDetail::AllOnes(
                        BitArrayDetail::Load(&data[word - WORDS_PER_VECTOR]))) {
                    break;
                }
            }
#endif
            for (; word > 0; --word) {
                if (data[word - 1] != ~0ULL) {
                    return word * BITS_PER_WORD - std::countl_one(data[word - 1]);
                }
            }
            return size_t(0);
        };
        const auto word_bits = [&](size_t index, u64 word) {
            const size_t empty_bits = std::countl_zero(word);
            const size_t ones_count = std::countl_one(word << empty_bits);
            const size_t end_bit = index * BITS_PER_WORD - empty_bits;
            if (empty_bits + ones_count < BITS_PER_WORD) {
                return Range{end_bit - ones_count, end_bit};
            }
            return Range{find_start_bit(index - 1), end_bit};
        };
        const size_t end_word = ((end - 1) / BITS_PER_WORD) + 1;
        const size_t end_bit = (end - 1) % BITS_PER_WORD;
        u64 masked_last = data[end_word - 1];
        if (end_bit < BITS_PER_WORD - 1) {
            masked_last &= (1ULL << (end_bit + 1)) - 1;
        }
        if (masked_last) {
            return word_bits(end_word, masked_last);
        }
        size_t word = end_word - 1;
#ifdef BIT_ARRAY_USE_VECTOR
        for (; word >= WORDS_PER_VECTOR; word -= WORDS_PER_VECTOR) {
            if (BitArrayDetail::AnySet(BitArrayDetail::Load(&data[word - WORDS_PER_VECTOR]))) {
                break;
            }
        }
#endif
        for (; word > 0; --word) {
            if (data[word - 1] != 0) {
                return word_bits(word, data[word - 1]);
            }
        }
        return {0, 0};
    }

    inline constexpr Range LastRange() const {
        return LastRangeFrom(N);
    }

    inline constexpr size_t Size() const {
        return N;
    }

    inline constexpr BitArray& operator|=(const BitArray& other) {
        size_t i = 0;
#ifdef BIT_ARRAY_USE_VECTOR
        if (!std::is_constant_evaluated()) {
            for (; i + WORDS_PER_VECTOR <= WORD_COUNT; i += WORDS_PER_VECTOR) {
                BitArrayDetail::Store(&data[i],
                                      BitArrayDetail::Or(BitArrayDetail::Load(&data[i]),
                                                         BitArrayDetail::Load(&other.data[i])));
            }
        }
#endif
        for (; i < WORD_COUNT; ++i) {
            data[i] |= other.data[i];
        }
        return *this;
    }

    inline constexpr BitArray& operator&=(const BitArray& other) {
        size_t i = 0;
#ifdef BIT_ARRAY_USE_VECTOR
        if (!std::is_constant_evaluated()) {
            for (; i + WORDS_PER_VECTOR <= WORD_COUNT; i += WORDS_PER_VECTOR) {
                BitArrayDetail::Store(&data[i],
                                      BitArrayDetail::And(BitArrayDetail::Load(&data[i]),
                                                          BitArrayDetail::Load(&other.data[i])));
            }
        }
#endif
        for (; i < WORD_COUNT; ++i) {
            data[i] &= other.data[i];
        }
        return *this;
    }

    inline constexpr BitArray& operator^=(const BitArray& other) {
        size_t i = 0;
#ifdef BIT_ARRAY_USE_VECTOR
        if (!std::is_constant_evaluated()) {
            for (; i + WORDS_PER_VECTOR <= WORD_COUNT; i += WORDS_PER_VECTOR) {
                BitArrayDetail::Store(&data[i],
                                      BitArrayDetail::Xor(BitArrayDetail::Load(&data[i]),
                                                          BitArrayDetail::Load(&other.data[i])));
            }
        }
#endif
        for (; i < WORD_COUNT; ++i) {
            data[i] ^= other.data[i];
        }
        return *this;
    }

    inline constexpr BitArray operator|(const BitArray& other) const {
        BitArray result = *this;
        result |= other;
        return result;
    }

    inline constexpr BitArray operator&(const BitArray& other) const {
        BitArray result = *this;
        result &= other;
        return result;
    }

    inline constexpr BitArray operator^(const BitArray& other) const {
        BitArray result = *this;
        result ^= other;
        return result;
    }

    inline constexpr BitArray operator~() const {
        BitArray result = *this;
        size_t i = 0;
#ifdef BIT_ARRAY_USE_VECTOR
        if (!std::is_constant_evaluated()) {
            for (; i + WORDS_PER_VECTOR <= WORD_COUNT; i += WORDS_PER_VECTOR) {
                BitArrayDetail::Store(&result.data[i],
                                      BitArrayDetail::Not(BitArrayDetail::Load(&data[i])));
            }
        }
#endif
        for (; i < WORD_COUNT; ++i) {
            result.data[i] = ~data[i];
        }
        return result;
    }

    inline constexpr bool operator==(const BitArray& other) const {
        size_t i = 0;
#ifdef BIT_ARRAY_USE_VECTOR
        if (!std::is_constant_evaluated()) {
            for (; i + WORDS_PER_VECTOR <= WORD_COUNT; i += WORDS_PER_VECTOR) {
                if (!BitArrayDetail::Equal(BitArrayDetail::Load(&data[i]),
                                           BitArrayDetail::Load(&other.data[i]))) {
                    return false;
                }
            }
        }
#endif
        u64 result = 0;
        for (; i < WORD_COUNT; ++i) {
            result |= data[i] ^ other.data[i];
        }
        return result == 0;
    }

    inline constexpr bool operator!=(const BitArray& other) const {
        return !(*this == other);
    }

    /// bbport: (~a ^ b) in one pass. The buffer cache's region manager derives the pages whose
    /// protection changes from exactly this expression on every update; writing it as operator~
    /// followed by operator^ copied the array twice and walked it twice.
    static constexpr BitArray NotXor(const BitArray& a, const BitArray& b) {
        BitArray result = a;
        size_t i = 0;
#ifdef BIT_ARRAY_USE_VECTOR
        if (!std::is_constant_evaluated()) {
            for (; i + WORDS_PER_VECTOR <= WORD_COUNT; i += WORDS_PER_VECTOR) {
                BitArrayDetail::Store(&result.data[i],
                                      BitArrayDetail::Not(BitArrayDetail::Xor(
                                          BitArrayDetail::Load(&a.data[i]),
                                          BitArrayDetail::Load(&b.data[i]))));
            }
        }
#endif
        for (; i < WORD_COUNT; ++i) {
            result.data[i] = ~(a.data[i] ^ b.data[i]);
        }
        return result;
    }

    /// bbport: ~(a | b) in one pass (De Morgan's other half -- NOT the same as NotXor, which
    /// clears where both operands are set).
    static constexpr BitArray NotOr(const BitArray& a, const BitArray& b) {
        BitArray result = a;
        size_t i = 0;
#ifdef BIT_ARRAY_USE_VECTOR
        if (!std::is_constant_evaluated()) {
            for (; i + WORDS_PER_VECTOR <= WORD_COUNT; i += WORDS_PER_VECTOR) {
                BitArrayDetail::Store(&result.data[i],
                                      BitArrayDetail::Not(BitArrayDetail::Or(
                                          BitArrayDetail::Load(&a.data[i]),
                                          BitArrayDetail::Load(&b.data[i]))));
            }
        }
#endif
        for (; i < WORD_COUNT; ++i) {
            result.data[i] = ~(a.data[i] | b.data[i]);
        }
        return result;
    }

private:
    std::array<u64, WORD_COUNT> data{};
};

} // namespace Common
