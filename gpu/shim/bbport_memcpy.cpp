// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: see bbport_memcpy.h.
#include "bbport_memcpy.h"

#include <cstdlib>
#include <cstdint>

#if defined(__AVX__)
#include <immintrin.h>
#define BBPORT_STREAM_COPY 1
#endif

namespace BbCopy {
namespace {

/// BB_STREAM_COPY=0 turns the streaming copy off (A/B comparisons, or a machine where the
/// destination is read back by the CPU straight away).
bool StreamingEnabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("BB_STREAM_COPY");
        return !(env && env[0] == '0');
    }();
    return enabled;
}

} // namespace

bool HaveStreamingCopy() noexcept {
#ifdef BBPORT_STREAM_COPY
    return true;
#else
    return false;
#endif
}

#if defined(BBPORT_STREAM_COPY)

void Copy(void* destination, const void* source, std::size_t size,
          CopyPolicy policy) noexcept {
    const bool stream =
        StreamingEnabled() && (policy == CopyPolicy::Stream ||
                               (policy == CopyPolicy::Auto && size >= AutoStreamBytes));
    if (!stream) {
        std::memcpy(destination, source, size);
        return;
    }
    auto* dst = static_cast<unsigned char*>(destination);
    const auto* src = static_cast<const unsigned char*>(source);

    // `vmovntdq` needs a 32-byte aligned destination: copy the head byte-wise until it is.
    const std::size_t head = (32 - (reinterpret_cast<std::uintptr_t>(dst) & 31)) & 31;
    if (head != 0) {
        const std::size_t n = head < size ? head : size;
        std::memcpy(dst, src, n);
        dst += n;
        src += n;
        size -= n;
    }
    // Body: 128 bytes per iteration (four 32-byte stores), the shape recent cores prefetch
    // best. The source is read with unaligned loads: guest pages are only 4 KiB aligned.
    while (size >= 128) {
        const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src));
        const __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 32));
        const __m256i c = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 64));
        const __m256i d = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 96));
        _mm256_stream_si256(reinterpret_cast<__m256i*>(dst), a);
        _mm256_stream_si256(reinterpret_cast<__m256i*>(dst + 32), b);
        _mm256_stream_si256(reinterpret_cast<__m256i*>(dst + 64), c);
        _mm256_stream_si256(reinterpret_cast<__m256i*>(dst + 96), d);
        dst += 128;
        src += 128;
        size -= 128;
    }
    while (size >= 32) {
        _mm256_stream_si256(reinterpret_cast<__m256i*>(dst),
                            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src)));
        dst += 32;
        src += 32;
        size -= 32;
    }
    if (size != 0) {
        std::memcpy(dst, src, size);
    }
    // The stores are weakly ordered: make them visible before the GPU (or another thread) is
    // told that the copy is done.
    _mm_sfence();
}

#else

void Copy(void* destination, const void* source, std::size_t size, CopyPolicy) noexcept {
    std::memcpy(destination, source, size);
}

#endif

} // namespace BbCopy
