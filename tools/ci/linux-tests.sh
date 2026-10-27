#!/usr/bin/env bash
# The port's host-side tests that need no GPU, no Vulkan and no SDL3: what CI can run on Linux
# (and what can be run in a plain shell before touching the graphics side).
#
#   bash tools/ci/linux-tests.sh
#
# Needs gcc/g++ and Zydis: from the distribution (Debian/Ubuntu: libzydis-dev), or the script
# builds Zydis v4.1.1 and Zycore v1.5.0 from source into out/tests/zydis/ (needs curl and tar).
set -euo pipefail
cd -- "$(dirname -- "$0")/../.."
out=out/tests
rm -rf "$out"
mkdir -p "$out"

# --- Zydis -----------------------------------------------------------------------------------
zydis_inc=()
zydis_lib=()
if [[ -f /usr/include/Zydis/Zydis.h || -f /usr/local/include/Zydis/Zydis.h ]]; then
    zydis_lib=(-lZydis)
    echo 'Zydis: system'
else
    echo 'Zydis: building v4.1.1 from source'
    src=$out/zydis/src
    mkdir -p "$src"
    fetch() { curl -fsSL --retry 3 -o "$src/$2" "https://codeload.github.com/$1/tar.gz/refs/tags/$2"; }
    fetch zyantific/zydis v4.1.1
    fetch zyantific/zycore-c v1.5.0
    mkdir -p "$src/zydis" "$src/zycore"
    tar xzf "$src/v4.1.1" -C "$src/zydis" --strip-components=1
    tar xzf "$src/v1.5.0" -C "$src/zycore" --strip-components=1
    zy=$src/zycore
    zd=$src/zydis
    flags=(-std=gnu99 -O2 -fPIC -DZYAN_INCLUDE_LIBC=1 -I"$zy/include" -I"$zy/src")
    for source in "$zy"/src/*.c; do
        gcc "${flags[@]}" -c "$source" -o "$src/zc_$(basename "${source%.c}").o" 2>>"$out/zydis-build.log"
    done
    zflags=(-std=gnu99 -O2 -fPIC -DZYDIS_STATIC_BUILD=1 -DZYAN_INCLUDE_LIBC=1
            -I"$zy/include" -I"$zd/include" -I"$zd/src")
    for source in "$zd"/src/*.c; do
        gcc "${zflags[@]}" -c "$source" -o "$src/zd_$(basename "${source%.c}").o" 2>>"$out/zydis-build.log"
    done
    ar rcs "$src/libzydis.a" "$src"/*.o
    zydis_inc=(-I"$zd/include" -I"$zy/include")
    zydis_lib=("$src/libzydis.a")
fi

c11=(-std=c11 -O2 -g -Wall -Wextra -Werror)

# --- CPU instruction rewriting ----------------------------------------------------------------
# The test runs each instruction natively first and then rewritten, so the CPU must have them.
if grep -qw bmi1 /proc/cpuinfo && grep -qw movbe /proc/cpuinfo && grep -qw abm /proc/cpuinfo; then
    gcc "${c11[@]}" -pthread -Isrc "${zydis_inc[@]}" tests/test_cpu_patch.c src/cpu_patch.c \
        "${zydis_lib[@]}" -o "$out/cpu-patch-test"
    echo '== cpu_patch: every rewritten instruction against the native one'
    BB_CPU_PATCH=all "$out/cpu-patch-test" | tee "$out/cpu-patch-auto.log"
    # The sweep runs on one thread per core: its summary must be exactly what one thread gives.
    BB_CPU_PATCH=all BB_CPU_PATCH_THREADS=1 "$out/cpu-patch-test" > "$out/cpu-patch-serial.log"
    diff <(grep '^CPU: rewrite' "$out/cpu-patch-serial.log") \
         <(grep '^CPU: rewrite' "$out/cpu-patch-auto.log")
else
    echo '== cpu_patch: skipped (this CPU lacks BMI1/LZCNT/MOVBE, which the test runs natively)'
fi

# --- HLE runtime -----------------------------------------------------------------------------
echo '== guest file operations'
gcc "${c11[@]}" -pthread -Isrc tests/test_file_mods.c -o "$out/file-mods-test"
"$out/file-mods-test"
echo '== content profile'
gcc "${c11[@]}" -D_GNU_SOURCE -I. -Isrc tests/test_content.c src/runtime_content.c -o "$out/content-test"
"$out/content-test"
echo '== display configuration patch'
# test_display.c and runtime_display.c define _GNU_SOURCE themselves.
gcc "${c11[@]}" -I. -Isrc tests/test_display.c src/runtime_display.c -o "$out/display-test"
"$out/display-test"

# --- guest runtime build shape ----------------------------------------------------------------
echo '== guest runtime compiles with the port code generation'
# The HLE runtime is plain C and is built at -O3 with -Werror: warnings that only appear with
# that optimization level and the build's instruction set (a helper that is used on Windows
# only, snprintf truncation into a fixed guest field) break the game build. The files that
# need SDL3/libavcodec headers are built by build.sh, not here.
rt_flags=(-std=c11 -O3 -g -Wall -Wextra -Werror -pthread -I. -Isrc "${zydis_inc[@]}")
if [[ $(uname -m) == x86_64 ]]; then
    rt_flags+=(-march=x86-64-v2 -mavx -mf16c -mno-avx2 -mno-fma -mno-bmi -mno-bmi2
               -mtune=core-avx-i -mprefer-vector-width=128)
fi
for source in src/probe.c src/cpu_patch.c $(ls src/runtime_*.c | grep -v -e runtime_audio.c -e runtime_pad.c -e runtime_ajm.c); do
    gcc "${rt_flags[@]}" -fsyntax-only "$source"
done
echo '   ok'

# --- host helpers the port adds ---------------------------------------------------------------
cxx=${CXX:-g++}
echo '== bit array (vector scans: AVX 256-bit, SSE2, scalar)'
for variant in "-O0" "-O2" "-O2 -mavx -mno-avx2" "-O2 -mavx2" "-O2 -msse4.2 -mno-avx"; do
    # shellcheck disable=SC2086
    "$cxx" -std=c++20 -g -Wall -Wextra -Werror $variant -Igpu/shadps4 tests/test_bit_array.cpp \
        -o "$out/bit-array-test"
    printf '   %-22s ' "$variant"
    "$out/bit-array-test" | tail -1
done
echo '== bit array timings (AVX 256-bit against SSE2, same source)'
for variant in "-mavx -mno-avx2" "-mno-avx"; do
    # shellcheck disable=SC2086
    "$cxx" -std=c++20 -O2 -g $variant -Igpu/shadps4 tests/bench_bit_array.cpp \
        -o "$out/bench-bit-array"
    printf '   %-18s ' "$variant"
    "$out/bench-bit-array"
done
echo '== guest semaphores and rwlocks'
# Both files are self-contained C: the test drives them through the same resolver the guest
# import table uses (many live objects, FIFO waiters, timeouts, cancel, delete, rwlock owner rules).
gcc "${c11[@]}" -pthread -I. -Isrc tests/test_runtime_sync.c src/runtime_sema.c \
    src/runtime_rwlock.c -o "$out/runtime-sync-test"
"$out/runtime-sync-test" | tail -1
# The register-block mask the draw pipe and the draw-preparation workers walk on every
# submission (video_core/amdgpu/block_mask.h): header-only, so it needs nothing else of the GPU.
"$cxx" -std=c++20 -O2 -g -Wall -Wextra -Werror -Igpu/shadps4 tests/test_block_mask.cpp \
    -o "$out/block-mask-test"
"$out/block-mask-test"
# The page-use table the buffer cache stamps for every binding, and the residency span it keeps
# for the next one (video_core/buffer_cache/page_stamp.h, common/interval_set.h).
"$cxx" -std=c++20 -O2 -g -Wall -Wextra -Werror -Igpu/shadps4 tests/test_page_stamp.cpp \
    -o "$out/page-stamp-test"
"$out/page-stamp-test"
"$cxx" -std=c++20 -O2 -g -Wall -Wextra -Werror -Igpu/shadps4 tests/test_resident_span.cpp \
    -o "$out/resident-span-test"
"$out/resident-span-test"
# ... and both again with the sanitizers, because the stamping reads and writes a page table by
# vector and the span is memory the previous query left behind.
if "$cxx" -fsanitize=address,undefined -x c++ -c /dev/null -o /dev/null 2>/dev/null; then
    echo '== page stamp and residency span, ASan + UBSan'
    for source in tests/test_page_stamp.cpp tests/test_resident_span.cpp; do
        "$cxx" -std=c++20 -O1 -g -fsanitize=address,undefined -fno-sanitize-recover=all \
            -Igpu/shadps4 "$source" -o "$out/sanitized-test"
        "$out/sanitized-test" | tail -1
    done
fi
gcc -std=c11 -O2 -g -Wall -Wextra -Werror -pthread tests/bench_runtime_sync.c -o "$out/runtime-sync-bench"
"$out/runtime-sync-bench"
# The same two files under 10 threads at once (timeouts racing signals, delete with waiters).
gcc "${c11[@]}" -pthread -I. -Isrc tests/stress_runtime_sync.c src/runtime_sema.c \
    -o "$out/runtime-sync-stress"
"$out/runtime-sync-stress" | tail -1
# ThreadSanitizer, when the compiler carries it: the semaphore and rwlock state is shared by guest
# threads and by the GPU side, and a data race there is a hang or a lost wakeup in the game rather
# than a crash here. Any report fails the run (halt_on_error + exitcode).
if gcc -fsanitize=thread -x c -c /dev/null -o /dev/null 2>/dev/null; then
    echo "== ThreadSanitizer"
    tsan=(-fsanitize=thread -g -O1 -pthread -I. -Isrc)
    gcc -std=c11 "${tsan[@]}" tests/test_runtime_sync.c src/runtime_sema.c src/runtime_rwlock.c \
        -o "$out/runtime-sync-tsan"
    gcc -std=c11 "${tsan[@]}" tests/stress_runtime_sync.c src/runtime_sema.c \
        -o "$out/runtime-sync-stress-tsan"
    export TSAN_OPTIONS="halt_on_error=0 exitcode=66 second_deadlock_stack=1"
    "$out/runtime-sync-tsan" >"$out/runtime-sync-tsan.log" 2>&1 || true
    "$out/runtime-sync-stress-tsan" >>"$out/runtime-sync-tsan.log" 2>&1 || true
    if grep -q 'WARNING: ThreadSanitizer' "$out/runtime-sync-tsan.log"; then
        grep -m 5 'WARNING: ThreadSanitizer' -A 20 "$out/runtime-sync-tsan.log" >&2
        echo 'ThreadSanitizer reported a data race (out/runtime-sync-tsan.log)' >&2
        exit 1
    fi
    echo "   no data races reported"
else
    echo '== ThreadSanitizer not available in this compiler, skipped'
fi

echo '== streaming guest-memory copy'
"$cxx" -std=c++20 -O2 -g -Wall -Wextra -Werror -mavx -Igpu/shim tests/test_stream_copy.cpp \
    gpu/shim/bbport_memcpy.cpp -o "$out/stream-copy-test"
"$out/stream-copy-test"

echo 'All host-side tests passed.'
