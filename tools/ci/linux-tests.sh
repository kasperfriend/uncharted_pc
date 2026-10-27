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
echo '== streaming guest-memory copy'
"$cxx" -std=c++20 -O2 -g -Wall -Wextra -Werror -mavx -Igpu/shim tests/test_stream_copy.cpp \
    gpu/shim/bbport_memcpy.cpp -o "$out/stream-copy-test"
"$out/stream-copy-test"

echo 'All host-side tests passed.'
