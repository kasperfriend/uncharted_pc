#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
mkdir -p out
# Windows: MSYS2 CLANG64 (packages in packaging/windows/README.md). The GPU library is static
# there and CMake links bb-probe.exe as well (ThinLTO through lld, PGO through tools/pgo.sh).
case $(uname -s) in
MINGW*|MSYS*)
    # shellcheck source=msys2-env.sh
    source ./msys2-env.sh
    for tool in clang cmake ninja pkg-config; do
        command -v "$tool" >/dev/null || { echo "Need $tool from MSYS2 CLANG64 (packaging/windows/README.md)." >&2; exit 1; }
    done
    if [[ ! -f gpu/third_party/fsr-vulkan/CMakeLists.txt || ! -f gpu/third_party/imgui/imgui.h ||
          ! -f gpu/third_party/xbyak/xbyak/xbyak.h ]]; then
        git submodule update --init --recursive
    fi
    for patch in gpu/patches/fsr-vulkan/*.patch; do
        if ! git -C gpu/third_party/fsr-vulkan apply --reverse --check "$PWD/$patch" 2>/dev/null; then
            git -C gpu/third_party/fsr-vulkan apply "$PWD/$patch"
        fi
    done
    # Host code generation: BB_ARCH (default avx: every PC the game runs on, and no AVX2),
    # BB_LTO (default THIN: ThinLTO through lld) and BB_PGO (generate, then use after
    # tools/pgo.sh merged the profile; it also builds the loader with it).
    pgo=${BB_PGO:-}
    if [[ -z $pgo ]]; then
        # tools/pgo.sh merge leaves the profile here: use it without being asked.
        [[ -f pgo/bbport.profdata ]] && pgo=use || pgo=off
    fi
    if [[ $pgo == use && ! -f pgo/bbport.profdata ]]; then
        echo 'BB_PGO=use needs pgo/bbport.profdata: bash tools/pgo.sh generate, run the game, bash tools/pgo.sh merge' >&2
        exit 1
    fi
    pgo_dir=$(cygpath -w "$PWD/pgo" 2>/dev/null || printf '%s' "$PWD/pgo")
    # Extra CMake options (CI passes ccache here): BB_CMAKE_ARGS="-DCMAKE_C_COMPILER_LAUNCHER=ccache".
    read -r -a extra_cmake <<< "${BB_CMAKE_ARGS:-}"
    if ! cmake -S gpu -B out/gpu -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DBB_ARCH="${BB_ARCH:-avx}" -DBB_LTO="${BB_LTO:-THIN}" -DBB_PGO="$pgo" \
        -DBB_PGO_DIR="$pgo_dir" -DBB_FRAME_POINTERS="${BB_FRAME_POINTERS:-ON}" \
        "${extra_cmake[@]}" \
        -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ > out/gpu-configure.log 2>&1; then
        tail -40 out/gpu-configure.log >&2
        echo 'CMake configure failed (full log: out/gpu-configure.log)' >&2; exit 1
    fi
    echo "GPU library and loader: ${BB_ARCH:-avx} code generation, LTO ${BB_LTO:-THIN}, PGO $pgo"
    if ! ninja -C out/gpu bb-probe bb-gpu-capabilities bb-play > out/gpu-build.log 2>&1; then
        grep -v '^\[' out/gpu-build.log | tail -40 >&2
        echo 'Build failed (full log: out/gpu-build.log)' >&2; exit 1
    fi
    cp out/gpu/bb-probe.exe out/gpu/bb-gpu-capabilities.exe out/gpu/bb-play.exe out/
    echo "Built $PWD/out/bb-probe.exe"
    exit 0
    ;;
esac
# Host code generation for the C runtime and the tools; the GPU library gets the same flags
# from gpu/CMakeLists.txt (see BB_ARCH there). The port's floor is AVX without AVX2: the guest
# game code needs AVX and the loader rewrites the BMI1/LZCNT/MOVBE instructions older CPUs lack.
arch_flags=()
if [[ $(uname -m) == x86_64 ]]; then
    case ${BB_ARCH:-avx} in
    native) arch_flags=(-march=native) ;;
    generic) arch_flags=() ;;
    avx2) arch_flags=(-march=x86-64-v3 -mtune=core-avx-i) ;;
    avx) arch_flags=(-march=x86-64-v2 -mavx -mf16c -mno-avx2 -mno-fma -mno-bmi -mno-bmi2
                     -mtune=core-avx-i -mprefer-vector-width=128) ;;
    *) arch_flags=(-march="${BB_ARCH:-avx}") ;;
    esac
fi
if [[ -z ${CC:-} ]]; then
    CC=$(command -v cc || command -v gcc || true)
    if [[ -z $CC ]]; then
        for candidate in /nix/store/*-gcc-wrapper-*/bin/gcc; do
            if [[ -x $candidate ]]; then CC=$candidate; break; fi
        done
    fi
fi
if [[ -z ${CC:-} ]]; then echo 'Install GCC/Clang or set CC.' >&2; exit 1; fi
# Dependencies come from pkg-config (Vulkan loader/headers, SDL3). On NixOS the
# environment is provided by shell.nix; re-enter it automatically if needed.
if ! { command -v pkg-config >/dev/null && pkg-config --exists vulkan sdl3 && command -v cmake >/dev/null && command -v ninja >/dev/null; }; then
    if [[ -z ${BB_IN_NIX_SHELL:-} ]] && command -v nix-shell >/dev/null; then
        exec env BB_IN_NIX_SHELL=1 nix-shell shell.nix --run "bash build.sh $*"
    fi
    echo 'Need pkg-config with vulkan and sdl3, cmake and ninja (see shell.nix).' >&2; exit 1
fi
read -r -a includes <<< "$(pkg-config --cflags vulkan sdl3)"
# Newer Vulkan headers than distributions ship (third_party/Vulkan-Headers; the video core uses
# VK_VALVE_shader_mixed_float_dot_product and other recent names).
[[ -f third_party/Vulkan-Headers/include/vulkan/vulkan.h ]] || git submodule update --init third_party/Vulkan-Headers
vulkan_headers=$PWD/third_party/Vulkan-Headers/include
includes=(-I"$vulkan_headers" "${includes[@]}")
read -r -a libraries <<< "$(pkg-config --libs vulkan sdl3)"
# GPU library (shadPS4 video core + drivers), built by CMake into out/gpu/libbbgpu.so.
# BB_PGO: generate (instrumented build that writes pgo/ while the game runs), use, off.
# Default: use the profile in pgo/ when there is one. BB_LTO: THIN (default), ON, OFF.
# BB_ARCH: avx (default, AVX without AVX2), avx2, native, generic or an -march= value.
pgo=${BB_PGO:-}
if [[ -z $pgo ]]; then
    # clang: pgo/bbport.profdata (tools/pgo.sh merge); GCC: the pgo/*.gcda files.
    if [[ -f pgo/bbport.profdata || -n $(find pgo -name '*.gcda' -print -quit 2>/dev/null) ]]; then
        pgo=use
    else
        pgo=off
    fi
fi
mkdir -p pgo
# Submodules (git clone --recursive, or: git submodule update --init) and this port's changes
# to FSR-Vulkan (gpu/patches/fsr-vulkan), applied to its working tree once.
if [[ ! -f gpu/third_party/fsr-vulkan/CMakeLists.txt || ! -f gpu/third_party/imgui/imgui.h ]]; then
    git submodule update --init --recursive
fi
for patch in gpu/patches/fsr-vulkan/*.patch; do
    if ! git -C gpu/third_party/fsr-vulkan apply --reverse --check "$PWD/$patch" 2>/dev/null; then
        git -C gpu/third_party/fsr-vulkan apply "$PWD/$patch"
    fi
done
read -r -a extra_cmake <<< "${BB_CMAKE_ARGS:-}"
if ! cmake -S gpu -B out/gpu -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBB_PGO="$pgo" \
    -DBB_ARCH="${BB_ARCH:-avx}" -DBB_LTO="${BB_LTO:-THIN}" -DBB_PGO_DIR="$PWD/pgo" \
    -DBB_FRAME_POINTERS="${BB_FRAME_POINTERS:-ON}" "${extra_cmake[@]}" \
    -DVulkan_INCLUDE_DIR="$vulkan_headers" > out/gpu-configure.log 2>&1; then
    tail -40 out/gpu-configure.log >&2
    echo 'CMake configure failed (full log: out/gpu-configure.log)' >&2; exit 1
fi
echo "GPU library: PGO $pgo, LTO ${BB_LTO:-THIN}, code generation ${BB_ARCH:-avx}"
# A failed GPU build must stop here: an older libbbgpu.so would otherwise be used silently.
if ! ninja ${BB_JOBS:+-j"$BB_JOBS"} -C out/gpu bbgpu > out/gpu-build.log 2>&1; then
    grep -v '^\[' out/gpu-build.log | tail -40 >&2
    echo 'GPU library build failed (full log: out/gpu-build.log)' >&2; exit 1
fi
# $ORIGIN/gpu: packaged copies keep the library next to the binary without patching it.
gpu=(-Lout/gpu -lbbgpu -Wl,-rpath,'$ORIGIN/gpu' -Wl,-rpath,"$PWD/out/gpu" -rdynamic)
runtime=(src/runtime*.c)
# Third-party decoders: compiled once, without this project's -Werror policy.
atrac9=(third_party/LibAtrac9/C/src/*.c)
if [[ ! -f out/libatrac9.a || -n $(find third_party/LibAtrac9/C/src -newer out/libatrac9.a -name '*.c') ]]; then
    rm -rf out/atrac9 && mkdir -p out/atrac9
    for source in "${atrac9[@]}"; do "$CC" -std=c99 -O2 -g -w -c "$source" -o "out/atrac9/$(basename "${source%.c}").o"; done
    ar rcs out/libatrac9.a out/atrac9/*.o
fi
# The C runtime is called from the guest on every file read, thread and memory operation:
# -O3 and the host instruction set (checked once: an older GCC/clang without -march=x86-64-v2
# falls back to the explicit feature flags).
cc_flags=("${arch_flags[@]}")
if [[ ${#cc_flags[@]} -gt 0 ]] && ! "$CC" "${cc_flags[@]}" -x c -E /dev/null >/dev/null 2>&1; then
    cc_flags=(-msse4.2 -mpopcnt -mcx16 -mavx -mf16c)
fi
# The runtime is one program spread over a dozen files: compile it once into out/rt (an
# unchanged file is not compiled again) and link with LTO, so the small helpers (error
# translation, file lookups, region queries) inline into the HLE functions that call them
# instead of crossing a file boundary, and the caller does not pay for the dead argument
# setup. -fno-plt keeps the calls into the GPU library down to one indirect call each, which
# matters for bbgpu_handle_fault: every guest write fault goes through it. BB_RT_LTO=OFF
# builds the runtime without LTO.
runtime_flags=(-std=c11 -O3 -g -Wall -Wextra -Werror -pthread "${cc_flags[@]}" "${includes[@]}" -I. -Isrc)
runtime_link=(-no-pie)
runtime_lto=off
if [[ ${BB_RT_LTO:-AUTO} != OFF ]] && ! "$CC" --version 2>/dev/null | grep -qi clang; then
    for candidate in -flto=auto -flto; do
        if "$CC" "$candidate" -x c -c /dev/null -o /dev/null >/dev/null 2>&1; then
            runtime_lto=$candidate
            break
        fi
    done
fi
if [[ $runtime_lto != off ]]; then
    runtime_flags+=("$runtime_lto")
    runtime_link+=("$runtime_lto")
fi
if "$CC" -fno-plt -x c -E /dev/null >/dev/null 2>&1; then
    runtime_flags+=(-fno-plt)
fi
mkdir -p out/rt
runtime_objects=()
for source in src/probe.c src/cpu_patch.c "${runtime[@]}" src/vulkan_smoke.c; do
    object="out/rt/$(basename "${source%.c}").o"
    runtime_objects+=("$object")
    if [[ ! -f $object || $source -nt $object || -n $(find src -name '*.h' -newer "$object" -print -quit) ]]; then
        "$CC" "${runtime_flags[@]}" -c "$source" -o "$object"
    fi
done
"$CC" "${runtime_link[@]}" "${runtime_objects[@]}" out/libatrac9.a -lm -lZydis "${gpu[@]}" "${libraries[@]}" -o out/bb-probe
echo "Built $PWD/out/bb-probe"
# GPU check for run.sh (live_resolution=auto): links only the Vulkan loader.
"$CC" -std=c11 -O2 -Wall -Wextra -Werror tools/gpu_capabilities.c "${libraries[@]}" -o out/bb-gpu-capabilities
if [[ ${1:-} == --test ]]; then
    "$CC" -std=c11 -O2 -g -Wall -Wextra -Werror -pthread "${includes[@]}" -I. -Isrc tests/test_pad.c "${libraries[@]}" -o out/pad-test
    out/pad-test
    "$CC" -std=c11 -O2 -g -Wall -Wextra -Werror -pthread -I. -Isrc tests/test_runtime.c "${runtime[@]}" out/libatrac9.a -lm "${gpu[@]}" "${libraries[@]}" -o out/runtime-test
    out/runtime-test
    "$CC" -std=c11 -O2 -g -Wall -Wextra -Werror -pthread -Isrc tests/test_file_mods.c -o out/file-mods-test
    out/file-mods-test
    "$CC" -std=c11 -O2 -g -Wall -Wextra -Werror -pthread -I. -Isrc tests/test_sema.c "${runtime[@]}" out/libatrac9.a -lm "${gpu[@]}" "${libraries[@]}" -o out/sema-test
    out/sema-test
    "$CC" -std=c11 -D_GNU_SOURCE -O2 -g -Wall -Wextra -Werror -I. -Isrc tests/test_content.c src/runtime_content.c -o out/content-test
    out/content-test
    "$CC" -std=c11 -O2 -g -Wall -Wextra -Werror -pthread -Isrc tests/test_cpu_patch.c src/cpu_patch.c -lZydis -o out/cpu-patch-test
    out/cpu-patch-test
    # Guest sync and memory-tracking fast paths (registry lookups, wait setup, region cache).
    "$CC" -std=c11 -O2 -g -Wall -Wextra -Werror -pthread tests/bench_runtime_sync.c -o out/runtime-bench
    out/runtime-bench
    # Guest semaphores and rwlocks: both files are self-contained, so they are tested on their
    # own (the whole-runtime tests above need the audio and pad libraries).
    "$CC" -std=c11 -O2 -g -Wall -Wextra -Werror -pthread -I. -Isrc tests/test_runtime_sync.c \
        src/runtime_sema.c src/runtime_rwlock.c -o out/runtime-sync-test
    out/runtime-sync-test
    # Host-side helpers; the C++ compiler of the toolchain builds them. Two builds of the bit
    # array: this machine's code generation and the SSE2-only one (cores without AVX2).
    cxx=$(command -v clang++ || command -v g++ || true)
    if [[ -n $cxx ]]; then
        "$cxx" -std=c++20 -O2 -g -Wall -Wextra -Werror -Igpu/shadps4 tests/test_bit_array.cpp -o out/bit-array-test
        out/bit-array-test
        if [[ -n ${cc_flags[*]:-} ]]; then
            "$cxx" -std=c++20 -O2 -g -Wall -Wextra -Werror "${cc_flags[@]}" -Igpu/shadps4 tests/test_bit_array.cpp -o out/bit-array-test-target
            out/bit-array-test-target
        fi
        "$cxx" -std=c++20 -O2 -g -Wall -Wextra -Werror -Igpu/shim tests/test_stream_copy.cpp gpu/shim/bbport_memcpy.cpp -o out/stream-copy-test
        out/stream-copy-test
    fi
fi
