#!/usr/bin/env bash
# Profile-guided optimization for the port (BB_PGO, gpu/CMakeLists.txt).
#
# PGO is the biggest single win left on a CPU-bound build: the compiler learns which branches
# of the shader recompiler, the PM4 decoder, the buffer and texture caches and the HLE runtime
# are taken, lays the hot code out together and inlines where it pays. Expect several percent
# of frame time, more in the load-time and shader-compilation paths.
#
# The profile has to come from the game itself (there is no way to train it without running
# it), so this is a three-step, one-time-per-build-setup job:
#
#   bash tools/pgo.sh generate     # instrumented build (writes pgo/ while the game runs)
#   <play for a few minutes: a level load, combat, a menu, a cutscene>
#   bash tools/pgo.sh merge        # pgo/*.profraw -> pgo/bbport.profdata
#   bash tools/pgo.sh use          # the optimized build, then package it as usual
#
# After that, `bash build.sh` picks the profile up on its own (BB_PGO=use is the default when
# pgo/bbport.profdata exists). `bash build.sh --test` and `bash packaging/windows/package.sh`
# are unchanged.
set -euo pipefail
cd -- "$(dirname -- "$0")/.."

profdata=pgo/bbport.profdata

profdata_tool() {
    # MSYS2 CLANG64 and Xcode name it llvm-profdata; Debian/Ubuntu put a version on it.
    for tool in llvm-profdata llvm-profdata-21 llvm-profdata-20 llvm-profdata-19 \
                llvm-profdata-18 llvm-profdata-17 llvm-profdata-16; do
        if command -v "$tool" >/dev/null; then echo "$tool"; return 0; fi
    done
    return 1
}

case ${1:-} in
generate)
    mkdir -p pgo
    rm -f pgo/*.profraw pgo/default_*.profraw
    echo "Instrumented build: pgo/ collects the profile while the game runs."
    BB_PGO=generate bash build.sh
    echo "Now start the game (Uncharted.exe / run.py) and play a few minutes."
    ;;
merge)
    tool=$(profdata_tool) || { echo "No llvm-profdata in PATH (MSYS2: pacman -S llvm; Debian: apt install llvm)." >&2; exit 1; }
    shopt -s nullglob
    files=(pgo/*.profraw pgo/default_*.profraw)
    shopt -u nullglob
    if [[ ${#files[@]} -eq 0 ]]; then
        echo "No pgo/*.profraw: run 'bash tools/pgo.sh generate' and play first." >&2
        exit 1
    fi
    # Several runs, several processes: one profile out of all of them.
    "$tool" merge --failure-mode=all -o "$profdata" "${files[@]}"
    rm -f "${files[@]}"
    echo "Profile: $profdata ($(du -h "$profdata" | cut -f1)) from ${#files[@]} file(s)."
    echo "Now: bash tools/pgo.sh use"
    ;;
use)
    [[ -f $profdata ]] || { echo "$profdata missing: bash tools/pgo.sh merge first." >&2; exit 1; }
    BB_PGO=use bash build.sh
    ;;
clean)
    rm -f pgo/*.profraw pgo/default_*.profraw "$profdata"
    echo "pgo/ cleaned: builds fall back to no PGO (BB_PGO=off)."
    ;;
status)
    if [[ -f $profdata ]]; then
        echo "profile: $profdata ($(du -h "$profdata" | cut -f1)): builds use it"
    else
        echo "profile: none: builds are not PGO-optimized (bash tools/pgo.sh generate)"
    fi
    ;;
*)
    sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
    ;;
esac
