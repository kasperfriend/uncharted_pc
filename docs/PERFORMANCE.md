# CPU performance: what the port does and how to build for it

The games run natively: the PS4 binaries are the PC binaries, the HLE runtime in `src/` serves
their system calls, and `libbbgpu` (in `gpu/`, derived from shadPS4) translates the GPU command
streams to Vulkan and recompiles the shaders. At 60 FPS the GPU thread is the largest single
consumer of one core, the guest threads run game logic on the rest, and the preparation and copy
threads feed both. Everything below makes those cores do less per frame, or the same work with
instructions the CPU actually has.

None of this changes what the games do: the same files, the same saves, the same picture. The
knobs that change behaviour are marked "changes behaviour" and default to on.

## Building for the CPU the game runs on

The floor is the PS4's own instruction set requirement: **AVX**, plus SSE4.2, POPCNT and
CMPXCHG16B (any 64-bit x86 CPU that runs the games has them). AVX2, FMA and BMI are *not*
required: on CPUs without them, the loader rewrites the guest instructions that need them (see
below), and the host code is built without them on purpose.

`gpu/CMakeLists.txt` and `build.sh` share one set of options:

| Option | Default | What it does |
|---|---|---|
| `BB_ARCH` | `avx` | `avx`: `-march=x86-64-v2 -mavx -mf16c -mno-avx2 -mno-fma -mno-bmi`; `avx2`: AVX2/FMA/BMI2 for newer CPUs; `native`: `-march=native` (local benchmark builds); `generic`: plain x86-64 SSE2; anything else: `-march=<value>` |
| `BB_TUNE` | follows `BB_ARCH` | `-mtune`; `avx` schedules for a 128-bit-wide core (`-mtune=core-avx-i`) |
| `BB_OPT_LEVEL` | `3` | `-O3` for release builds |
| `BB_LTO` | `THIN` | ThinLTO through lld; `ON` for full LTO, `OFF` for none |
| `BB_USE_LLD` | `AUTO` | links with `lld` when clang and lld are available |
| `BB_PGO` | `off` | `generate`, then `use` after training (see below); `build.sh` uses `pgo/bbport.profdata` automatically once it exists |
| `BB_FRAME_POINTERS` | `ON` | keep frame pointers for the stack dumps in crash reports; `OFF` frees a register |
| `BB_JOBS` | all cores | `ninja -j` |
| `BB_CMAKE_ARGS` | empty | extra `-D` options (`CI` passes the ccache launchers here) |

Command line, after `bash build.sh` once:

```bash
BB_ARCH=avx2 bash build.sh          # a build for a newer CPU (AVX2/FMA/BMI2)
BB_ARCH=native bash build.sh        # this machine, for benchmarks only
BB_FRAME_POINTERS=OFF bash build.sh # a little faster, worse crash reports
```

Why `BB_ARCH=avx` is not just "AVX on": it keeps 256-bit vectors out of the host code
(`-mprefer-vector-width=128`). The cores in this class execute a 256-bit operation as two
128-bit halves, so it buys nothing, and mixing 256-bit "upper state" into code that calls the
guest's legacy-SSE code costs a transition on every boundary. The build still gets SSE4.2
(`pcmpistri`, `pblendvb`, `popcnt`, `round*`) and AVX's three-operand forms, which is where the
gain over the default SSE2 build comes from: code generation for a 64-bit baseline makes every
float and every memcpy loop use SSE2 with two operands and no `popcnt`.

The C runtime (`src/`, the HLE system calls the guest calls) is built with the same flags and
`-O3` by `build.sh`; on Windows CMake builds it together with the loader, so it gets the same
treatment through the same options.

## Profile-guided optimization

```bash
bash tools/pgo.sh generate     # instrumented build (bb-probe + libbbgpu), writes pgo/
BB_GAME_DIR=... python run.py  # play the way you want it optimized: first areas, heavy combat
bash tools/pgo.sh merge        # llvm-profdata merge -> pgo/bbport.profdata
bash tools/pgo.sh use          # -fprofile-use build; build.sh picks the profile up from now on
bash tools/pgo.sh status
```

The instrumented build writes the profile every 30 seconds as well as at exit (the game often
ends with `_exit`, which skips the exit write): clang's `__llvm_profile_write_file` rewrites the
totals, GCC's `__gcov_dump`/`__gcov_reset` accumulate intervals. Train on the content you play,
then rebuild: functions reordered by frequency, inlining and register allocation decided by the
counters. `-fprofile-partial-training` keeps code you did not train on optimized normally, so a
short session is still useful.

## Guest instruction rewriting (`src/cpu_patch.c`)

The games' code contains LZCNT, TZCNT, BMI1 (ANDN, BEXTR, BLSI, BLSMSK, BLSR) and MOVBE. On a
CPU that lacks them they either fault or compute something else. Before the guest starts, every
executable segment is swept and each such instruction is replaced:

- **In place, when the replacement fits the same bytes**: a 32-bit `movbe` load becomes
  `mov r32, m` + `bswap r32` (the same ModRM bytes, NOP padding when the MOVBE was longer), and a
  store becomes `bswap` + `mov`. No jump, no thunk: the guest keeps running straight-line code.
- **A jump to a thunk** near the image when it does not fit (64-bit and 16-bit forms, the
  BLS* family, ANDN): the thunk recomputes the same result and the same defined flags with older
  instructions and jumps back. A short instruction (`lzcnt eax, ecx` is 4 bytes) moves up to two
  neighbouring instructions into the thunk when no direct branch lands between them.
- **A breakpoint** (rare) only when even that is impossible: the exception handler sends the
  instruction to its thunk and resumes after it.

The sweep is a linear Zydis decode at about **41–47 MB/s per core**, so it runs on several cores
(`BB_CPU_PATCH_THREADS`, default one per processor, at most eight) and stops mattering at
launch: 64 MB of code took 1553 ms on one core and 849 ms on two, with identical results —
the test below checks that every thread count produces the same rewrites.

Nothing is rewritten on a CPU that has the instructions (`BB_CPU_PATCH=0` turns it off,
`BB_CPU_PATCH=all` forces it for testing).

`tests/test_cpu_patch.c` runs each of 54 instructions natively and then rewritten, 400 trials
each, and compares every register, the defined flags and memory; `tools/ci/linux-tests.sh` builds
and runs it on one and several threads. `BB_CPU_PATCH_STATS=1` prints the sweep.

## Memory copies (`gpu/shim/bbport_memcpy.*`)

Guest memory that the GPU reads is copied out of the game's buffers every frame (streaming
buffers, texture uploads, buffer uploads). Those bytes are consumed once, by the GPU, over PCIe,
so writing them with non-temporal stores avoids the read-for-ownership traffic a normal `rep
movsb` would cause:

| Copy | Cached (`memcpy`) | Streamed (`movntdq`) |
|---|---|---|
| 256 MiB, cold, then 96 MiB of other data | 42.5 ms | **32.5 ms** (+29 %) |
| 128 MiB into a 16 MiB ring, warm | faster | −9…−12 % |
| 256–512 KiB | faster | −51 % |
| 1 MiB | faster | −28 % |
| 2 MiB | −33 % | faster |
| 4–8 MiB | −19…−28 % | faster |

So `BbCopy::Copy(dst, src, size, policy)` streams only when it pays: `Auto` (the default the
memory manager uses) streams from `AutoStreamBytes` = 4 MiB up, `Cached` always uses the fast
`memcpy` path, `Stream` always streams. Streaming ends with `_mm_sfence()`; `BB_STREAM_COPY=0`
turns streaming off entirely. The copy itself is AVX 32-byte stores with 128-byte blocks and an
alignment head, so it works on any destination alignment.

Large uploads are split across the copy pool (`gpu/shim/bbport_copy.*`: a quarter of the
hardware threads, at most four, `BB_COPY_THREADS` overrides), which also runs the deferred
buffer-cache copies; the GPU thread waits for them only at a fence that needs the data.

## Synchronization

- `Common::SpinLock` (`gpu/shadps4/common/spin_lock.*`, used by the buffer cache's region
  manager and the page manager, taken per draw) is one `lock bts` when uncontended, and
  test-and-test-and-set with a pause backoff when not: waiting cores probe with a plain load
  instead of pulling the cache line across on every attempt. Two threads taking the same lock in
  turn: **36–45 ns → 10–13 ns** per acquire; uncontended cost unchanged.
- `Common::SpinWait`/`SpinWaitShort` (`common/spin_wait.h`) are the same idea for the flags the
  copy pool, the scheduler's fence waits and the presenter's frame pacing spin on: pauses for a
  short wait, then a yield, never a sleep.
- The presenter paces frames with pauses and one clock read in 32 instead of a yield per
  iteration, which cost a reschedule per frame.

## Bit arrays (`gpu/shadps4/common/bit_array.h`)

`BitArray` (the buffer cache's 1024-page region bitmap: which pages of a 4 MiB region the CPU or
the GPU wrote) scans and fills 4 words per instruction with AVX2, 2 words with SSE2, and falls
back to a scalar loop. `BIT_ARRAY_USE_AVX`/`BIT_ARRAY_USE_SSE`/`BIT_ARRAY_USE_VECTOR` say which
one a build compiled; `tests/test_bit_array.cpp` checks all of them (ranges, iterators,
copy/and/xor) against a scalar model, in `tools/ci/linux-tests.sh` for each build variant.

## Measuring

- `BB_DCB_STATS=1` prints the GPU thread's per-frame counters (see `docs/parallel_gpu.md`).
- `BB_CPU_PATCH_STATS=1` prints the sweep; `BB_STREAM_COPY=0` and `BB_CPU_PATCH=0` isolate the
  two rewrites above.
- Linux: `perf record -g -- ./run.sh`; the toggles in `src/bbport_toggles.h` (bit per
  optimisation, settable from the launcher's Debug page) isolate one at a time.
- Build a benchmark binary with `BB_ARCH=native` to see what the code *could* do on the machine;
  keep `avx` for anything that ships.

## Continuous integration

`.github/workflows/ci.yml` builds the Windows package with the MSYS2 CLANG64 toolchain (the
same `.github/actions/windows-build` the release uses) and runs the host-side tests on Linux
(`tools/ci/linux-tests.sh` plus the launcher's tests). `.github/workflows/release.yml` builds
`dist/uncharted-windows.zip` on demand and attaches it, with its SHA-256, to a GitHub release.
