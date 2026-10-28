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

On Linux `build.sh` also compiles the runtime once into `out/rt/` (an unchanged file is not
compiled again) and links it with LTO and `-fno-plt` (`BB_RT_LTO=OFF` turns the LTO off):

- The runtime is a dozen separate files of small helpers — error translation, file lookups,
  range queries — that the HLE calls walk through on every guest operation. LTO inlines them
  into their callers: on the runtime's own sources it removes **8.6 % of the call sites**
  (2118 → 1936) and 5.7 % of the text.
- `-fno-plt` turns the call into the GPU library (`bbgpu_handle_fault`, the entry every guest
  write fault goes through) into one GOT-indirect call instead of a PLT stub jump.

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
- Debug markers (`BB_VK_MARKERS`) used to be checked *after* `DrainDrawPipe()`: with markers
  off — the default — every marker call still flushed the parallel draw pipeline, and the three
  host sites in the rasterizer built their label with `fmt::format` first. The clear/resolve/
  depth-copy paths now drain explicitly, once each instead of twice through their begin/end
  markers, and the label is only formatted when it is written.

## Guest synchronization and range queries (`src/`)

The guest's rwlocks, semaphores and the GPU side's guest-range queries are the runtime's three
hottest data structures; none of them was indexed.

- **Registry lookups.** The PS4 rwlock and semaphore handles live in a registry that was walked
  in full on *every* operation (read lock, write lock, unlock, wait, signal, poll) while holding
  a global mutex. Both registries are now hashed buckets chained through the objects' own `next`
  pointer: one bucket probe instead of the whole list. `tests/bench_runtime_sync.c`:

  | Live objects | List scan | Hashed | |
  |---|---|---|---|
  | 4 | 1.5 ns | 0.6 ns | 2.3× |
  | 16 | 4.9 ns | 1.1 ns | 4.5× |
  | 64 | 20.7 ns | 1.1 ns | 19× |
  | 128 | 58.4 ns | 0.6 ns | 94× |

- **Blocking waits.** A blocking wait on a guest semaphore used to create and destroy a
  condition variable (plus its attribute object and clock setting) around *every* wait; on
  winpthreads those calls are not cheap. A thread blocks on one semaphore at a time, so each
  thread now keeps one condition variable, created on its first blocking wait and reused after
  that, and the FIFO list appends at a tail pointer instead of walking to the end (job queues
  keep many waiters). Waiter bookkeeping keeps the FIFO semantics exactly: the waiter still owns
  its slot, and only that slot is signalled. `tests/test_runtime_sync.c` covers timeouts, cancel,
  delete-with-waiter and concurrent readers; 8 waiters and 2 signallers over 4 semaphores with
  mixed timeouts and signals leave no waiter behind. The setup cost removed from each wait is the
  `13.2 ns` line of `out/runtime-bench` (it is paid once per thread now).

- **Region cache.** `runtime_memory.c` keeps a per-thread cache of recently resolved guest
  mappings (`cached_region`), consulted for every GPU-side `clamp`/`region`/`is_mapped` query —
  that is, for every buffer and texture binding. It was 8 entries probed one by one and replaced
  round-robin, so unrelated mappings evicted each other and every miss took the runtime's
  read lock. It is now a table indexed by the address (page bits mixed, so consecutive pages
  spread out): **7.9 ns → 1.1 ns** per query.

  Its size was then raised from 32 to 256 slots, because a frame does not stay within 32
  distinct guest ranges: a scene binds its vertex and index buffers, the textures it samples and
  its render targets, and each range that is not cached costs a read lock and a search of the
  mapping table. `tests/bench_runtime_sync.c` sweeps the capacity with the runtime's own lock and
  table search on the miss path:

  | Hot ranges in a frame | 32 slots | 256 slots | |
  |---|---|---|---|
  | 16 | 2.1 ns/query | 2.1 ns/query | — |
  | 40 | 2.9 ns/query | 2.1 ns/query | 1.4× |
  | 100 | 19.7 ns/query | 2.0 ns/query | **10×** |

  The table costs 8 KiB per thread; the queries it serves are a handful per draw, so the larger
  table is a pure win once a frame touches more than ~30 ranges.

- **Register deltas.** `AmdGpu::RegDelta::Capture` (`video_core/amdgpu/regs.h`) copies the
  registers a submission wrote into the draw-preparation worker's delta, out of a bitmap of the
  register file's 1664 blocks. It tested all 1664 blocks, on every standard library (`std::bitset`
  has no portable set-bit walk; the draw pipe's own copy of the walk was libstdc++-only). The
  bitmap is now a 26-word mask of its own with `ForEachSet`, which the register writes mark with
  a few word writes (`SetRange`) instead of one per bit: **~1.4 µs → ~50 ns** per submission
  (≈25×), checked against the block scan for empty, partial, tail, field and reset captures, and
  the draw pipe fills its packet from the same walk on every platform, Windows included.

- **Register hashing.** Every register-writing PM4 packet is folded into the running checksum
  that the draw-preparation workers compare against (`Liverpool::HashRegisterPacket`), on the GPU
  thread and on every worker. The fold carried a per-word shift-xor (`h ^= h >> 29`) that sat in
  the dependency chain, ~7 cycles per word; the bare fold `h = (h ^ word) * odd_constant` is ~4
  and measures **1.5×** on 8- to 150-word packets. Nothing is lost for this use: xor and
  multiplication by an odd constant are invertible, so two streams that differ in one word always
  end up with different checksums (10 400 single-bit cases, no collisions), and the fold stays
  independent of where a caller starts and stops, so a packet split differently in two places
  still hashes the same.

- **What that walk must not do (a fixed regression).** The region manager's write mask was
  briefly rewritten as the fused `~(cpu ^ writeable)` to match the read mask's `~(gpu ^ readable)`
  form. The two are not symmetric: the read mask *is* the complement of the difference, the write
  mask is the plain difference. Complementing it marked every page that had not changed, so the
  page manager's watcher counts drifted and an unprotect eventually found a page with no watcher —
  the GPU library stopped with `page_manager.cpp: Not enough watchers` on the first seconds of a
  launch. Both masks now live in `region_definitions.h` as named functions, and
  `tests/test_bit_array.cpp` checks each against the arithmetic it has to mean (the test fails with
  4163 checks on the inverted form), which is the part the bit-array tests alone never covered:
  they verified `NotXor` against its own definition, not against the call site.

- **Block size: measured, not guessed.** The delta above copies whole 32-word blocks. Smaller
  blocks copy fewer words but touch more of them (and more block indices), smaller masks being
  no help: with a submission's worth of scattered context writes and 4–16-word SH runs, 32-word
  blocks are the fastest of the three, and the copies stay smallest where it matters:

  | Marks per submission | 32-word blocks | 16-word | 8-word |
  |---|---|---|---|
  | 8 | **242 ns**, 284 words | 292 ns (0.83×), 157 words | 321 ns (0.75×), 93 words |
  | 40 | **778 ns**, 1371 words | 854 ns (0.91×), 770 words | 1048 ns (0.74×), 461 words |
  | 150 | 2659 ns, 4571 words | **2328 ns (1.14×)**, 2699 words | 2550 ns (1.04×), 1657 words |

  So the granularity stays at 32 words; the mask itself now lives in its own header
  (`video_core/amdgpu/block_mask.h`, no fmt/logging dependency) and `tests/test_block_mask.cpp`
  walks it against a bool reference at every 64-bit word boundary and for block counts that are
  not a multiple of 64 (a mistyped tail mask fails 1257 checks there).

- **Blocks that only the game build sees.** `tests/ci` now compiles the runtime with the port's
  own `-O3 -Werror` and instruction set, which is where two Linux-only breakages were hiding:
  a thread-priority helper that only Windows uses (`-Wunused-function`) and an `snprintf` into a
  128-byte guest trophy field (`-Wformat-truncation`).

`tests/bench_runtime_sync.c` prints all of the numbers above (`out/runtime-bench`, or the
`guest synchronization and memory-tracking fast paths` block of `tools/ci/linux-tests.sh`).

Both files are also run under **ThreadSanitizer**: `tests/test_runtime_sync.c` (the contract) and
`tests/stress_runtime_sync.c` (8 waiting and 2 signalling threads over 4 semaphores for a second,
~32 000 waits with ~11 000 acquisitions and ~21 000 timeouts, none left behind). No data race is
reported in either, which is the property a per-thread condition variable and the hashed bucket
lists have to keep; `tools/ci/linux-tests.sh` fails the run if ThreadSanitizer reports one.

## Bit arrays (`gpu/shadps4/common/bit_array.h`)

`BitArray` is the buffer cache's page bitmap: for every 4 MiB region it records which of its 1024
pages the CPU wrote (`cpu`), the GPU wrote (`gpu`), and which are readable/writeable, and the
region manager asks it something on **every buffer binding** and every change of page
protection.

The 256-bit path needs **AVX1 only**, which is not obvious: AVX1 has no 256-bit *integer*
instructions (those arrived with AVX2), but the scans only ever load, store, AND, OR, XOR and
test, and all of those exist on 256 bits in AVX1 — `vptest` for the tests, and the bitwise
single-precision forms (`vandps`, `vorps`, `vxorps`, `vandnps`) for the rest, which are pure
bitwise operations with no arithmetic, exceptions or NaN handling. So on this instruction set:

| Loop | SSE2 (2 words) | AVX (4 words) | Speed-up |
|---|---|---|---|
| `AnyInRange` (a buffer binding's pages) | 3.8–4.8 ns | 1.2–1.6 ns | **~3×** |
| `None` (did anything change) | 3.0–3.2 ns | 0.6–0.8 ns | **~4×** |
| `NotXor` (pages whose protection changes) | 6.1–6.5 ns | 3.3–4.1 ns | **~1.6×** |
| `==` | 1.5–2.1 ns | 0.5–0.6 ns | **~3×** |

Each test is a single `vptest` for four words; before, SSE2 spent `pcmpeq` + `pmovmskb` + `cmp`
per two words, and the set operators (`|=`, `&=`, `^=`, `~`, `==`) were 16 scalar word loops that
GCC could only vectorize to 128 bits without AVX2. `None()` combines whole vectors into one
accumulator and tests once, so the common "nothing changed" answer has no branch in it, and
`NotXor`/`NotOr` (the two De Morgan forms the region manager needs — they differ where both
operands are set) fuse what used to be a temporary array, a copy and two walks.

`BIT_ARRAY_USE_AVX` (any `-mavx` build, AVX2 included), `BIT_ARRAY_USE_SSE` and
`BIT_ARRAY_USE_VECTOR` say which path a build compiled. `tests/test_bit_array.cpp` checks all of
them against a scalar model (ranges, iterators, the set operators, the fused forms);
`tests/bench_bit_array.cpp` prints the timings above for the build at hand, and
`tools/ci/linux-tests.sh` builds it twice (AVX and SSE2) so the difference is visible in CI.

## Per-binding costs in the buffer cache (`video_core/buffer_cache/`)

Two costs sit on the path of *every* buffer binding, which the profile puts at 32 % of the GPU
command thread's time (`BindResources`), and neither showed up in a profiler because both are a
handful of instructions done a few thousand times a frame.

- **Is the range resident?** `EnsureResident` bisects the port's list of resident block ranges
  before a buffer is used, and that list only ever grows: after a few minutes of play it holds
  hundreds of ranges over dozens of KiB, so the search is a handful of dependent loads into cold
  memory. A buffer, though, is bound by several draws in a row and the buffers of a pass are
  allocated next to each other, so the span the last search proved gap-free is kept per thread
  (`ResidentSpan` in `common/interval_set.h`) and **widened** over each new binding that touches
  it: one binding pays the search, the following ones are a compare.

  | Resident ranges | Search, warm | Search, cold | Span hit | |
  |---|---|---|---|---|
  | 64 | 7–8 ns | 36–39 ns | 0.7–1.0 ns | 36–54× |
  | 256 | 7–10 ns | 48–56 ns | 0.7–1.2 ns | 47–72× |
  | 1024 | 9–10 ns | 62–71 ns | 0.6–0.8 ns | 92–103× |
  | 4096 | 11–13 ns | 82 ns | 0.6–0.7 ns | 125–129× |
  | 65536 | 17–18 ns | 129–140 ns | 0.8 ns | 161–171× |

  (`tests/bench_resident_hint.cpp`; "warm" is a binding of the pass whose residents the last
  search touched, "cold" one of another pass, and the span hit is what a repeat binding costs.)
  The cache is sound because residency is monotone: a gap-free span stays gap-free. That is what
  `tests/test_resident_span.cpp` checks over 20 000 queries against a list with holes in it — a
  span that claims a block the list does not have, or a widening that reaches across a hole,
  fails the test.

- **Which frame last used each page?** `NoteGpuUse` stamps the frames of the 8 MB table the buffer
  cache keeps for the GPU heap window, so GPU data still in use is never dropped as dead. Most
  pages of a range already carry the frame being written (the same buffer bound again), so
  stamping them again only dirties cache lines. Ranges of 16 pages (64 KiB) and up are compared
  8 pages at a time with two 128-bit integer compares and stored only where they differ
  (`PageStamp::Mark` in `video_core/buffer_cache/page_stamp.h`):

  | Pages per bind | Plain stores | Compare and skip | |
  |---|---|---|---|
  | 1–8 | 3.5–5.1 ns | 3.5–5.1 ns | — (plain stores are used) |
  | 16 | 7.6–7.9 ns | 7.0–7.3 ns | 1.06–1.09× |
  | 32 | 12.7–13.5 ns | 10.4–10.8 ns | 1.22–1.25× |
  | 64 (256 KiB) | 23.8–25.7 ns | 16.8–17.9 ns | 1.41–1.45× |
  | 256 (1 MiB) | 93–101 ns | 72 ns | 1.31–1.41× |
  | 1024 (4 MiB) | 357–367 ns | 200–207 ns | 1.73–1.82× |

  `tests/bench_page_stamp.cpp` prints that table. `tests/test_page_stamp.cpp` checks the fast
  path against the plain loop at every length around the threshold, from every offset that
  straddles a step, in both toggle states and over random frames — inverting its store condition
  fails 2 494 of its 2 801 checks.

Both can be switched off at runtime (`BB_TOGGLE_FILE`): `GpuUseStamp` for the stamping; the
residency span has no toggle because it changes nothing observable, only how the same answer is
reached.

## Per-draw memos in the draw path (`video_core/renderer_vulkan/`)

A frame submits a few thousand draws, and each one asks the same four questions before any
command is recorded: which image is this render target, which set of images does this shader
sample, does this draw begin a new render area, and what is the state behind it? The answers
barely change between consecutive draws, so each question has a memo — and every memo used to be
validated against `TextureCache::RegistryGeneration()`, a counter that changes whenever *any*
image is registered or collected. A streaming frame registers and collects images constantly, so
all four memos were thrown away several times per frame and most of their lookups became the
image search they were meant to skip.

- **Lifetime token.** An image's `image_uid` is handed out once and never reused, so "the image
  in this slot still carries the uid I stored" is exactly "my image is still alive", and nothing
  else. The memos (render targets, the render-state signature, the texture set, the per-slot
  image descriptions) validate by uid instead. `TextureCache::ImageUid` answers 0 for an empty id
  and for a slot that has been freed, which is what makes a memoized id safe to check: no memo
  indexes the slot array before the uid matches. A memo now dies when, and only when, its own
  image does.
- **Key size.** A target lookup keys on the raw bytes of the target registers (56 bytes for
  depth, 68 for a color target). It used to zero-fill a 256-byte buffer per lookup, keep 256
  bytes of it in the last-hit and memo entries, and hash it word at a time; it now stores exactly
  the bytes of the call's parts (`common/byte_key.h`) and hashes those with XXH3:

  | Lookup | 256-byte zero-filled key | Per-call key | |
  |---|---|---|---|
  | target unchanged since the previous draw | 53 ns | 12 ns | 4.4–4.6× |
  | target changes every draw | 72 ns | 28 ns | 2.5–2.6× |
  | uid validation of a memo hit | — | 1.6 ns | |

  (`tests/bench_target_key.cpp`, which prints exactly that table; `tests/test_byte_key.cpp`
  checks that the key is byte-exact — a single flipped bit in any part, or a shorter key with the
  same prefix, must not match — under ASan and UBSan.)
- **Image misses.** A texel buffer usually aliases no image at all, and `SynchronizeMemoryFromImage`
  remembers that so the range query is not repeated. Collecting an image cannot make a range
  covered, so that memo keys on registrations only (`RegistrationGeneration`) and survives the
  eviction side of a streaming frame.
- **Texture sets.** A memoized set validates per entry by uid and prefetches the next two images
  of the set: the images of one draw are scattered over a large slot array and each check
  depends on the loads of the previous one.
- **Samplers.** Every stage of every draw asks the texture cache for the sampler of each S# it
  binds, and the answer depends on the S# bytes, the depth flag, the LOD bias and the border
  color base — nothing else. The cache hashes the S#, takes a mutex, probes a hash map and
  touches an LRU entry; a frame binds the same handful of S#s over and over. A 512-entry
  direct-mapped table of the answers serves the repeats: two multiplies, a mask and five
  compares. It is not shared with the texture binding helper, which binds on another thread, and
  every entry carries the texture cache's `SamplerEpoch()`, which a sampler collection bumps: a
  collected sampler's handle is destroyed, and an entry filled before that is no hit.
  `BB_FRAME_STATS=1` reports its hits and misses next to the render target memo's.
- **Descriptor writes.** A draw's write list and the infos each write points at are copied into
  the recording chunk that the recording thread executes later. That was one `RecordData()` call
  per write -- an allocation in the chunk, its headroom checks and its two prefetches, for the 24
  to 32 bytes of one info. A stage appends its buffers, then its images, then its samplers, so
  the infos of consecutive writes are consecutive entries of one vector: the writes are copied
  one run at a time (`vk_pipeline_common.cpp`), 20 writes in 4 calls instead of 21. The loop
  costs 1.5× less (70 → 46 ns per draw, `tests/bench_record_data.cpp`).
- **Dynamic state.** Every draw derives its Vulkan dynamic state from a handful of register
  groups: the viewports and the three scissors, depth and stencil state, the primitive state, the
  line width and the blend state. That is ~30-45 ns of arithmetic per draw
  (`tests/bench_dynamic_state.cpp`), and inside a batch the registers it reads have not changed
  since the previous draw. The draw pipe's packets already carry the 32-word register blocks they
  wrote (`RegDirty`), so the recording thread classifies those blocks once
  (`video_core/amdgpu/dynamic_state_blocks.h`: one byte per block, built from the members'
  offsets at first use) and recomputes only the groups a touched block belongs to — one table
  lookup per block, 0.5 ns, against the 30-45 ns of work it elides. `CommitWith()` still runs
  every draw, so nothing left dirty by an earlier draw is lost. Three cases force the full path:
  no draw pipe (nothing says what changed), a `ClearState` packet (defaults every register), and
  a new command buffer (`DynamicState::Invalidate()`), which marks every state dirty precisely so
  it is emitted again — the memo compares the scheduler's invalidation count for that.
  `tests/test_dynamic_state_blocks.cpp` checks the classification against the real register file
  (every register a group reads must report its group; a shader-register write must report none),
  and `BB_FRAME_STATS=1` prints how often each group was recomputed against skipped.
- **Samplers on the texture binding helper.** The helper thread binds a draw's textures while the
  GPU thread binds its buffers, and it used to skip the sampler memo above — the table belonged
  to the GPU thread -- so every S# it bound took the texture cache's path: hash the S#, take the
  sampler mutex, probe the hash map, touch the LRU entry, with the GPU thread taking the same
  mutex for its own bindings. Each thread now has its own table (`helper_sampler_memo`, written
  only by the helper), keyed and epoch-guarded exactly like the GPU thread's, and the stats line
  reports both halves. `tests/bench_draw_setup.cpp` measures the shape of the two lookups (a hash,
  an uncontended mutex, a map probe and an LRU touch against two multiplies and five compares):
  **4.4 ns against 2.5 ns** for a sampler a frame binds over and over, and the cache's own path
  adds what the harness cannot show — the XXH3 call over the S#, a second `at()` lookup on a
  miss, and a lock the GPU thread also takes.
- **Vertex input state.** A draw sets the vertex formats, strides and attribute offsets as dynamic
  state (`setVertexInputEXT`), and a batch of draws of one pipeline describes the same set. The
  state stays live in the command buffer until something sets it again, so a draw whose description
  equals the last one emitted records nothing: two `memcmp`s (nine 16-byte entries) stand in front
  of the two copies into the recording chunk and the command that carries them, 3.1 ns against
  3.6 ns of chunk work in `tests/bench_draw_setup.cpp` — and the recorded command itself, a driver
  call that validates the description and writes it into the command buffer, is gone. The memo is
  dropped when the scheduler's invalidating count moves (a new command buffer needs the state
  again), which is the same edge `UpdateDynamicState` uses. `BB_FRAME_STATS=1` prints emitted
  against reused, and `BbToggle::VertexInputMemo` turns it off.
- **Buffer uses.** `NoteGpuUse` stamps the frame on every 4 KiB page a binding touches, in an
  8 MiB table of `u32`. A batch of draws binds the same vertex stream, the same constants and
  the same storage buffers again and again, and stamping a page that already carries this frame
  writes the value it holds. A per-thread table of the 1024 ranges stamped so far in the frame
  (`StampedSlots`) turns a repeat into one compare; a range that starts where a stamped one did
  widens it, anything else replaces the entry. Stamping a batch's repeated bindings costs 2.3×
  less (5.8 → 2.5 ns per bind), and a frame that walks more distinct ranges than the table holds
  pays 15 % more per bind than the plain stamp it would have done — the tails of
  `tests/bench_page_stamp.cpp`.

## Settings the hot paths ask about

Every one of the draw path's diagnostics (`BB_FRAME_STATS`, `BB_TRACE_BUFFERS`, `BB_VERTEX_CHECK`,
`BB_BONE_CHECK`, `BB_FULL_BARRIER`, the buffer cache's `BB_CLOBBER_CHECK` / `BB_STABLE_CHECK` /
`BB_BUFFER_STATS`, the texture cache's `BB_TINY_READBACK`, the draw preparation's `BB_PREP_COMPUTE`)
was read through a function-local static, which is initialized once but *asked* on every draw,
binding or submission: the ask is a guard variable's acquire load and a branch. They are now
`const` objects in static storage, read at load. `tests/bench_draw_setup.cpp` measures the
difference the way the call sites see it — **0.69 ns against 0.47 ns** per ask, times the handful
of asks per draw -- and the same treatment was given to the settings the draw-preparation workers
and the scheduler ask per submission.

The texture cache's memory query is the other side of the same coin: `TextureCache`'s collector
asks the driver for the current usage and budget of its heaps on every submission
(`vkGetPhysicalDeviceMemoryProperties2`), and a budget that other processes move over hundreds of
milliseconds does not need a fresh look every few milliseconds. `Instance::QueryMemory` keeps its
answer for 4 ms, which is well below the rate at which the collector's own accounting moves (it
decides about images the frame just used, not about a heap that changed in the last millisecond).

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

A failed build step's log is only reachable through a signed URL, so the Windows build step
tees the build output and turns the error lines into annotations of the run and the tail into the
job summary; `build.sh` keeps `out/gpu-configure.log` and `out/gpu-build.log` for the same
reason. Two things about that job are easy to trip over: the compiler cache needs the compiler
path CMake passes it, and a POSIX `ccache` does not accept the absolute Windows path form, so the
step probes it and falls back to a wrapper (or to no cache) instead of failing.
