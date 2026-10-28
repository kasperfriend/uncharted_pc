// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: what the setup around one recorded draw costs (video_core/renderer_vulkan/):
//
//  - a diagnostics flag read from a function-local static against the same flag kept in static
//    storage. Each of the draw path's env-gated diagnostics used to be asked that way, so every
//    draw (or every packet) paid a guard variable's acquire load and branch to answer a question
//    whose answer never changes;
//  - the vertex input state of a draw: copying the binding/attribute descriptions into the
//    recording chunk and recording the command that carries them, against comparing them with the
//    state already live in the command buffer and recording nothing (VertexInputMemo);
//  - one sampler binding: the texture cache's answer (a hash, a mutex, a hash-map probe and an
//    LRU touch) against the sampler memo's direct-mapped table, which is what the texture binding
//    helper now uses too (it used to skip the memo because the table belonged to the GPU thread).
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <mutex>
#include <unordered_map>

namespace {

double NowNs() {
    timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return static_cast<double>(t.tv_sec) * 1e9 + static_cast<double>(t.tv_nsec);
}

// ---------------------------------------------------------------------------------------------
// 1. A flag asked about per draw.
// ---------------------------------------------------------------------------------------------
bool ReadFlag() {
    static const bool flag = [] {
        const char* env = getenv("BB_BENCH_UNSET_DIAGNOSTIC");
        return env != nullptr;
    }();
    return flag;
}

const bool GlobalFlag = [] {
    const char* env = getenv("BB_BENCH_UNSET_DIAGNOSTIC");
    return env != nullptr;
}();

std::atomic<std::uint64_t> sink{0};

__attribute__((noinline)) void TouchGuarded() {
    if (ReadFlag()) {
        sink.fetch_add(1, std::memory_order_relaxed);
    }
}

__attribute__((noinline)) void TouchGlobal() {
    if (GlobalFlag) {
        sink.fetch_add(1, std::memory_order_relaxed);
    }
}

// ---------------------------------------------------------------------------------------------
// 2. Vertex input state of a draw (vk::VertexInputBindingDescription2EXT is four words).
// ---------------------------------------------------------------------------------------------
struct Binding {
    std::uint32_t binding, stride, input_rate, divisor;
};
struct Attribute {
    std::uint32_t location, binding, format, offset;
};

constexpr std::size_t Bindings = 3, Attributes = 6;

/// The parts of Scheduler::RecordData()/RecordChunk this measures: headroom test, allocation,
/// copy, and a command closure (virtual Execute) carrying the two spans.
struct Chunk {
    alignas(64) std::byte storage[128 * 1024];
    std::size_t used = 0;

    void Reserve(std::size_t bytes) {
        if (sizeof(storage) - used < bytes + 1024) {
            used = 0;
        }
    }
    void* Allocate(std::size_t bytes, std::size_t align) {
        const std::size_t offset = (used + align - 1) & ~(align - 1);
        used = offset + bytes;
        __builtin_prefetch(storage + used + 384, 1, 3);
        __builtin_prefetch(storage + used + 448, 1, 3);
        return storage + offset;
    }
};

struct CommandBase {
    virtual ~CommandBase() = default;
    virtual void Execute(void* cmdbuf) = 0;
};

std::uint64_t emitted_bindings = 0;

struct VertexInputCommand final : CommandBase {
    VertexInputCommand(const Binding* bindings_, std::size_t num_bindings_,
                       const Attribute* attributes_, std::size_t num_attributes_)
        : bindings{bindings_}, num_bindings{num_bindings_}, attributes{attributes_},
          num_attributes{num_attributes_} {}
    const Binding* bindings;
    std::size_t num_bindings;
    const Attribute* attributes;
    std::size_t num_attributes;
    void Execute(void*) override {
        emitted_bindings += num_bindings + num_attributes;
    }
};

// ---------------------------------------------------------------------------------------------
// 3. One sampler binding.
// ---------------------------------------------------------------------------------------------
struct SamplerEntry {
    std::uint64_t raw0 = 0, raw1 = 0, border = 0;
    std::uint64_t handle{};
    float lod_bias = 0.0f;
    std::uint32_t is_depth = 0, epoch = 0;
    bool valid = false;
};
constexpr std::size_t SamplerMemoSize = 512;

} // namespace

int main() {
    // 1. Guard variable against static storage: the same question, asked 100M times.
    constexpr std::uint64_t Reads = 100'000'000;
    double guarded = 0, global = 0;
    for (int rep = 0; rep < 3; ++rep) {
        double t0 = NowNs();
        for (std::uint64_t i = 0; i < Reads; ++i) {
            TouchGuarded();
        }
        double t1 = NowNs();
        for (std::uint64_t i = 0; i < Reads; ++i) {
            TouchGlobal();
        }
        double t2 = NowNs();
        guarded += (t1 - t0) / double(Reads);
        global += (t2 - t1) / double(Reads);
    }
    std::printf("env flag per draw: function-local static %.2f ns, static storage %.2f ns "
                "(%.1fx cheaper)\n",
                guarded / 3, global / 3, guarded / global);

    // 2. Vertex input: the state of one draw, emitted against reused.
    std::array<Binding, Bindings> bindings{};
    std::array<Attribute, Attributes> attributes{};
    for (std::size_t i = 0; i < Bindings; ++i) {
        bindings[i] = {std::uint32_t(i), std::uint32_t(16 + 12 * i), 1, 1};
    }
    for (std::size_t i = 0; i < Attributes; ++i) {
        attributes[i] = {std::uint32_t(i), std::uint32_t(i % Bindings), 106, std::uint32_t(12 * i)};
    }
    std::array<Binding, Bindings> live_bindings{};
    std::array<Attribute, Attributes> live_attributes{};
    constexpr int Draws = 400'000;
    double emit_ns = 0, reuse_ns = 0, changed_ns = 0;
    for (int rep = 0; rep < 5; ++rep) {
        Chunk chunk;
        double t0 = NowNs();
        for (int d = 0; d < Draws; ++d) {
            chunk.Reserve((Bindings + Attributes) * sizeof(std::uint32_t) * 4 + 128);
            auto* b = static_cast<Binding*>(chunk.Allocate(Bindings * sizeof(Binding),
                                                           alignof(Binding)));
            auto* a = static_cast<Attribute*>(chunk.Allocate(Attributes * sizeof(Attribute),
                                                             alignof(Attribute)));
            std::memcpy(b, bindings.data(), sizeof(bindings));
            std::memcpy(a, attributes.data(), sizeof(attributes));
            auto* command = new (chunk.Allocate(sizeof(VertexInputCommand),
                                                alignof(VertexInputCommand)))
                VertexInputCommand{b, Bindings, a, Attributes};
            CommandBase* base = command;
            base->Execute(nullptr);
            base->~CommandBase();
        }
        double t1 = NowNs();
        // Reuse: the state is already what the command buffer holds.
        std::memcpy(live_bindings.data(), bindings.data(), sizeof(bindings));
        std::memcpy(live_attributes.data(), attributes.data(), sizeof(attributes));
        for (int d = 0; d < Draws; ++d) {
            chunk.Reserve(128);
            const bool same =
                std::memcmp(live_bindings.data(), bindings.data(), sizeof(bindings)) == 0 &&
                std::memcmp(live_attributes.data(), attributes.data(), sizeof(attributes)) == 0;
            if (!same) {
                std::printf("unreachable\n");
            }
        }
        double t2 = NowNs();
        // Miss: the first binding differs, so the comparison stops early and the state is emitted.
        for (int d = 0; d < Draws; ++d) {
            live_bindings[0].stride += 16;
            chunk.Reserve((Bindings + Attributes) * sizeof(std::uint32_t) * 4 + 128);
            const bool same =
                std::memcmp(live_bindings.data(), bindings.data(), sizeof(bindings)) == 0 &&
                std::memcmp(live_attributes.data(), attributes.data(), sizeof(attributes)) == 0;
            if (!same) {
                auto* b = static_cast<Binding*>(chunk.Allocate(Bindings * sizeof(Binding),
                                                               alignof(Binding)));
                auto* a = static_cast<Attribute*>(chunk.Allocate(Attributes * sizeof(Attribute),
                                                                 alignof(Attribute)));
                std::memcpy(b, bindings.data(), sizeof(bindings));
                std::memcpy(a, attributes.data(), sizeof(attributes));
                live_bindings[0] = bindings[0];
            }
        }
        double t3 = NowNs();
        emit_ns += (t1 - t0) / Draws;
        reuse_ns += (t2 - t1) / Draws;
        changed_ns += (t3 - t2) / Draws;
    }
    std::printf("vertex input state per draw: emit %.1f ns, reused (compare only) %.1f ns "
                "(%.1fx), changed (compare + emit) %.1f ns\n",
                emit_ns / 5, reuse_ns / 5, emit_ns / reuse_ns, changed_ns / 5);

    // The rule the memo has to keep (Rasterizer::EmitVertexBuffers): a draw may skip only when
    // what it needs is already what the command buffer holds, and a new command buffer (a new
    // scheduler invalidation generation) makes the answer unknown, so the draw after one emits.
    // Here a sequence of draws changes the formats and the generation, and the state the
    // "command buffer" holds is checked against every draw's own state.
    {
        std::uint32_t generation = 1;
        bool known = false;
        std::array<Binding, Bindings> live{};
        std::array<Attribute, Attributes> live_attrs{};
        std::array<Binding, Bindings> memo_bindings{};
        std::array<Attribute, Attributes> memo_attributes{};
        std::uint32_t memo_generation = ~0u;
        std::uint64_t emits = 0;
        std::uint32_t state = 12345;
        std::uint32_t mismatches = 0;
        for (int d = 0; d < 20000; ++d) {
            state = state * 1664525u + 1013904223u;
            if ((state >> 26) == 0) {
                generation++; // a new command buffer
            }
            std::array<Binding, Bindings> next = bindings;
            std::array<Attribute, Attributes> next_attrs = attributes;
            if (((state >> 24) & 0x7) == 0) {
                next[0].stride = 16 + (state & 0xF0);
                next_attrs[1].offset = state & 0xFF;
            }
            if (generation != memo_generation) {
                known = false;
            }
            const bool emit =
                !known ||
                std::memcmp(memo_bindings.data(), next.data(), sizeof(next)) != 0 ||
                std::memcmp(memo_attributes.data(), next_attrs.data(), sizeof(next_attrs)) != 0;
            if (emit) {
                ++emits;
                live = next;
                live_attrs = next_attrs;
                memo_bindings = next;
                memo_attributes = next_attrs;
                memo_generation = generation;
                known = true;
            }
            if (!known || std::memcmp(live.data(), next.data(), sizeof(next)) != 0 ||
                std::memcmp(live_attrs.data(), next_attrs.data(), sizeof(next_attrs)) != 0) {
                ++mismatches;
            }
        }
        std::printf("vertex input memo model: 20000 draws, %llu emitted, %u states wrong after the "
                    "decision\n",
                    static_cast<unsigned long long>(emits), mismatches);
        if (mismatches != 0) {
            std::printf("FAIL: a skipped emit left the wrong vertex input state live\n");
            return 1;
        }
    }

    // 3. One sampler binding through the texture cache against the memo's table.
    constexpr std::size_t Sharps = 16;
    std::array<std::uint64_t, Sharps> raw{};
    for (std::size_t i = 0; i < Sharps; ++i) {
        raw[i] = 0x9E3779B97F4A7C15ull * (i + 3);
    }
    std::unordered_map<std::uint64_t, SamplerEntry> cache;
    for (std::size_t i = 0; i < Sharps; ++i) {
        cache.emplace(raw[i], SamplerEntry{raw[i], raw[i] >> 3, 0x1000, raw[i] ^ 0xFF, 0.0f, 0, 1,
                                           true});
    }
    std::mutex cache_mutex;
    // The sampler LRU the cache touches on every hit: the entry's last-use tick is rewritten.
    std::array<std::uint32_t, Sharps> lru_stamp{};
    std::uint64_t lru_tick = 0;
    std::array<SamplerEntry, SamplerMemoSize> memo{};
    constexpr int Binds = 4'000'000;
    double cache_ns = 0, memo_ns = 0;
    std::uint64_t hash_sink = 0;
    for (int rep = 0; rep < 3; ++rep) {
        double t0 = NowNs();
        for (int i = 0; i < Binds; ++i) {
            // The cache's path: hash the 16 S# bytes, take the mutex, probe the map, touch the LRU.
            const std::size_t index = std::size_t(i) & (Sharps - 1);
            const std::uint64_t raw0 = raw[index], raw1 = raw[index] >> 3;
            std::uint64_t hash = raw0 * 0x9E3779B97F4A7C15ull;
            hash ^= raw1 * 0xC2B2AE3D27D4EB4Full;
            hash ^= hash >> 33;
            hash *= 0xFF51AFD7ED558CCDull;
            hash ^= hash >> 29;
            std::scoped_lock lock{cache_mutex};
            const auto it = cache.find(hash);
            const std::uint32_t id = it != cache.end() ? std::uint32_t(it->second.handle & 0xF) : 0;
            // LRU touch (the cache's `sampler_lru_cache.Touch`): the entry's last-use tick.
            ++lru_tick;
            lru_stamp[id] = std::uint32_t(lru_tick);
            hash_sink += hash + lru_stamp[id];
        }
        double t1 = NowNs();
        const std::uint32_t epoch = 1;
        constexpr std::uint64_t border = 0x1000;
        for (int i = 0; i < Binds; ++i) {
            const std::uint64_t sharp = raw[std::size_t(i) & (Sharps - 1)];
            const std::uint64_t hash =
                sharp * 0x9E3779B97F4A7C15ull ^ (sharp >> 3) * 0xFF51AFD7ED558CCDull ^
                border * 0xC4CEB9FE1A85EC53ull;
            SamplerEntry& entry = memo[(hash ^ hash >> 29) & (SamplerMemoSize - 1)];
            if (entry.valid && entry.epoch == epoch && entry.raw0 == sharp &&
                entry.raw1 == sharp >> 3 && entry.border == border) {
                hash_sink += entry.handle;
            } else {
                entry = SamplerEntry{sharp, sharp >> 3, border, sharp ^ 0xFF, 0.0f, 0, epoch, true};
                hash_sink += entry.handle;
            }
        }
        double t2 = NowNs();
        cache_ns += (t1 - t0) / Binds;
        memo_ns += (t2 - t1) / Binds;
    }
    std::printf("one sampler binding: texture cache (hash, mutex, map probe, LRU relink) %.1f ns, "
                "memo table %.1f ns (%.1fx)\n",
                cache_ns / 3, memo_ns / 3, cache_ns / memo_ns);
    std::printf("  a draw binds %.0f samplers on its way through BindResources: %.0f ns of cache "
                "lookups against %.0f ns of table reads\n",
                8.0, 8 * cache_ns / 3, 8 * memo_ns / 3);
    if (hash_sink == 0xDEADBEEF || emitted_bindings == 0) {
        std::printf("unreachable\n");
    }
    return 0;
}
