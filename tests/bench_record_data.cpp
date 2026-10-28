// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: what one draw's descriptor writes cost in the recording chunk (Scheduler::RecordData,
// video_core/renderer_vulkan/vk_scheduler.h).
//
// A draw copies its write list into the chunk, then the infos each write points at. That used to
// be one RecordData call per write -- an allocation in the chunk, the headroom checks and the two
// prefetches, for 24 to 32 bytes of payload. A stage appends its buffers, then its images, then
// its samplers, so the infos of consecutive writes are consecutive vector entries: one call per
// run copies the same bytes. This is the mechanics of both loops over the same layout (the chunk
// here mirrors the real one: 128 KiB, aligned, prefetching ahead); the numbers are per draw.
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <span>
#include <vector>

namespace {

constexpr std::size_t ChunkCapacity = 128 * 1024;
constexpr std::size_t Headroom = 1024;

double NowNs() {
    timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return static_cast<double>(t.tv_sec) * 1e9 + static_cast<double>(t.tv_nsec);
}

/// The parts of Scheduler::RecordData this measures: the headroom test, the allocation, the
/// prefetches of the block being written, and the copy.
struct Chunk {
    alignas(64) std::byte storage[ChunkCapacity];
    std::size_t used = 0;

    bool Deferred() const {
        return true;
    }

    void Reserve(std::size_t bytes) {
        if (ChunkCapacity - used < bytes + Headroom) {
            used = 0; // a fresh chunk
        }
    }

    void* Allocate(std::size_t bytes, std::size_t align) {
        const std::size_t offset = (used + align - 1) & ~(align - 1);
        if (offset + bytes > ChunkCapacity) {
            return nullptr;
        }
        used = offset + bytes;
        __builtin_prefetch(storage + used + 384, 1, 3);
        __builtin_prefetch(storage + used + 448, 1, 3);
        return storage + offset;
    }

    template <typename T>
    std::span<const T> RecordData(std::span<const T> data) {
        if (!Deferred() || data.empty()) {
            return data;
        }
        Reserve(data.size_bytes() + alignof(T));
        void* storage_ = Allocate(data.size_bytes(), alignof(T));
        if (!storage_) {
            return data;
        }
        std::memcpy(storage_, data.data(), data.size_bytes());
        return {static_cast<T*>(storage_), data.size()};
    }
};

struct Write {
    std::uint64_t a, b, c;
    const void* infos;
    std::uint32_t count;
    std::uint32_t type;
};

struct Info {
    std::uint64_t a, b, c, d;
};

} // namespace

int main() {
    // A draw's bindings: 8 vertex/storage buffers, 8 textures and 4 samplers of one stage.
    constexpr std::size_t Buffers = 8, Images = 8, Samplers = 4;
    std::vector<Info> buffer_infos(Buffers), image_infos(Images + Samplers);
    std::vector<Write> writes;
    for (std::size_t i = 0; i < Buffers; ++i) {
        writes.push_back({i, i, i, &buffer_infos[i], 1, 0});
    }
    for (std::size_t i = 0; i < Images + Samplers; ++i) {
        writes.push_back({i, i, i, &image_infos[i], 1, 1});
    }
    constexpr int Draws = 200000;
    const std::size_t reserve =
        writes.size() * sizeof(Write) + 64 + writes.size() * (2 * sizeof(Info) + 32);

    double per_write = 0, coalesced = 0;
    for (int rep = 0; rep < 5; ++rep) {
        Chunk chunk;
        const double t0 = NowNs();
        for (int d = 0; d < Draws; ++d) {
            chunk.Reserve(reserve);
            const auto copied = chunk.RecordData<Write>(std::span{writes});
            auto* patched = const_cast<Write*>(copied.data());
            for (std::size_t i = 0; i < copied.size(); ++i) {
                auto& write = patched[i];
                write.infos = chunk
                                  .RecordData<Info>(std::span{
                                      static_cast<const Info*>(write.infos), write.count})
                                  .data();
            }
        }
        const double t1 = NowNs();
        for (int d = 0; d < Draws; ++d) {
            chunk.Reserve(reserve);
            const auto copied = chunk.RecordData<Write>(std::span{writes});
            auto* patched = const_cast<Write*>(copied.data());
            const std::size_t count = copied.size();
            for (std::size_t i = 0; i < count;) {
                const auto* base = static_cast<const Info*>(patched[i].infos);
                if (!base) {
                    ++i;
                    continue;
                }
                std::size_t total = patched[i].count;
                std::size_t j = i + 1;
                while (j < count && static_cast<const Info*>(patched[j].infos) == base + total &&
                       patched[j].type == patched[i].type) {
                    total += patched[j].count;
                    ++j;
                }
                const auto run = chunk.RecordData<Info>(std::span{base, total});
                std::size_t offset = 0;
                for (std::size_t k = i; k < j; ++k) {
                    patched[k].infos = run.data() + offset;
                    offset += patched[k].count;
                }
                i = j;
            }
        }
        const double t2 = NowNs();
        per_write += (t1 - t0) / Draws;
        coalesced += (t2 - t1) / Draws;
    }
    std::printf("descriptor writes, %zu writes per draw: one RecordData per write %.1f ns, per run "
                "%.1f ns (%.2fx)\n",
                writes.size(), per_write / 5, coalesced / 5, per_write / coalesced);
    return 0;
}
