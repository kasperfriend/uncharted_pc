// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: what a draw's commands cost to record, and what the binding echo
// (common/binding_echo.h) takes off that. Two things are measured:
//
//  1. the emulator side: the closure that carries a command into the recording chunk (the same
//     placement-new, list link and two prefetches the real RecordChunk does), the copies of the
//     arrays a command points at, and for the echo path the comparison that replaces them;
//  2. the descriptor staging of one draw: a std::vector cleared and resized per draw (resize()
//     value-initializes every entry it adds) against a buffer of slots that is never resized.
//
// The driver's own cost for a vkCmd* call is not measurable here (there is no driver on the
// build machines); the numbers are reported per draw for a range of them instead, so the table
// shows what a dropped command is worth on a machine where one costs that much.
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <vector>

#include "common/binding_echo.h"

namespace {

constexpr std::size_t ChunkCapacity = 128 * 1024;
constexpr std::size_t Headroom = 1024;
constexpr std::size_t WriteDescriptorSetBytes = 80; ///< sizeof(vk::WriteDescriptorSet)
constexpr std::size_t PushBlockBytes = 96;         ///< sizeof(Shader::PushData) and change

double NowNs() {
    timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return static_cast<double>(t.tv_sec) * 1e9 + static_cast<double>(t.tv_nsec);
}

/// The parts of Scheduler::Record/RecordData this measures: the headroom test, the aligned
/// placement of the closure (with a trivial destructor, as the commands here have), the list
/// link and the two prefetches of the block being written.
struct Chunk {
    alignas(64) std::byte storage[ChunkCapacity];
    std::size_t used = 0;
    std::size_t commands = 0;

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

    template <typename Command>
    void Push(const Command& command) {
        void* slot = Allocate(sizeof(Command), alignof(Command));
        if (!slot) {
            used = 0;
            slot = Allocate(sizeof(Command), alignof(Command));
        }
        std::memcpy(slot, &command, sizeof(Command));
        ++commands;
    }

    void Reset() {
        used = 0;
        commands = 0;
    }
};

/// One draw's commands as the recorder writes them: push constants, the descriptor set with its
/// write list and infos, the vertex buffers with their arrays, the index buffer, and the draw
/// (with its pipeline bind). Counts are a G-buffer draw: 18 descriptors, 2 vertex streams.
struct PushConstantsCommand {
    std::uint64_t layout;
    std::uint32_t stages;
    std::uint32_t size;
    std::byte data[PushBlockBytes];
};
struct DescriptorSetCommand {
    std::uint64_t layout;
    std::uint64_t data;
    std::uint32_t count;
    std::uint32_t bind_point;
};
struct VertexBufferCommand {
    std::uint32_t count;
    std::uint64_t buffers;
    std::uint64_t offsets;
    std::uint64_t sizes;
    std::uint64_t strides;
};
struct IndexBufferCommand {
    std::uint64_t handle;
    std::uint64_t offset;
    std::uint32_t type;
};
struct DrawCommand {
    std::uint64_t pipeline;
    std::uint32_t count;
    std::uint32_t instances;
    std::uint32_t first_index;
    std::uint32_t first_instance;
};

constexpr std::uint32_t Descriptors = 18;
constexpr std::uint32_t VertexStreams = 2;

struct Draw {
    std::uint64_t pipeline;
    std::uint64_t mesh;
    std::uint64_t index_buffer;
    std::uint64_t push_checksum; // stands in for "the push block is the same as last draw's"
};

/// The commands of one draw through the recorder, with the echo consulted first when `echo` is
/// not null. Returns nothing: the chunk's counters are the result.
void RecordDraw(Chunk& chunk, const Draw& draw, Common::BindingEcho* echo,
                std::uint64_t& recorded) {
    static thread_local std::vector<std::uint64_t> infos(Descriptors * 4);

    const bool push =
        echo ? echo->PushConstants(0x1000, 0x10, &draw.push_checksum, sizeof(draw.push_checksum))
             : true;
    if (push) {
        PushConstantsCommand command{0x1000, 0x10, PushBlockBytes, {}};
        chunk.Reserve(sizeof(command) + 16);
        chunk.Push(command);
        ++recorded;
    }

    // The descriptor list is a real change on most draws (a texture or a buffer moved): it is
    // always recorded, which is what the draw costs whoever binds it.
    chunk.Reserve(Descriptors * WriteDescriptorSetBytes + Descriptors * 40 + 1024);
    std::memcpy(chunk.Allocate(Descriptors * WriteDescriptorSetBytes, 8), infos.data(),
                Descriptors * WriteDescriptorSetBytes);
    std::memcpy(chunk.Allocate(Descriptors * 40, 8), infos.data(), Descriptors * 40);
    DescriptorSetCommand set{0x2000, 0, Descriptors, 0};
    chunk.Push(set);
    ++recorded;

    std::uint64_t streams[Common::BindingEcho::MaxVertexBuffers]{};
    std::uint64_t offsets[Common::BindingEcho::MaxVertexBuffers]{};
    std::uint64_t sizes[Common::BindingEcho::MaxVertexBuffers]{};
    std::uint64_t strides[Common::BindingEcho::MaxVertexBuffers]{};
    for (std::uint32_t i = 0; i < VertexStreams; ++i) {
        streams[i] = draw.mesh + i * 0x1000;
        offsets[i] = 0;
        sizes[i] = 4096;
        strides[i] = 16;
    }
    const bool vertex =
        echo ? echo->VertexBuffers(VertexStreams, streams, offsets, sizes, strides) : true;
    if (vertex) {
        chunk.Reserve(4 * VertexStreams * sizeof(std::uint64_t) + 128);
        std::memcpy(chunk.Allocate(VertexStreams * sizeof(std::uint64_t), 8), streams,
                    VertexStreams * sizeof(std::uint64_t));
        std::memcpy(chunk.Allocate(VertexStreams * sizeof(std::uint64_t), 8), offsets,
                    VertexStreams * sizeof(std::uint64_t));
        std::memcpy(chunk.Allocate(VertexStreams * sizeof(std::uint64_t), 8), sizes,
                    VertexStreams * sizeof(std::uint64_t));
        std::memcpy(chunk.Allocate(VertexStreams * sizeof(std::uint64_t), 8), strides,
                    VertexStreams * sizeof(std::uint64_t));
        VertexBufferCommand command{VertexStreams, 0, 0, 0, 0};
        chunk.Push(command);
        ++recorded;
    }

    const bool index = echo ? echo->IndexBuffer(draw.index_buffer, 0, 0) : true;
    if (index) {
        IndexBufferCommand command{draw.index_buffer, 0, 0};
        chunk.Reserve(sizeof(command) + 16);
        chunk.Push(command);
        ++recorded;
    }

    const bool pipeline = echo ? echo->Pipeline(true, draw.pipeline) : true;
    DrawCommand command{draw.pipeline, 3, 1, 0, 0};
    chunk.Reserve(sizeof(command) + 16);
    if (!pipeline) {
        // The bind is gone from the closure; the draw itself stays.
        DrawCommand without_pipeline{0, 3, 1, 0, 0};
        chunk.Push(without_pipeline);
    } else {
        chunk.Push(command);
    }
    ++recorded;
}

/// A scene pass: 40 materials, 8 draws each, every mesh drawn twice in a row (as in
/// tests/test_binding_echo.cpp), one push block per material.
std::vector<Draw> Scene() {
    std::vector<Draw> draws;
    draws.reserve(320);
    for (std::uint64_t material = 0; material < 40; ++material) {
        for (std::uint32_t instance = 0; instance < 8; ++instance) {
            Draw draw{};
            draw.pipeline = 0x1000 + material;
            draw.mesh = 0x100000 + material * 0x1000 + (instance / 2) * 0x100;
            draw.index_buffer = 0x900000 + material * 0x200 + (instance / 2) * 0x10;
            draw.push_checksum = material;
            draws.push_back(draw);
        }
    }
    return draws;
}

struct Timing {
    double ns_per_draw = 0;
    double commands_per_draw = 0;
};

Timing RunScene(const std::vector<Draw>& draws, bool use_echo, double& checksum) {
    Chunk chunk;
    Common::BindingEcho echo;
    constexpr int Repetitions = 400;
    std::uint64_t recorded = 0;
    // Warm up the chunk (first-touch of its pages) before the clock starts.
    for (const auto& draw : draws) {
        RecordDraw(chunk, draw, use_echo ? &echo : nullptr, recorded);
    }
    chunk.Reset();
    echo.Clear();
    recorded = 0;
    const double start = NowNs();
    for (int repetition = 0; repetition < Repetitions; ++repetition) {
        echo.Clear(); // one command buffer per pass
        for (const auto& draw : draws) {
            RecordDraw(chunk, draw, use_echo ? &echo : nullptr, recorded);
        }
    }
    const double elapsed = NowNs() - start;
    checksum += static_cast<double>(chunk.used + recorded);
    Timing timing;
    timing.ns_per_draw = elapsed / (double(Repetitions) * double(draws.size()));
    timing.commands_per_draw = double(recorded) / (double(Repetitions) * double(draws.size()));
    return timing;
}

/// One descriptor write as the binding code fills it (Rasterizer::DescriptorWrite): nine fields,
/// 80 bytes with the padding vulkan.hpp's structure has.
struct WriteSlot {
    void* p_next;
    std::uint64_t dst_set;
    std::uint32_t dst_binding;
    std::uint32_t dst_array_element;
    std::uint32_t descriptor_count;
    std::uint32_t descriptor_type;
    const void* p_image_info;
    const void* p_buffer_info;
    const void* p_texel_buffer_view;
    std::byte padding[WriteDescriptorSetBytes - 56];
};
static_assert(sizeof(WriteSlot) == WriteDescriptorSetBytes, "the model is the whole write");

inline void Fill(WriteSlot& write, std::uint32_t binding) {
    write.p_next = nullptr;
    write.dst_set = 0;
    write.dst_binding = binding;
    write.dst_array_element = 0;
    write.descriptor_count = 1;
    write.descriptor_type = 1;
    write.p_image_info = nullptr;
    write.p_buffer_info = &write;
    write.p_texel_buffer_view = nullptr;
}

/// Descriptor staging: the vector the port cleared and resized per draw against the buffer of
/// slots it keeps now. The old path value-initializes every entry (all 80 bytes) and then fills
/// it; the new one only fills it.
Timing RunStaging(std::uint32_t descriptors) {
    constexpr int Draws = 200'000;
    std::vector<std::byte> vector_staging;
    std::vector<std::byte> slot_staging;
    const std::size_t bytes = std::size_t{descriptors} * WriteDescriptorSetBytes;
    slot_staging.resize(std::max<std::size_t>(bytes, 64 * WriteDescriptorSetBytes));
    double checksum = 0;

    const double vector_start = NowNs();
    for (int draw = 0; draw < Draws; ++draw) {
        vector_staging.clear();
        vector_staging.resize(bytes); // value-initializes every entry
        for (std::uint32_t i = 0; i < descriptors; ++i) {
            Fill(*reinterpret_cast<WriteSlot*>(vector_staging.data() + i * WriteDescriptorSetBytes),
                 i);
        }
        checksum += static_cast<double>(vector_staging.size() & 0xFF);
    }
    const double vector_ns = (NowNs() - vector_start) / Draws;

    const double slot_start = NowNs();
    for (int draw = 0; draw < Draws; ++draw) {
        for (std::uint32_t i = 0; i < descriptors; ++i) {
            Fill(*reinterpret_cast<WriteSlot*>(slot_staging.data() + i * WriteDescriptorSetBytes),
                 i);
        }
        checksum += static_cast<double>(bytes & 0xFF);
    }
    const double slot_ns = (NowNs() - slot_start) / Draws;

    std::printf("  %2u descriptors: clear+resize %.1f ns/draw, kept slots %.1f ns/draw "
                "(%.1f us per 2000-draw frame)\n",
                descriptors, vector_ns, slot_ns, (vector_ns - slot_ns) * 2000.0 / 1000.0);
    Timing timing;
    timing.ns_per_draw = vector_ns - slot_ns;
    checksum += slot_ns;
    std::printf("%s", checksum == 0.0 ? "" : "");
    return timing;
}

} // namespace

int main() {
    std::printf("binding echo + descriptor slots (x86-64, this host)\n");

    const auto draws = Scene();
    double checksum = 0;

    // Warm the clocks and the chunk.
    RunScene(draws, false, checksum);
    RunScene(draws, true, checksum);

    double before = 1e18, after = 1e18;
    Timing recorded{}, echoed{};
    for (int repetition = 0; repetition < 5; ++repetition) {
        const Timing plain = RunScene(draws, false, checksum);
        const Timing with_echo = RunScene(draws, true, checksum);
        if (plain.ns_per_draw < before) {
            before = plain.ns_per_draw;
            recorded = plain;
        }
        if (with_echo.ns_per_draw < after) {
            after = with_echo.ns_per_draw;
            echoed = with_echo;
        }
    }

    std::printf("\nA scene pass of %zu draws, recorded into the chunk (emulator side only):\n",
                draws.size());
    std::printf("  commands per draw: %.2f recorded, %.2f with the echo (%.0f%% dropped)\n",
                recorded.commands_per_draw, echoed.commands_per_draw,
                100.0 * (recorded.commands_per_draw - echoed.commands_per_draw) /
                    recorded.commands_per_draw);
    std::printf("  recording:         %.0f ns/draw, %.0f ns/draw with the echo (%.0f ns saved, "
                "%.1f us per 2000-draw frame)\n",
                before, after, before - after, (before - after) * 2000.0 / 1000.0);

    std::printf("\nWhat the dropped commands are worth when one vkCmd* costs (per 2000-draw "
                "frame):\n");
    const double dropped = recorded.commands_per_draw - echoed.commands_per_draw;
    for (const double driver_ns : {50.0, 100.0, 200.0, 400.0}) {
        const double per_frame_us = dropped * 2000.0 * driver_ns / 1000.0;
        std::printf("  %3.0f ns: %6.0f us/frame (%4.1f%% of a 7 ms frame)\n", driver_ns,
                    per_frame_us, 100.0 * per_frame_us / 1000.0 / 7.0);
    }

    std::printf("\nDescriptor writes of one draw, staged the old way and the new way:\n");
    double staging = 0;
    for (const std::uint32_t descriptors : {8u, 16u, 24u, 32u}) {
        staging += RunStaging(descriptors).ns_per_draw;
    }
    std::printf("  (the sum over the four shapes: %.0f ns/draw)\n", staging);

    if (checksum == 12345.0) {
        std::printf("(checksum %f)\n", checksum); // keeps the measured work alive
    }
    return 0;
}
