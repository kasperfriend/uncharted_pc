// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: the binding echo (docs/PERFORMANCE.md, "Commands that restate the command buffer").
//
// A draw is a handful of commands, and most of them only restate state the command buffer
// already holds: every draw of a material binds the same pipeline, every draw of a pass pushes
// the same constants, the next batch of a mesh binds the same vertex and index buffer. Each of
// those commands costs a driver call and the bytes that carry it into the recording chunk, and
// the recorder cannot know that it is redundant -- only the state holder can.
//
// This is that state holder: one memo per kind of binding, holding what the command buffer has
// live, with the question the recorder asks instead of recording:
//
//     if (echo.Pipeline(handle)) { cmdbuf.bindPipeline(..., handle); }
//
// Handles are plain integers here so the memos can be tested without Vulkan (tests/
// test_binding_echo.cpp), and so a comparison is one 64-bit compare rather than a copy.
//
// The memos describe one command buffer: clear() them when a new one starts (state does not
// carry over between command buffers).

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace Common {

class BindingEcho {
public:
    using Handle = std::uint64_t;
    static constexpr std::uint32_t MaxVertexBuffers = 32;
    /// The largest push constant block the port uses (Vulkan guarantees 128 bytes).
    static constexpr std::uint32_t MaxPushSize = 128;

    /// Records one command buffer: everything it states is forgotten.
    void Clear() {
        graphics_valid = compute_valid = false;
        push_valid = vertex_valid = index_valid = false;
        vertex_count = 0;
    }

    /// True while nothing is known about the command buffer (a fresh one).
    [[nodiscard]] bool Empty() const noexcept {
        return !graphics_valid && !compute_valid && !push_valid && !vertex_valid && !index_valid;
    }

    /// True when the pipeline has to be bound (the command buffer does not have it live).
    bool Pipeline(bool graphics, Handle pipeline) {
        bool& valid = graphics ? graphics_valid : compute_valid;
        Handle& slot = graphics ? graphics_pipeline : compute_pipeline;
        if (valid && slot == pipeline) {
            return false;
        }
        slot = pipeline;
        valid = true;
        return true;
    }

    /// True when the push constant block has to be recorded. The layout and the stage mask are
    /// part of the comparison: two pipelines that share a block still have their own layout, and
    /// the same layout can be pushed for different stages (compute and graphics).
    bool PushConstants(Handle layout, std::uint32_t stages, const void* data, std::uint32_t size) {
        if (size > MaxPushSize) {
            push_valid = false; // not compared: forget what was live rather than compare against it
            return true;
        }
        if (push_valid && push_layout == layout && push_stages == stages && push_size == size &&
            (size == 0 || std::memcmp(push_data.data(), data, size) == 0)) {
            return false;
        }
        push_layout = layout;
        push_stages = stages;
        push_size = size;
        if (size) {
            std::memcpy(push_data.data(), data, size);
        }
        push_valid = true;
        return true;
    }

    /// True when the vertex buffers have to be bound. `sizes` and `strides` are null when the
    /// draw leaves them to the dynamic state (bindVertexBuffers, not bindVertexBuffers2).
    bool VertexBuffers(std::uint32_t count, const Handle* buffers, const std::uint64_t* offsets,
                       const std::uint64_t* sizes, const std::uint64_t* strides) {
        if (count > MaxVertexBuffers) {
            vertex_valid = false; // not compared from now on; forget what was live
            return true;
        }
        const bool ranges = sizes != nullptr;
        // A different first handle is the common case by far (each mesh has its own vertex
        // buffer): one comparison decides it, the rest of the list is compared only for a batch
        // that really repeats a binding.
        if (vertex_valid && vertex_count == count && vertex_ranges == ranges &&
            (count == 0 || (vertex_buffers[0] == buffers[0] && vertex_offsets[0] == offsets[0])) &&
            std::memcmp(vertex_buffers.data(), buffers, count * sizeof(Handle)) == 0 &&
            std::memcmp(vertex_offsets.data(), offsets, count * sizeof(std::uint64_t)) == 0 &&
            (!ranges ||
             (std::memcmp(vertex_sizes.data(), sizes, count * sizeof(std::uint64_t)) == 0 &&
              std::memcmp(vertex_strides.data(), strides, count * sizeof(std::uint64_t)) == 0))) {
            return false;
        }
        vertex_count = count;
        vertex_ranges = ranges;
        if (count) {
            std::memcpy(vertex_buffers.data(), buffers, count * sizeof(Handle));
            std::memcpy(vertex_offsets.data(), offsets, count * sizeof(std::uint64_t));
            if (ranges) {
                std::memcpy(vertex_sizes.data(), sizes, count * sizeof(std::uint64_t));
                std::memcpy(vertex_strides.data(), strides, count * sizeof(std::uint64_t));
            }
        }
        vertex_valid = true;
        return true;
    }

    /// True when the index buffer has to be bound.
    bool IndexBuffer(Handle buffer, std::uint64_t offset, std::uint32_t type) {
        if (index_valid && index_buffer == buffer && index_offset == offset && index_type == type) {
            return false;
        }
        index_buffer = buffer;
        index_offset = offset;
        index_type = type;
        index_valid = true;
        return true;
    }

private:
    Handle graphics_pipeline{};
    Handle compute_pipeline{};
    bool graphics_valid{};
    bool compute_valid{};

    Handle push_layout{};
    std::uint32_t push_stages{};
    std::uint32_t push_size{};
    std::array<std::byte, MaxPushSize> push_data{};
    bool push_valid{};

    std::uint32_t vertex_count{};
    bool vertex_ranges{};
    bool vertex_valid{};
    std::array<Handle, MaxVertexBuffers> vertex_buffers{};
    std::array<std::uint64_t, MaxVertexBuffers> vertex_offsets{};
    std::array<std::uint64_t, MaxVertexBuffers> vertex_sizes{};
    std::array<std::uint64_t, MaxVertexBuffers> vertex_strides{};

    Handle index_buffer{};
    std::uint64_t index_offset{};
    std::uint32_t index_type{};
    bool index_valid{};

};

} // namespace Common
