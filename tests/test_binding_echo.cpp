// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: the binding echo (common/binding_echo.h) must never drop a command that changes state.
// A wrong skip is a draw with a stale pipeline, constants or vertex buffer -- a picture that is
// wrong in a way no assertion in the port would catch. So the memos are tested here without
// Vulkan: every kind of state, the exact repeats that must be dropped, and the near misses (one
// byte, one handle, one offset, the count, the optional ranges) that must not be.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "common/binding_echo.h"

namespace {

int failures = 0;

void Check(bool condition, const char* what) {
    if (!condition) {
        std::printf("FAILED: %s\n", what);
        ++failures;
    }
}

using Common::BindingEcho;

void TestPipeline() {
    BindingEcho echo;
    Check(echo.Pipeline(true, 0x1000), "first graphics pipeline is recorded");
    Check(!echo.Pipeline(true, 0x1000), "the same pipeline again is dropped");
    Check(!echo.Pipeline(true, 0x1000), "and every time after that");
    Check(echo.Pipeline(true, 0x2000), "another pipeline is recorded");
    Check(!echo.Pipeline(true, 0x2000), "then it is the live one");
    Check(echo.Pipeline(true, 0x1000), "switching back is recorded");
    // Compute has its own memo: a compute pipeline that happens to have the same handle value
    // as the live graphics one is still its own bind point.
    Check(echo.Pipeline(false, 0x3000), "first compute pipeline is recorded");
    Check(!echo.Pipeline(false, 0x3000), "the same compute pipeline again is dropped");
    Check(!echo.Pipeline(true, 0x1000), "a compute bind does not disturb the graphics memo");
    echo.Clear();
    Check(echo.Empty(), "clearing forgets every memo");
    Check(echo.Pipeline(true, 0x1000), "a new command buffer records everything again");
}

void TestPushConstants() {
    BindingEcho echo;
    alignas(8) std::uint32_t block[32]{};
    const std::uint32_t graphics = 0x10;
    const std::uint32_t compute = 0x20;

    Check(echo.PushConstants(0x1000, graphics, block, 96), "the first block is recorded");
    Check(!echo.PushConstants(0x1000, graphics, block, 96), "the same block is dropped");
    block[7] = 1;
    Check(echo.PushConstants(0x1000, graphics, block, 96), "one changed word is recorded");
    Check(!echo.PushConstants(0x1000, graphics, block, 96), "and is the live block after that");
    Check(echo.PushConstants(0x1000, compute, block, 96),
          "the same bytes for another stage mask are recorded");
    Check(echo.PushConstants(0x2000, compute, block, 96),
          "the same bytes through another layout are recorded");
    Check(echo.PushConstants(0x2000, compute, block, 92), "a shorter block is recorded");
    Check(!echo.PushConstants(0x2000, compute, block, 92), "then it is the live one");

    // Over the guaranteed size: never compared, and what was live is forgotten.
    std::vector<std::uint8_t> big(256, 0);
    Check(echo.PushConstants(0x2000, compute, big.data(), 256), "an oversized block is recorded");
    Check(echo.PushConstants(0x2000, compute, block, 92),
          "the block that was live before the oversized one is recorded again");
}

void TestVertexBuffers() {
    BindingEcho echo;
    const std::uint64_t buffers[3]{0xAA, 0xBB, 0xCC};
    const std::uint64_t offsets[3]{0, 16, 32};
    const std::uint64_t sizes[3]{4096, 4096, 4096};
    const std::uint64_t strides[3]{12, 16, 32};

    Check(echo.VertexBuffers(3, buffers, offsets, sizes, strides), "the first bind is recorded");
    Check(!echo.VertexBuffers(3, buffers, offsets, sizes, strides), "the same bind is dropped");
    Check(echo.VertexBuffers(3, buffers, offsets, nullptr, nullptr),
          "the same handles without ranges are a different bind");
    Check(!echo.VertexBuffers(3, buffers, offsets, nullptr, nullptr), "which then repeats");
    Check(echo.VertexBuffers(3, buffers, offsets, sizes, strides),
          "going back to the ranged bind is recorded");

    std::uint64_t other[3]{0xAA, 0xBB, 0xCD};
    Check(echo.VertexBuffers(3, other, offsets, sizes, strides),
          "a different last handle is recorded");
    std::uint64_t moved[3]{0, 16, 32};
    Check(echo.VertexBuffers(3, buffers, moved, sizes, strides), "a different offset is recorded");
    std::uint64_t bigger[3]{4096, 4096, 8192};
    Check(echo.VertexBuffers(3, buffers, offsets, bigger, strides), "a different size is recorded");
    std::uint64_t packed[3]{12, 16, 24};
    Check(echo.VertexBuffers(3, buffers, offsets, sizes, packed), "a different stride is recorded");
    Check(echo.VertexBuffers(2, buffers, offsets, sizes, strides), "a different count is recorded");

    // 32 streams is the most the port binds (MaxVertexBufferCount); more is never compared.
    std::uint64_t many_buffers[33]{};
    std::uint64_t many_offsets[33]{};
    std::uint64_t many_sizes[33]{};
    std::uint64_t many_strides[33]{};
    for (std::uint32_t i = 0; i < 33; ++i) {
        many_buffers[i] = 0x1000 + i;
    }
    Check(echo.VertexBuffers(33, many_buffers, many_offsets, many_sizes, many_strides),
          "an oversized bind is recorded");
    Check(echo.VertexBuffers(33, many_buffers, many_offsets, many_sizes, many_strides),
          "and is never dropped");
}

void TestIndexBuffer() {
    BindingEcho echo;
    Check(echo.IndexBuffer(0x1234, 0, 0), "the first index buffer is recorded");
    Check(!echo.IndexBuffer(0x1234, 0, 0), "the same one is dropped");
    Check(echo.IndexBuffer(0x1234, 4, 0), "another offset is recorded");
    Check(echo.IndexBuffer(0x1234, 4, 1), "another type is recorded");
    Check(echo.IndexBuffer(0x9999, 4, 1), "another handle is recorded");
    Check(!echo.IndexBuffer(0x9999, 4, 1), "then it is the live one");
}

/// The memos are independent: a draw that repeats its vertex bind but changes its index buffer
/// must record the index bind, and the drop of the vertex one must not touch anything else.
void TestIndependence() {
    BindingEcho echo;
    const std::uint64_t buffers[1]{0xAA};
    const std::uint64_t offsets[1]{0};
    Check(echo.VertexBuffers(1, buffers, offsets, nullptr, nullptr), "vertex bind recorded");
    Check(echo.IndexBuffer(0x1234, 0, 0), "index bind recorded");
    Check(echo.Pipeline(true, 0x1000), "pipeline recorded");
    Check(!echo.VertexBuffers(1, buffers, offsets, nullptr, nullptr), "vertex bind dropped");
    Check(echo.IndexBuffer(0x1235, 0, 0), "a new index buffer is still recorded");
    Check(!echo.Pipeline(true, 0x1000), "the pipeline is still live");
}

/// A frame's worth of a scene pass: 40 materials, each drawn 8 times, every mesh twice in a row
/// (the shape of a prepass and a colour pass over the same geometry). The pipeline repeats for
/// all 8 draws of a material and each vertex binding for its pair, so the echo may drop exactly
/// those; anything else would be a state the command buffer does not have.
void TestScenePattern() {
    BindingEcho echo;
    std::uint64_t dropped = 0, recorded = 0;
    const std::uint64_t offsets[2]{0, 0};
    std::uint32_t draws = 0;
    for (std::uint64_t material = 0; material < 40; ++material) {
        const std::uint64_t pipeline = 0x1000 + material;
        for (std::uint32_t instance = 0; instance < 8; ++instance) {
            ++draws;
            const std::uint64_t mesh = 0x100000 + material * 0x1000 + (instance / 2) * 0x100;
            std::uint64_t stream[2]{mesh, mesh};
            (echo.Pipeline(true, pipeline) ? recorded : dropped) += 1;
            (echo.VertexBuffers(2, stream, offsets, nullptr, nullptr) ? recorded : dropped) += 1;
        }
    }
    // 320 draws: 40 pipelines and 160 vertex bindings are recorded, the other 280 commands only
    // restate what the one before them left live.
    Check(draws == 320, "the pattern is 320 draws");
    Check(recorded == 40 + 160, "the scene pass records what it must");
    Check(dropped == 280 + 160, "and drops the repeats");
    Check(recorded + dropped == 2 * draws, "every command was either recorded or dropped");
    std::printf("Scene pattern: %u draws, %llu binds recorded, %llu dropped (%.0f%%)\n", draws,
                static_cast<unsigned long long>(recorded),
                static_cast<unsigned long long>(dropped),
                100.0 * double(dropped) / double(recorded + dropped));
}

} // namespace

int main() {
    TestPipeline();
    TestPushConstants();
    TestVertexBuffers();
    TestIndexBuffer();
    TestIndependence();
    TestScenePattern();
    if (failures != 0) {
        std::printf("%d binding echo checks failed\n", failures);
        return 1;
    }
    std::printf("binding echo: all checks passed\n");
    return 0;
}
