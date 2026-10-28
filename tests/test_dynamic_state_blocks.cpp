// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: the dynamic-state register classification (video_core/amdgpu/dynamic_state_blocks.h).
//
// Rasterizer::UpdateDynamicState skips a group when the packet did not write any of its register
// blocks. A block that belongs to no group while a register the group reads lives in it would
// silently keep a stale viewport, depth state or cull mode -- so the classification is checked
// here against the real register file: for every register a group's update function reads, the
// block that register lives in must come back as (at least) that group.
#include <cstdio>
#include <cstdint>

// The register file's headers pull in the GPU build's fmt and spdlog, which a plain checkout does
// not have (the host tests run without the submodules): tests/stubs stands in for both, and the
// build line for this test puts it first (-Itests/stubs), see tools/ci/linux-tests.sh.
#include "video_core/amdgpu/dynamic_state_blocks.h"

namespace {

int failures = 0;

void Expect(bool condition, const char* what, unsigned block, unsigned groups) {
    if (!condition) {
        ++failures;
        std::printf("FAIL: %s (block %u, groups 0x%x)\n", what, block, groups);
    }
}

/// The groups reported for a packet that wrote the block holding `field`.
template <typename Field>
unsigned GroupsOfField(const AmdGpu::Regs& regs, const Field& field) {
    AmdGpu::RegDirty dirty;
    dirty.MarkField(regs, field);
    return AmdGpu::DynamicStateRegGroups::Get().GroupsOf(dirty);
}

} // namespace

int main() {
    using Groups = AmdGpu::DynamicStateRegGroups;
    AmdGpu::Regs regs{};
    const auto& groups = Groups::Get();

    // Every group has blocks, and no block claims a group that does not exist.
    std::array<unsigned, Groups::GroupCount> per_group{};
    for (const auto block_groups : groups.group_of_block) {
        Expect((block_groups & ~Groups::AllGroups) == 0, "unknown group bit", 0, block_groups);
        for (unsigned bit = 0; bit < Groups::GroupCount; ++bit) {
            per_group[bit] += (block_groups >> bit) & 1;
        }
    }
    for (unsigned bit = 0; bit < Groups::GroupCount; ++bit) {
        Expect(per_group[bit] != 0, "a dynamic-state group has no register blocks", bit,
               per_group[bit]);
    }

    // Each register an update function reads: its block must report the group.
    const auto expect_group = [&](auto& field, Groups::Group group, const char* name) {
        const unsigned reported = GroupsOfField(regs, field);
        Expect((reported & group) != 0, name, 0, reported);
    };
    expect_group(regs.screen_scissor, Groups::Viewport, "screen_scissor");
    expect_group(regs.window_scissor, Groups::Viewport, "window_scissor");
    expect_group(regs.generic_scissor, Groups::Viewport, "generic_scissor");
    expect_group(regs.window_offset, Groups::Viewport, "window_offset");
    expect_group(regs.viewport_control, Groups::Viewport, "viewport_control");
    expect_group(regs.viewports, Groups::Viewport, "viewports");
    expect_group(regs.viewport_depths, Groups::Viewport, "viewport_depths");
    expect_group(regs.viewport_scissors, Groups::Viewport, "viewport_scissors");
    expect_group(regs.mode_control, Groups::Viewport, "mode_control");
    expect_group(regs.clipper_control, Groups::Viewport, "clipper_control (viewport)");
    expect_group(regs.primitive_type, Groups::Viewport, "primitive_type (viewport)");

    expect_group(regs.depth_control, Groups::DepthStencil, "depth_control");
    expect_group(regs.depth_buffer, Groups::DepthStencil, "depth_buffer");
    expect_group(regs.depth_render_control, Groups::DepthStencil, "depth_render_control");
    expect_group(regs.depth_bounds_min, Groups::DepthStencil, "depth_bounds_min");
    expect_group(regs.depth_bounds_max, Groups::DepthStencil, "depth_bounds_max");
    expect_group(regs.poly_offset, Groups::DepthStencil, "poly_offset");
    expect_group(regs.stencil_control, Groups::DepthStencil, "stencil_control");
    expect_group(regs.stencil_ref_front, Groups::DepthStencil, "stencil_ref_front");
    expect_group(regs.stencil_ref_back, Groups::DepthStencil, "stencil_ref_back");

    expect_group(regs.enable_primitive_restart, Groups::Primitive, "enable_primitive_restart");
    expect_group(regs.primitive_type, Groups::Primitive, "primitive_type");
    expect_group(regs.primitive_restart_index, Groups::Primitive, "primitive_restart_index");
    expect_group(regs.polygon_control, Groups::Primitive, "polygon_control");
    expect_group(regs.clipper_control, Groups::Primitive, "clipper_control");

    expect_group(regs.line_control, Groups::Rasterization, "line_control");
    expect_group(regs.blend_constants, Groups::ColorBlend, "blend_constants");

    // A draw packet in a batch writes its SH registers and the draw's counts: that must not make
    // it recompute every group (this is what the optimisation is for).
    const unsigned sh_only =
        GroupsOfField(regs, regs.ps_program) | GroupsOfField(regs, regs.vs_program);
    Expect((sh_only & Groups::AllGroups) == 0, "a shader-register write forces a dynamic-state "
                                               "update", 0, sh_only);

    // Several blocks at once: the union of their groups, and only those.
    AmdGpu::RegDirty dirty;
    dirty.MarkField(regs, regs.viewports);
    dirty.MarkField(regs, regs.blend_constants);
    const unsigned both = groups.GroupsOf(dirty);
    Expect((both & Groups::Viewport) != 0 && (both & Groups::ColorBlend) != 0, "two groups", 0,
           both);

    // A reset (ClearState) replaces the whole register set.
    AmdGpu::RegDirty reset;
    reset.reset = true;
    Expect(groups.GroupsOf(reset) == Groups::AllGroups, "reset", 0, groups.GroupsOf(reset));

    // No blocks at all: nothing changed.
    AmdGpu::RegDirty none;
    Expect(groups.GroupsOf(none) == 0, "empty packet", 0, groups.GroupsOf(none));

    // The classification is cheap enough to sit in the draw path: one byte per register block.
    std::printf("dynamic state blocks: %u blocks, groups(viewports %u, depth %u, primitive %u, "
                "raster %u, blend %u), %zu bytes of table\n",
                AmdGpu::RegDirty::NumBlocks, per_group[0], per_group[1], per_group[2],
                per_group[3], per_group[4], sizeof(AmdGpu::DynamicStateRegGroups));
    std::printf("  block of the viewports: %u; a shader-register write reports 0x%x\n",
                unsigned((reinterpret_cast<const std::uint8_t*>(&regs.viewports) -
                          reinterpret_cast<const std::uint8_t*>(regs.reg_array.data())) /
                         (AmdGpu::RegDirty::BlockWords * sizeof(std::uint32_t))),
                sh_only);
    if (failures != 0) {
        std::printf("dynamic state blocks: %d failures\n", failures);
        return 1;
    }
    std::printf("dynamic state blocks: all checks passed\n");
    return 0;
}
