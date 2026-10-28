// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: which dynamic-state group each register block of AmdGpu::Regs belongs to.
//
// Rasterizer::UpdateDynamicState derives the Vulkan dynamic state from a handful of register
// groups, once per draw. On the draw pipeline the packet says which 32-word register blocks it
// wrote (AmdGpu::RegDirty), so a packet that wrote none of a group's blocks cannot have changed
// that group, and recomputing it is pure waste. The classification here is built once from the
// members' offsets with the same math the delta capture uses (RegDirty::MarkField), so it cannot
// drift from the struct layout, and it errs towards recomputing: any register sharing a block
// with one of a group's registers marks the whole block for that group.
//
// Deliberately free of the Vulkan headers so tests/test_dynamic_state_blocks.cpp can check the
// classification on its own.
#pragma once

#include <array>
#include "video_core/amdgpu/regs.h"

namespace AmdGpu {

struct DynamicStateRegGroups {
    /// One bit per group; a block's entry is the union of the groups that read a register in it.
    enum Group : u8 {
        Viewport = 1 << 0,
        DepthStencil = 1 << 1,
        Primitive = 1 << 2,
        Rasterization = 1 << 3,
        ColorBlend = 1 << 4,
        GroupCount = 5,
        AllGroups = (1 << GroupCount) - 1,
    };

    std::array<u8, RegDirty::NumBlocks> group_of_block{};

    /// The groups a packet that wrote `blocks` can have changed.
    u8 GroupsOf(const RegDirty& dirty) const {
        if (dirty.reset) {
            return AllGroups;
        }
        u8 groups = 0;
        dirty.blocks.ForEachSet([this, &groups](u32 block) { groups |= group_of_block[block]; });
        return groups;
    }

    [[nodiscard]] static const DynamicStateRegGroups& Get() {
        static const DynamicStateRegGroups groups = Build();
        return groups;
    }

private:
    static DynamicStateRegGroups Build() {
        DynamicStateRegGroups result;
        Regs regs{};
        const auto mark = [&regs, &result](u8 group, auto&& fields) {
            RegDirty dirty;
            fields(regs, dirty);
            dirty.blocks.ForEachSet([&result, group](u32 block) {
                result.group_of_block[block] |= group;
            });
        };
        // Rasterizer::UpdateViewportScissorState: the three scissors with their offsets, the
        // viewports and their depths, the per-viewport scissors, and the control registers that
        // enable each part (IsClipDisabled also reads the primitive type).
        mark(Viewport, [](const Regs& r, RegDirty& d) {
            d.MarkField(r, r.screen_scissor);
            d.MarkField(r, r.window_scissor);
            d.MarkField(r, r.generic_scissor);
            d.MarkField(r, r.window_offset);
            d.MarkField(r, r.polygon_control);
            d.MarkField(r, r.viewport_control);
            d.MarkField(r, r.viewports);
            d.MarkField(r, r.viewport_depths);
            d.MarkField(r, r.viewport_scissors);
            d.MarkField(r, r.clipper_control);
            d.MarkField(r, r.mode_control);
            d.MarkField(r, r.primitive_type);
        });
        // UpdateDepthStencilState: depth and stencil control, the buffer registers the two
        // validity checks read, the bounds and the polygon offset.
        mark(DepthStencil, [](const Regs& r, RegDirty& d) {
            d.MarkField(r, r.depth_control);
            d.MarkField(r, r.depth_buffer);
            d.MarkField(r, r.depth_render_control);
            d.MarkField(r, r.depth_bounds_min);
            d.MarkField(r, r.depth_bounds_max);
            d.MarkField(r, r.polygon_control);
            d.MarkField(r, r.poly_offset);
            d.MarkField(r, r.stencil_control);
            d.MarkField(r, r.stencil_ref_front);
            d.MarkField(r, r.stencil_ref_back);
        });
        // UpdatePrimitiveState: the topology and restart registers, the polygon control that
        // carries the cull mode and face, the clip control that kills rasterization.
        mark(Primitive, [](const Regs& r, RegDirty& d) {
            d.MarkField(r, r.enable_primitive_restart);
            d.MarkField(r, r.primitive_type);
            d.MarkField(r, r.primitive_restart_index);
            d.MarkField(r, r.polygon_control);
            d.MarkField(r, r.clipper_control);
        });
        // UpdateRasterizationState: the line width.
        mark(Rasterization, [](const Regs& r, RegDirty& d) { d.MarkField(r, r.line_control); });
        // UpdateColorBlendingState: the blend constants (the write masks come from the pipeline,
        // which the rasterizer compares by value).
        mark(ColorBlend, [](const Regs& r, RegDirty& d) { d.MarkField(r, r.blend_constants); });
        return result;
    }
};

} // namespace AmdGpu
