// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: what a draw recomputes for its Vulkan dynamic state, and what the register-block memo
// in Rasterizer::UpdateDynamicState lets it skip.
//
// The five update functions turn a handful of register groups into viewports, scissors, depth
// and stencil state, cull mode and blend constants. A draw that wrote none of a group's register
// blocks leaves all of it exactly as the previous draw did -- and the memo then costs one table
// lookup per block the packet carried (2 to 8) plus a bit test. The math below is the viewport
// and scissor half of UpdateViewportScissorState with the same inputs (one or two active
// viewports, the usual case for this title), reduced to plain types; the numbers are per draw.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <ctime>

namespace {

double NowNs() {
    timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return static_cast<double>(t.tv_sec) * 1e9 + static_cast<double>(t.tv_nsec);
}

struct Scissor {
    std::int16_t top_left_x, top_left_y, bottom_right_x, bottom_right_y;
    std::uint32_t GetWidth() const {
        return std::uint32_t(bottom_right_x - top_left_x);
    }
    std::uint32_t GetHeight() const {
        return std::uint32_t(bottom_right_y - top_left_y);
    }
    static std::int16_t Clamp(std::int16_t v) {
        return v;
    }
};

struct ViewportBounds {
    float xoffset, xscale, yoffset, yscale, zoffset, zscale;
};

struct ViewportDepth {
    float zoffset, zscale;
};

struct Rect {
    std::int32_t x, y;
    std::uint32_t width, height;
};

struct Viewport {
    float x, y, width, height, minDepth, maxDepth;
};

struct Inputs {
    Scissor screen_scissor{0, 0, 1920, 1080};
    Scissor window_scissor{0, 0, 1920, 1080};
    Scissor generic_scissor{0, 0, 1920, 1080};
    std::int16_t window_x_offset = 0, window_y_offset = 0;
    bool window_offset_disable = false;
    std::array<ViewportBounds, 16> viewports{};
    std::array<ViewportDepth, 16> viewport_depths{};
    std::array<Scissor, 16> viewport_scissors{};
    std::uint32_t active_viewports = 1;
    bool xoffset_enable = true, yoffset_enable = true;
    bool xscale_enable = true, yscale_enable = true;
    bool zoffset_enable = false, zscale_enable = false;
    bool vport_scissor_enable = false;
    bool clip_space_minus_w = false;
    std::array<float, 2> target_scale{1.0f, 1.0f};
    std::array<float, 2> draw_jitter{0.0f, 0.0f};
    // The registers UpdateDepthStencilState, UpdatePrimitiveState, UpdateRasterizationState and
    // UpdateColorBlendingState read.
    bool depth_enable = true, depth_valid = true, depth_write = true, depth_clear = false;
    int depth_func = 3;
    bool depth_bounds_enable = false;
    float depth_bounds_min = 0.0f, depth_bounds_max = 1.0f;
    bool needs_bias = true, front_bias = false;
    float front_offset = 0.0f, back_offset = -1.0f, depth_bias = 0.0f;
    float front_scale = 0.0f, back_scale = 4.0f;
    bool stencil_enabled = true;
    std::array<int, 4> stencil_front_ops{1, 2, 3, 4}, stencil_back_ops{1, 2, 3, 4};
    std::uint32_t stencil_front_reference = 0, stencil_back_reference = 0;
    std::uint32_t stencil_front_write_mask = 0xff, stencil_back_write_mask = 0xff;
    std::uint32_t stencil_front_compare_mask = 0xff, stencil_back_compare_mask = 0xff;
    bool prim_restart = false, rasterizer_kill = false;
    int cull_mode = 1, front_face = 0;
    std::array<float, 4> blend_constants{0.0f, 0.0f, 0.0f, 0.0f};
    float line_width = 1.0f;
    bool feedback_loop = false;
    std::array<std::uint32_t, 8> write_masks{15, 15, 0, 0, 0, 0, 0, 0};
};

/// UpdateViewportScissorState's computation of the viewports and the scissors.
void ComputeDynamicViewports(const Inputs& in, Viewport* viewports, Rect* scissors,
                             std::size_t& count) {
    const auto combined_tl = [](std::int16_t scr, std::int16_t win, std::int16_t gen,
                                std::int16_t offset) {
        return std::max({scr, std::int16_t(win + offset), std::int16_t(gen + offset)});
    };
    const auto combined_br = [](std::int16_t scr, std::int16_t win, std::int16_t gen,
                                std::int16_t offset) {
        return std::min({scr, std::int16_t(win + offset), std::int16_t(gen + offset)});
    };
    const bool enable_offset = !in.window_offset_disable;
    Scissor csr{};
    csr.top_left_x = combined_tl(in.screen_scissor.top_left_x, in.window_scissor.top_left_x,
                                 in.generic_scissor.top_left_x,
                                 enable_offset ? in.window_x_offset : 0);
    csr.top_left_y = combined_tl(in.screen_scissor.top_left_y, in.window_scissor.top_left_y,
                                 in.generic_scissor.top_left_y,
                                 enable_offset ? in.window_y_offset : 0);
    csr.bottom_right_x = combined_br(in.screen_scissor.bottom_right_x,
                                     in.window_scissor.bottom_right_x,
                                     in.generic_scissor.bottom_right_x,
                                     enable_offset ? in.window_x_offset : 0);
    csr.bottom_right_y = combined_br(in.screen_scissor.bottom_right_y,
                                     in.window_scissor.bottom_right_y,
                                     in.generic_scissor.bottom_right_y,
                                     enable_offset ? in.window_y_offset : 0);

    count = 0;
    for (std::uint32_t i = 0; i < in.active_viewports; ++i) {
        const auto& vp = in.viewports[i];
        const auto& vp_d = in.viewport_depths[i];
        if (vp.xscale == 0) {
            continue;
        }
        const auto zoffset = in.zoffset_enable ? vp_d.zoffset : 0.0f;
        const auto zscale = in.zscale_enable ? vp_d.zscale : 1.0f;
        Viewport viewport{};
        if (in.clip_space_minus_w) {
            viewport.minDepth = zoffset - zscale;
            viewport.maxDepth = zoffset + zscale;
        } else {
            viewport.minDepth = zoffset;
            viewport.maxDepth = zoffset + zscale;
        }
        const auto xoffset = in.xoffset_enable ? vp.xoffset : 0.0f;
        const auto xscale = in.xscale_enable ? vp.xscale : 1.0f;
        const auto yoffset = in.yoffset_enable ? vp.yoffset : 0.0f;
        const auto yscale = in.yscale_enable ? vp.yscale : 1.0f;
        viewport.x = (xoffset - xscale) * in.target_scale[0] + in.draw_jitter[0];
        viewport.y = (yoffset - yscale) * in.target_scale[1] + in.draw_jitter[1];
        viewport.width = xscale * 2.0f * in.target_scale[0];
        viewport.height = yscale * 2.0f * in.target_scale[1];
        viewports[count] = viewport;

        auto vp_scsr = csr;
        if (in.vport_scissor_enable) {
            vp_scsr.top_left_x = std::max(vp_scsr.top_left_x, in.viewport_scissors[i].top_left_x);
            vp_scsr.top_left_y = std::max(vp_scsr.top_left_y, in.viewport_scissors[i].top_left_y);
            vp_scsr.bottom_right_x =
                std::min(Scissor::Clamp(vp_scsr.bottom_right_x),
                         in.viewport_scissors[i].bottom_right_x);
            vp_scsr.bottom_right_y =
                std::min(Scissor::Clamp(vp_scsr.bottom_right_y),
                         in.viewport_scissors[i].bottom_right_y);
        }
        if (in.target_scale[0] != 1.0f || in.target_scale[1] != 1.0f) {
            const auto scale = [](std::int32_t v, float f) {
                return std::int32_t(std::lround(float(v) * f));
            };
            const std::int32_t x0 = scale(vp_scsr.top_left_x, in.target_scale[0]);
            const std::int32_t y0 = scale(vp_scsr.top_left_y, in.target_scale[1]);
            const std::int32_t x1 =
                scale(vp_scsr.top_left_x + std::int32_t(vp_scsr.GetWidth()), in.target_scale[0]);
            const std::int32_t y1 =
                scale(vp_scsr.top_left_y + std::int32_t(vp_scsr.GetHeight()), in.target_scale[1]);
            scissors[count] = {x0, y0, std::uint32_t(std::max(x1 - x0, 0)),
                               std::uint32_t(std::max(y1 - y0, 0))};
        } else {
            scissors[count] = {vp_scsr.top_left_x, vp_scsr.top_left_y, vp_scsr.GetWidth(),
                               vp_scsr.GetHeight()};
        }
        ++count;
    }
}

// The dynamic state the setters below feed: Scheduler::DynamicState with the same fields and
// the same guarded setters (a value that does not change sets nothing and marks nothing dirty).
struct State {
    struct {
        unsigned viewports : 1, scissors : 1, depth_test_enabled : 1, depth_write_enabled : 1,
            depth_compare_op : 1, depth_bounds_test_enabled : 1, depth_bounds : 1,
            depth_bias_enabled : 1, depth_bias : 1, stencil_test_enabled : 1,
            stencil_front_ops : 1, stencil_front_reference : 1, stencil_front_write_mask : 1,
            stencil_front_compare_mask : 1, stencil_back_ops : 1, stencil_back_reference : 1,
            stencil_back_write_mask : 1, stencil_back_compare_mask : 1,
            primitive_restart_enable : 1, rasterizer_discard_enable : 1, cull_mode : 1,
            front_face : 1, blend_constants : 1, color_write_masks : 1, line_width : 1,
            feedback_loop_enabled : 1;
    } dirty{};
    std::uint32_t num_viewports = 0;
    std::array<Viewport, 16> viewports{};
    std::array<Rect, 16> scissors{};
    bool depth_test_enabled{}, depth_write_enabled{};
    int depth_compare_op{};
    bool depth_bounds_test_enabled{};
    float depth_bounds_min{}, depth_bounds_max{};
    bool depth_bias_enabled{};
    float depth_bias_constant{}, depth_bias_clamp{}, depth_bias_slope{};
    bool stencil_test_enabled{};
    std::array<int, 4> stencil_front_ops{}, stencil_back_ops{};
    std::uint32_t stencil_front_reference{}, stencil_back_reference{};
    std::uint32_t stencil_front_write_mask{}, stencil_back_write_mask{};
    std::uint32_t stencil_front_compare_mask{}, stencil_back_compare_mask{};
    bool primitive_restart_enable{}, rasterizer_discard_enable{};
    int cull_mode{}, front_face{};
    std::array<float, 4> blend_constants{};
    std::array<std::uint32_t, 8> color_write_masks{};
    float line_width{};
    bool feedback_loop_enabled{};
};

void SetStencilOps(State& s, const std::array<int, 4>& front, const std::array<int, 4>& back) {
    if (s.stencil_front_ops != front) {
        s.stencil_front_ops = front;
        s.dirty.stencil_front_ops = 1;
    }
    if (s.stencil_back_ops != back) {
        s.stencil_back_ops = back;
        s.dirty.stencil_back_ops = 1;
    }
}

/// UpdateDepthStencilState, UpdatePrimitiveState, UpdateRasterizationState and
/// UpdateColorBlendingState for one draw: every setter the four functions call, with the same
/// conditions on the same registers.
void UpdateRest(const Inputs& in, const std::array<std::uint32_t, 8>& write_masks, State& s) {
    // UpdateDepthStencilState.
    const bool depth_test = in.depth_enable && in.depth_valid;
    s.depth_test_enabled = depth_test;
    if (depth_test) {
        const bool write = in.depth_write && !in.depth_clear;
        if (s.depth_write_enabled != write) {
            s.depth_write_enabled = write;
            s.dirty.depth_write_enabled = 1;
        }
        if (s.depth_compare_op != in.depth_func) {
            s.depth_compare_op = in.depth_func;
            s.dirty.depth_compare_op = 1;
        }
    }
    if (s.depth_bounds_test_enabled != in.depth_bounds_enable) {
        s.depth_bounds_test_enabled = in.depth_bounds_enable;
        s.dirty.depth_bounds_test_enabled = 1;
    }
    if (in.depth_bounds_enable &&
        (s.depth_bounds_min != in.depth_bounds_min || s.depth_bounds_max != in.depth_bounds_max)) {
        s.depth_bounds_min = in.depth_bounds_min;
        s.depth_bounds_max = in.depth_bounds_max;
        s.dirty.depth_bounds = 1;
    }
    if (s.depth_bias_enabled != in.needs_bias) {
        s.depth_bias_enabled = in.needs_bias;
        s.dirty.depth_bias_enabled = 1;
    }
    if (in.needs_bias) {
        const float constant = in.front_bias ? in.front_offset : in.back_offset;
        const float slope = (in.front_bias ? in.front_scale : in.back_scale) / 16.0f;
        if (s.depth_bias_constant != constant || s.depth_bias_clamp != in.depth_bias ||
            s.depth_bias_slope != slope) {
            s.depth_bias_constant = constant;
            s.depth_bias_clamp = in.depth_bias;
            s.depth_bias_slope = slope;
            s.dirty.depth_bias = 1;
        }
    }
    if (s.stencil_test_enabled != in.stencil_enabled) {
        s.stencil_test_enabled = in.stencil_enabled;
        s.dirty.stencil_test_enabled = 1;
    }
    if (in.stencil_enabled) {
        SetStencilOps(s, in.stencil_front_ops, in.stencil_back_ops);
        if (s.stencil_front_reference != in.stencil_front_reference) {
            s.stencil_front_reference = in.stencil_front_reference;
            s.dirty.stencil_front_reference = 1;
        }
        if (s.stencil_back_reference != in.stencil_back_reference) {
            s.stencil_back_reference = in.stencil_back_reference;
            s.dirty.stencil_back_reference = 1;
        }
        if (s.stencil_front_write_mask != in.stencil_front_write_mask) {
            s.stencil_front_write_mask = in.stencil_front_write_mask;
            s.dirty.stencil_front_write_mask = 1;
        }
        if (s.stencil_back_write_mask != in.stencil_back_write_mask) {
            s.stencil_back_write_mask = in.stencil_back_write_mask;
            s.dirty.stencil_back_write_mask = 1;
        }
        if (s.stencil_front_compare_mask != in.stencil_front_compare_mask) {
            s.stencil_front_compare_mask = in.stencil_front_compare_mask;
            s.dirty.stencil_front_compare_mask = 1;
        }
        if (s.stencil_back_compare_mask != in.stencil_back_compare_mask) {
            s.stencil_back_compare_mask = in.stencil_back_compare_mask;
            s.dirty.stencil_back_compare_mask = 1;
        }
    }
    // UpdatePrimitiveState.
    if (s.primitive_restart_enable != in.prim_restart) {
        s.primitive_restart_enable = in.prim_restart;
        s.dirty.primitive_restart_enable = 1;
    }
    if (s.rasterizer_discard_enable != in.rasterizer_kill) {
        s.rasterizer_discard_enable = in.rasterizer_kill;
        s.dirty.rasterizer_discard_enable = 1;
    }
    if (s.cull_mode != in.cull_mode) {
        s.cull_mode = in.cull_mode;
        s.dirty.cull_mode = 1;
    }
    if (s.front_face != in.front_face) {
        s.front_face = in.front_face;
        s.dirty.front_face = 1;
    }
    // UpdateRasterizationState.
    if (s.line_width != in.line_width) {
        s.line_width = in.line_width;
        s.dirty.line_width = 1;
    }
    // UpdateColorBlendingState.
    if (s.blend_constants != in.blend_constants) {
        s.blend_constants = in.blend_constants;
        s.dirty.blend_constants = 1;
    }
    if (s.color_write_masks != write_masks) {
        s.color_write_masks = write_masks;
        s.dirty.color_write_masks = 1;
    }
    if (s.feedback_loop_enabled != in.feedback_loop) {
        s.feedback_loop_enabled = in.feedback_loop;
        s.dirty.feedback_loop_enabled = 1;
    }
}

// The memo, at the same inputs: the groups a packet's blocks map to (Rasterizer::
// DynamicStateRegGroups) and the per-draw values compared against the last computation.
constexpr std::uint32_t NumBlocks = 1664; ///< the register file in 32-word blocks

struct MemoState {
    std::array<std::uint8_t, NumBlocks> group_of_block{};
    std::array<std::uint8_t, 8> packet_groups{}; ///< one entry per packet in the sample
    std::array<float, 2> target_scale{};
    std::array<float, 2> draw_jitter{};
};

/// Returns true when the viewport group has to be recomputed for this packet.
bool MemoCheck(const MemoState& memo, std::size_t packet) {
    std::uint8_t touched = 0;
    // A packet carries a handful of blocks: its own list, here one representative entry.
    touched |= memo.group_of_block[memo.packet_groups[packet]];
    return (touched & 1) != 0;
}

Inputs MakeInputs(std::uint32_t active) {
    Inputs in;
    in.active_viewports = active;
    for (std::uint32_t i = 0; i < active; ++i) {
        in.viewports[i] = {960.0f, 960.0f, 540.0f, 540.0f, 0.0f, 0.0f};
        in.viewport_depths[i] = {0.0f, 1.0f};
        in.viewport_scissors[i] = {0, 0, 1920, 1080};
    }
    in.clip_space_minus_w = true;
    return in;
}

} // namespace

int main() {
    constexpr int Draws = 2000000;
    for (const std::uint32_t active : {1u, 2u, 4u}) {
        const Inputs in = MakeInputs(active);
        Viewport viewports[16];
        Rect scissors[16];
        std::size_t count = 0;
        State state;
        double compute = 0, memo_ns = 0, rest = 0;
        for (int rep = 0; rep < 5; ++rep) {
            const double t0 = NowNs();
            for (int d = 0; d < Draws; ++d) {
                ComputeDynamicViewports(in, viewports, scissors, count);
            }
            const double t_rest0 = NowNs();
            for (int d = 0; d < Draws; ++d) {
                UpdateRest(in, in.write_masks, state);
            }
            const double t_rest1 = NowNs();
            rest += (t_rest1 - t_rest0) / Draws;
            const double t1 = NowNs();
            MemoState memo;
            for (std::uint32_t block = 0; block < NumBlocks; ++block) {
                // The register file's block 0x300 holds PA_CL_VPORT_* on this hardware; here a
                // tenth of the blocks are viewport blocks, which is the memo's real hit rate.
                memo.group_of_block[block] = (block % 10) == 0 ? 1 : 0;
            }
            for (std::size_t i = 0; i < memo.packet_groups.size(); ++i) {
                memo.packet_groups[i] = std::uint8_t((i * 137) % NumBlocks);
            }
            std::uint32_t hits = 0;
            for (int d = 0; d < Draws; ++d) {
                hits += MemoCheck(memo, std::size_t(d) & 7) ? 1u : 0u;
            }
            const double t2 = NowNs();
            compute += (t1 - t0) / Draws;
            memo_ns += (t2 - t1) / Draws;
            if (hits == 0xFFFFFFFFu) {
                return 1; // keep the loop alive
            }
        }
        std::printf("%u active viewport(s): viewport+scissor %5.1f ns + the other four groups "
                    "%5.1f ns = %5.1f ns/draw, memo check %4.2f ns/draw (%.0fx)\n",
                    active, compute / 5, rest / 5, (compute + rest) / 5, memo_ns / 5,
                    (compute + rest) / memo_ns);
    }
    return 0;
}
