// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include "video_core/texture_cache/image.h"
#include "video_core/texture_cache/image_info.h"
#include "video_core/texture_cache/image_view.h"
#include "video_core/texture_cache/tile_manager.h"

#include "video_core/host_shaders/tiling_macro_128_comp.h"
#include "video_core/host_shaders/tiling_macro_16_comp.h"
#include "video_core/host_shaders/tiling_macro_32_comp.h"
#include "video_core/host_shaders/tiling_macro_64_comp.h"
#include "video_core/host_shaders/tiling_macro_8_comp.h"
#include "video_core/host_shaders/tiling_macro_96_comp.h"
#include "video_core/host_shaders/tiling_micro_128_comp.h"
#include "video_core/host_shaders/tiling_micro_16_comp.h"
#include "video_core/host_shaders/tiling_micro_32_comp.h"
#include "video_core/host_shaders/tiling_micro_64_comp.h"
#include "video_core/host_shaders/tiling_micro_8_comp.h"
#include "video_core/host_shaders/tiling_micro_96_comp.h"

#include <atomic>
#include <cstdio>
#include <limits>
#include <magic_enum/magic_enum.hpp>
#include <vk_mem_alloc.h>

namespace Vulkan {
/// bbport: printed by the device-lost report (vk_semaphore.cpp).
extern void (*g_gpu_crash_detail)();
} // namespace Vulkan

namespace VideoCore {

struct TilingInfo {
    u32 bank_swizzle;
    u32 num_slices;
    u32 num_mips;
    u32 num_texels;  ///< the texels the filled mips hold (the shader's hard bound)
    u32 tiled_bytes; ///< the bytes the tiled side has (the shader's bound on that side)
    std::array<ImageInfo::MipInfo, 16> mips;
};

namespace {
/// bbport: the bytes the filled mips of a tiling dispatch cover. The dispatch is sized from this
/// and not from the image's guest size: the shader walks the mip list by subtracting mip sizes,
/// so a texel past the last filled mip used to step onto the entry after it - which the host does
/// not fill - and address the surface with that entry's pitch, height and offset (a write to a
/// garbage offset, i.e. a device lost). Clamping the shader's walk fixes the addressing; sizing
/// the dispatch from the covered mips makes the mismatch impossible in the first place.
u64 CoveredBytes(const TilingInfo& params) {
    u64 covered = 0;
    for (u32 mip = 0; mip < params.num_mips && mip < params.mips.size(); ++mip) {
        covered += params.mips[mip].size;
    }
    return covered;
}

/// bbport: the parameters of a tiling dispatch that does not fit its buffer, printed a few times
/// (the crash report of a device lost then names the dispatch that could write outside it).
void ReportTilingMismatch(const char* what, const ImageInfo& info, const TilingInfo& params,
                          u64 covered) {
    static std::atomic<u32> printed{0};
    if (printed.fetch_add(1, std::memory_order_relaxed) >= 8) {
        return;
    }
    std::printf("Tiling: %s of a %ux%u image (%u levels, %u filled, %u bits): the filled mips "
                "cover 0x%llx bytes of the 0x%llx-byte guest size; dispatching the covered bytes\n",
                what, info.size.width, info.size.height, info.resources.levels, params.num_mips,
                info.num_bits, static_cast<unsigned long long>(covered),
                static_cast<unsigned long long>(info.guest_size));
}
} // namespace

TileManager::TileManager(const Vulkan::Instance& instance_, Vulkan::Scheduler& scheduler_,
                         Vulkan::Runtime& runtime_, StreamBuffer& stream_buffer_)
    : instance{instance_}, scheduler{scheduler_}, runtime{runtime_}, stream_buffer{stream_buffer_} {
    // bbport: a device lost prints the tiling dispatches recorded so far (see DispatchRecord).
    static TileManager* crash_report;
    crash_report = this;
    Vulkan::g_gpu_crash_detail = [] { crash_report->DumpDispatches(); };
    const auto device = instance.GetDevice();
    const std::array<vk::DescriptorSetLayoutBinding, 3> bindings = {{
        {
            .binding = 0,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
        {
            .binding = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
        {
            .binding = 2,
            .descriptorType = vk::DescriptorType::eUniformBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
    }};

    const vk::DescriptorSetLayoutCreateInfo desc_layout_ci = {
        .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
        .bindingCount = static_cast<u32>(bindings.size()),
        .pBindings = bindings.data(),
    };
    auto desc_layout_result = device.createDescriptorSetLayoutUnique(desc_layout_ci);
    ASSERT_MSG(desc_layout_result.result == vk::Result::eSuccess,
               "Failed to create descriptor set layout: {}",
               vk::to_string(desc_layout_result.result));
    desc_layout = std::move(desc_layout_result.value);

    const vk::DescriptorSetLayout set_layout = *desc_layout;
    const vk::PipelineLayoutCreateInfo layout_info = {
        .setLayoutCount = 1U,
        .pSetLayouts = &set_layout,
        .pushConstantRangeCount = 0U,
        .pPushConstantRanges = nullptr,
    };
    auto [layout_result, layout] = device.createPipelineLayoutUnique(layout_info);
    ASSERT_MSG(layout_result == vk::Result::eSuccess, "Failed to create pipeline layout: {}",
               vk::to_string(layout_result));
    pl_layout = std::move(layout);
}

TileManager::~TileManager() = default;

vk::Pipeline TileManager::GetTilingPipeline(const ImageInfo& info, bool is_tiler) {
    const TilingKey key{
        .tile_mode = info.tile_mode,
        .num_bits = info.num_bits,
        .num_samples = info.num_samples,
        .is_tiler = is_tiler,
    };
    if (const auto it = tiling_pipelines.find(key); it != tiling_pipelines.end()) {
        return *it->second;
    }

    const auto device = instance.GetDevice();
    const auto micro_tile_mode = AmdGpu::GetMicroTileMode(info.tile_mode);
    const bool is_macro = AmdGpu::IsMacroTiled(info.array_mode);
    std::array<u32, 12> spec_data{
        info.num_samples,
        u32(micro_tile_mode),
        AmdGpu::GetMicroTileThickness(info.array_mode),
        u32(is_tiler),
    };
    if (is_macro) {
        const auto macro_tile_mode =
            AmdGpu::CalculateMacrotileMode(info.tile_mode, info.num_bits, info.num_samples);
        spec_data[4] = u32(info.array_mode);
        spec_data[5] = u32(AmdGpu::GetPipeConfig(info.tile_mode));
        spec_data[6] = AmdGpu::GetBankWidth(macro_tile_mode);
        spec_data[7] = AmdGpu::GetBankHeight(macro_tile_mode);
        spec_data[8] = AmdGpu::GetNumBanks(macro_tile_mode);
        spec_data[9] = std::bit_width(spec_data[8]) - 1;
        spec_data[10] = AmdGpu::CalculateTileSplit(info.tile_mode, info.array_mode, micro_tile_mode,
                                                   info.num_bits);
        spec_data[11] = AmdGpu::GetMacrotileAspect(macro_tile_mode);
    }

    std::span<const u32> code;
    switch (info.num_bits) {
    case 8:
        code = is_macro ? std::span<const u32>{TILING_MACRO_8_COMP}
                        : std::span<const u32>{TILING_MICRO_8_COMP};
        break;
    case 16:
        code = is_macro ? std::span<const u32>{TILING_MACRO_16_COMP}
                        : std::span<const u32>{TILING_MICRO_16_COMP};
        break;
    case 32:
        code = is_macro ? std::span<const u32>{TILING_MACRO_32_COMP}
                        : std::span<const u32>{TILING_MICRO_32_COMP};
        break;
    case 64:
        code = is_macro ? std::span<const u32>{TILING_MACRO_64_COMP}
                        : std::span<const u32>{TILING_MICRO_64_COMP};
        break;
    case 96:
        code = is_macro ? std::span<const u32>{TILING_MACRO_96_COMP}
                        : std::span<const u32>{TILING_MICRO_96_COMP};
        break;
    case 128:
        code = is_macro ? std::span<const u32>{TILING_MACRO_128_COMP}
                        : std::span<const u32>{TILING_MICRO_128_COMP};
        break;
    default:
        UNREACHABLE_MSG("Unsupported tiling pixel width {}", info.num_bits);
    }

    const auto module = Vulkan::CompileSPV(code, device);
    static constexpr auto spec_entries = [] {
        std::array<vk::SpecializationMapEntry, 12> entries{};
        for (u32 i = 0; i < entries.size(); ++i) {
            entries[i] = vk::SpecializationMapEntry{i, i * u32(sizeof(u32)), sizeof(u32)};
        }
        return entries;
    }();

    const u32 spec_count = is_macro ? 12 : 4;
    const vk::SpecializationInfo specialization{
        .mapEntryCount = spec_count,
        .pMapEntries = spec_entries.data(),
        .dataSize = spec_count * sizeof(u32),
        .pData = spec_data.data(),
    };

    const auto module_name = fmt::format("{}_{} {}", magic_enum::enum_name(info.tile_mode),
                                         info.num_bits, is_tiler ? "tiler" : "detiler");
    LOG_INFO(Render_Vulkan, "Creating tiling pipeline {}", module_name);
    Vulkan::SetObjectName(device, module, module_name);

    const vk::PipelineShaderStageCreateInfo shader_ci = {
        .stage = vk::ShaderStageFlagBits::eCompute,
        .module = module,
        .pName = "main",
        .pSpecializationInfo = &specialization,
    };
    const vk::ComputePipelineCreateInfo compute_pipeline_ci = {
        .stage = shader_ci,
        .layout = *pl_layout,
    };

    auto [result, pipeline] =
        device.createComputePipelineUnique(VK_NULL_HANDLE, compute_pipeline_ci);
    ASSERT_MSG(result == vk::Result::eSuccess, "Detiler pipeline creation failed {}",
               vk::to_string(result));
    const auto handle = *pipeline;
    tiling_pipelines.emplace(key, std::move(pipeline));
    device.destroyShaderModule(module);
    return handle;
}

void TileManager::NoteDispatch(const DispatchRecord& record) {
    dispatches[dispatch_seq.fetch_add(1, std::memory_order_relaxed) % DispatchRecords] = record;
}

void TileManager::DumpDispatches() const {
    const u64 total = dispatch_seq.load(std::memory_order_relaxed);
    if (total == 0) {
        return;
    }
    std::printf("GPU crash: last tiling dispatches (%llu recorded):\n",
                static_cast<unsigned long long>(total));
    const u64 oldest = total > DispatchRecords ? total - DispatchRecords : 0;
    u32 printed = 0;
    for (u64 n = total; n > oldest && printed < 8; --n) {
        const auto& r = dispatches[(n - 1) % DispatchRecords];
        std::printf("GPU crash:   %s #%llu %u mips %u bits %u texels (%u groups): tiled "
                    "0x%llx+0x%llx, linear 0x%llx+0x%llx, covers 0x%llx of guest 0x%llx%s\n",
                    r.tiler ? "tile" : "detile", static_cast<unsigned long long>(n - 1),
                    r.num_mips, r.num_bits, r.texels, r.dim_x,
                    static_cast<unsigned long long>(r.tiled_address),
                    static_cast<unsigned long long>(r.tiled_size),
                    static_cast<unsigned long long>(r.linear_address),
                    static_cast<unsigned long long>(r.linear_size),
                    static_cast<unsigned long long>(r.covered),
                    static_cast<unsigned long long>(r.guest_size),
                    r.covered > r.tiled_size ? " OUT OF THE TILED BUFFER" : "");
        ++printed;
    }
}

void TileManager::RecordTilingDispatch(vk::Pipeline pipeline, const vk::DescriptorBufferInfo& tiled,
                                       const vk::DescriptorBufferInfo& linear,
                                       const vk::DescriptorBufferInfo& params, u32 dim_x) {
    // The tiling shader binds its own pipeline: the echo must forget what it knew.
    scheduler.ClearEcho();
    scheduler.Record([pipeline, layout = *pl_layout, tiled, linear, params,
                      dim_x](vk::CommandBuffer cmdbuf) {
        cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
        const std::array<vk::WriteDescriptorSet, 3> set_writes = {{
            {
                .dstBinding = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eStorageBuffer,
                .pBufferInfo = &tiled,
            },
            {
                .dstBinding = 1,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eStorageBuffer,
                .pBufferInfo = &linear,
            },
            {
                .dstBinding = 2,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eUniformBuffer,
                .pBufferInfo = &params,
            },
        }};
        cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, layout, 0, set_writes);
        cmdbuf.dispatch(dim_x, 1, 1);
    });
}

std::pair<const Buffer*, u64> TileManager::DetileImage(const VideoCore::Buffer* in_buffer,
                                                       u64 in_offset, const ImageInfo& info) {
    if (!info.props.is_tiled) {
        return {in_buffer, in_offset};
    }

    TilingInfo params{};
    params.bank_swizzle = info.bank_swizzle;
    params.num_slices = info.props.is_volume ? info.size.depth : info.resources.layers;
    // bbport: the mip array of the shader's parameter block is 16 entries; an image with more
    // levels wrote past it (the stack of this function) and the shader read whatever followed.
    params.num_mips = std::min<u32>(info.resources.levels, static_cast<u32>(params.mips.size()));
    for (u32 mip = 0; mip < params.num_mips; ++mip) {
        auto& mip_info = params.mips[mip];
        mip_info = info.mips_layout[mip];
        if (info.props.is_block) {
            mip_info.pitch = std::max((mip_info.pitch + 3) / 4, 1U);
            mip_info.height = std::max((mip_info.height + 3) / 4, 1U);
        }
    }

    // bbport: the texels the filled mips hold, not the image's guest size (see CoveredBytes).
    const u64 covered = CoveredBytes(params);
    if (covered != info.guest_size) {
        ReportTilingMismatch("detile", info, params, covered);
    }
    if (covered == 0 || info.num_bits == 0) {
        return {in_buffer, in_offset};
    }
    // bbport: every texel of the filled mips, and no texel past the last one (the shader stops
    // at num_texels), so the dispatch cannot address outside the buffers it was given.
    const u32 texels = static_cast<u32>(
        std::min<u64>(covered / (info.num_bits / 8), std::numeric_limits<u32>::max()));
    const u32 dim_x = (texels + 63) / 64;
    params.num_texels = texels;

    // bbport: the tiled side is bound for the image's guest size, but the buffer behind it can
    // hold less than that after the offset (an image at the end of an arena chunk, a staging
    // block): the descriptor range is cut to what is there and the shader drops the texels the
    // surface maths puts past it (tiling.comp). A read or a write past the buffer is a lost
    // device, and neither descriptor used to be checked.
    if (in_offset >= in_buffer->SizeBytes()) {
        return {in_buffer, in_offset};
    }
    const u64 tiled_bytes = std::min<u64>(info.guest_size, in_buffer->SizeBytes() - in_offset);
    params.tiled_bytes = static_cast<u32>(std::min<u64>(tiled_bytes, ~u32{0}));

    const vk::DescriptorBufferInfo params_buffer_info{
        .buffer = stream_buffer.Handle(),
        .offset = stream_buffer.Copy(&params, sizeof(params), instance.UniformMinAlignment()),
        .range = sizeof(params),
    };

    const auto staging = runtime.GetStagingPool().Request(info.guest_size, MemoryType::DeviceLocal);
    if (staging.size < info.guest_size) {
        return {in_buffer, in_offset};
    }

    scheduler.EndRendering();
    runtime.FlushBarriers();

    const vk::DescriptorBufferInfo tiled_buffer_info{
        .buffer = in_buffer->Handle(),
        .offset = in_offset,
        .range = tiled_bytes,
    };
    const vk::DescriptorBufferInfo linear_buffer_info{
        .buffer = staging.buffer->Handle(),
        .offset = staging.offset,
        .range = info.guest_size,
    };
    NoteDispatch({.tiled_address = in_buffer->BufferDeviceAddress(),
                  .tiled_size = in_buffer->SizeBytes(),
                  .linear_address = staging.buffer->BufferDeviceAddress(),
                  .linear_size = staging.buffer->SizeBytes(),
                  .covered = covered,
                  .guest_size = info.guest_size,
                  .dim_x = dim_x,
                  .texels = texels,
                  .num_mips = params.num_mips,
                  .num_bits = info.num_bits,
                  .tiler = false});
    // bbport: recorded with copies of the descriptor infos (threaded recording).
    RecordTilingDispatch(GetTilingPipeline(info, false), tiled_buffer_info, linear_buffer_info,
                         params_buffer_info, dim_x);

    runtime.AccessBuffer(staging.buffer, staging.offset, info.guest_size,
                         vk::PipelineStageFlagBits2::eComputeShader,
                         vk::AccessFlagBits2::eShaderWrite);

    return {staging.buffer, staging.offset};
}

void TileManager::TileImage(Image& in_image, std::span<vk::BufferImageCopy> buffer_copies,
                            const VideoCore::Buffer* out_buffer, u64 out_offset) {
    const auto& info = in_image.info;
    if (!info.props.is_tiled) {
        for (auto& copy : buffer_copies) {
            copy.bufferOffset += out_offset;
        }
        runtime.DownloadImage(&in_image, out_buffer, buffer_copies);
        return;
    }

    TilingInfo params{};
    params.bank_swizzle = info.bank_swizzle;
    params.num_slices = info.props.is_volume ? info.size.depth : info.resources.layers;
    // bbport: 16 mips is what the shader's parameter block holds (see DetileImage).
    params.num_mips = std::min<u32>(static_cast<u32>(buffer_copies.size()),
                                    static_cast<u32>(params.mips.size()));
    for (u32 mip = 0; mip < params.num_mips; ++mip) {
        auto& mip_info = params.mips[mip];
        mip_info = info.mips_layout[mip];
        if (info.props.is_block) {
            mip_info.pitch = std::max((mip_info.pitch + 3) / 4, 1U);
            mip_info.height = std::max((mip_info.height + 3) / 4, 1U);
        }
    }

    // bbport: the texels the filled mips hold, not the image's guest size (see CoveredBytes).
    const u64 covered = CoveredBytes(params);
    if (covered != info.guest_size) {
        ReportTilingMismatch("tile", info, params, covered);
    }
    if (covered == 0 || info.num_bits == 0) {
        return;
    }
    // bbport: every texel of the filled mips, and no texel past the last one (see DetileImage).
    const u32 texels = static_cast<u32>(
        std::min<u64>(covered / (info.num_bits / 8), std::numeric_limits<u32>::max()));
    const u32 dim_x = (texels + 63) / 64;
    params.num_texels = texels;

    // bbport: see DetileImage: the tiler writes the tiled side, so that range is cut to the
    // buffer and the shader drops the texels it would put past it.
    if (out_offset >= out_buffer->SizeBytes()) {
        return;
    }
    const u64 tiled_bytes = std::min<u64>(info.guest_size, out_buffer->SizeBytes() - out_offset);
    params.tiled_bytes = static_cast<u32>(std::min<u64>(tiled_bytes, ~u32{0}));

    const auto staging = runtime.GetStagingPool().Request(info.guest_size, MemoryType::DeviceLocal);
    if (staging.size < info.guest_size) {
        return;
    }

    const vk::DescriptorBufferInfo params_buffer_info{
        .buffer = stream_buffer.Handle(),
        .offset = stream_buffer.Copy(&params, sizeof(params), instance.UniformMinAlignment()),
        .range = sizeof(params),
    };
    for (auto& copy : buffer_copies) {
        copy.bufferOffset += staging.offset;
    }

    runtime.DownloadImage(&in_image, staging.buffer, buffer_copies);
    runtime.FlushBarriers();

    const vk::DescriptorBufferInfo tiled_buffer_info{
        .buffer = out_buffer->Handle(),
        .offset = out_offset,
        .range = tiled_bytes,
    };
    const vk::DescriptorBufferInfo linear_buffer_info{
        .buffer = staging.buffer->Handle(),
        .offset = staging.offset,
        .range = info.guest_size,
    };
    NoteDispatch({.tiled_address = out_buffer->BufferDeviceAddress(),
                  .tiled_size = out_buffer->SizeBytes(),
                  .linear_address = staging.buffer->BufferDeviceAddress(),
                  .linear_size = staging.buffer->SizeBytes(),
                  .covered = covered,
                  .guest_size = info.guest_size,
                  .dim_x = dim_x,
                  .texels = texels,
                  .num_mips = params.num_mips,
                  .num_bits = info.num_bits,
                  .tiler = true});
    // bbport: recorded with copies of the descriptor infos (threaded recording).
    RecordTilingDispatch(GetTilingPipeline(info, true), tiled_buffer_info, linear_buffer_info,
                         params_buffer_info, dim_x);

    runtime.AccessBuffer(out_buffer, out_offset, tiled_bytes,
                         vk::PipelineStageFlagBits2::eComputeShader,
                         vk::AccessFlagBits2::eShaderWrite);
}

} // namespace VideoCore
