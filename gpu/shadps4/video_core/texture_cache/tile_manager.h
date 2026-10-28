// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <unordered_map>
#include <utility>
#include <boost/container_hash/hash.hpp>

#include "common/types.h"
#include "video_core/amdgpu/tiling.h"
#include "video_core/buffer_cache/buffer.h"

namespace Vulkan {
struct Runtime;
}

namespace VideoCore {

struct ImageInfo;
struct Image;

class TileManager {
    struct TilingKey {
        AmdGpu::TileMode tile_mode;
        u32 num_bits;
        u32 num_samples;
        bool is_tiler;

        bool operator==(const TilingKey&) const = default;

        struct Hash {
            size_t operator()(const TilingKey& key) const {
                size_t hash = 0;
                boost::hash_combine(hash, key.tile_mode);
                boost::hash_combine(hash, key.num_bits);
                boost::hash_combine(hash, key.num_samples);
                boost::hash_combine(hash, key.is_tiler);
                return hash;
            }
        };
    };

public:
    explicit TileManager(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler,
                         Vulkan::Runtime& runtime, StreamBuffer& stream_buffer);
    ~TileManager();

    void TileImage(Image& in_image, std::span<vk::BufferImageCopy> buffer_copies,
                   const VideoCore::Buffer* out_buffer, u64 out_offset);

    std::pair<const Buffer*, u64> DetileImage(const VideoCore::Buffer* in_buffer, u64 in_offset,
                                              const ImageInfo& info);

    /// bbport: what the last tiling dispatches were given (device lost reports). A write outside
    /// a buffer is otherwise only visible as "the nearest tracked buffer was N bytes below the
    /// faulting address"; with this, the dispatch whose output range does not hold its own texels
    /// is named.
    struct DispatchRecord {
        u64 tiled_address{}, tiled_size{}, linear_address{}, linear_size{};
        u64 covered{}, guest_size{};
        u32 dim_x{}, texels{}, num_mips{}, num_bits{};
        bool tiler{};
    };
    static constexpr u32 DispatchRecords = 32;

private:
    void RecordTilingDispatch(vk::Pipeline pipeline, const vk::DescriptorBufferInfo& tiled,
                              const vk::DescriptorBufferInfo& linear,
                              const vk::DescriptorBufferInfo& params, u32 dim_x);
    void NoteDispatch(const DispatchRecord& record);
    void DumpDispatches() const;

    vk::Pipeline GetTilingPipeline(const ImageInfo& info, bool is_tiler);

private:
    const Vulkan::Instance& instance;
    Vulkan::Scheduler& scheduler;
    Vulkan::Runtime& runtime;
    StreamBuffer& stream_buffer;
    vk::UniqueDescriptorSetLayout desc_layout;
    vk::UniquePipelineLayout pl_layout;
    std::unordered_map<TilingKey, vk::UniquePipeline, TilingKey::Hash> tiling_pipelines;
    std::array<DispatchRecord, DispatchRecords> dispatches{};
    std::atomic<u64> dispatch_seq{0};
};

} // namespace VideoCore
