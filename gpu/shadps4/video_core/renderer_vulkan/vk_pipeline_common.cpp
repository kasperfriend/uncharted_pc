// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <span>

#include <boost/container/small_vector.hpp>
#include <boost/container/static_vector.hpp>

#include "shader_recompiler/resource.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_pipeline_common.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

namespace Vulkan {

Pipeline::Pipeline(const Instance& instance_, Scheduler& scheduler_, DescriptorHeap& desc_heap_,
                   const Shader::Profile& profile_, vk::PipelineCache pipeline_cache,
                   bool is_compute_ /*= false*/)
    : instance{instance_}, scheduler{scheduler_}, desc_heap{desc_heap_}, profile{profile_},
      is_compute{is_compute_} {}

Pipeline::~Pipeline() = default;


void Pipeline::BindResources(DescriptorWrites& set_writes,
                             const Shader::PushData& push_data) const {
    const auto bind_point =
        IsCompute() ? vk::PipelineBindPoint::eCompute : vk::PipelineBindPoint::eGraphics;
    const auto stage_flags = IsCompute() ? vk::ShaderStageFlagBits::eCompute : AllGraphicsStageBits;
    const vk::PipelineLayout layout = *pipeline_layout;
    scheduler.Record([layout, stage_flags, push_data](vk::CommandBuffer cmdbuf) {
        cmdbuf.pushConstants(layout, stage_flags, 0u, sizeof(push_data), &push_data);
    });

    // Bind descriptor set.
    if (set_writes.empty()) {
        return;
    }

    if (uses_push_descriptors) {
        if (!scheduler.IsRecordingDeferred()) {
            scheduler.Record([&](vk::CommandBuffer cmdbuf) {
                cmdbuf.pushDescriptorSetKHR(bind_point, layout, 0, set_writes);
            });
            return;
        }
        // Writes and the infos they point to are laid out in the recording chunk, with the
        // pointers already aimed at those copies: the command only carries a span.
        size_t bytes = set_writes.size() * sizeof(vk::WriteDescriptorSet) + 64;
        for (const auto& write : set_writes) {
            bytes += write.descriptorCount *
                         (sizeof(vk::DescriptorBufferInfo) + sizeof(vk::DescriptorImageInfo) +
                          sizeof(vk::BufferView)) +
                     32;
        }
        scheduler.ReserveRecordData(bytes);
        const auto writes = scheduler.RecordData(std::span<const vk::WriteDescriptorSet>{set_writes});
        auto* patched = const_cast<vk::WriteDescriptorSet*>(writes.data());
        // A stage appends its buffers, then its images, then its samplers, so the infos of
        // consecutive writes are usually consecutive entries of one vector: copy a run of them
        // with one RecordData() (an allocation in the chunk, its capacity checks and its
        // prefetches cost more than the few dozen bytes they carry) instead of one per write.
        const size_t count = writes.size();
        const auto coalesce = [&scheduler, patched, count](auto get, auto set) {
            for (size_t i = 0; i < count;) {
                const auto* base = get(patched[i]);
                if (!base) {
                    ++i;
                    continue;
                }
                size_t total = patched[i].descriptorCount;
                size_t j = i + 1;
                while (j < count && get(patched[j]) == base + total) {
                    total += patched[j].descriptorCount;
                    ++j;
                }
                const auto copied = scheduler.RecordData(std::span{base, total});
                size_t offset = 0;
                for (size_t k = i; k < j; ++k) {
                    set(patched[k], copied.data() + offset);
                    offset += patched[k].descriptorCount;
                }
                i = j;
            }
        };
        coalesce([](const vk::WriteDescriptorSet& write) { return write.pBufferInfo; },
                 [](vk::WriteDescriptorSet& write, const vk::DescriptorBufferInfo* infos) {
                     write.pBufferInfo = infos;
                 });
        coalesce([](const vk::WriteDescriptorSet& write) { return write.pImageInfo; },
                 [](vk::WriteDescriptorSet& write, const vk::DescriptorImageInfo* infos) {
                     write.pImageInfo = infos;
                 });
        coalesce([](const vk::WriteDescriptorSet& write) { return write.pTexelBufferView; },
                 [](vk::WriteDescriptorSet& write, const vk::BufferView* views) {
                     write.pTexelBufferView = views;
                 });
        scheduler.Record([bind_point, layout, writes](vk::CommandBuffer cmdbuf) {
            cmdbuf.pushDescriptorSetKHR(bind_point, layout, 0, writes);
        });
        return;
    }

    // Descriptor set updates are device calls: they stay on this thread.
    const auto desc_set = desc_heap.Commit(*desc_layout);
    for (auto& set_write : set_writes) {
        set_write.dstSet = desc_set;
    }
    instance.GetDevice().updateDescriptorSets(set_writes, {});
    scheduler.Record([bind_point, layout, desc_set](vk::CommandBuffer cmdbuf) {
        cmdbuf.bindDescriptorSets(bind_point, layout, 0, desc_set, {});
    });
}

std::string Pipeline::GetDebugString() const {
    std::string stage_desc;
    for (const auto& stage : stages) {
        if (stage) {
            const auto shader_name = PipelineCache::GetShaderName(stage->hw_stage, stage->pgm_hash);
            if (stage_desc.empty()) {
                stage_desc = shader_name;
            } else {
                stage_desc = fmt::format("{},{}", stage_desc, shader_name);
            }
        }
    }
    return stage_desc;
}

} // namespace Vulkan
