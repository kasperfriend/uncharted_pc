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
namespace {

/// bbport: copies each run of contiguous descriptor infos into the recording chunk and points the
/// writes that refer to them at the copies (the chunk is what outlives the vectors here). Returns
/// the index of the first write past the run that starts at `i`.
template <typename T>
size_t CopyRun(Scheduler& scheduler, vk::WriteDescriptorSet* writes, size_t i, size_t count,
               const T* vk::WriteDescriptorSet::*member) {
    const T* const base = writes[i].*member;
    size_t total = writes[i].descriptorCount;
    size_t j = i + 1;
    // An entry of another member has this one null, so the comparison alone ends the run.
    while (j < count && (writes[j].*member) == base + total) {
        total += writes[j].descriptorCount;
        ++j;
    }
    // One RecordData() per run: an allocation in the chunk, its capacity checks and its
    // prefetches cost more than the few dozen bytes they carry.
    const auto copied = scheduler.RecordData(std::span<const T>{base, total});
    size_t offset = 0;
    for (size_t k = i; k < j; ++k) {
        writes[k].*member = copied.data() + offset;
        offset += writes[k].descriptorCount;
    }
    return j;
}

/// The same walk, once over the writes instead of once per info member: each entry names the
/// member its infos are in (the other two are null), so the three passes that used to read every
/// entry are one pass that reads it once.
void CopyWriteInfos(Scheduler& scheduler, vk::WriteDescriptorSet* writes, size_t count) {
    for (size_t i = 0; i < count;) {
        const auto& write = writes[i];
        if (write.pBufferInfo) {
            i = CopyRun(scheduler, writes, i, count, &vk::WriteDescriptorSet::pBufferInfo);
        } else if (write.pImageInfo) {
            i = CopyRun(scheduler, writes, i, count, &vk::WriteDescriptorSet::pImageInfo);
        } else if (write.pTexelBufferView) {
            i = CopyRun(scheduler, writes, i, count, &vk::WriteDescriptorSet::pTexelBufferView);
        } else {
            ++i;
        }
    }
}

} // namespace

Pipeline::Pipeline(const Instance& instance_, Scheduler& scheduler_, DescriptorHeap& desc_heap_,
                   const Shader::Profile& profile_, vk::PipelineCache pipeline_cache,
                   bool is_compute_ /*= false*/)
    : instance{instance_}, scheduler{scheduler_}, desc_heap{desc_heap_}, profile{profile_},
      is_compute{is_compute_} {}

Pipeline::~Pipeline() = default;


void Pipeline::BindResources(DescriptorWrites set_writes,
                             const Shader::PushData& push_data) const {
    const auto bind_point =
        IsCompute() ? vk::PipelineBindPoint::eCompute : vk::PipelineBindPoint::eGraphics;
    const auto stage_flags = IsCompute() ? vk::ShaderStageFlagBits::eCompute : AllGraphicsStageBits;
    const vk::PipelineLayout layout = *pipeline_layout;
    // bbport: the push constants and the descriptor set are the same for every draw of a batch
    // (same layout, same block): the binding echo drops the command that would restate them.
    if (scheduler.EchoPushConstants(layout, stage_flags, &push_data, sizeof(push_data))) {
        scheduler.Record([layout, stage_flags, push_data](vk::CommandBuffer cmdbuf) {
            cmdbuf.pushConstants(layout, stage_flags, 0u, sizeof(push_data), &push_data);
        });
    }

    // Bind descriptor set.
    if (set_writes.Empty()) {
        return;
    }

    if (uses_push_descriptors) {
        if (!scheduler.IsRecordingDeferred()) {
            scheduler.Record([bind_point, layout, set_writes](vk::CommandBuffer cmdbuf) {
                cmdbuf.pushDescriptorSetKHR(
                    bind_point, layout, 0,
                    std::span<const vk::WriteDescriptorSet>{set_writes.data, set_writes.count});
            });
            return;
        }
        // Writes and the infos they point to are laid out in the recording chunk, with the
        // pointers already aimed at those copies: the command only carries a span.
        size_t bytes = set_writes.count * sizeof(vk::WriteDescriptorSet) + 64;
        for (u32 i = 0; i < set_writes.count; ++i) {
            bytes += set_writes[i].descriptorCount *
                         (sizeof(vk::DescriptorBufferInfo) + sizeof(vk::DescriptorImageInfo) +
                          sizeof(vk::BufferView)) +
                     32;
        }
        scheduler.ReserveRecordData(bytes);
        const auto writes = scheduler.RecordData(
            std::span<const vk::WriteDescriptorSet>{set_writes.data, set_writes.count});
        auto* patched = const_cast<vk::WriteDescriptorSet*>(writes.data());
        // A stage appends its buffers, then its images, then its samplers, so the infos of
        // consecutive writes are usually consecutive entries of one vector: copy a run of them
        // with one RecordData() (an allocation in the chunk, its capacity checks and its
        // prefetches cost more than the few dozen bytes they carry) instead of one per write.
        CopyWriteInfos(scheduler, patched, writes.size());
        scheduler.Record([bind_point, layout, writes](vk::CommandBuffer cmdbuf) {
            cmdbuf.pushDescriptorSetKHR(bind_point, layout, 0, writes);
        });
        return;
    }

    // Descriptor set updates are device calls: they stay on this thread.
    const auto desc_set = desc_heap.Commit(*desc_layout);
    for (u32 i = 0; i < set_writes.count; ++i) {
        set_writes[i].dstSet = desc_set;
    }
    instance.GetDevice().updateDescriptorSets(
        std::span<const vk::WriteDescriptorSet>{set_writes.data, set_writes.count}, {});
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
