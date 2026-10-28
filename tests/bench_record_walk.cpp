// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: what the recording thread pays per command and per draw after the round-9 diet.
//
// Two walks over a recording chunk:
//
//  1. command execution. Every command in a chunk used to have its virtual destructor called
//     after Execute(), although almost every recorded closure holds only handles, spans and
//     numbers (its destructor is empty). A chunk now knows whether any of its commands owns
//     anything and only then calls the destructors.
//  2. descriptor writes. Copying the infos a push-descriptor command points at walked the
//     writes once per info member (buffer, image, texel view) although each write uses exactly
//     one of them. One walk finds the same runs and reads each write once.
//
// The models below are the shapes of those loops with the same memory traffic, so the numbers
// can be applied to a frame: DrawsPerFrame draws of CommandsPerDraw recorded commands and
// WritesPerDraw descriptor writes.

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <span>
#include <utility>

namespace {

constexpr int DrawsPerFrame = 2000;
constexpr int CommandsPerDraw = 3;
constexpr int WritesPerDraw = 24;
constexpr int Rounds = 400;

using Clock = std::chrono::steady_clock;

volatile std::uint64_t Sink = 0;

// --- 1. the command walk ----------------------------------------------------------------

/// Stands in for the driver call a recorded closure ends in: opaque, so the compiler cannot
/// fold the two walks into each other and the numbers stay the ones a chunk of mixed closures
/// costs (one real body per command type, called through the vtable).
int OpaqueCalls = 0;
__attribute__((noinline)) void OpaqueCall(int& out) {
    ++OpaqueCalls;
    out += OpaqueCalls & 1;
}

struct CommandBase {
    virtual ~CommandBase() = default;
    virtual void Execute(int& out) = 0;
    CommandBase* next{};
};

struct HandleCommand final : CommandBase {
    explicit HandleCommand(std::uint64_t handle_) : handle{handle_} {}
    void Execute(int& out) override {
        OpaqueCall(out);
        out += static_cast<int>(handle & 1);
    }
    std::uint64_t handle;
};

/// A second closure type in the same chunk: with two dynamic types the destructor call and
/// Execute() both stay virtual, as they are for a chunk of recorded closures.
struct SpanCommand final : CommandBase {
    explicit SpanCommand(std::uint32_t count_) : count{count_} {}
    void Execute(int& out) override {
        OpaqueCall(out);
        out += static_cast<int>(count & 1);
    }
    std::uint32_t count;
    std::uint32_t pad{};
};

/// The old walk: every command's destructor is called through the base pointer.
std::uint64_t ExecuteWithDestructors(CommandBase* first, int& out) {
    std::uint64_t ran = 0;
    for (CommandBase* command = first; command;) {
        CommandBase* const next = command->next;
        command->Execute(out);
        command->~CommandBase();
        command = next;
        ++ran;
    }
    return ran;
}

/// The new walk: chunks of closures that own nothing skip the destructor calls.
std::uint64_t ExecuteExecuteOnly(CommandBase* first, int& out) {
    std::uint64_t ran = 0;
    for (CommandBase* command = first; command;) {
        CommandBase* const next = command->next;
        command->Execute(out);
        command = next;
        ++ran;
    }
    return ran;
}

// --- 2. the descriptor-write walk -------------------------------------------------------

struct BufferInfo {
    std::uint64_t buffer{};
    std::uint64_t offset{};
    std::uint64_t range{};
};

struct ImageInfo {
    std::uint64_t sampler{};
    std::uint64_t view{};
    std::uint64_t layout{};
};

struct ViewInfo {
    std::uint64_t view{};
};

struct WriteDesc {
    std::uint64_t dst_set{};
    std::uint32_t dst_binding{};
    std::uint32_t dst_array_element{};
    std::uint32_t count{};
    std::uint32_t type{};
    const BufferInfo* buffer_info{};
    const ImageInfo* image_info{};
    const ViewInfo* texel_view{};
    const void* next{};
};
static_assert(sizeof(WriteDesc) == 56);

/// Stands in for the recording chunk: RecordData() hands back chunk memory that it copied into.
struct Chunk {
    std::array<std::byte, 256 * 1024> storage{};
    std::size_t used{};

    template <typename T>
    std::span<const T> RecordData(std::span<const T> data) {
        const std::size_t align = alignof(T);
        used = (used + align - 1) & ~(align - 1);
        auto* dst = reinterpret_cast<T*>(storage.data() + used);
        std::memcpy(dst, data.data(), data.size_bytes());
        used += data.size_bytes();
        return {dst, data.size()};
    }
    void Reset() {
        used = 0;
    }
};

template <typename T>
void CopyRunLegacy(Chunk& chunk, WriteDesc* writes, std::size_t count,
                   const T* WriteDesc::*member) {
    for (std::size_t i = 0; i < count;) {
        const T* const base = writes[i].*member;
        if (!base) {
            ++i;
            continue;
        }
        std::size_t total = writes[i].count;
        std::size_t j = i + 1;
        while (j < count && (writes[j].*member) == base + total) {
            total += writes[j].count;
            ++j;
        }
        const auto copied = chunk.RecordData(std::span<const T>{base, total});
        std::size_t offset = 0;
        for (std::size_t k = i; k < j; ++k) {
            writes[k].*member = copied.data() + offset;
            offset += writes[k].count;
        }
        i = j;
    }
}

/// Three passes, one per info member: the shape the recording used before this round.
void CopyWriteInfosLegacy(Chunk& chunk, WriteDesc* writes, std::size_t count) {
    CopyRunLegacy(chunk, writes, count, &WriteDesc::buffer_info);
    CopyRunLegacy(chunk, writes, count, &WriteDesc::image_info);
    CopyRunLegacy(chunk, writes, count, &WriteDesc::texel_view);
}

template <typename T>
std::size_t CopyRun(Chunk& chunk, WriteDesc* writes, std::size_t i, std::size_t count,
                    const T* WriteDesc::*member) {
    const T* const base = writes[i].*member;
    std::size_t total = writes[i].count;
    std::size_t j = i + 1;
    while (j < count && (writes[j].*member) == base + total) {
        total += writes[j].count;
        ++j;
    }
    const auto copied = chunk.RecordData(std::span<const T>{base, total});
    std::size_t offset = 0;
    for (std::size_t k = i; k < j; ++k) {
        writes[k].*member = copied.data() + offset;
        offset += writes[k].count;
    }
    return j;
}

/// One pass over the writes: each entry names the member its infos are in.
void CopyWriteInfosOnce(Chunk& chunk, WriteDesc* writes, std::size_t count) {
    for (std::size_t i = 0; i < count;) {
        const auto& write = writes[i];
        if (write.buffer_info) {
            i = CopyRun(chunk, writes, i, count, &WriteDesc::buffer_info);
        } else if (write.image_info) {
            i = CopyRun(chunk, writes, i, count, &WriteDesc::image_info);
        } else if (write.texel_view) {
            i = CopyRun(chunk, writes, i, count, &WriteDesc::texel_view);
        } else {
            ++i;
        }
    }
}

// --- the scenes -------------------------------------------------------------------------

/// A frame's worth of commands: CommandsPerDraw per draw, one of them a closure with owner.
struct CommandScene {
    std::array<std::byte, 1024 * CommandsPerDraw> storage{};
    CommandBase* first{};

    /// A draw records a handful of commands of different types, one after another.
    void Build() {
        std::size_t used = 0;
        CommandBase* last = nullptr;
        for (int i = 0; i < CommandsPerDraw; ++i) {
            CommandBase* command = nullptr;
            if (i % 2 == 0) {
                used = (used + alignof(HandleCommand) - 1) & ~(alignof(HandleCommand) - 1);
                command = new (storage.data() + used) HandleCommand(std::uint64_t(i) * 37u);
                used += sizeof(HandleCommand);
            } else {
                used = (used + alignof(SpanCommand) - 1) & ~(alignof(SpanCommand) - 1);
                command = new (storage.data() + used) SpanCommand(std::uint32_t(i) * 11u);
                used += sizeof(SpanCommand);
            }
            if (last) {
                last->next = command;
            } else {
                first = command;
            }
            last = command;
        }
    }
};

/// A draw's descriptor writes: 8 buffers, 12 images and 4 texel views, each a run of its own.
struct WriteScene {
    std::array<BufferInfo, 8> buffers{};
    std::array<ImageInfo, 12> images{};
    std::array<ViewInfo, 4> views{};
    std::array<WriteDesc, WritesPerDraw> writes{};

    void Build() {
        for (int i = 0; i < WritesPerDraw; ++i) {
            auto& write = writes[i];
            write = {};
            write.dst_binding = std::uint32_t(i);
            write.count = 1;
            if (i < 8) {
                write.type = 6;
                write.buffer_info = &buffers[i];
            } else if (i < 20) {
                write.type = 2;
                write.image_info = &images[i - 8];
            } else {
                write.type = 4;
                write.texel_view = &views[i - 20];
            }
        }
    }
};

template <typename Work>
double MeasureNsPerDraw(Work&& work) {
    // Warm up: the first pass faults in the model's pages.
    for (int warm = 0; warm < 4; ++warm) {
        work();
    }
    std::uint64_t iterations = 0;
    const auto start = Clock::now();
    for (int round = 0; round < Rounds; ++round) {
        for (int draw = 0; draw < DrawsPerFrame; ++draw) {
            work();
        }
        iterations += DrawsPerFrame;
    }
    const auto done = Clock::now();
    const double ns =
        double(std::chrono::duration_cast<std::chrono::nanoseconds>(done - start).count());
    Sink = Sink + iterations;
    return ns / double(iterations);
}

} // namespace

int main() {
    CommandScene commands;
    commands.Build();
    WriteScene writes;
    writes.Build();
    Chunk chunk;
    int out = 0;

    const auto no_owner = [&] {
        Sink = Sink + ExecuteExecuteOnly(commands.first, out);
    };
    const auto with_owner = [&] {
        Sink = Sink + ExecuteWithDestructors(commands.first, out);
    };
    const auto legacy = [&] {
        chunk.Reset();
        auto copy = writes.writes;
        CopyWriteInfosLegacy(chunk, copy.data(), copy.size());
        Sink = Sink + copy[0].dst_binding;
    };
    const auto once = [&] {
        chunk.Reset();
        auto copy = writes.writes;
        CopyWriteInfosOnce(chunk, copy.data(), copy.size());
        Sink = Sink + copy[0].dst_binding;
    };

    // Best of five: a preemption only ever makes a run longer, so the minimum is the estimate
    // and the spread between the five says what the bench can resolve on this host.
    const auto best = [](auto&& work) {
        double lowest = MeasureNsPerDraw(work);
        double highest = lowest;
        for (int rep = 0; rep < 4; ++rep) {
            const double ns = MeasureNsPerDraw(work);
            lowest = std::min(lowest, ns);
            highest = std::max(highest, ns);
        }
        return std::pair{lowest, highest - lowest};
    };
    const auto [ns_execute_only, spread_execute] = best(no_owner);
    const auto [ns_with_destructors, spread_destructors] = best(with_owner);
    const auto [ns_writes_legacy, spread_legacy] = best(legacy);
    const auto [ns_writes_once, spread_once] = best(once);

    std::printf("Recorded commands per draw: %d\n", CommandsPerDraw);
    std::printf("  Execute only           : %7.2f ns/draw (spread %.2f)\n", ns_execute_only,
                spread_execute);
    std::printf("  Execute + destructors  : %7.2f ns/draw (spread %.2f)\n", ns_with_destructors,
                spread_destructors);
    const double destructor_delta = ns_with_destructors - ns_execute_only;
    const double destructor_spread = std::max(spread_execute, spread_destructors);
    std::printf("  saved                  : %7.2f ns/draw\n", destructor_delta);
    std::printf("  %s\n", std::abs(destructor_delta) <= 2.0 * destructor_spread
                               ? "under twice the spread: one indirect call per command is not"
                                 " resolvable here"
                               : "resolvable on this host");
    std::printf("Descriptor writes per draw: %d\n", WritesPerDraw);
    std::printf("  three passes           : %7.2f ns/draw (spread %.2f)\n", ns_writes_legacy,
                spread_legacy);
    std::printf("  one pass               : %7.2f ns/draw (spread %.2f)\n", ns_writes_once,
                spread_once);
    std::printf("  saved                  : %7.2f ns/draw\n", ns_writes_legacy - ns_writes_once);
    const double saved_per_frame = (ns_with_destructors - ns_execute_only) +
                                   (ns_writes_legacy - ns_writes_once);
    std::printf("Per frame (%d draws)      : %7.1f us\n", DrawsPerFrame,
                saved_per_frame * DrawsPerFrame / 1000.0);
    return 0;
}
