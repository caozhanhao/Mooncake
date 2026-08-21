#ifndef MOONCAKE_PG_DEVICE_COMM_DEVICE_COLLECTIVE_PROTOCOLS_FLAT_RING_ALL_REDUCE_TYPES_CUH
#define MOONCAKE_PG_DEVICE_COMM_DEVICE_COLLECTIVE_PROTOCOLS_FLAT_RING_ALL_REDUCE_TYPES_CUH

#include <cstdint>

#include <cuda_alike.h>

#include "common_types.h"
#include "device_comm/device_collective/device_collective_types.cuh"

namespace mooncake {

inline constexpr uint32_t kFlatRingPipelineSlots = 2;
static_assert(kFlatRingPipelineSlots >= 2);
inline constexpr uint64_t kFlatRingBufferBytes = 16ull << 20;
inline constexpr uint64_t kFlatRingStagingBytes = kFlatRingBufferBytes;
static_assert(kFlatRingBufferBytes <= kDeviceCollectiveBufferCapacity);

struct FlatRingPeerTarget {
    GlobalRank global_rank = kInvalidGlobalRank;
    InGroupRank in_group_rank = kInvalidInGroupRank;
    uint64_t buffer_offset = 0;
    uint64_t signal_offset = 0;
};

struct FlatRingAllReducePlan {
    InGroupRank self_rank = kInvalidInGroupRank;
    int32_t self_active_index = -1;
    uint32_t participant_count = 0;

    // Local shared payload buffer. The protocol receives predecessor payloads
    // here; direct writers on other ranks use the same offset from our
    // endpoint.
    uint64_t buffer_offset = 0;
    uint64_t buffer_size = 0;

    // Optional local-only staging for routes that cannot write directly into
    // successor.buffer_offset. The Flat Ring kernel produces outgoing payloads
    // here and PayloadWriter publishes them with put(). No additional staging
    // copy is inserted.
    uint64_t staging_offset = 0;
    uint64_t staging_size = 0;

    // Communicator-local signal prefix. Peers address the same prefix through
    // the signal offset published in our endpoint.
    uint64_t signal_offset = 0;
    uint32_t signal_count = 0;

    FlatRingPeerTarget predecessor;
    FlatRingPeerTarget successor;
};

using FlatRingAllReducePlanSlot = DevicePlanSlot<FlatRingAllReducePlan>;

// Owns the [kind][channel][peer][slot] interpretation of the generic signals.
struct FlatRingSignalLayout {
    uint32_t max_group_size = 0;
    uint32_t payload_ready_begin = 0;
    uint32_t payload_consumed_begin = 0;
    uint32_t signal_count = 0;

    [[nodiscard]] static FlatRingSignalLayout make(
        uint32_t max_group_size) noexcept {
        const uint32_t channel_peer_count =
            kMaxDeviceCollectiveChannels * max_group_size;
        const uint32_t pipelined_count =
            channel_peer_count * kFlatRingPipelineSlots;
        return FlatRingSignalLayout{
            .max_group_size = max_group_size,
            .payload_ready_begin = channel_peer_count,
            .payload_consumed_begin = channel_peer_count + pipelined_count,
            .signal_count = channel_peer_count + 2 * pipelined_count,
        };
    }

    [[nodiscard]] __device__ __forceinline__ uint32_t
    recvBufferReadyIndex(uint32_t channel_index,
                         InGroupRank peer_rank) const {
        return channelPeerIndex(channel_index, peer_rank);
    }

    [[nodiscard]] __device__ __forceinline__ uint32_t payloadReadyIndex(
        uint32_t channel_index, InGroupRank peer_rank,
        uint32_t payload_slot) const {
        return payload_ready_begin +
               pipelinedIndex(channel_index, peer_rank, payload_slot);
    }

    [[nodiscard]] __device__ __forceinline__ uint32_t payloadConsumedIndex(
        uint32_t channel_index, InGroupRank peer_rank,
        uint32_t payload_slot) const {
        return payload_consumed_begin +
               pipelinedIndex(channel_index, peer_rank, payload_slot);
    }

   private:
    [[nodiscard]] __device__ __forceinline__ uint32_t channelPeerIndex(
        uint32_t channel_index, InGroupRank peer_rank) const {
        return channel_index * max_group_size +
               static_cast<uint32_t>(peer_rank);
    }

    [[nodiscard]] __device__ __forceinline__ uint32_t pipelinedIndex(
        uint32_t channel_index, InGroupRank peer_rank,
        uint32_t payload_slot) const {
        return channelPeerIndex(channel_index, peer_rank) *
                   kFlatRingPipelineSlots +
               payload_slot;
    }
};

struct FlatRingPersistentStateView {
    FlatRingAllReducePlanSlot* plan = nullptr;
    uint64_t* next_step_sequences = nullptr;
    uint64_t* next_recv_buffer_ready_sequences = nullptr;
};

// The generic resource owner allocates this many opaque bytes. Flat Ring alone
// maps those bytes to its Plan and rolling sequence cursors.
struct FlatRingPersistentStateLayout {
    static constexpr uint64_t kAlignment = 256;

    uint64_t size = 0;
    uint64_t plan_offset = 0;
    uint64_t next_step_sequences_offset = 0;
    uint64_t next_recv_buffer_ready_sequences_offset = 0;

    [[nodiscard]] static FlatRingPersistentStateLayout make() noexcept {
        FlatRingPersistentStateLayout layout;
        uint64_t cursor = 0;
        layout.plan_offset = reserve<FlatRingAllReducePlanSlot>(cursor);
        layout.next_step_sequences_offset = reserve<uint64_t>(
            cursor, kMaxDeviceCollectiveChannels * kFlatRingPipelineSlots);
        layout.next_recv_buffer_ready_sequences_offset = reserve<uint64_t>(
            cursor, kMaxDeviceCollectiveChannels);
        layout.size = alignUp(cursor, kAlignment);
        return layout;
    }

    [[nodiscard]] FlatRingPersistentStateView map(void* state) const noexcept {
        return FlatRingPersistentStateView{
            .plan = itemAt<FlatRingAllReducePlanSlot>(state, plan_offset),
            .next_step_sequences =
                itemAt<uint64_t>(state, next_step_sequences_offset),
            .next_recv_buffer_ready_sequences = itemAt<uint64_t>(
                state, next_recv_buffer_ready_sequences_offset),
        };
    }

   private:
    [[nodiscard]] static constexpr uint64_t alignUp(
        uint64_t value, uint64_t alignment) noexcept {
        return (value + alignment - 1) / alignment * alignment;
    }

    template <typename Item>
    [[nodiscard]] static uint64_t reserve(uint64_t& cursor,
                                          uint64_t count = 1) noexcept {
        cursor = alignUp(cursor, alignof(Item));
        const uint64_t offset = cursor;
        cursor += count * sizeof(Item);
        return offset;
    }

    template <typename Item>
    [[nodiscard]] static Item* itemAt(void* base, uint64_t offset) noexcept {
        return reinterpret_cast<Item*>(static_cast<char*>(base) + offset);
    }
};

struct FlatRingAllReduceDeviceResources {
    DeviceCollectiveKernelResources common;
    FlatRingPersistentStateView state;
    FlatRingSignalLayout signal_layout;
};

struct FlatRingAllReduceKernelArgs {
    const void* send_buffer = nullptr;
    void* recv_buffer = nullptr;
    uint64_t count = 0;
    DataType datatype = DataType::Float32;
    ReduceOp op = ReduceOp::Sum;
    int32_t* failed_ranks_hint = nullptr;
};

cudaError_t launchFlatRingAllReduceKernel(
    const FlatRingAllReduceKernelArgs& request,
    const FlatRingAllReduceDeviceResources& resources, uint32_t channel_count,
    cudaStream_t stream);

}  // namespace mooncake

#endif  // MOONCAKE_PG_DEVICE_COMM_DEVICE_COLLECTIVE_PROTOCOLS_FLAT_RING_ALL_REDUCE_TYPES_CUH
