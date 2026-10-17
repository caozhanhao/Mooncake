#ifndef MOONCAKE_PG_DEVICE_COMM_DEVICE_COLLECTIVE_ALGORITHMS_ONE_SHOT_KERNEL_CUH
#define MOONCAKE_PG_DEVICE_COMM_DEVICE_COLLECTIVE_ALGORITHMS_ONE_SHOT_KERNEL_CUH

#include <cooperative_groups.h>

#include "device_comm/device_collective/algorithms/one_shot/one_shot_algorithm.cuh"
#include "device_comm/device_collective/device_collective_kernel.cuh"
#include "device_comm/device_collective/device_collective_local.cuh"
#include "device_comm/device_collective/protocols/ll/ll_primitives.cuh"
#include "device_comm/device_collective/protocols/ll/ll_value_pack.cuh"
#include "device_comm/device_utils/device_utils.cuh"

namespace mooncake {

// One channel's contiguous input range, measured in T elements. Channel
// boundaries align to LL packs so CTAs never share a pack.
struct OneShotChannelPartition {
    uint64_t offset;
    uint64_t count;

    // Channel capacity is ceil(pack_count / channel_count) packs. Clip each
    // range at count; trailing channels may be empty.
    //
    // For count = 10, pack size = 4 and channel_count = 4:
    //
    //             channel 0  channel 1  channel 2  channel 3
    // offset              0          4          8         12
    // count               4          4          2          0
    template <typename T>
    [[nodiscard]] __device__ __forceinline__ static OneShotChannelPartition
    make(uint64_t count, uint32_t channel, uint32_t channel_count) {
        if (channel_count == 1) return {0, count};
        const uint64_t packs = ceilDiv(count, LLValuePack<T>::kValueCount);
        const uint64_t capacity =
            ceilDiv(packs, channel_count) * LLValuePack<T>::kValueCount;
        const uint64_t offset = uint64_t{channel} * capacity;
        if (offset >= count) return {offset, 0};
        return {offset, minimum(count - offset, capacity)};
    }
};

// Send to every peer and finish all receives before advancing LL progress.
template <typename T>
class OneShotExchange {
    using Pack = LLValuePack<T>;

   public:
    __device__ __forceinline__ OneShotExchange(const OneShotPlan& plan,
                                               LLState& ll, uint32_t channel,
                                               uint64_t timeout_ticks)
        : plan_(plan),
          ll_(ll),
          channel_(channel),
          timeout_ticks_(timeout_ticks) {}

    [[nodiscard]] __device__ __forceinline__ uint64_t
    chunkElementCapacity() const {
        return OneShotBufferLayout::kChunkBytes / sizeof(T);
    }

    [[nodiscard]] __device__ __forceinline__ static uint64_t packCount(
        uint64_t count) {
        return ceilDiv(count, Pack::kValueCount);
    }

    [[nodiscard]] __device__ __forceinline__ CollectiveStepResult
    begin(cooperative_groups::thread_block block) {
        const auto result =
            LLControl(ll_, channel_, timeout_ticks_)
                .begin(plan_.view_epoch, plan_.remotePeers(),
                       plan_.packets + plan_.layout.packetIndex(channel_, 0, 0),
                       plan_.layout.channelPacketBytes(), block);
        sequence_ = ll_.progress[channel_].next_sequence;
        return result;
    }

    // Before X sends n+2, it receives Y's n+1, so Y has consumed X's n.
    // Empty/local-only calls consume no sequence, preserving this dependency.
    __device__ __forceinline__ void sendAll(
        CollectiveChunk<T> chunk,
        cooperative_groups::thread_block block) const {
        const uint64_t pack_count = packCount(chunk.count);
        const auto remote_peers = plan_.remotePeers();
        const uint64_t work = pack_count * remote_peers.size();
        const LLPacketOps packets(sequence_, timeout_ticks_);
        for (uint64_t index = block.thread_rank(); index < work;
             index += block.size()) {
            const auto& peer = remote_peers.atIndex(index / pack_count);
            const uint64_t pack_index = index % pack_count;
            const auto value =
                Pack::load(chunk.source, chunk.count, pack_index);
            const uint64_t remote_offset =
                peer.workspace_offset +
                plan_.layout.packetIndex(
                    channel_, sequence_ % OneShotBufferLayout::kSlots,
                    plan_.self_rank, pack_index * Pack::kPayloadWords) *
                    LLPacket::kStorageBytes;
            auto* destination =
                static_cast<LLPacket*>(ll_.bindings.transfer_handle->remotePtr(
                    peer.global_rank, remote_offset));
            packets.storePack<Pack>(destination, value);
        }
        block.sync();
    }

    [[nodiscard]] __device__ __forceinline__ bool readReceived(
        uint32_t active_index, uint64_t pack_index,
        typename Pack::Value& value) const {
        const auto& peer = plan_.remoteParticipant(active_index);
        const auto* packet =
            plan_.packets +
            plan_.layout.packetIndex(
                channel_, sequence_ % OneShotBufferLayout::kSlots,
                peer.in_group_rank, pack_index * Pack::kPayloadWords);
        return LLPacketOps(sequence_, timeout_ticks_)
            .loadPack<Pack>(packet, value);
    }

    [[nodiscard]] __device__ __forceinline__ CollectiveStepResult
    finish(InGroupRank failed, cooperative_groups::thread_block block) {
        return LLControl(ll_, channel_, timeout_ticks_)
            .finish(sequence_, failed, block);
    }

   private:
    const OneShotPlan& plan_;
    LLState& ll_;
    uint32_t channel_;
    uint64_t timeout_ticks_;
    uint64_t sequence_ = 0;
};

// Every CTA enters and leaves through the same invocation protocol, including
// channels with no input. The launch determines the channel count.
template <CollectiveExecution Execution, typename Request, typename RunChannel>
__device__ __forceinline__ void runOneShotInvocation(const Request& request,
                                                     OneShotDeviceState* state,
                                                     RunChannel run_channel) {
    const auto block = cooperative_groups::this_thread_block();
    PG_ASSERT(state);
    const auto& plan = state->plan;
    const auto& collective = state->collective;
    auto result =
        beginCollectiveInvocation<Execution>(&state->plan, collective, block);
    if (result.succeeded()) {
        if (plan.participant_count == 1) {
            executeLocalOnlyCollective(request, collective.self_rank, block);
        } else {
            result = run_channel(blockIdx.x, gridDim.x, block);
        }
    }
    finishCollectiveInvocation<Execution>(collective, plan.remotePeers(), block,
                                          result.failed_rank,
                                          request.failed_ranks_hint);
}

}  // namespace mooncake

#endif  // MOONCAKE_PG_DEVICE_COMM_DEVICE_COLLECTIVE_ALGORITHMS_ONE_SHOT_KERNEL_CUH
