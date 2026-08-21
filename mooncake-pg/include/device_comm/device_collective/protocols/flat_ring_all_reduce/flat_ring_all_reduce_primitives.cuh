#ifndef MOONCAKE_PG_DEVICE_COMM_DEVICE_COLLECTIVE_PROTOCOLS_FLAT_RING_ALL_REDUCE_PRIMITIVES_CUH
#define MOONCAKE_PG_DEVICE_COMM_DEVICE_COLLECTIVE_PROTOCOLS_FLAT_RING_ALL_REDUCE_PRIMITIVES_CUH

#include <cstdint>

#include <cooperative_groups.h>

#include "device_comm/device_assert.cuh"
#include "device_comm/device_collective/device_collective_kernel.cuh"
#include "device_comm/device_collective/protocols/flat_ring_all_reduce/flat_ring_all_reduce_types.cuh"
#include "device_comm/device_collective/reduction_traits.cuh"
#include "device_comm/device_primitives/payload_writer.cuh"
#include "device_comm/device_transfer/transfer_lane.cuh"

namespace mooncake {
namespace flat_ring_device {

namespace cg = cooperative_groups;

[[nodiscard]] __device__ __forceinline__ uint64_t minimum(uint64_t left,
                                                          uint64_t right) {
    return left < right ? left : right;
}

[[nodiscard]] __device__ __forceinline__ uint64_t divideRoundUp(
    uint64_t value, uint64_t divisor) {
    return value / divisor + (value % divisor != 0 ? 1 : 0);
}

[[nodiscard]] __device__ __forceinline__ uint64_t signalOffset(
    uint64_t region_offset, uint32_t signal_index) {
    return region_offset +
           static_cast<uint64_t>(signal_index) * sizeof(uint64_t);
}

[[nodiscard]] __device__ __forceinline__ uint32_t wrapActiveIndex(
    int64_t value, uint32_t participants) {
    value %= static_cast<int64_t>(participants);
    if (value < 0) value += participants;
    return static_cast<uint32_t>(value);
}

template <typename T>
struct alignas(16) ValuePack {
    static_assert(16 % sizeof(T) == 0);
    T values[16 / sizeof(T)];
};

template <typename T>
[[nodiscard]] __device__ __forceinline__ uint64_t packedValueCount(
    const T* destination, const T* source, uint64_t count) {
    static_assert(sizeof(ValuePack<T>) == 16);
    const auto combined = reinterpret_cast<uintptr_t>(destination) |
                          reinterpret_cast<uintptr_t>(source);
    return (combined & (alignof(ValuePack<T>) - 1)) == 0
               ? count / (sizeof(ValuePack<T>) / sizeof(T))
               : 0;
}

template <typename T>
__device__ __forceinline__ void copyValues(T* destination, const T* source,
                                           uint64_t count,
                                           cg::thread_block block) {
    const uint64_t pack_count =
        packedValueCount(destination, source, count);
    auto* destination_packs = reinterpret_cast<ValuePack<T>*>(destination);
    const auto* source_packs =
        reinterpret_cast<const ValuePack<T>*>(source);
    for (uint64_t index = block.thread_rank(); index < pack_count;
         index += block.size()) {
        destination_packs[index] = source_packs[index];
    }

    constexpr uint64_t kValuesPerPack = sizeof(ValuePack<T>) / sizeof(T);
    const uint64_t tail_begin = pack_count * kValuesPerPack;
    for (uint64_t index = tail_begin + block.thread_rank(); index < count;
         index += block.size()) {
        destination[index] = source[index];
    }
}

template <typename T>
[[nodiscard]] __device__ __forceinline__ uint64_t packedValueCount(
    const T* first, const T* second, const T* third, uint64_t count) {
    const auto combined = reinterpret_cast<uintptr_t>(first) |
                          reinterpret_cast<uintptr_t>(second) |
                          reinterpret_cast<uintptr_t>(third);
    return (combined & (alignof(ValuePack<T>) - 1)) == 0
               ? count / (sizeof(ValuePack<T>) / sizeof(T))
               : 0;
}

template <typename T>
[[nodiscard]] __device__ __forceinline__ uint64_t packedValueCount(
    const T* first, const T* second, const T* third, const T* fourth,
    uint64_t count) {
    const auto combined = reinterpret_cast<uintptr_t>(first) |
                          reinterpret_cast<uintptr_t>(second) |
                          reinterpret_cast<uintptr_t>(third) |
                          reinterpret_cast<uintptr_t>(fourth);
    return (combined & (alignof(ValuePack<T>) - 1)) == 0
               ? count / (sizeof(ValuePack<T>) / sizeof(T))
               : 0;
}

template <typename T, ReduceOp Op>
__device__ __forceinline__ void reduceValuesTo(
    T* destination, const T* local_values, const T* received_values,
    uint64_t count, cg::thread_block block) {
    constexpr uint64_t kValuesPerPack = sizeof(ValuePack<T>) / sizeof(T);
    const uint64_t pack_count = packedValueCount(
        destination, local_values, received_values, count);
    auto* destination_packs = reinterpret_cast<ValuePack<T>*>(destination);
    const auto* local_packs =
        reinterpret_cast<const ValuePack<T>*>(local_values);
    const auto* received_packs =
        reinterpret_cast<const ValuePack<T>*>(received_values);
    for (uint64_t index = block.thread_rank(); index < pack_count;
         index += block.size()) {
        ValuePack<T> result;
#pragma unroll
        for (uint64_t item = 0; item < kValuesPerPack; ++item) {
            result.values[item] = DeviceReductionTraits<T, Op>::apply(
                local_packs[index].values[item],
                received_packs[index].values[item]);
        }
        destination_packs[index] = result;
    }

    const uint64_t tail_begin = pack_count * kValuesPerPack;
    for (uint64_t index = tail_begin + block.thread_rank(); index < count;
         index += block.size()) {
        destination[index] = DeviceReductionTraits<T, Op>::apply(
            local_values[index], received_values[index]);
    }
}

template <typename T, ReduceOp Op>
__device__ __forceinline__ void reduceValuesToTwo(
    T* first_destination, T* second_destination, const T* local_values,
    const T* received_values, uint64_t count, cg::thread_block block) {
    constexpr uint64_t kValuesPerPack = sizeof(ValuePack<T>) / sizeof(T);
    const uint64_t pack_count =
        packedValueCount(first_destination, second_destination, local_values,
                         received_values, count);
    auto* first_packs =
        reinterpret_cast<ValuePack<T>*>(first_destination);
    auto* second_packs =
        reinterpret_cast<ValuePack<T>*>(second_destination);
    const auto* local_packs =
        reinterpret_cast<const ValuePack<T>*>(local_values);
    const auto* received_packs =
        reinterpret_cast<const ValuePack<T>*>(received_values);
    for (uint64_t index = block.thread_rank(); index < pack_count;
         index += block.size()) {
        ValuePack<T> result;
#pragma unroll
        for (uint64_t item = 0; item < kValuesPerPack; ++item) {
            result.values[item] = DeviceReductionTraits<T, Op>::apply(
                local_packs[index].values[item],
                received_packs[index].values[item]);
        }
        first_packs[index] = result;
        second_packs[index] = result;
    }

    const uint64_t tail_begin = pack_count * kValuesPerPack;
    for (uint64_t index = tail_begin + block.thread_rank(); index < count;
         index += block.size()) {
        const T result = DeviceReductionTraits<T, Op>::apply(
            local_values[index], received_values[index]);
        first_destination[index] = result;
        second_destination[index] = result;
    }
}

template <typename T>
__device__ __forceinline__ void copyValuesToTwo(
    T* first_destination, T* second_destination, const T* source,
    uint64_t count, cg::thread_block block) {
    constexpr uint64_t kValuesPerPack = sizeof(ValuePack<T>) / sizeof(T);
    const uint64_t pack_count = packedValueCount(
        first_destination, second_destination, source, count);
    auto* first_packs =
        reinterpret_cast<ValuePack<T>*>(first_destination);
    auto* second_packs =
        reinterpret_cast<ValuePack<T>*>(second_destination);
    const auto* source_packs =
        reinterpret_cast<const ValuePack<T>*>(source);
    for (uint64_t index = block.thread_rank(); index < pack_count;
         index += block.size()) {
        const auto value = source_packs[index];
        first_packs[index] = value;
        second_packs[index] = value;
    }

    const uint64_t tail_begin = pack_count * kValuesPerPack;
    for (uint64_t index = tail_begin + block.thread_rank(); index < count;
         index += block.size()) {
        const T value = source[index];
        first_destination[index] = value;
        second_destination[index] = value;
    }
}

enum class RingStepResult : uint32_t {
    Succeeded,
    InvocationFailed,
    PayloadReadyTimedOut,
    PayloadConsumedTimedOut,
};

// The shared bulk buffer may have been used by another communicator. The
// StrongStream orders local kernels, while this edge-local signal prevents a
// remote predecessor from overwriting our buffer too early.
__device__ __forceinline__ void publishRecvBufferReady(
    const FlatRingAllReduceDeviceResources& resources,
    const TransferLane& transfer_lane, const FlatRingPeerTarget& sender,
    InGroupRank recv_rank, uint32_t channel_index, cg::thread_block block) {
    SignalRequest ready;
    ready.signal.kind = SignalAction::Kind::Add;
    ready.signal.add.remote_offset = signalOffset(
        sender.signal_offset,
        resources.signal_layout.recvBufferReadyIndex(channel_index,
                                                       recv_rank));
    ready.timeout_ticks = resources.common.timeout_ticks;
    (void)transfer_lane.signal(sender.global_rank, ready, block);
}

[[nodiscard]] __device__ __forceinline__ SignalWaitStatus
waitForRecvBufferReady(const FlatRingAllReduceDeviceResources& resources,
                       const FlatRingAllReducePlan& plan,
                       const TransferLane& transfer_lane,
                       InGroupRank recv_rank, uint64_t ready_sequence,
                       uint32_t channel_index, cg::thread_block block) {
    const auto recv_ready = transfer_lane.waitSignal(
        SignalWaitRequest{
            .local_offset = signalOffset(
                plan.signal_offset,
                resources.signal_layout.recvBufferReadyIndex(channel_index,
                                                               recv_rank)),
            .least = ready_sequence,
            .timeout_ticks = resources.common.timeout_ticks,
        },
        block);
    if (recv_ready.status == SignalWaitStatus::TimedOut) {
        return SignalWaitStatus::TimedOut;
    }
    PG_DEVICE_ASSERT(recv_ready.observed == ready_sequence);
    return SignalWaitStatus::Reached;
}

[[nodiscard]] __device__ __forceinline__ InGroupRank failedRankForRingStep(
    const FlatRingAllReducePlan& plan, RingStepResult result) {
    switch (result) {
        case RingStepResult::InvocationFailed:
            return kInvalidInGroupRank;
        case RingStepResult::PayloadReadyTimedOut:
            return plan.predecessor.in_group_rank;
        case RingStepResult::PayloadConsumedTimedOut:
            return plan.successor.in_group_rank;
        case RingStepResult::Succeeded:
            PG_DEVICE_UNREACHABLE();
            return kInvalidInGroupRank;
    }
    PG_DEVICE_UNREACHABLE();
    return kInvalidInGroupRank;
}

struct RingTile {
    uint64_t begin = 0;
    uint64_t count = 0;
};

struct RingPayloadSlot {
    uint32_t index = 0;
    uint64_t sequence = 0;
};

[[nodiscard]] __device__ __forceinline__ RingPayloadSlot nextRingPayloadSlot(
    uint64_t* payload_slot_sequences, uint64_t& operation_index) {
    const uint32_t payload_slot =
        static_cast<uint32_t>(operation_index % kFlatRingPipelineSlots);
    ++operation_index;
    return RingPayloadSlot{
        .index = payload_slot,
        .sequence = payload_slot_sequences[payload_slot]++,
    };
}

[[nodiscard]] __device__ __forceinline__ RingTile ringTile(
    uint64_t channel_elements, uint64_t shard_element_capacity,
    uint32_t shard_index, uint64_t tile_index,
    uint64_t tile_element_capacity) {
    const uint64_t shard_begin =
        static_cast<uint64_t>(shard_index) * shard_element_capacity;
    const uint64_t shard_elements =
        shard_begin < channel_elements
            ? minimum(shard_element_capacity, channel_elements - shard_begin)
            : 0;
    const uint64_t tile_begin = tile_index * tile_element_capacity;
    const uint64_t count =
        tile_begin < shard_elements
            ? minimum(tile_element_capacity, shard_elements - tile_begin)
            : 0;
    return RingTile{
        .begin = count == 0 ? 0 : shard_begin + tile_begin,
        .count = count,
    };
}

struct RingTileLayout {
    uint64_t channel_elements = 0;
    uint64_t shard_element_capacity = 0;
    uint64_t tile_element_capacity = 0;

    [[nodiscard]] __device__ __forceinline__ RingTile tile(
        uint32_t shard_index, uint64_t tile_index) const {
        return ringTile(channel_elements, shard_element_capacity, shard_index,
                        tile_index, tile_element_capacity);
    }

    [[nodiscard]] __device__ __forceinline__ uint64_t tileCount() const {
        return divideRoundUp(shard_element_capacity, tile_element_capacity);
    }
};

[[nodiscard]] __device__ __forceinline__ uint32_t ringShardAtDistance(
    const FlatRingAllReducePlan& plan, uint64_t distance) {
    return wrapActiveIndex(static_cast<int64_t>(plan.self_active_index) -
                               static_cast<int64_t>(distance),
                           plan.participant_count);
}

[[nodiscard]] __device__ __forceinline__ char* localRegionAt(
    const DeviceTransferHandle& transfer_handle, uint64_t offset) {
    return static_cast<char*>(transfer_handle.local_region) + offset;
}

[[nodiscard]] __device__ __forceinline__ StagingRegion stagingForChannel(
    const DeviceTransferHandle& transfer_handle,
    const FlatRingAllReducePlan& plan, uint64_t channel_offset,
    uint64_t channel_size) {
    if (plan.staging_size == 0) return {};

    PG_DEVICE_ASSERT(channel_offset <= plan.staging_size);
    PG_DEVICE_ASSERT(channel_size <= plan.staging_size - channel_offset);
    PG_DEVICE_ASSERT(plan.staging_offset <= UINT64_MAX - channel_offset);
    return StagingRegion{
        .addr = localRegionAt(transfer_handle,
                              plan.staging_offset + channel_offset),
        .region_offset = plan.staging_offset + channel_offset,
        .size = channel_size,
    };
}

// CTA-collective operations fused around one ring receive/send step. This is
// deliberately Flat-Ring-specific: it owns slot reuse, peer signal indices and
// the exact reduction/copy placement around publication.
template <typename T, ReduceOp Op>
class FlatRingPrimitives {
   public:
    __device__ __forceinline__ FlatRingPrimitives(
        const FlatRingAllReduceDeviceResources& resources,
        const TransferLane& transfer_lane, const FlatRingAllReducePlan& plan,
        const T* input, T* output, uint64_t channel_buffer_size,
        uint32_t channel_index)
        : resources_(resources),
          transfer_lane_(transfer_lane),
          self_rank_(plan.self_rank),
          successor_(plan.successor),
          predecessor_(plan.predecessor),
          input_(input),
          output_(output),
          incoming_payload_(localRegionAt(
              *resources.common.transfer_handle,
              plan.buffer_offset + static_cast<uint64_t>(channel_index) *
                                       channel_buffer_size)),
          send_writer_(
              transfer_lane, plan.successor.global_rank,
              stagingForChannel(
                  *resources.common.transfer_handle, plan,
                  static_cast<uint64_t>(channel_index) * channel_buffer_size,
                  channel_buffer_size),
              RemotePayloadRegion{
                  .region_offset =
                      plan.successor.buffer_offset +
                      static_cast<uint64_t>(channel_index) *
                          channel_buffer_size,
                  .size = channel_buffer_size,
              }),
          payload_slot_size_(channel_buffer_size / kFlatRingPipelineSlots),
          signal_offset_(plan.signal_offset),
          channel_index_(channel_index) {}

    [[nodiscard]] __device__ __forceinline__ RingStepResult send(
        const RingTile& tile, RingPayloadSlot outgoing,
        cg::thread_block block) const {
        const auto available = waitSendSlotAvailable(outgoing, block);
        if (available != RingStepResult::Succeeded) return available;

        const auto payload = outgoingPayload(outgoing);
        copyValues(payload.template dataAs<T>(), input_ + tile.begin,
                   tile.count, block);
        publish(payload, outgoing, tile.count, block);
        return RingStepResult::Succeeded;
    }

    [[nodiscard]] __device__ __forceinline__ RingStepResult recvReduceSend(
        const RingTile& tile, RingPayloadSlot incoming,
        RingPayloadSlot outgoing, cg::thread_block block) const {
        const auto arrived = waitIncoming(incoming, block);
        if (arrived != RingStepResult::Succeeded) return arrived;
        const auto available = waitSendSlotAvailable(outgoing, block);
        if (available != RingStepResult::Succeeded) return available;

        const auto* const received = incomingPayload(incoming);
        const auto payload = outgoingPayload(outgoing);
        reduceValuesTo<T, Op>(payload.template dataAs<T>(),
                              input_ + tile.begin, received, tile.count, block);
        publish(payload, outgoing, tile.count, block);
        releaseIncomingSlot(incoming.index, block);
        return RingStepResult::Succeeded;
    }

    [[nodiscard]] __device__ __forceinline__ RingStepResult
    recvReduceCopySend(const RingTile& tile, RingPayloadSlot incoming,
                       RingPayloadSlot outgoing,
                       cg::thread_block block) const {
        const auto arrived = waitIncoming(incoming, block);
        if (arrived != RingStepResult::Succeeded) return arrived;
        const auto available = waitSendSlotAvailable(outgoing, block);
        if (available != RingStepResult::Succeeded) return available;

        const auto* const received = incomingPayload(incoming);
        const auto payload = outgoingPayload(outgoing);
        reduceValuesToTwo<T, Op>(output_ + tile.begin,
                                 payload.template dataAs<T>(),
                                 input_ + tile.begin, received, tile.count,
                                 block);
        publish(payload, outgoing, tile.count, block);
        releaseIncomingSlot(incoming.index, block);
        return RingStepResult::Succeeded;
    }

    [[nodiscard]] __device__ __forceinline__ RingStepResult recvCopySend(
        const RingTile& tile, RingPayloadSlot incoming,
        RingPayloadSlot outgoing, cg::thread_block block) const {
        const auto arrived = waitIncoming(incoming, block);
        if (arrived != RingStepResult::Succeeded) return arrived;
        const auto available = waitSendSlotAvailable(outgoing, block);
        if (available != RingStepResult::Succeeded) return available;

        const auto* const received = incomingPayload(incoming);
        const auto payload = outgoingPayload(outgoing);
        copyValuesToTwo(output_ + tile.begin, payload.template dataAs<T>(),
                        received, tile.count, block);
        publish(payload, outgoing, tile.count, block);
        releaseIncomingSlot(incoming.index, block);
        return RingStepResult::Succeeded;
    }

    [[nodiscard]] __device__ __forceinline__ RingStepResult recvCopyAndDrain(
        const RingTile& tile, RingPayloadSlot payload,
        cg::thread_block block) const {
        const auto arrived = waitIncoming(payload, block);
        if (arrived != RingStepResult::Succeeded) return arrived;

        copyValues(output_ + tile.begin, incomingPayload(payload), tile.count,
                   block);
        releaseIncomingSlot(payload.index, block);
        return waitSendConsumed(payload, block);
    }

   private:
    [[nodiscard]] __device__ __forceinline__ const T* incomingPayload(
        RingPayloadSlot payload) const {
        return reinterpret_cast<const T*>(
            incoming_payload_ +
            static_cast<uint64_t>(payload.index) * payload_slot_size_);
    }

    [[nodiscard]] __device__ __forceinline__ PayloadWriteView outgoingPayload(
        RingPayloadSlot payload) const {
        const uint64_t staging_offset =
            static_cast<uint64_t>(payload.index) * payload_slot_size_;
        const uint64_t remote_offset =
            static_cast<uint64_t>(payload.index) * payload_slot_size_;
        return send_writer_.view(staging_offset, remote_offset,
                                 payload_slot_size_);
    }

    [[nodiscard]] __device__ __forceinline__ uint64_t
    payloadReadySignalRemoteOffset(uint32_t payload_slot) const {
        return signalOffset(
            successor_.signal_offset,
            resources_.signal_layout.payloadReadyIndex(
                channel_index_, self_rank_, payload_slot));
    }

    __device__ __forceinline__ void publish(
        const PayloadWriteView& payload, RingPayloadSlot outgoing,
        uint64_t count, cg::thread_block block) const {
        PayloadPublishRequest publication;
        publication.size = count * sizeof(T);
        publication.signal.kind = SignalAction::Kind::Add;
        publication.signal.add.remote_offset =
            payloadReadySignalRemoteOffset(outgoing.index);
        publication.signal.add.delta = 1;
        publication.timeout_ticks = resources_.common.timeout_ticks;
        (void)payload.publish(publication, block);
    }

    [[nodiscard]] __device__ __forceinline__ RingStepResult waitIncoming(
        RingPayloadSlot incoming, cg::thread_block block) const {
        const auto arrival = transfer_lane_.waitSignal(
            SignalWaitRequest{
                .local_offset = signalOffset(
                    signal_offset_,
                    resources_.signal_layout.payloadReadyIndex(
                        channel_index_, predecessor_.in_group_rank,
                        incoming.index)),
                .least = incoming.sequence,
                .timeout_ticks = resources_.common.timeout_ticks,
            },
            block);
        if (arrival.status == SignalWaitStatus::TimedOut) {
            return RingStepResult::PayloadReadyTimedOut;
        }
        PG_DEVICE_ASSERT(arrival.observed == incoming.sequence);
        return RingStepResult::Succeeded;
    }

    __device__ __forceinline__ void releaseIncomingSlot(
        uint32_t payload_slot, cg::thread_block block) const {
        SignalRequest ack;
        ack.signal.kind = SignalAction::Kind::Add;
        ack.signal.add.remote_offset = signalOffset(
            predecessor_.signal_offset,
            resources_.signal_layout.payloadConsumedIndex(
                channel_index_, self_rank_, payload_slot));
        ack.timeout_ticks = resources_.common.timeout_ticks;
        (void)transfer_lane_.signal(predecessor_.global_rank, ack, block);
    }

    [[nodiscard]] __device__ __forceinline__ RingStepResult
    waitPayloadConsumed(uint32_t payload_slot, uint64_t sequence,
                        cg::thread_block block) const {
        const auto ack = transfer_lane_.waitSignal(
            SignalWaitRequest{
                .local_offset = signalOffset(
                    signal_offset_,
                    resources_.signal_layout.payloadConsumedIndex(
                        channel_index_, successor_.in_group_rank,
                        payload_slot)),
                .least = sequence,
                .timeout_ticks = resources_.common.timeout_ticks,
            },
            block);
        if (ack.status == SignalWaitStatus::TimedOut) {
            return RingStepResult::PayloadConsumedTimedOut;
        }
        PG_DEVICE_ASSERT(ack.observed == sequence);
        return RingStepResult::Succeeded;
    }

    [[nodiscard]] __device__ __forceinline__ RingStepResult
    waitSendSlotAvailable(RingPayloadSlot outgoing,
                          cg::thread_block block) const {
        return waitPayloadConsumed(outgoing.index, outgoing.sequence - 1,
                                   block);
    }

    [[nodiscard]] __device__ __forceinline__ RingStepResult waitSendConsumed(
        RingPayloadSlot outgoing, cg::thread_block block) const {
        return waitPayloadConsumed(outgoing.index, outgoing.sequence, block);
    }

    const FlatRingAllReduceDeviceResources& resources_;
    TransferLane transfer_lane_;
    InGroupRank self_rank_;
    FlatRingPeerTarget successor_;
    FlatRingPeerTarget predecessor_;
    const T* input_;
    T* output_;
    const char* incoming_payload_;
    PayloadWriter send_writer_;
    uint64_t payload_slot_size_;
    uint64_t signal_offset_;
    uint32_t channel_index_;
};

template <typename T, ReduceOp Op>
[[nodiscard]] __device__ __forceinline__ RingStepResult runRingTile(
    const FlatRingAllReduceDeviceResources& resources,
    const FlatRingAllReducePlan& plan,
    const FlatRingPrimitives<T, Op>& primitives,
    const RingTileLayout& layout, uint64_t tile_index,
    uint64_t* payload_slot_sequences, uint32_t* shared_result,
    cg::thread_block block) {
    if (collective_device::invocationFailed(resources.common, shared_result,
                                            block)) {
        return RingStepResult::InvocationFailed;
    }

    const uint64_t ring_steps = plan.participant_count - 1;
    uint64_t operation_index = 0;

    auto current =
        nextRingPayloadSlot(payload_slot_sequences, operation_index);
    auto result = primitives.send(
        layout.tile(ringShardAtDistance(plan, 0), tile_index), current, block);
    if (result != RingStepResult::Succeeded) return result;

    for (uint64_t step = 1; step < ring_steps; ++step) {
        if (collective_device::invocationFailed(
                resources.common, shared_result, block)) {
            return RingStepResult::InvocationFailed;
        }
        const auto next =
            nextRingPayloadSlot(payload_slot_sequences, operation_index);
        result = primitives.recvReduceSend(
            layout.tile(ringShardAtDistance(plan, step), tile_index), current,
            next, block);
        if (result != RingStepResult::Succeeded) return result;
        current = next;
    }

    if (collective_device::invocationFailed(resources.common, shared_result,
                                            block)) {
        return RingStepResult::InvocationFailed;
    }
    auto next = nextRingPayloadSlot(payload_slot_sequences, operation_index);
    result = primitives.recvReduceCopySend(
        layout.tile(ringShardAtDistance(plan, ring_steps), tile_index), current,
        next, block);
    if (result != RingStepResult::Succeeded) return result;
    current = next;

    for (uint64_t step = 0; step + 1 < ring_steps; ++step) {
        if (collective_device::invocationFailed(
                resources.common, shared_result, block)) {
            return RingStepResult::InvocationFailed;
        }
        next = nextRingPayloadSlot(payload_slot_sequences, operation_index);
        result = primitives.recvCopySend(
            layout.tile(ringShardAtDistance(plan, step), tile_index), current,
            next, block);
        if (result != RingStepResult::Succeeded) return result;
        current = next;
    }

    if (collective_device::invocationFailed(resources.common, shared_result,
                                            block)) {
        return RingStepResult::InvocationFailed;
    }
    return primitives.recvCopyAndDrain(
        layout.tile(ringShardAtDistance(plan, ring_steps - 1), tile_index),
        current, block);
}

}  // namespace flat_ring_device
}  // namespace mooncake

#endif  // MOONCAKE_PG_DEVICE_COMM_DEVICE_COLLECTIVE_PROTOCOLS_FLAT_RING_ALL_REDUCE_PRIMITIVES_CUH
