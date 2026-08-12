#include "device_comm/device_collective/device_collective_types.cuh"

#include <cstdint>

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "device_comm/device_transfer/transfer_lane.cuh"

namespace mooncake {
namespace cg = cooperative_groups;

namespace {

template <typename T>
__device__ __forceinline__ T addValue(T left, T right) {
    return left + right;
}

template <>
__device__ __forceinline__ __half addValue(__half left, __half right) {
    return __float2half_rn(__half2float(left) + __half2float(right));
}

template <>
__device__ __forceinline__ __nv_bfloat16 addValue(__nv_bfloat16 left,
                                                  __nv_bfloat16 right) {
    return __float2bfloat16(__bfloat162float(left) + __bfloat162float(right));
}

__device__ __forceinline__ uint64_t minimum(uint64_t left, uint64_t right) {
    return left < right ? left : right;
}

__device__ __forceinline__ uint64_t divideRoundUp(uint64_t value,
                                                  uint64_t divisor) {
    return value / divisor + (value % divisor != 0 ? 1 : 0);
}

__device__ __forceinline__ uint32_t wrapActiveIndex(int64_t value,
                                                    uint32_t participants) {
    value %= static_cast<int64_t>(participants);
    if (value < 0) value += participants;
    return static_cast<uint32_t>(value);
}

bool validChannelCount(uint32_t channel_count) {
    return channel_count == 1 || channel_count == 2 ||
           channel_count == kMaxDeviceCollectiveChannels;
}

__device__ __forceinline__ bool invocationFailed(
    const DeviceCollectiveKernelResources& resources, uint32_t* shared_result,
    cg::thread_block block) {
    bool failed = false;
    if (block.thread_rank() == 0) {
        auto* const latched = reinterpret_cast<unsigned int*>(
            &resources.invocation->failure_latched);
        failed = atomicAdd(latched, 0u) != 0;
        *shared_result = failed ? 1 : 0;
    }
    block.sync();
    failed = *shared_result != 0;
    block.sync();
    return failed;
}

// Successful CTAs may retire before another channel discovers a failure. The
// last CTA therefore provides the device-wide safe-point proof: only it
// publishes the failure generation and waits for the host to install a usable
// Plan.
//
// finishInvocation() and failInvocation() synchronize their CTA before
// entering this common tail. That entry barrier proves that every thread in
// the CTA has stopped touching the Plan, peer table, and transfer buffers.
__device__ __forceinline__ void retireInvocationAtSafePoint(
    const DeviceAllReduceKernelArgs& request,
    const DeviceCollectiveKernelResources& resources, cg::thread_block block) {
    if (block.thread_rank() == 0) {
        auto* const invocation = resources.invocation;
        auto* const arrived =
            reinterpret_cast<unsigned int*>(&invocation->arrived_channels);
        auto* const failed =
            reinterpret_cast<unsigned int*>(&invocation->failure_latched);

        // Publish this CTA's safe point after its failure diagnostic, if any.
        __threadfence();
        const uint32_t previous = atomicAdd(arrived, 1u);
        if (previous + 1 == request.channel_count) {
            if (atomicAdd(failed, 0u) != 0u) {
                auto* const mailbox = resources.recovery;
                const uint64_t generation =
                    device::mc_ld_acquire_u64(&mailbox->failure_generation) + 1;
                __threadfence_system();
                device::mc_st_release_u64(&mailbox->failure_generation,
                                          generation);
                while (device::mc_ld_acquire_u64(&mailbox->ready_generation) <
                       generation) {
                }
            }

            // StrongStream prevents the next invocation from starting until
            // this kernel exits, so the last CTA can prepare the fixed state
            // for a different 1/2/4-channel invocation without a grid barrier.
            atomicExch(failed, 0u);
            __threadfence();
            atomicExch(arrived, 0u);
        }
    }
    block.sync();
}

__device__ __forceinline__ void finishInvocation(
    const DeviceAllReduceKernelArgs& request,
    const DeviceCollectiveKernelResources& resources, cg::thread_block block) {
    block.sync();
    retireInvocationAtSafePoint(request, resources, block);
}

__device__ __forceinline__ void failInvocation(
    const DeviceAllReduceKernelArgs& request,
    const DeviceCollectiveKernelResources& resources,
    DeviceCollectiveKernelError error, InGroupRank failed_rank,
    uint64_t view_epoch, cg::thread_block block) {
    block.sync();
    if (block.thread_rank() == 0) {
        auto* const failed = reinterpret_cast<unsigned int*>(
            &resources.invocation->failure_latched);
        if (atomicCAS(failed, 0u, 1u) == 0u) {
            auto* const mailbox = resources.recovery;
            mailbox->error_code = static_cast<int32_t>(error);
            mailbox->failed_rank = failed_rank;
            mailbox->view_epoch = view_epoch;
            mailbox->failed_hint_address =
                reinterpret_cast<uint64_t>(request.failed_ranks_hint);
            mailbox->failed_hint_count = request.failed_ranks_hint_count;
            __threadfence_system();
        }
    }
    retireInvocationAtSafePoint(request, resources, block);
}

template <typename T>
__device__ __forceinline__ void copyValues(T* destination, const T* source,
                                           uint64_t count,
                                           cg::thread_block block) {
    for (uint64_t index = block.thread_rank(); index < count;
         index += block.size()) {
        destination[index] = source[index];
    }
}

template <typename T>
__device__ __forceinline__ void reduceValues(T* destination, const T* source,
                                             uint64_t count,
                                             cg::thread_block block) {
    for (uint64_t index = block.thread_rank(); index < count;
         index += block.size()) {
        destination[index] = addValue(destination[index], source[index]);
    }
}

enum class RingStepResult : uint32_t {
    Succeeded,
    DataSignalTimedOut,
    ConsumedAckTimedOut,
};

__device__ __forceinline__ void parkAfterRingFailure(
    const DeviceAllReduceKernelArgs& request,
    const DeviceCollectiveKernelResources& resources,
    const DeviceAllReducePlanImage& plan, RingStepResult result,
    cg::thread_block block) {
    InGroupRank failed_rank = kInvalidInGroupRank;
    switch (result) {
        case RingStepResult::DataSignalTimedOut:
            failed_rank = plan.predecessor_rank;
            break;
        case RingStepResult::ConsumedAckTimedOut:
            failed_rank = plan.successor_rank;
            break;
        case RingStepResult::Succeeded:
            __trap();
            return;
    }
    failInvocation(request, resources,
                   DeviceCollectiveKernelError::IncomingTransferTimedOut,
                   failed_rank, plan.view_epoch, block);
}

template <typename T>
__device__ __forceinline__ RingStepResult runRingStep(
    const DeviceCollectiveKernelResources& resources,
    const TransferLane& transfer_lane, const DeviceAllReducePlanImage& plan,
    const DeviceCollectivePeerBinding& succ,
    const DeviceCollectivePeerBinding& pred, T* output,
    const DeviceCollectiveTransferBuffer& send_region,
    const DeviceCollectiveTransferBuffer& recv_region,
    uint64_t channel_elements, uint64_t shard_element_capacity,
    uint32_t send_shard_index, uint32_t recv_shard_index, uint64_t tile_index,
    uint64_t tile_element_capacity, uint64_t step_sequence,
    uint32_t channel_index, bool reduce_received_values,
    cg::thread_block block) {
    const uint64_t send_shard_begin =
        static_cast<uint64_t>(send_shard_index) * shard_element_capacity;
    const uint64_t recv_shard_begin =
        static_cast<uint64_t>(recv_shard_index) * shard_element_capacity;
    const uint64_t send_shard_elements =
        send_shard_begin < channel_elements
            ? minimum(shard_element_capacity,
                      channel_elements - send_shard_begin)
            : 0;
    const uint64_t recv_shard_elements =
        recv_shard_begin < channel_elements
            ? minimum(shard_element_capacity,
                      channel_elements - recv_shard_begin)
            : 0;
    const uint64_t tile_begin = tile_index * tile_element_capacity;
    const uint64_t send_count =
        tile_begin < send_shard_elements
            ? minimum(tile_element_capacity, send_shard_elements - tile_begin)
            : 0;
    const uint64_t recv_count =
        tile_begin < recv_shard_elements
            ? minimum(tile_element_capacity, recv_shard_elements - tile_begin)
            : 0;
    // Every rank executes the same phase/step/tile schedule. A tail shard may
    // contribute zero elements to this tile, but its peers still exchange the
    // arrival and consumed notifications. Use offset zero for an empty
    // direction so we never form a pointer outside the caller buffer.
    const uint64_t send_begin =
        send_count == 0 ? 0 : send_shard_begin + tile_begin;
    const uint64_t recv_begin =
        recv_count == 0 ? 0 : recv_shard_begin + tile_begin;

    auto* const send_buffer = reinterpret_cast<T*>(send_region.addr);
    const auto* const recv_buffer =
        reinterpret_cast<const T*>(recv_region.addr);
    copyValues(send_buffer, output + send_begin, send_count, block);
    block.sync();

    PutAndSignalRequest data_send;
    data_send.local_offset = send_region.region_offset;
    data_send.remote_offset = recv_region.region_offset;
    data_send.size = send_count * sizeof(T);
    data_send.signal.remote_offset =
        succ.remote_control_offset +
        resources.signal_slots.remoteSlotOffset(channel_index, plan.self_rank);
    data_send.timeout_ticks = resources.timeout_ticks;
    // The consumed notification waited below proves that the successor has
    // finished reading this payload, which is stronger than waiting for the
    // optional local-completion ticket here.
    (void)transfer_lane.putAndSignal(succ.peer_idx, data_send, block);

    const auto arrival = transfer_lane.waitSignal(
        SignalWaitRequest{
            .local_offset = resources.signal_slots.localSlotOffset(
                channel_index, plan.predecessor_rank),
            .least = step_sequence,
            .timeout_ticks = resources.timeout_ticks,
        },
        block);
    if (arrival.status == SignalWaitStatus::TimedOut) {
        return RingStepResult::DataSignalTimedOut;
    }
    // This Ring permits only one outstanding step per peer and channel. The
    // generic Service accepts any value at least `step_sequence`; observing a
    // later value here means the collective schedules have diverged.
    if (arrival.observed != step_sequence) {
        __trap();
        return RingStepResult::DataSignalTimedOut;
    }

    if (reduce_received_values) {
        reduceValues(output + recv_begin, recv_buffer, recv_count, block);
    } else {
        copyValues(output + recv_begin, recv_buffer, recv_count, block);
    }
    block.sync();

    // Arrival means the receive buffer is readable. This second signal says
    // every thread has consumed it and the sender may reuse that channel.
    SignalRequest consumed_ack;
    consumed_ack.signal.remote_offset =
        pred.remote_control_offset +
        resources.consumed_ack_slots.remoteSlotOffset(channel_index,
                                                      plan.self_rank);
    consumed_ack.timeout_ticks = resources.timeout_ticks;
    (void)transfer_lane.signal(pred.peer_idx, consumed_ack, block);

    const auto ack = transfer_lane.waitSignal(
        SignalWaitRequest{
            .local_offset = resources.consumed_ack_slots.localSlotOffset(
                channel_index, plan.successor_rank),
            .least = step_sequence,
            .timeout_ticks = resources.timeout_ticks,
        },
        block);
    if (ack.status == SignalWaitStatus::TimedOut) {
        return RingStepResult::ConsumedAckTimedOut;
    }
    if (ack.observed != step_sequence) {
        __trap();
        return RingStepResult::ConsumedAckTimedOut;
    }
    return RingStepResult::Succeeded;
}

template <typename T>
__global__ void flatRingAllReduceKernel(
    DeviceAllReduceKernelArgs request,
    DeviceCollectiveKernelResources resources) {
    const auto block = cg::this_thread_block();
    __shared__ uint32_t shared_result;
    const uint32_t channel = blockIdx.x;
    const auto plan = *resources.all_reduce_plan;
    // Recovery may update the stable Plan between Graph replays, so status
    // must be read on every execution. Plan structure was validated on host
    // before publication.
    if (plan.status != DeviceCollectivePlanStatus::Ready) {
        // A non-Ready Plan is an internal launch-contract violation, not a
        // peer failure. Do not fabricate a failed rank or enter recovery.
        __trap();
        return;
    }
    if (request.count == 0) {
        finishInvocation(request, resources, block);
        return;
    }

    // Split the full request evenly across the active channel CTAs. Channel
    // sizes differ by at most one element; any remainder is assigned to the
    // leading channels.
    const uint64_t elements_per_channel = request.count / request.channel_count;
    const uint64_t extra_elements = request.count % request.channel_count;
    const uint64_t channel_elements =
        elements_per_channel + (channel < extra_elements ? 1 : 0);
    // Let r = extra_elements and c = channel. The first r channels each get
    // one extra element, so the number of extra elements before channel c is:
    // - c < r: all c preceding channels have one extra, giving c;
    // - c >= r: only the first r channels have extras, giving r.
    // Therefore the offset adjustment is min(c, r).
    const uint64_t channel_offset =
        static_cast<uint64_t>(channel) * elements_per_channel +
        minimum(channel, extra_elements);

    const auto* input =
        static_cast<const T*>(request.send_buffer) + channel_offset;
    auto* output = static_cast<T*>(request.recv_buffer) + channel_offset;
    if (input != output) {
        copyValues(output, input, channel_elements, block);
    }

    // finishInvocation() supplies the copy barrier for this terminal path.
    if (plan.participant_count == 1) {
        finishInvocation(request, resources, block);
        return;
    }
    block.sync();

    // This AllReduce binds each algorithm channel one-to-one to a Service
    // lane. `channel` partitions data and collective control tables; the lane
    // selects fixed device-transfer execution resources.
    const auto transfer_lane = resources.transfer_handle->lane(channel);
    const uint64_t channel_buffer_size =
        resources.send_buffer.size / request.channel_count;
    const DeviceCollectiveTransferBuffer send_region{
        .addr = static_cast<char*>(resources.send_buffer.addr) +
                static_cast<uint64_t>(channel) * channel_buffer_size,
        .region_offset = resources.send_buffer.region_offset +
                         static_cast<uint64_t>(channel) * channel_buffer_size,
        .size = channel_buffer_size,
    };
    const DeviceCollectiveTransferBuffer recv_region{
        .addr = static_cast<char*>(resources.recv_buffer.addr) +
                static_cast<uint64_t>(channel) * channel_buffer_size,
        .region_offset = resources.recv_buffer.region_offset +
                         static_cast<uint64_t>(channel) * channel_buffer_size,
        .size = channel_buffer_size,
    };

    // This CTA processes one channel range. The ring further partitions that
    // range, and the fixed transfer buffers may require one more subdivision:
    //
    //   full request
    //     -> channel range          one per CTA
    //       -> ring shard           one per participant
    //         -> transfer tile      fits one channel send/recv buffer
    //
    // Each participant owns one fixed-capacity shard slot. Tail slots may be
    // partially filled or empty, but every rank uses the same capacity and
    // therefore the same tile-loop bound and step sequence.
    const uint64_t shard_element_capacity =
        divideRoundUp(channel_elements, plan.participant_count);
    const uint64_t tile_element_capacity = send_region.size / sizeof(T);
    const uint64_t tile_iterations_per_shard =
        divideRoundUp(shard_element_capacity, tile_element_capacity);

    // ReduceScatter and AllGather each contain participant_count - 1 ring
    // steps. Each tile advances this channel's persistent step sequence once.
    const uint64_t steps = plan.participant_count - 1;
    auto* const next_step_sequence = resources.next_step_sequences + channel;
    uint64_t step_sequence = device::mc_ld_acquire_u64(next_step_sequence);

    const auto& succ = resources.peers.entries[plan.successor_rank];
    const auto& pred = resources.peers.entries[plan.predecessor_rank];

    // ReduceScatter: forward one shard around the ring per step and reduce the
    // shard received from the predecessor into this rank's output.
    for (uint64_t step = 0; step < steps; ++step) {
        const uint32_t send_shard =
            wrapActiveIndex(static_cast<int64_t>(plan.self_active_index) - step,
                            plan.participant_count);
        const uint32_t recv_shard = wrapActiveIndex(
            static_cast<int64_t>(plan.self_active_index) - step - 1,
            plan.participant_count);
        for (uint64_t tile = 0; tile < tile_iterations_per_shard; ++tile) {
            if (invocationFailed(resources, &shared_result, block)) {
                finishInvocation(request, resources, block);
                return;
            }
            const auto result = runRingStep(
                resources, transfer_lane, plan, succ, pred, output, send_region,
                recv_region, channel_elements, shard_element_capacity,
                send_shard, recv_shard, tile, tile_element_capacity,
                step_sequence, channel, true, block);
            if (result != RingStepResult::Succeeded) {
                parkAfterRingFailure(request, resources, plan, result, block);
                return;
            }
            ++step_sequence;
        }
    }

    // AllGather: circulate the reduced shards without modifying their values.
    for (uint64_t step = 0; step < steps; ++step) {
        const uint32_t send_shard = wrapActiveIndex(
            static_cast<int64_t>(plan.self_active_index) + 1 - step,
            plan.participant_count);
        const uint32_t recv_shard =
            wrapActiveIndex(static_cast<int64_t>(plan.self_active_index) - step,
                            plan.participant_count);
        for (uint64_t tile = 0; tile < tile_iterations_per_shard; ++tile) {
            if (invocationFailed(resources, &shared_result, block)) {
                finishInvocation(request, resources, block);
                return;
            }
            const auto result = runRingStep(
                resources, transfer_lane, plan, succ, pred, output, send_region,
                recv_region, channel_elements, shard_element_capacity,
                send_shard, recv_shard, tile, tile_element_capacity,
                step_sequence, channel, false, block);
            if (result != RingStepResult::Succeeded) {
                parkAfterRingFailure(request, resources, plan, result, block);
                return;
            }
            ++step_sequence;
        }
    }

    // Failed invocations return above without committing; recovery resets the
    // step-sequence and signal state before another invocation can use the
    // new view.
    if (block.thread_rank() == 0) {
        device::mc_st_release_u64(next_step_sequence, step_sequence);
    }
    finishInvocation(request, resources, block);
}

}  // namespace

cudaError_t launchDeviceAllReduceKernel(
    const DeviceAllReduceKernelArgs& request,
    const DeviceCollectiveKernelResources& resources, cudaStream_t stream) {
    const bool valid_partition = request.count == 0
                                     ? request.channel_count == 1
                                     : request.count >= request.channel_count;
    if (!validChannelCount(request.channel_count) || !valid_partition) {
        return cudaErrorInvalidValue;
    }

    constexpr int kProtocolThreads = 256;
    const dim3 grid(request.channel_count);
    switch (request.datatype) {
        case DataType::Float16:
            flatRingAllReduceKernel<__half>
                <<<grid, kProtocolThreads, 0, stream>>>(request, resources);
            break;
        case DataType::Bfloat16:
            flatRingAllReduceKernel<__nv_bfloat16>
                <<<grid, kProtocolThreads, 0, stream>>>(request, resources);
            break;
        case DataType::Float32:
            flatRingAllReduceKernel<float>
                <<<grid, kProtocolThreads, 0, stream>>>(request, resources);
            break;
        default:
            return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}

}  // namespace mooncake
