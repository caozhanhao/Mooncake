#include "device_comm/device_collective/protocols/flat_ring_all_reduce/flat_ring_all_reduce_types.cuh"

#include <cstdint>

#include <cooperative_groups.h>

#include "device_comm/device_assert.cuh"
#include "device_comm/device_collective/device_collective_kernel.cuh"
#include "device_comm/device_collective/protocols/flat_ring_all_reduce/flat_ring_all_reduce_primitives.cuh"
#include "device_comm/device_transfer/transfer_lane.cuh"

namespace mooncake {
namespace {

namespace cg = cooperative_groups;
using namespace flat_ring_device;

template <typename T, ReduceOp Op>
__global__ void flatRingAllReduceKernel(
    FlatRingAllReduceKernelArgs request,
    FlatRingAllReduceDeviceResources resources) {
    const auto block = cg::this_thread_block();
    __shared__ uint32_t shared_result;
    const uint32_t channel = blockIdx.x;
    const auto* const plan_slot = resources.state.plan;

    // Recovery may update this host-constructed Plan between Graph replays.
    PG_DEVICE_ASSERT(plan_slot->status == DevicePlanStatus::Ready);
    const auto plan = plan_slot->plan;
    if (request.count == 0) {
        collective_device::completeChannel(resources.common, block);
        return;
    }

    const uint32_t channel_count = gridDim.x;
    const uint64_t elements_per_channel = request.count / channel_count;
    const uint64_t extra_elements = request.count % channel_count;
    const uint64_t channel_elements =
        elements_per_channel + (channel < extra_elements ? 1 : 0);
    const uint64_t channel_offset =
        static_cast<uint64_t>(channel) * elements_per_channel +
        minimum(channel, extra_elements);

    const auto* input =
        static_cast<const T*>(request.send_buffer) + channel_offset;
    auto* output = static_cast<T*>(request.recv_buffer) + channel_offset;

    if (plan.participant_count == 1) {
        if (input != output) {
            copyValues(output, input, channel_elements, block);
        }
        collective_device::completeChannel(resources.common, block);
        return;
    }
    block.sync();

    // Flat Ring binds one algorithm channel to one fixed DTS lane. The current
    // Plan supplies the buffer, optional staging and signal prefix as offsets
    // in the DTS arena.
    const auto* const transfer_handle = resources.common.transfer_handle;
    PG_DEVICE_ASSERT(transfer_handle);
    PG_DEVICE_ASSERT(transfer_handle->local_region);
    const uint64_t local_region_size = transfer_handle->local_region_size;

    PG_DEVICE_ASSERT(plan.buffer_offset <= local_region_size);
    PG_DEVICE_ASSERT(plan.buffer_size >= kFlatRingBufferBytes);
    PG_DEVICE_ASSERT(plan.buffer_size <=
                     local_region_size - plan.buffer_offset);
    if (plan.staging_size != 0) {
        PG_DEVICE_ASSERT(plan.staging_offset <= local_region_size);
        PG_DEVICE_ASSERT(plan.staging_size >= kFlatRingStagingBytes);
        PG_DEVICE_ASSERT(plan.staging_size <=
                         local_region_size - plan.staging_offset);
    }
    PG_DEVICE_ASSERT(plan.signal_count >=
                     resources.signal_layout.signal_count);
    const uint64_t signal_bytes =
        static_cast<uint64_t>(plan.signal_count) * sizeof(uint64_t);
    PG_DEVICE_ASSERT(plan.signal_offset <= local_region_size);
    PG_DEVICE_ASSERT(signal_bytes <=
                     local_region_size - plan.signal_offset);

    const auto transfer_lane = transfer_handle->lane(channel);
    const uint64_t channel_buffer_size =
        kFlatRingBufferBytes / channel_count;
    PG_DEVICE_ASSERT(channel_buffer_size % kFlatRingPipelineSlots == 0 &&
                     channel_buffer_size / kFlatRingPipelineSlots >=
                         sizeof(T));
    const uint64_t payload_slot_size =
        channel_buffer_size / kFlatRingPipelineSlots;

    // full request -> channel -> ring shard -> payload-sized tile
    const RingTileLayout tile_layout{
        .channel_elements = channel_elements,
        .shard_element_capacity =
            divideRoundUp(channel_elements, plan.participant_count),
        .tile_element_capacity = payload_slot_size / sizeof(T),
    };

    auto* const next_step_sequences =
        resources.state.next_step_sequences +
        static_cast<uint64_t>(channel) * kFlatRingPipelineSlots;
    uint64_t step_sequences[kFlatRingPipelineSlots];
#pragma unroll
    for (uint32_t payload_slot = 0;
         payload_slot < kFlatRingPipelineSlots; ++payload_slot) {
        step_sequences[payload_slot] =
            device::mc_ld_acquire_u64(next_step_sequences + payload_slot);
    }
    auto* const next_recv_buffer_ready_sequence =
        resources.state.next_recv_buffer_ready_sequences + channel;
    const uint64_t recv_buffer_ready_sequence =
        device::mc_ld_acquire_u64(next_recv_buffer_ready_sequence);

    publishRecvBufferReady(resources, transfer_lane, plan.predecessor,
                           plan.self_rank, channel, block);
    if (waitForRecvBufferReady(
            resources, plan, transfer_lane, plan.successor.in_group_rank,
            recv_buffer_ready_sequence, channel,
            block) != SignalWaitStatus::Reached) {
        collective_device::completeChannel(
            resources.common, block, plan.successor.in_group_rank,
            request.failed_ranks_hint);
        return;
    }

    const FlatRingPrimitives<T, Op> primitives(
        resources, transfer_lane, plan, input, output, channel_buffer_size,
        channel);
    const uint64_t tile_count = tile_layout.tileCount();

    // Complete a full reduce-scatter/all-gather traversal for one tile before
    // reusing its two pipeline slots for the next tile.
    for (uint64_t tile_index = 0; tile_index < tile_count; ++tile_index) {
        const auto result = runRingTile(
            resources, plan, primitives, tile_layout, tile_index,
            step_sequences, &shared_result, block);
        if (result != RingStepResult::Succeeded) {
            collective_device::completeChannel(
                resources.common, block, failedRankForRingStep(plan, result),
                request.failed_ranks_hint);
            return;
        }
    }

    // Recovery resets the protocol prefix after a failure, so only successful
    // invocations commit their next rolling sequences.
    if (block.thread_rank() == 0) {
#pragma unroll
        for (uint32_t payload_slot = 0;
             payload_slot < kFlatRingPipelineSlots; ++payload_slot) {
            device::mc_st_release_u64(next_step_sequences + payload_slot,
                                      step_sequences[payload_slot]);
        }
        device::mc_st_release_u64(next_recv_buffer_ready_sequence,
                                  recv_buffer_ready_sequence + 1);
    }
    collective_device::completeChannel(resources.common, block);
}

template <typename T>
cudaError_t launchReduction(
    const FlatRingAllReduceKernelArgs& request,
    const FlatRingAllReduceDeviceResources& resources, dim3 grid,
    int protocol_threads, cudaStream_t stream) {
    switch (request.op) {
        case ReduceOp::Sum:
            flatRingAllReduceKernel<T, ReduceOp::Sum>
                <<<grid, protocol_threads, 0, stream>>>(request, resources);
            break;
        case ReduceOp::Product:
            flatRingAllReduceKernel<T, ReduceOp::Product>
                <<<grid, protocol_threads, 0, stream>>>(request, resources);
            break;
        case ReduceOp::Min:
            flatRingAllReduceKernel<T, ReduceOp::Min>
                <<<grid, protocol_threads, 0, stream>>>(request, resources);
            break;
        case ReduceOp::Max:
            flatRingAllReduceKernel<T, ReduceOp::Max>
                <<<grid, protocol_threads, 0, stream>>>(request, resources);
            break;
        default:
            return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}

}  // namespace

cudaError_t launchFlatRingAllReduceKernel(
    const FlatRingAllReduceKernelArgs& request,
    const FlatRingAllReduceDeviceResources& resources, uint32_t channel_count,
    cudaStream_t stream) {
    constexpr int kProtocolThreads = 256;
    const dim3 grid(channel_count);
    switch (request.datatype) {
        case DataType::Float16:
            return launchReduction<__half>(request, resources, grid,
                                           kProtocolThreads, stream);
        case DataType::Uint8:
            return launchReduction<uint8_t>(request, resources, grid,
                                            kProtocolThreads, stream);
        case DataType::Int8:
            return launchReduction<int8_t>(request, resources, grid,
                                           kProtocolThreads, stream);
        case DataType::Int16:
            return launchReduction<int16_t>(request, resources, grid,
                                            kProtocolThreads, stream);
        case DataType::Int32:
            return launchReduction<int32_t>(request, resources, grid,
                                            kProtocolThreads, stream);
        case DataType::Int64:
            return launchReduction<int64_t>(request, resources, grid,
                                            kProtocolThreads, stream);
        case DataType::Bfloat16:
            return launchReduction<__nv_bfloat16>(
                request, resources, grid, kProtocolThreads, stream);
        case DataType::Float32:
            return launchReduction<float>(request, resources, grid,
                                          kProtocolThreads, stream);
        case DataType::Float64:
            return launchReduction<double>(request, resources, grid,
                                           kProtocolThreads, stream);
        case DataType::Bool:
            return launchReduction<bool>(request, resources, grid,
                                         kProtocolThreads, stream);
        default:
            return cudaErrorInvalidValue;
    }
}

}  // namespace mooncake
