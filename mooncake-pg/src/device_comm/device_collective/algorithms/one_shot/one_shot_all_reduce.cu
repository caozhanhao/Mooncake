#include "device_comm/device_collective/algorithms/one_shot/one_shot_kernel.cuh"

#include <cstdint>

#include <cooperative_groups.h>

#include "device_comm/device_collective/protocols/ll/ll_value_pack.cuh"
#include "device_comm/device_primitives/value_primitives.cuh"
#include "device_comm/device_utils/device_utils.cuh"
#include "pg_assert.h"

namespace mooncake {
namespace {

inline constexpr int kOneShotTileSize = 8;

template <typename T, ReduceOp Op>
[[nodiscard]] __device__ __forceinline__ InGroupRank
reducePeersScalar(CollectiveChunk<T> chunk, const OneShotPlan& plan,
                  const OneShotExchange<T>& exchange, uint64_t pack_count,
                  cooperative_groups::thread_block block) {
    using Pack = LLValuePack<T>;
    for (uint64_t pack_index = block.thread_rank(); pack_index < pack_count;
         pack_index += block.size()) {
        typename Pack::Value reduced = 0;
        for (uint32_t peer = 0; peer < plan.participant_count; ++peer) {
            typename Pack::Value value;
            if (peer == plan.self_active_index) {
                value = Pack::load(chunk.source, chunk.count, pack_index);
            } else if (!exchange.readReceived(peer, pack_index, value)) {
                return plan.remoteParticipant(peer).in_group_rank;
            }
            // Start with the first participant; no identity value is needed.
            reduced =
                peer == 0 ? value : Pack::template reduce<Op>(reduced, value);
        }
        Pack::store(chunk.destination, chunk.count, pack_index, reduced);
    }
    return kInvalidInGroupRank;
}

template <typename T, ReduceOp Op>
[[nodiscard]] __device__ __forceinline__ InGroupRank
reducePeersTiled(CollectiveChunk<T> chunk, const OneShotPlan& plan,
                 const OneShotExchange<T>& exchange, uint64_t pack_count,
                 cooperative_groups::thread_block block) {
    using Pack = LLValuePack<T>;
    const auto tile =
        cooperative_groups::tiled_partition<kOneShotTileSize>(block);
    for (uint64_t pack_index = tile.meta_group_rank(); pack_index < pack_count;
         pack_index += tile.meta_group_size()) {
        typename Pack::Value reduced = 0;
        for (uint32_t base = 0; base < plan.participant_count;
             base += kOneShotTileSize) {
            const uint32_t peer = base + tile.thread_rank();
            typename Pack::Value value = 0;
            InGroupRank failed = kInvalidInGroupRank;
            if (peer < plan.participant_count) {
                if (peer == plan.self_active_index) {
                    value = Pack::load(chunk.source, chunk.count, pack_index);
                } else if (!exchange.readReceived(peer, pack_index, value)) {
                    failed = plan.remoteParticipant(peer).in_group_rank;
                }
            }
            const auto failures = tile.ballot(failed != kInvalidInGroupRank);
            if (failures) return tile.shfl(failed, __ffs(failures) - 1);
#pragma unroll
            for (uint32_t lane = 0; lane < kOneShotTileSize; ++lane) {
                const uint32_t low =
                    tile.shfl(static_cast<uint32_t>(value), lane);
                typename Pack::Value received = low;
                if constexpr (Pack::kPayloadWords == 2) {
                    const uint32_t high =
                        tile.shfl(static_cast<uint32_t>(value >> 32), lane);
                    received |= static_cast<typename Pack::Value>(high) << 32;
                }
                if (tile.thread_rank() == 0 &&
                    base + lane < plan.participant_count)
                    reduced = base + lane == 0 ? received
                                               : Pack::template reduce<Op>(
                                                     reduced, received);
            }
        }
        if (tile.thread_rank() == 0) {
            Pack::store(chunk.destination, chunk.count, pack_index, reduced);
        }
    }
    return kInvalidInGroupRank;
}

template <typename T, ReduceOp Op>
[[nodiscard]] __device__ __forceinline__ CollectiveStepResult runOneShotChunks(
    CollectiveChunk<T> values, const OneShotPlan& plan,
    OneShotExchange<T>& exchange, cooperative_groups::thread_block block) {
    uint64_t offset = 0;
    while (offset < values.count) {
        PG_COLL_TRY(exchange.begin(block));
        const uint64_t remaining = values.count - offset;
        const CollectiveChunk<T> chunk{
            .source = values.source + offset,
            .destination = values.destination + offset,
            .count = remaining < exchange.chunkElementCapacity()
                         ? remaining
                         : exchange.chunkElementCapacity(),
        };
        exchange.sendAll(chunk, block);
        const uint64_t pack_count = OneShotExchange<T>::packCount(chunk.count);
        // Fill each tile with peers and give it only one pack.
        const bool tiled = plan.participant_count >= kOneShotTileSize &&
                           pack_count <= block.size() / kOneShotTileSize;
        const auto failed = tiled
                                ? reducePeersTiled<T, Op>(chunk, plan, exchange,
                                                          pack_count, block)
                                : reducePeersScalar<T, Op>(
                                      chunk, plan, exchange, pack_count, block);
        PG_COLL_TRY(exchange.finish(failed, block));
        offset += chunk.count;
    }
    return {};
}

template <typename T, ReduceOp Op>
[[nodiscard]] __device__ __forceinline__ CollectiveStepResult
runOneShotChannel(const AllReduceRequest& request, OneShotDeviceState* state,
                  uint32_t channel, uint32_t channel_count,
                  cooperative_groups::thread_block block) {
    const auto& plan = state->plan;
    const auto partition =
        OneShotChannelPartition::make<T>(request.count, channel, channel_count);
    if (partition.count == 0) return {};
    const CollectiveChunk<T> values{
        .source = static_cast<const T*>(request.send_buffer) + partition.offset,
        .destination = static_cast<T*>(request.recv_buffer) + partition.offset,
        .count = partition.count,
    };
    PG_ASSERT(state->ll);
    OneShotExchange<T> exchange(plan, *state->ll, channel,
                                state->collective.timeout_ticks);
    return runOneShotChunks<T, Op>(values, plan, exchange, block);
}

template <typename T, ReduceOp Op, CollectiveExecution Execution>
__global__ __launch_bounds__(kOneShotMaxThreads) void oneShotAllReduceKernel(
    AllReduceRequest request, OneShotDeviceState* state) {
    runOneShotInvocation<Execution>(
        request, state,
        [&](uint32_t channel, uint32_t channel_count,
            cooperative_groups::thread_block block) {
            return runOneShotChannel<T, Op>(request, state, channel,
                                            channel_count, block);
        });
}

template <typename T, ReduceOp Op>
cudaError_t launchKernel(const AllReduceRequest& request,
                         OneShotDeviceState* state, int threads,
                         uint32_t channels, cudaStream_t stream) {
    if (channels > 1) {
        oneShotAllReduceKernel<T, Op, CollectiveExecution::MultiCTA>
            <<<channels, threads, 0, stream>>>(request, state);
    } else {
        oneShotAllReduceKernel<T, Op, CollectiveExecution::SingleCTA>
            <<<1, threads, 0, stream>>>(request, state);
    }
    return cudaGetLastError();
}

template <typename T>
cudaError_t launchForType(const AllReduceRequest& request,
                          OneShotDeviceState* state, int threads,
                          uint32_t channels, cudaStream_t stream) {
    switch (request.op) {
        case ReduceOp::Sum:
            return launchKernel<T, ReduceOp::Sum>(request, state, threads,
                                                  channels, stream);
        case ReduceOp::Product:
            return launchKernel<T, ReduceOp::Product>(request, state, threads,
                                                      channels, stream);
        case ReduceOp::Min:
            return launchKernel<T, ReduceOp::Min>(request, state, threads,
                                                  channels, stream);
        case ReduceOp::Max:
            return launchKernel<T, ReduceOp::Max>(request, state, threads,
                                                  channels, stream);
        default:
            return cudaErrorInvalidValue;
    }
}

}  // namespace

cudaError_t launchOneShotAllReduceKernel(const AllReduceRequest& request,
                                         OneShotDeviceState* state,
                                         cudaStream_t stream) {
    if (!isDeviceReductionCombinationSupported(request.datatype, request.op))
        return cudaErrorInvalidValue;
    const uint64_t bytes = request.count * elementSize(request.datatype);
    // Size is fixed in a captured call; smaller blocks suit tiny messages.
    const int threads = bytes <= 64    ? 128
                        : bytes <= 128 ? 256
                                       : kOneShotMaxThreads;
    const uint32_t channels = bytes <= 2048 ? 1 : kOneShotMaxChannels;
    switch (request.datatype) {
        case DataType::Float16:
            return launchForType<__half>(request, state, threads, channels,
                                         stream);
        case DataType::Uint8:
            return launchForType<uint8_t>(request, state, threads, channels,
                                          stream);
        case DataType::Int8:
            return launchForType<int8_t>(request, state, threads, channels,
                                         stream);
        case DataType::Int16:
            return launchForType<int16_t>(request, state, threads, channels,
                                          stream);
        case DataType::Int64:
            return launchForType<int64_t>(request, state, threads, channels,
                                          stream);
        case DataType::Bfloat16:
            return launchForType<__nv_bfloat16>(request, state, threads,
                                                channels, stream);
        case DataType::Float32:
            return launchForType<float>(request, state, threads, channels,
                                        stream);
        case DataType::Float64:
            return launchForType<double>(request, state, threads, channels,
                                         stream);
        case DataType::Int32:
            return launchForType<int32_t>(request, state, threads, channels,
                                          stream);
        case DataType::Bool:
            return launchForType<bool>(request, state, threads, channels,
                                       stream);
        default:
            return cudaErrorInvalidValue;
    }
}

}  // namespace mooncake
