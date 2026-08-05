#ifndef MOONCAKE_PG_COLLECTIVE_EXECUTOR_ALGORITHM_FLAT_RING_CUH
#define MOONCAKE_PG_COLLECTIVE_EXECUTOR_ALGORITHM_FLAT_RING_CUH

#include <cstdint>
#include <type_traits>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "collective/executor/allreduce_device.cuh"
#include "collective/executor/tensor_partition.cuh"
#include "collective/runtime/peer_buffer_exchange.cuh"
#include "collective/transport/transfer.cuh"

namespace mooncake::flat_ring {

using namespace mooncake::device;

enum class Phase : uint32_t {
    BufferExchange = 0,
    ReduceScatter = 1,
    AllGather = 2,
};

inline constexpr uint64_t kReadySignalOffset = 0;
inline constexpr uint64_t kAckSignalOffset = sizeof(uint64_t);
inline constexpr uint64_t kPredecessorBufferOffset = sizeof(uint64_t);
inline constexpr uint64_t kPredecessorBufferReady = 2 * sizeof(uint64_t);
inline constexpr uint64_t kSuccessorBufferOffset = 3 * sizeof(uint64_t);
inline constexpr uint64_t kSuccessorBufferReady = 4 * sizeof(uint64_t);

inline __device__ uint64_t
stageBytes(const CollectiveKernelResources& resources) {
    return resources.buffer.staging_bytes / 4;
}

inline __device__ uint64_t
inboxOffset(const CollectiveKernelResources& resources, uint32_t inbox) {
    return resources.buffer.staging_offset +
           (2 + inbox) * stageBytes(resources);
}

inline __device__ uint64_t mixToken(uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

inline __device__ uint64_t protocolToken(uint64_t view_epoch,
                                         uint64_t collective_sequence,
                                         uint64_t transfer_chunk, Phase phase,
                                         uint32_t round) {
    // Signals are deliberately not cleared between invocations. View epoch,
    // view-scoped monotonic per-lane sequence and transfer coordinates jointly
    // separate wire domains across graph replay and membership changes.
    uint64_t token = mixToken(0x464c415452494e47ULL);
    token = mixToken(token ^ mixToken(view_epoch + 0x101ULL));
    token = mixToken(token ^ mixToken(collective_sequence + 0x202ULL));
    token = mixToken(token ^ mixToken(transfer_chunk + 0x404ULL));
    token = mixToken(token ^ mixToken(static_cast<uint32_t>(phase) + 0x505ULL));
    token = mixToken(token ^ mixToken(round + 0x606ULL));
    return token == 0 ? 1 : token;
}

template <typename T>
inline __device__ T
valueFromBits(std::conditional_t<sizeof(T) == 2, uint16_t, uint32_t> bits) {
    T value;
    auto* output = reinterpret_cast<uint8_t*>(&value);
    const auto* input = reinterpret_cast<const uint8_t*>(&bits);
#pragma unroll
    for (uint32_t index = 0; index < sizeof(T); ++index) {
        output[index] = input[index];
    }
    return value;
}

template <typename T>
inline __device__ T loadInboxNoncoherent(const T* source, uint64_t index) {
    if constexpr (sizeof(T) == sizeof(uint16_t)) {
        const uint32_t word = static_cast<uint32_t>(
            mc_ld_nc_s32(reinterpret_cast<const int*>(source) + index / 2));
        const uint16_t bits =
            static_cast<uint16_t>(index % 2 == 0 ? word : word >> 16);
        return valueFromBits<T>(bits);
    } else {
        const uint32_t bits = static_cast<uint32_t>(
            mc_ld_nc_s32(reinterpret_cast<const int*>(source) + index));
        return valueFromBits<T>(bits);
    }
}

template <typename T>
inline __device__ T addValues(T left, T right) {
    return left + right;
}

template <>
inline __device__ __half addValues(__half left, __half right) {
    return __float2half(__half2float(left) + __half2float(right));
}

template <>
inline __device__ __nv_bfloat16 addValues(__nv_bfloat16 left,
                                          __nv_bfloat16 right) {
    return __float2bfloat16(__bfloat162float(left) + __bfloat162float(right));
}

template <typename T>
inline __device__ void reduceIntoFromInbox(T* destination, const T* source,
                                           uint64_t elements) {
    for (uint64_t index = threadIdx.x; index < elements; index += blockDim.x) {
        destination[index] =
            addValues(destination[index], loadInboxNoncoherent(source, index));
    }
}

template <typename T>
inline __device__ void copyFromInbox(T* destination, const T* source,
                                     uint64_t elements) {
    for (uint64_t index = threadIdx.x; index < elements; index += blockDim.x) {
        destination[index] = loadInboxNoncoherent(source, index);
    }
}

inline __device__ void reduceFromInbox(DataType datatype, void* destination,
                                       const void* source, uint64_t elements) {
    if (datatype == DataType::Float16) {
        reduceIntoFromInbox(static_cast<__half*>(destination),
                            static_cast<const __half*>(source), elements);
    } else if (datatype == DataType::Bfloat16) {
        reduceIntoFromInbox(static_cast<__nv_bfloat16*>(destination),
                            static_cast<const __nv_bfloat16*>(source),
                            elements);
    } else {
        reduceIntoFromInbox(static_cast<float*>(destination),
                            static_cast<const float*>(source), elements);
    }
}

inline __device__ void copyFromInbox(DataType datatype, void* destination,
                                     const void* source, uint64_t elements) {
    if (datatype == DataType::Float16) {
        copyFromInbox(static_cast<__half*>(destination),
                      static_cast<const __half*>(source), elements);
    } else if (datatype == DataType::Bfloat16) {
        copyFromInbox(static_cast<__nv_bfloat16*>(destination),
                      static_cast<const __nv_bfloat16*>(source), elements);
    } else {
        copyFromInbox(static_cast<float*>(destination),
                      static_cast<const float*>(source), elements);
    }
}

inline __device__ TensorShard transferSlice(uint64_t total_elements,
                                            uint32_t participant_count,
                                            uint32_t owner,
                                            uint64_t transfer_offset,
                                            uint64_t transfer_elements) {
    const TensorShard partition =
        tensorShardUnchecked(total_elements, participant_count, owner);
    const uint64_t offset_in_partition =
        transfer_offset < partition.length_elements ? transfer_offset
                                                    : partition.length_elements;
    const uint64_t remaining = partition.length_elements - offset_in_partition;
    return TensorShard{
        .offset_elements = partition.offset_elements + offset_in_partition,
        .length_elements =
            remaining < transfer_elements ? remaining : transfer_elements,
    };
}

struct CopyOverlap {
    void* destination = nullptr;
    const void* source = nullptr;
    uint64_t bytes = 0;

    inline __device__ void operator()(uint32_t worker_index,
                                      uint32_t worker_count) const {
        copyCollectiveBytes(destination, source, bytes, worker_index,
                            worker_count);
    }
};

inline __device__ bool transferFailed(const AllReduceKernelArgs& args,
                                      const PeerRoute& edge) {
    int32_t error_code = args.common.resources.control->first_error_code;
    if (error_code == 0) {
        error_code = static_cast<int32_t>(CollectiveProtocolError::Transport);
    }
    setCollectiveError(args, error_code, edge.peer_in_group_rank);
    __syncthreads();
    return false;
}

inline __device__ bool runReduceScatter(
    const AllReduceKernelArgs& args, const FlatRingKernelPlan& ring,
    const void* input, uint64_t view_epoch, uint64_t collective_sequence,
    uint64_t transfer_offset, uint64_t transfer_elements,
    uint64_t transfer_index, void* work_stages[2], uint32_t* current_stage) {
    const auto& resources = args.common.resources;
    const uint32_t count = ring.participant_count;
    const uint32_t self = ring.self_ordinal;
    const uint32_t bytes_per_element = allReduceElementBytes(args.datatype);
    auto* signals = static_cast<char*>(resources.buffer.base) +
                    resources.buffer.protocol_offset;
    auto* ready = reinterpret_cast<uint64_t*>(signals + kReadySignalOffset);
    auto* ack = reinterpret_cast<uint64_t*>(signals + kAckSignalOffset);

    const uint32_t initial_owner = (self + count - 1) % count;
    const TensorShard initial =
        transferSlice(args.element_count, count, initial_owner, transfer_offset,
                      transfer_elements);
    copyCollectiveBytes(work_stages[0],
                        static_cast<const char*>(input) +
                            initial.offset_elements * bytes_per_element,
                        initial.length_elements * bytes_per_element);
    __syncthreads();
    *current_stage = 0;

    for (uint32_t round = 0; round + 1 < count; ++round) {
        const uint32_t outgoing_owner = (self + count - round - 1) % count;
        const uint32_t incoming_owner = (self + count - round - 2) % count;
        const TensorShard outgoing =
            transferSlice(args.element_count, count, outgoing_owner,
                          transfer_offset, transfer_elements);
        const TensorShard incoming =
            transferSlice(args.element_count, count, incoming_owner,
                          transfer_offset, transfer_elements);
        const uint32_t inbox = round & 1U;
        const uint32_t next_stage = *current_stage ^ 1U;
        const uint64_t token =
            protocolToken(view_epoch, collective_sequence, transfer_index,
                          Phase::ReduceScatter, round);
        const uint64_t command_id =
            (transfer_index << 16) |
            (static_cast<uint64_t>(Phase::ReduceScatter) << 8) | round;

        const CopyOverlap copy_next{
            .destination = work_stages[next_stage],
            .source = static_cast<const char*>(input) +
                      incoming.offset_elements * bytes_per_element,
            .bytes = incoming.length_elements * bytes_per_element,
        };
        if (!putAndSignal(resources, ring.successor,
                          work_stages[*current_stage],
                          outgoing.length_elements * bytes_per_element,
                          inboxOffset(resources, inbox),
                          resources.buffer.protocol_offset + kReadySignalOffset,
                          token, command_id << 1, copy_next)) {
            return transferFailed(args, ring.successor);
        }
        if (!waitForCollectiveToken(ready, token, args,
                                    ring.predecessor.peer_in_group_rank)) {
            return false;
        }

        const auto* receive_source =
            static_cast<const char*>(resources.buffer.base) +
            inboxOffset(resources, inbox);
        reduceFromInbox(args.datatype, work_stages[next_stage], receive_source,
                        incoming.length_elements);
        mc_fence();
        __syncthreads();

        if (!signal(resources, ring.predecessor,
                    resources.buffer.protocol_offset + kAckSignalOffset, token,
                    (command_id << 1) | 1)) {
            return transferFailed(args, ring.predecessor);
        }
        if (!waitForCollectiveToken(ack, token, args,
                                    ring.successor.peer_in_group_rank)) {
            return false;
        }
        *current_stage = next_stage;
    }
    return true;
}

inline __device__ bool runAllGather(
    const AllReduceKernelArgs& args, const FlatRingKernelPlan& ring,
    uint64_t view_epoch, uint64_t collective_sequence, uint64_t transfer_offset,
    uint64_t transfer_elements, uint64_t transfer_index, void* work_stages[2],
    uint32_t* current_stage) {
    const auto& resources = args.common.resources;
    const uint32_t count = ring.participant_count;
    const uint32_t self = ring.self_ordinal;
    const uint32_t bytes_per_element = allReduceElementBytes(args.datatype);
    auto* signals = static_cast<char*>(resources.buffer.base) +
                    resources.buffer.protocol_offset;
    auto* ready = reinterpret_cast<uint64_t*>(signals + kReadySignalOffset);
    auto* ack = reinterpret_cast<uint64_t*>(signals + kAckSignalOffset);

    for (uint32_t round = 0; round + 1 < count; ++round) {
        const uint32_t outgoing_owner = (self + count - round) % count;
        const uint32_t incoming_owner = (self + count - round - 1) % count;
        const TensorShard outgoing =
            transferSlice(args.element_count, count, outgoing_owner,
                          transfer_offset, transfer_elements);
        const TensorShard incoming =
            transferSlice(args.element_count, count, incoming_owner,
                          transfer_offset, transfer_elements);
        const uint32_t inbox = round & 1U;
        const uint32_t next_stage = *current_stage ^ 1U;
        const uint64_t token =
            protocolToken(view_epoch, collective_sequence, transfer_index,
                          Phase::AllGather, round);
        const uint64_t command_id =
            (transfer_index << 16) |
            (static_cast<uint64_t>(Phase::AllGather) << 8) | round;

        const CopyOverlap copy_outgoing{
            .destination = static_cast<char*>(args.output) +
                           outgoing.offset_elements * bytes_per_element,
            .source = work_stages[*current_stage],
            .bytes = outgoing.length_elements * bytes_per_element,
        };
        if (!putAndSignal(resources, ring.successor,
                          work_stages[*current_stage],
                          outgoing.length_elements * bytes_per_element,
                          inboxOffset(resources, inbox),
                          resources.buffer.protocol_offset + kReadySignalOffset,
                          token, command_id << 1, copy_outgoing)) {
            return transferFailed(args, ring.successor);
        }
        if (!waitForCollectiveToken(ready, token, args,
                                    ring.predecessor.peer_in_group_rank)) {
            return false;
        }

        const auto* receive_source =
            static_cast<const char*>(resources.buffer.base) +
            inboxOffset(resources, inbox);
        copyFromInbox(args.datatype, work_stages[next_stage], receive_source,
                      incoming.length_elements);
        mc_fence();
        __syncthreads();

        if (!signal(resources, ring.predecessor,
                    resources.buffer.protocol_offset + kAckSignalOffset, token,
                    (command_id << 1) | 1)) {
            return transferFailed(args, ring.predecessor);
        }
        if (!waitForCollectiveToken(ack, token, args,
                                    ring.successor.peer_in_group_rank)) {
            return false;
        }
        *current_stage = next_stage;
    }

    const uint32_t final_owner = (self + 1) % count;
    const TensorShard final_slice =
        transferSlice(args.element_count, count, final_owner, transfer_offset,
                      transfer_elements);
    copyCollectiveBytes(static_cast<char*>(args.output) +
                            final_slice.offset_elements * bytes_per_element,
                        work_stages[*current_stage],
                        final_slice.length_elements * bytes_per_element);
    __syncthreads();
    return true;
}

inline __device__ bool run(const AllReduceKernelArgs& args,
                           const FlatRingKernelPlan& ring, const void* input,
                           uint64_t view_epoch, uint64_t collective_sequence) {
    const auto& resources = args.common.resources;
    if (args.element_count == 0) return true;
    if (ring.participant_count <= 1) {
        copyCollectiveBytes(
            args.output, input,
            args.element_count * allReduceElementBytes(args.datatype));
        __syncthreads();
        return true;
    }

    __shared__ FlatRingKernelPlan resolved_ring;
    __shared__ PeerBufferExchange buffer_exchanges[2];
    if (threadIdx.x == 0) {
        resolved_ring.participant_count = ring.participant_count;
        resolved_ring.self_ordinal = ring.self_ordinal;
        buffer_exchanges[0] = PeerBufferExchange{
            .route = ring.predecessor,
            .remote_offset_target = kSuccessorBufferOffset,
            .remote_ready_target = kSuccessorBufferReady,
            .local_offset_source = kPredecessorBufferOffset,
            .local_ready_source = kPredecessorBufferReady,
            .resolved_route = &resolved_ring.predecessor,
        };
        buffer_exchanges[1] = PeerBufferExchange{
            .route = ring.successor,
            .remote_offset_target = kPredecessorBufferOffset,
            .remote_ready_target = kPredecessorBufferReady,
            .local_offset_source = kSuccessorBufferOffset,
            .local_ready_source = kSuccessorBufferReady,
            .resolved_route = &resolved_ring.successor,
        };
    }
    __syncthreads();
    const uint64_t exchange_token = protocolToken(
        view_epoch, collective_sequence, 0, Phase::BufferExchange, 0);
    if (!exchangePeerBufferOffsets(resources, buffer_exchanges, exchange_token,
                                   0)) {
        return false;
    }

    // The logical Ring units are participant shards of the whole tensor.
    // Registered staging capacity only determines an internal transport tile.
    // Two work stages overlap transfer with the next local copy-in during
    // ReduceScatter and with result copy-out during AllGather.
    const uint64_t bytes_per_element = allReduceElementBytes(args.datatype);
    const uint64_t stage_bytes = stageBytes(resources);
    const uint64_t transfer_elements = stage_bytes / bytes_per_element;
    if (transfer_elements == 0) {
        setCollectiveError(
            args, static_cast<int32_t>(CollectiveProtocolError::InvalidPlan),
            -1);
        __syncthreads();
        return false;
    }

    auto* work = static_cast<char*>(resources.buffer.base) +
                 resources.buffer.staging_offset;
    void* work_stages[2]{work, work + stage_bytes};
    const uint64_t max_partition_elements =
        (args.element_count + resolved_ring.participant_count - 1) /
        resolved_ring.participant_count;
    uint64_t transfer_index = 0;
    for (uint64_t transfer_offset = 0; transfer_offset < max_partition_elements;
         transfer_offset += transfer_elements, ++transfer_index) {
        uint32_t current_stage = 0;
        if (!runReduceScatter(args, resolved_ring, input, view_epoch,
                              collective_sequence, transfer_offset,
                              transfer_elements, transfer_index, work_stages,
                              &current_stage) ||
            !runAllGather(args, resolved_ring, view_epoch, collective_sequence,
                          transfer_offset, transfer_elements, transfer_index,
                          work_stages, &current_stage)) {
            return false;
        }
    }
    return true;
}

}  // namespace mooncake::flat_ring

#endif  // MOONCAKE_PG_COLLECTIVE_EXECUTOR_ALGORITHM_FLAT_RING_CUH
