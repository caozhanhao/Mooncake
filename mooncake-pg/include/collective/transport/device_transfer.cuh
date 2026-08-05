#ifndef MOONCAKE_PG_COLLECTIVE_TRANSPORT_DEVICE_TRANSFER_CUH
#define MOONCAKE_PG_COLLECTIVE_TRANSPORT_DEVICE_TRANSFER_CUH

#include <cstdint>

#include "collective/transport/kernel_resources.cuh"
#include "collective/transport/peer_route.h"
#include "transport/device/device_ops.cuh"
#include "transport/device/ibgda_device.cuh"

namespace mooncake::device_transfer {

using namespace mooncake::device;

inline __device__ bool timedOut(uint64_t start, uint64_t timeout_ticks) {
    return timeout_ticks != 0 && clock64() - start >= timeout_ticks;
}

inline __device__ IbgdaContext rdmaContext(const PeerRoute& edge) {
    IbgdaContext context{};
#ifdef MOONCAKE_EP_USE_MACA
    context.qp_devctxs = edge.device_rdma.qp_contexts;
#else
    context.qp_devctxs =
        reinterpret_cast<mlx5gda_qp_devctx*>(edge.device_rdma.qp_contexts);
#endif
    context.rkeys = edge.device_rdma.remote_keys;
    return context;
}

struct RdmaCompletion {
#ifdef MOONCAKE_EP_USE_MACA
    bool submitted = false;
#else
    mlx5gda_qp_devctx* qp = nullptr;
    uint16_t wqe = 0;
#endif
};

inline __device__ RdmaCompletion postRdmaWrite(const PeerRoute& edge,
                                               const void* source,
                                               uint64_t remote_offset,
                                               uint64_t bytes,
                                               uint64_t channel) {
    const auto context = rdmaContext(edge);
#ifdef MOONCAKE_EP_USE_MACA
    mc_ibgda_put(context, static_cast<int>(channel),
                 edge.device_rdma.peer_index, edge.device_rdma.local_peer_index,
                 edge.device_rdma.qps_per_peer, source,
                 edge.device_rdma.remote_buffer_address + remote_offset,
                 static_cast<uint32_t>(bytes));
    return RdmaCompletion{.submitted = true};
#else
    auto* qp = mc_ibgda_channel(context, static_cast<int>(channel),
                                edge.device_rdma.peer_index,
                                edge.device_rdma.qps_per_peer);
    mc_ibgda_lock(qp);
    const uint16_t wqe = qp->wq_head;
    mc_ibgda_write_rdma_write_wqe(
        qp, reinterpret_cast<uint64_t>(source),
        mc_bswap32(context.rkeys[edge.device_rdma.local_peer_index]),
        edge.device_rdma.remote_buffer_address + remote_offset,
        mc_bswap32(context.rkeys[edge.device_rdma.peer_index]),
        static_cast<uint32_t>(bytes));
    mc_ibgda_post_send_db(qp);
    mc_ibgda_unlock(qp);
    return RdmaCompletion{.qp = qp, .wqe = wqe};
#endif
}

inline __device__ bool waitRdmaCompletion(
    const CollectiveKernelResources& resources,
    const RdmaCompletion& completion, InGroupRank peer) {
#ifdef MOONCAKE_EP_USE_MACA
    return completion.submitted;
#else
    auto* qp = completion.qp;
    mc_ibgda_lock(qp);
    uint16_t wq_tail = qp->wq_tail;
    const uint64_t start = clock64();
    while (static_cast<int16_t>(wq_tail - completion.wqe) <= 0) {
        const uint16_t cq_be =
            *reinterpret_cast<volatile uint16_t*>(&qp->cq->wqe_counter);
        const uint8_t opcode = qp->cq->op_own >> 4;
        if (!(opcode == 0x0 || opcode == 0xF)) {
            resources.control->first_error_code =
                static_cast<int32_t>(CollectiveProtocolError::Transport);
            resources.control->failed_peer = peer;
            mc_ibgda_unlock(qp);
            return false;
        }
        wq_tail = mc_bswap16(cq_be) + 1;
        if (static_cast<int16_t>(wq_tail - completion.wqe) <= 0 &&
            timedOut(start, resources.timeout_device_ticks)) {
            resources.control->first_error_code =
                static_cast<int32_t>(CollectiveProtocolError::Timeout);
            resources.control->failed_peer = peer;
            mc_ibgda_unlock(qp);
            return false;
        }
    }
    if (wq_tail != qp->wq_tail) qp->wq_tail = wq_tail;
    mc_ibgda_unlock(qp);
    return true;
#endif
}

inline __device__ void storePeerPayload(void* destination, const void* source,
                                        uint64_t bytes, uint32_t worker_index,
                                        uint32_t worker_count) {
    auto* output_words = static_cast<int*>(destination);
    const auto* input = static_cast<const uint8_t*>(source);
    const uint64_t word_count = bytes / sizeof(uint32_t);
    for (uint64_t word = worker_index; word < word_count;
         word += worker_count) {
        const uint64_t offset = word * sizeof(uint32_t);
        const uint32_t value = static_cast<uint32_t>(input[offset]) |
                               static_cast<uint32_t>(input[offset + 1]) << 8 |
                               static_cast<uint32_t>(input[offset + 2]) << 16 |
                               static_cast<uint32_t>(input[offset + 3]) << 24;
        mc_st_na_s32(output_words + word, static_cast<int>(value));
    }
    if (worker_index == 0 && bytes % sizeof(uint32_t) != 0) {
        const uint64_t offset = word_count * sizeof(uint32_t);
        const uint16_t value = static_cast<uint16_t>(input[offset]) |
                               static_cast<uint16_t>(input[offset + 1]) << 8;
        mc_st_na_u16(reinterpret_cast<uint16_t*>(output_words + word_count),
                     value);
    }
}

template <typename Overlap>
inline __device__ bool putAndSignalP2p(const PeerRoute& edge,
                                       const void* source, uint64_t bytes,
                                       uint64_t remote_inbox_offset,
                                       uint64_t remote_signal_offset,
                                       uint64_t token, Overlap overlap) {
    const uint32_t transport_workers = blockDim.x / 2;
    if (threadIdx.x < transport_workers) {
        auto* destination = static_cast<char*>(edge.device_p2p.mapped_buffer) +
                            remote_inbox_offset;
        storePeerPayload(destination, source, bytes, threadIdx.x,
                         transport_workers);
    } else {
        overlap(threadIdx.x - transport_workers,
                blockDim.x - transport_workers);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        mc_fence();
        auto* ready = reinterpret_cast<uint64_t*>(
            static_cast<char*>(edge.device_p2p.mapped_buffer) +
            remote_signal_offset);
        mc_st_release_u64(ready, token);
    }
    __syncthreads();
    return true;
}

template <typename Overlap>
inline __device__ bool putAndSignalRdma(
    const CollectiveKernelResources& resources, const PeerRoute& edge,
    const void* source, uint64_t bytes, uint64_t remote_inbox_offset,
    uint64_t remote_signal_offset, uint64_t token, uint64_t command_id,
    Overlap overlap) {
    __shared__ int success;
    if (threadIdx.x == 0) {
        mc_st_release_u32(&resources.control->resource_idle, 0);
        auto* signal_source = reinterpret_cast<uint64_t*>(
            static_cast<char*>(resources.buffer.base) +
            resources.buffer.protocol_offset + kTransferSignalSourceOffset);
        mc_st_release_u64(signal_source, token);
        if (bytes != 0) {
            (void)postRdmaWrite(edge, source, remote_inbox_offset, bytes,
                                command_id);
        }
        const auto completion =
            postRdmaWrite(edge, signal_source, remote_signal_offset,
                          sizeof(token), command_id);
        success =
            waitRdmaCompletion(resources, completion, edge.peer_in_group_rank)
                ? 1
                : 0;
        if (success) {
            mc_st_release_u32(&resources.control->resource_idle, 1);
        }
    } else {
        overlap(threadIdx.x - 1, blockDim.x - 1);
    }
    __syncthreads();
    return success != 0;
}

inline __device__ bool signalP2p(const PeerRoute& edge,
                                 uint64_t remote_signal_offset,
                                 uint64_t token) {
    if (threadIdx.x == 0) {
        auto* target = reinterpret_cast<uint64_t*>(
            static_cast<char*>(edge.device_p2p.mapped_buffer) +
            remote_signal_offset);
        mc_st_release_u64(target, token);
    }
    __syncthreads();
    return true;
}

inline __device__ bool signalRdma(const CollectiveKernelResources& resources,
                                  const PeerRoute& edge,
                                  uint64_t remote_signal_offset, uint64_t token,
                                  uint64_t command_id) {
    __shared__ int success;
    if (threadIdx.x == 0) {
        mc_st_release_u32(&resources.control->resource_idle, 0);
        auto* signal_source = reinterpret_cast<uint64_t*>(
            static_cast<char*>(resources.buffer.base) +
            resources.buffer.protocol_offset + kTransferSignalSourceOffset);
        mc_st_release_u64(signal_source, token);
        const auto completion =
            postRdmaWrite(edge, signal_source, remote_signal_offset,
                          sizeof(token), command_id);
        success =
            waitRdmaCompletion(resources, completion, edge.peer_in_group_rank)
                ? 1
                : 0;
        if (success) {
            mc_st_release_u32(&resources.control->resource_idle, 1);
        }
    }
    __syncthreads();
    return success != 0;
}

}  // namespace mooncake::device_transfer

#endif  // MOONCAKE_PG_COLLECTIVE_TRANSPORT_DEVICE_TRANSFER_CUH
