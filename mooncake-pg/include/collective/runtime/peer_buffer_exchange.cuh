#ifndef MOONCAKE_PG_COLLECTIVE_RUNTIME_PEER_BUFFER_EXCHANGE_CUH
#define MOONCAKE_PG_COLLECTIVE_RUNTIME_PEER_BUFFER_EXCHANGE_CUH

#include <cstddef>
#include <cstdint>

#include "collective/runtime/device_protocol.cuh"
#include "collective/transport/transfer.cuh"
#include "transport/device/device_ops.cuh"

namespace mooncake {

using namespace mooncake::device;

inline constexpr uint64_t kOutgoingBufferOffset = 0;

struct NoBufferExchangeOverlap {
    inline __device__ void operator()(uint32_t, uint32_t) const {}
};

struct PeerBufferExchange {
    PeerRoute route;
    uint64_t remote_offset_target = 0;
    uint64_t remote_ready_target = 0;
    uint64_t local_offset_source = 0;
    uint64_t local_ready_source = 0;
    PeerRoute* resolved_route = nullptr;
};

inline __device__ PeerRoute setRemoteBufferOffset(PeerRoute peer,
                                                  uint64_t buffer_offset) {
    switch (peer.kind) {
        case PeerRouteKind::DevP2p:
            peer.device_p2p.mapped_buffer =
                static_cast<char*>(peer.device_p2p.mapped_arena) +
                buffer_offset;
            break;
        case PeerRouteKind::DevRdma:
            peer.device_rdma.remote_buffer_address =
                peer.device_rdma.remote_arena_address + buffer_offset;
            break;
        case PeerRouteKind::Host:
            peer.host.remote_buffer_address =
                peer.host.remote_arena_address + buffer_offset;
            break;
    }
    return peer;
}

// A process-wide pool may hand different offsets to different ranks. The
// caller describes the peer exchanges required by its topology; this common
// protocol publishes the local lease and resolves invocation-local peer
// addresses without changing the selected algorithm or route.
template <size_t PeerCount>
inline __device__ bool exchangePeerBufferOffsets(
    const CollectiveKernelResources& resources,
    const PeerBufferExchange (&exchanges)[PeerCount], uint64_t token,
    uint64_t first_command_id) {
    auto* protocol = static_cast<uint8_t*>(resources.buffer.base) +
                     resources.buffer.protocol_offset;
    for (uint64_t index = threadIdx.x; index < resources.buffer.protocol_bytes;
         index += blockDim.x) {
        protocol[index] = 0;
    }
    __syncthreads();

    auto* control_signals = static_cast<char*>(resources.peer_signals.base) +
                            resources.peer_signals.offset;
    auto* outgoing =
        reinterpret_cast<uint64_t*>(control_signals + kOutgoingBufferOffset);
    if (threadIdx.x == 0) {
        mc_st_release_u64(outgoing, resources.buffer.arena_offset);
    }
    __syncthreads();

    for (size_t index = 0; index < PeerCount; ++index) {
        const auto& exchange = exchanges[index];
        if (!putAndSignal(
                resources, exchange.route, outgoing, sizeof(uint64_t),
                resources.peer_signals.offset + exchange.remote_offset_target,
                resources.peer_signals.offset + exchange.remote_ready_target,
                token, first_command_id + index, NoBufferExchangeOverlap{})) {
            return false;
        }
    }

    for (size_t index = 0; index < PeerCount; ++index) {
        const auto& exchange = exchanges[index];
        const auto* ready = reinterpret_cast<const uint64_t*>(
            control_signals + exchange.local_ready_source);
        if (!waitForCollectiveToken(ready, token, resources,
                                    exchange.route.peer_in_group_rank)) {
            return false;
        }
        if (threadIdx.x == 0) {
            const auto offset =
                mc_ld_acquire_u64(reinterpret_cast<const uint64_t*>(
                    control_signals + exchange.local_offset_source));
            *exchange.resolved_route =
                setRemoteBufferOffset(exchange.route, offset);
        }
    }
    __syncthreads();
    return true;
}

}  // namespace mooncake

#endif  // MOONCAKE_PG_COLLECTIVE_RUNTIME_PEER_BUFFER_EXCHANGE_CUH
