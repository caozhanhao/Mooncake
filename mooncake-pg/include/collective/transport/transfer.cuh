#ifndef MOONCAKE_PG_COLLECTIVE_TRANSPORT_TRANSFER_CUH
#define MOONCAKE_PG_COLLECTIVE_TRANSPORT_TRANSFER_CUH

#include <cstdint>

#include "collective/transport/device_transfer.cuh"
#include "collective/transport/kernel_resources.cuh"
#include "collective/transport/host_transfer.cuh"
#include "collective/transport/peer_binding.h"

namespace mooncake {

// Dispatches an already materialized end-to-end route. Different directed
// peers may use different routes; ranks only need to agree on the collective
// algorithm. A failed attempt never changes route inside the kernel.
template <typename Overlap>
inline __device__ bool putAndSignal(const CollectiveKernelResources& resources,
                                    const CollectivePeerBinding& edge,
                                    const void* source, uint64_t bytes,
                                    uint64_t remote_inbox_offset,
                                    uint64_t remote_signal_offset,
                                    uint64_t token, uint64_t command_id,
                                    Overlap overlap) {
    switch (edge.route) {
        case CollectiveRoute::DevP2p:
            return device_transfer::putAndSignalP2p(
                edge, source, bytes, remote_inbox_offset, remote_signal_offset,
                token, overlap);
        case CollectiveRoute::DevRdma:
            return device_transfer::putAndSignalRdma(
                resources, edge, source, bytes, remote_inbox_offset,
                remote_signal_offset, token, command_id, overlap);
        case CollectiveRoute::Host:
            return host_transfer::putAndSignal(
                resources, edge, source, bytes, remote_inbox_offset,
                remote_signal_offset, token, overlap);
    }
    return false;
}

inline __device__ bool signal(const CollectiveKernelResources& resources,
                              const CollectivePeerBinding& edge,
                              uint64_t remote_signal_offset, uint64_t token,
                              uint64_t command_id) {
    switch (edge.route) {
        case CollectiveRoute::DevP2p:
            return device_transfer::signalP2p(edge, remote_signal_offset,
                                              token);
        case CollectiveRoute::DevRdma:
            return device_transfer::signalRdma(
                resources, edge, remote_signal_offset, token, command_id);
        case CollectiveRoute::Host:
            return host_transfer::signal(resources, edge, remote_signal_offset,
                                         token);
    }
    return false;
}

}  // namespace mooncake

#endif  // MOONCAKE_PG_COLLECTIVE_TRANSPORT_TRANSFER_CUH
