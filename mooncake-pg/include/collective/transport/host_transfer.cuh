#ifndef MOONCAKE_PG_COLLECTIVE_TRANSPORT_HOST_TRANSFER_CUH
#define MOONCAKE_PG_COLLECTIVE_TRANSPORT_HOST_TRANSFER_CUH

#include <cstdint>

#include "collective/transport/kernel_resources.cuh"
#include "collective/transport/peer_route.h"
#include "transport/device/device_ops.cuh"

namespace mooncake::host_transfer {

using namespace mooncake::device;

inline __device__ bool timedOut(uint64_t start, uint64_t timeout_ticks) {
    return timeout_ticks != 0 && clock64() - start >= timeout_ticks;
}

inline __device__ bool issue(const CollectiveKernelResources& resources,
                             HostTransferCommandKind kind,
                             const PeerRoute& edge, const void* source,
                             uint64_t target, uint64_t bytes,
                             uint64_t signal_target, uint64_t token) {
    auto* command = resources.host_command;
    auto* control = resources.control;
    auto* signal_source = reinterpret_cast<uint64_t*>(
        static_cast<char*>(resources.buffer.base) +
        resources.buffer.protocol_offset + kTransferSignalSourceOffset);
    mc_st_release_u64(signal_source, token);

    command->kind = static_cast<uint32_t>(kind);
    command->peer_host_link = edge.host.link;
    command->peer_in_group_rank = edge.peer_in_group_rank;
    command->source_address = reinterpret_cast<uint64_t>(source);
    command->target_address = target;
    command->bytes = bytes;
    command->signal_source_address = reinterpret_cast<uint64_t>(signal_source);
    command->signal_target_address = signal_target;
    control->first_error_code = 0;
    control->failed_peer = -1;
    mc_fence();
    mc_st_release_u32(&command->state,
                      static_cast<uint32_t>(HostTransferCommandState::Ready));

    const uint64_t start = clock64();
    while (true) {
        const auto state = static_cast<HostTransferCommandState>(
            mc_ld_acquire(reinterpret_cast<const int*>(&command->state)));
        if (state == HostTransferCommandState::Completed) {
            mc_st_release_u32(
                &command->state,
                static_cast<uint32_t>(HostTransferCommandState::Idle));
            return true;
        }
        if (state == HostTransferCommandState::Failed) return false;
        if (timedOut(start, resources.timeout_device_ticks)) {
            control->first_error_code =
                static_cast<int32_t>(CollectiveProtocolError::Timeout);
            control->failed_peer = edge.peer_in_group_rank;
            return false;
        }
    }
}

template <typename Overlap>
inline __device__ bool putAndSignal(const CollectiveKernelResources& resources,
                                    const PeerRoute& edge, const void* source,
                                    uint64_t bytes,
                                    uint64_t remote_inbox_offset,
                                    uint64_t remote_signal_offset,
                                    uint64_t token, Overlap overlap) {
    __shared__ int success;
    if (threadIdx.x == 0) {
        success =
            issue(resources,
                  bytes == 0 ? HostTransferCommandKind::Signal
                             : HostTransferCommandKind::PutAndSignal,
                  edge, source,
                  edge.host.remote_buffer_address + remote_inbox_offset, bytes,
                  edge.host.remote_buffer_address + remote_signal_offset, token)
                ? 1
                : 0;
    } else {
        overlap(threadIdx.x - 1, blockDim.x - 1);
    }
    __syncthreads();
    return success != 0;
}

inline __device__ bool signal(const CollectiveKernelResources& resources,
                              const PeerRoute& edge,
                              uint64_t remote_signal_offset, uint64_t token) {
    __shared__ int success;
    if (threadIdx.x == 0) {
        success =
            issue(resources, HostTransferCommandKind::Signal, edge, nullptr, 0,
                  0, edge.host.remote_buffer_address + remote_signal_offset,
                  token)
                ? 1
                : 0;
    }
    __syncthreads();
    return success != 0;
}

}  // namespace mooncake::host_transfer

#endif  // MOONCAKE_PG_COLLECTIVE_TRANSPORT_HOST_TRANSFER_CUH
