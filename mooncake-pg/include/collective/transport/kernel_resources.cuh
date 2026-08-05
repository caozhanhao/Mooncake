#ifndef MOONCAKE_PG_COLLECTIVE_TRANSPORT_KERNEL_RESOURCES_CUH
#define MOONCAKE_PG_COLLECTIVE_TRANSPORT_KERNEL_RESOURCES_CUH

#include <cstdint>

#include "collective/runtime/control_block.cuh"
#include "collective/transport/host_transfer_command.cuh"

namespace mooncake {

struct CollectiveKernelBuffer {
    void* base = nullptr;
    uint64_t arena_offset = 0;
    uint64_t staging_offset = 0;
    uint64_t staging_bytes = 0;
    uint64_t protocol_offset = 0;
    uint64_t protocol_bytes = 0;
};

struct CollectivePeerControl {
    void* buffer = nullptr;
    uint64_t signals_offset = 0;
};

// Stable device-side resources shared by the kernel protocol, algorithms
// and route dispatch. This view contains no GroupView, GlobalRank, algorithm,
// participant set or host-side invocation identity.
struct CollectiveKernelResources {
    CollectiveKernelBuffer buffer;
    CollectivePeerControl peer_control;
    CollectiveControlBlock* control = nullptr;
    HostTransferCommand* host_command = nullptr;
    uint64_t timeout_device_ticks = 0;
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_COLLECTIVE_TRANSPORT_KERNEL_RESOURCES_CUH
