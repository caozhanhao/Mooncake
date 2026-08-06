#ifndef MOONCAKE_PG_COLLECTIVE_DEVICE_CONTEXT_CUH
#define MOONCAKE_PG_COLLECTIVE_DEVICE_CONTEXT_CUH

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

struct CollectivePeerSignals {
    void* base = nullptr;
    uint64_t offset = 0;
};

// Device resources bound to a submission. Their addresses may have either
// communicator or submission lifetime. This value contains no GroupView,
// global rank, participant set or algorithm policy.
struct CollectiveKernelResources {
    CollectiveKernelBuffer buffer;
    CollectivePeerSignals peer_signals;
    CollectiveControlBlock* control = nullptr;
    HostTransferCommand* host_command = nullptr;
    uint64_t timeout_device_ticks = 0;
};

// Common runtime-to-kernel ABI embedded by each collective operation.
struct CollectiveKernelArgs {
    CollectiveKernelResources resources;
    // All collective operations sharing a physical channel increment the same
    // sequence. This keeps their wire-token domains disjoint without making
    // invocation state part of an operation plan.
    uint64_t* invocation_sequence = nullptr;
    uint64_t failure_target_id = 0;
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_COLLECTIVE_DEVICE_CONTEXT_CUH
