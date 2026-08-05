#ifndef MOONCAKE_PG_COLLECTIVE_RUNTIME_CONTROL_BLOCK_CUH
#define MOONCAKE_PG_COLLECTIVE_RUNTIME_CONTROL_BLOCK_CUH

#include <cstdint>

#include "collective/types.h"

namespace mooncake {

// Compact device/host protocol code. Public error propagation remains
// PGResult<...>; this enum only crosses mapped control memory.
enum class CollectiveProtocolError : int32_t {
    None = 0,
    Timeout,
    Transport,
    InvalidBinding,
    Unsupported,
};

enum class CollectiveFailureGateState : uint32_t {
    Open = 0,
    FailurePending,
    Recovering,
    Closed,
};

// A captured kernel cannot call the control plane. It closes this
// mapped gate after publishing failure evidence; CPU progress applies the
// Coordinator's new GroupView and only reopens the same stable address when
// both the new binding and the transport resources are safe to reuse.
struct CollectiveFailureGate {
    uint32_t state = static_cast<uint32_t>(CollectiveFailureGateState::Open);
    int32_t error_code = 0;
    InGroupRank failed_peer = -1;
    uint64_t failure_cookie = 0;
    uint64_t failure_view_epoch = 0;
};

// Common host-visible state for device and host transports. It is not owned by
// HostTransferProxy; every invocation gets the same control ABI regardless of
// the route selected by its current binding.
struct alignas(64) CollectiveControlBlock {
    int32_t first_error_code = 0;
    InGroupRank failed_peer = -1;
    uint32_t resource_idle = 1;
    CollectiveFailureGate failure_gate;
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_COLLECTIVE_RUNTIME_CONTROL_BLOCK_CUH
