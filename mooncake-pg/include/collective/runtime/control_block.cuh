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

enum class CollectiveFailureState : uint32_t {
    Idle = 0,
    Pending,
    Handling,
    Acknowledged,
};

// Device-to-host failure handshake. A failed invocation publishes evidence and
// waits until CPU progress has finished its sync-after-failure attempt. The
// acknowledgement only lets that invocation finish; it never authorizes the
// same invocation to run the collective again.
struct CollectiveFailureReport {
    uint32_t state = static_cast<uint32_t>(CollectiveFailureState::Idle);
    int32_t error_code = 0;
    InGroupRank failed_peer = -1;
    uint64_t failure_cookie = 0;
};

// Common host-visible state for device and host transports. It is not owned by
// HostTransferProxy; every invocation gets the same control ABI regardless of
// the route selected by its current binding.
struct alignas(64) CollectiveControlBlock {
    int32_t first_error_code = 0;
    InGroupRank failed_peer = -1;
    uint32_t resource_idle = 1;
    CollectiveFailureReport failure;
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_COLLECTIVE_RUNTIME_CONTROL_BLOCK_CUH
