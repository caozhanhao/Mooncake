#ifndef MOONCAKE_PG_COLLECTIVE_TRANSPORT_HOST_TRANSFER_COMMAND_CUH
#define MOONCAKE_PG_COLLECTIVE_TRANSPORT_HOST_TRANSFER_COMMAND_CUH

#include <cstdint>

#include "collective/transport/peer_route.h"
#include "collective/types.h"

namespace mooncake {

enum class HostTransferCommandState : uint32_t {
    Idle = 0,
    Ready,
    RunningData,
    RunningSignal,
    Completed,
    Failed,
};

enum class HostTransferCommandKind : uint32_t {
    PutAndSignal = 0,
    Signal,
};

// Device is the single producer and HostTransferExecutor is the single
// consumer. Device-only routes never touch this command.
struct alignas(64) HostTransferCommand {
    uint32_t state = static_cast<uint32_t>(HostTransferCommandState::Idle);
    uint32_t kind =
        static_cast<uint32_t>(HostTransferCommandKind::PutAndSignal);
    HostLinkHandle peer_host_link = kInvalidHostLinkHandle;
    InGroupRank peer_in_group_rank = -1;
    uint64_t source_address = 0;
    uint64_t target_address = 0;
    uint64_t bytes = 0;
    uint64_t signal_source_address = 0;
    uint64_t signal_target_address = 0;
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_COLLECTIVE_TRANSPORT_HOST_TRANSFER_COMMAND_CUH
