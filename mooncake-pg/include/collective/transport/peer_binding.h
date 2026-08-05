#ifndef MOONCAKE_PG_COLLECTIVE_TRANSPORT_PEER_BINDING_H
#define MOONCAKE_PG_COLLECTIVE_TRANSPORT_PEER_BINDING_H

#include <cstdint>

#include "collective/transport/link.h"
#include "collective/types.h"

namespace mooncake {

struct HostPeerBinding {
    HostLinkHandle link = kInvalidHostLinkHandle;
    uint64_t remote_arena_address = 0;
    uint64_t remote_buffer_address = 0;
};

struct DeviceP2pPeerBinding {
    void* mapped_arena = nullptr;
    void* mapped_buffer = nullptr;
};

struct DeviceRdmaPeerBinding {
    void* qp_contexts = nullptr;
    const uint32_t* remote_keys = nullptr;
    int32_t local_peer_index = -1;
    int32_t peer_index = -1;
    int32_t qps_per_peer = 0;
    uint64_t remote_arena_address = 0;
    uint64_t remote_buffer_address = 0;
};

// One group-local, already materialized edge. Device transport indices are
// opaque process-level slots; collective code uses only peer_in_group_rank.
struct CollectivePeerBinding {
    CollectiveRoute route = CollectiveRoute::Host;
    InGroupRank peer_in_group_rank = -1;
    HostPeerBinding host;
    DeviceP2pPeerBinding device_p2p;
    DeviceRdmaPeerBinding device_rdma;
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_COLLECTIVE_TRANSPORT_PEER_BINDING_H
