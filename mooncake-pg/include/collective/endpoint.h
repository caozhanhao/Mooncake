#ifndef MOONCAKE_PG_COLLECTIVE_ENDPOINT_H
#define MOONCAKE_PG_COLLECTIVE_ENDPOINT_H

#include <cstdint>
#include <optional>
#include <vector>

#include "collective/types.h"

namespace mooncake {

struct CollectiveControlLaneLayout {
    BufferSpan signals;

    bool operator==(const CollectiveControlLaneLayout&) const = default;
};

struct CollectiveControlLayout {
    uint64_t version = 0;
    uint64_t total_bytes = 0;
    uint64_t alignment = 0;
    uint32_t lane_count = 0;
    std::vector<CollectiveControlLaneLayout> lanes;

    bool operator==(const CollectiveControlLayout&) const = default;
};

// Device API bindings describe the process-level registered arena. A group
// endpoint publishes only its small stable control subspan. Collective buffer
// offsets are exchanged by the data plane for each invocation.
struct P2pArenaBinding {
    std::vector<int32_t> opaque_handle;

    bool operator==(const P2pArenaBinding&) const = default;
};

struct RdmaArenaBinding {
    uint64_t remote_access_address = 0;
    uint32_t remote_key = 0;
    uint64_t subnet_prefix = 0;
    uint64_t interface_id = 0;
    std::vector<int32_t> qpns;
    std::vector<int32_t> lids;
    bool is_roce = false;

    bool operator==(const RdmaArenaBinding&) const = default;
};

// V2 is intentional during the incremental migration: this is the semantic
// replacement for GroupEndpointInfo, while legacy collectives keep using the
// old endpoint until the final cleanup PR.
struct GroupEndpointV2 {
    uint64_t endpoint_epoch = 0;
    DeviceId device = kInvalidDeviceId;
    uint64_t arena_generation = 0;
    uint64_t arena_base_address = 0;
    uint64_t arena_bytes = 0;
    uint64_t control_base_address = 0;
    CollectiveControlLayout control_layout;
    std::optional<P2pArenaBinding> device_p2p;
    std::optional<RdmaArenaBinding> device_rdma;

    bool operator==(const GroupEndpointV2&) const = default;
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_COLLECTIVE_ENDPOINT_H
