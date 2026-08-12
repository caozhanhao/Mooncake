#ifndef MOONCAKE_PG_DEVICE_COMM_DEVICE_TRANSFER_TRANSFER_ENDPOINT_H
#define MOONCAKE_PG_DEVICE_COMM_DEVICE_TRANSFER_TRANSFER_ENDPOINT_H

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace mooncake {

struct P2pEndpoint {
    std::vector<int32_t> ipc_handle;

    bool operator==(const P2pEndpoint&) const = default;
};

struct HostProxyEndpoint {
    std::string te_server_name;

    bool operator==(const HostProxyEndpoint&) const = default;
};

// Peer-scoped bootstrap metadata for one CUDA device. The
// registered region is shared by every available route; route-specific
// endpoint presence advertises which implementations may reach it.
struct DeviceTransferEndpoint {
    int32_t device_index = -1;

    uint64_t region_address = 0;
    uint64_t region_size = 0;

    std::optional<P2pEndpoint> p2p;
    std::optional<HostProxyEndpoint> host_proxy;

    bool operator==(const DeviceTransferEndpoint&) const = default;
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_DEVICE_COMM_DEVICE_TRANSFER_TRANSFER_ENDPOINT_H
