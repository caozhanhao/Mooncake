#ifndef MOONCAKE_PG_DEVICE_COMM_DEVICE_TRANSFER_ROUTES_P2P_ROUTE_P2P_ROUTE_H
#define MOONCAKE_PG_DEVICE_COMM_DEVICE_TRANSFER_ROUTES_P2P_ROUTE_P2P_ROUTE_H

#include <cstdint>
#include <optional>
#include <vector>

#include "error_types.h"

namespace mooncake {

namespace device {
class P2pTransport;
}

// Owns only the TE P2P endpoint snapshot and the imported peer mappings.
// Route policy remains in DeviceTransferService.
class P2pRoute {
   public:
    P2pRoute(device::P2pTransport& transport, void* local_region,
             int device_index, uint32_t self_peer_index,
             uint32_t peer_capacity);

    [[nodiscard]] std::vector<int32_t> localHandle() const;

    void installPeerHandle(uint32_t peer_index,
                           const std::vector<int32_t>& ipc_handle);

    // importPeerHandles() is a repeatable full-snapshot replacement by
    // contract. TE owns the lifetime of the mappings it replaces.
    PGResult<void> refreshMappings();

    [[nodiscard]] std::optional<uint64_t> resolve(uint32_t peer_index) const;

   private:
    struct PeerMapping {
        std::vector<int32_t> ipc_handle;
        uint64_t remote_region_address = 0;
    };

    device::P2pTransport& transport_;
    void* local_region_ = nullptr;
    int device_index_ = -1;
    uint32_t self_peer_index_ = 0;
    uint32_t peer_capacity_ = 0;
    std::vector<PeerMapping> peers_;
    bool refresh_needed_ = false;
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_DEVICE_COMM_DEVICE_TRANSFER_ROUTES_P2P_ROUTE_P2P_ROUTE_H
