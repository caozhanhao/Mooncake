#ifndef MOONCAKE_PG_DEVICE_COMM_DEVICE_TRANSFER_ROUTES_HOST_PROXY_ROUTE_HOST_PROXY_ROUTE_H
#define MOONCAKE_PG_DEVICE_COMM_DEVICE_TRANSFER_ROUTES_HOST_PROXY_ROUTE_HOST_PROXY_ROUTE_H

#include <cstdint>
#include <optional>

#include "control_plane/control_types.h"
#include "error_types.h"

namespace mooncake {

class HostTransferProxy;
struct HostProxyCommandSlot;

// Binds one device's fixed GPU -> host command slots to the shared host worker
// and forwards peer-endpoint changes. HostTransferProxy owns the slots for the
// lifetime of DeviceTransferService.
class HostProxyRoute {
   public:
    explicit HostProxyRoute(HostTransferProxy& proxy);

    PGResult<void> initialize(int device_index);
    [[nodiscard]] HostProxyCommandSlot* deviceCommandSlots() const noexcept;

    void installPeerEndpoint(
        uint32_t peer_index, const std::optional<HostProxyEndpoint>& endpoint);

    PGResult<void> waitUntilIdle();
    PGResult<void> shutdown();

   private:
    HostTransferProxy& proxy_;
    HostProxyCommandSlot* device_slots_ = nullptr;
    bool initialized_ = false;
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_DEVICE_COMM_DEVICE_TRANSFER_ROUTES_HOST_PROXY_ROUTE_HOST_PROXY_ROUTE_H
