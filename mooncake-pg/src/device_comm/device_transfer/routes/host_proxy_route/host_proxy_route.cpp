#include "device_comm/device_transfer/routes/host_proxy_route/host_proxy_route.h"

#include <chrono>

#include "device_comm/device_transfer/routes/host_proxy_route/host_transfer_proxy.h"

namespace mooncake {

HostProxyRoute::HostProxyRoute(HostTransferProxy& proxy) : proxy_(proxy) {}

PGResult<void> HostProxyRoute::initialize(int device_index) {
    if (initialized_) return {};
    PG_TRY(device_slots_, proxy_.initializeDevice(device_index));
    initialized_ = true;
    return {};
}

HostProxyCommandSlot* HostProxyRoute::deviceCommandSlots() const noexcept {
    return device_slots_;
}

void HostProxyRoute::installPeerEndpoint(
    uint32_t peer_index, const std::optional<HostProxyEndpoint>& endpoint) {
    proxy_.installPeerEndpoint(peer_index, endpoint);
}

PGResult<void> HostProxyRoute::waitUntilIdle() {
    PG_VALIDATE_STATE(initialized_, "HostProxyRoute is not initialized");
    return proxy_.waitUntilIdle();
}

PGResult<void> HostProxyRoute::shutdown() {
    if (!initialized_) return {};
    PG_TRY(proxy_.waitUntilIdle(std::chrono::milliseconds(0)));
    // The fixed lane set remains owned by HostTransferProxy until the
    // complete DeviceTransferService shuts down.
    device_slots_ = nullptr;
    initialized_ = false;
    return {};
}

}  // namespace mooncake
