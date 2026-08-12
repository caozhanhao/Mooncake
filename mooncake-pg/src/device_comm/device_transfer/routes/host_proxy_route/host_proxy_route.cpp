#include "device_comm/device_transfer/routes/host_proxy_route/host_proxy_route.h"

#include "device_comm/device_transfer/routes/host_proxy_route/host_transfer_proxy.h"

namespace mooncake {

HostProxyRoute::HostProxyRoute(HostTransferProxy& proxy) : proxy_(proxy) {}

PGResult<void> HostProxyRoute::initialize(int device_index) {
    if (initialized_) return {};
    PG_TRY(device_slots_, proxy_.addDevice(device_index));
    device_index_ = device_index;
    initialized_ = true;
    return {};
}

HostProxyCommandSlot* HostProxyRoute::deviceCommandSlots() const noexcept {
    return device_slots_;
}

PGResult<void> HostProxyRoute::installPeerEndpoint(
    uint32_t peer_index, const std::optional<HostProxyEndpoint>& endpoint) {
    return proxy_.installPeerEndpoint(peer_index, endpoint);
}

PGResult<void> HostProxyRoute::waitUntilIdle() {
    PG_VALIDATE_STATE(initialized_, "HostProxyRoute is not initialized");
    return proxy_.waitUntilIdle(device_index_);
}

PGResult<void> HostProxyRoute::waitUntilIdle(
    std::chrono::milliseconds timeout) {
    PG_VALIDATE_STATE(initialized_, "HostProxyRoute is not initialized");
    return proxy_.waitUntilIdle(device_index_, timeout);
}

PGResult<void> HostProxyRoute::shutdown() {
    if (!initialized_) return {};
    PG_TRY(proxy_.waitUntilIdle(device_index_, std::chrono::milliseconds(0)));
    // The fixed lane set remains owned by HostTransferProxy until the
    // complete DeviceTransferService shuts down.
    device_slots_ = nullptr;
    device_index_ = -1;
    initialized_ = false;
    return {};
}

}  // namespace mooncake
