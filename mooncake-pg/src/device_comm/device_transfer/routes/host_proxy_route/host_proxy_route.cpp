#include <algorithm>
#include <chrono>
#include <string>

#include "device_comm/device_transfer/routes/host_proxy_route/host_proxy_route.h"
#include "device_comm/device_transfer/routes/host_proxy_route/host_transfer_proxy.h"

namespace mooncake {
namespace {

PGResult<std::string> decodeServerName(const RouteEndpoint& endpoint) {
    PG_VALIDATE_ARG(endpoint.route_key == HostProxyRoute::kRouteKey,
                    "host-proxy route endpoint key does not match");
    PG_VALIDATE_ARG(endpoint.version == HostProxyRoute::kEndpointVersion,
                    "unsupported host-proxy route endpoint version");
    PG_VALIDATE_ARG(!endpoint.metadata.empty(),
                    "host-proxy route endpoint metadata is empty");
    PG_VALIDATE_ARG(
        std::find(endpoint.metadata.begin(), endpoint.metadata.end(),
                  uint8_t{0}) == endpoint.metadata.end(),
        "host-proxy route endpoint contains a NUL byte");
    return std::string(endpoint.metadata.begin(), endpoint.metadata.end());
}

}  // namespace

HostProxyRoute::HostProxyRoute(TransferEngine& engine, uint32_t peer_capacity)
    : proxy_(std::make_unique<HostTransferProxy>(engine, peer_capacity)),
      peer_capacity_(peer_capacity) {}

HostProxyRoute::~HostProxyRoute() noexcept = default;

PGResult<void> HostProxyRoute::initialize(int device_index) {
    PG_VALIDATE_STATE(!shutdown_requested_, "HostProxyRoute is shutting down");
    if (initialized_) return {};
    PG_TRY(proxy_->start());
    PG_TRY(device_slots_, proxy_->initializeDevice(device_index));
    initialized_ = true;
    return {};
}

HostProxyCommandSlot* HostProxyRoute::deviceCommandSlots() const noexcept {
    return device_slots_;
}

std::string_view HostProxyRoute::routeKey() const noexcept { return kRouteKey; }

uint32_t HostProxyRoute::routeVersion() const noexcept {
    return kEndpointVersion;
}

std::optional<RouteEndpoint> HostProxyRoute::localEndpoint() {
    const auto server_name = proxy_->localServerName();
    if (server_name.empty()) return std::nullopt;
    return RouteEndpoint{
        .route_key = std::string(kRouteKey),
        .version = routeVersion(),
        .metadata =
            std::vector<uint8_t>(server_name.begin(), server_name.end()),
    };
}

PGResult<std::vector<DevicePeerRoute>> HostProxyRoute::resolveRoutes(
    std::span<const std::optional<DeviceTransferEndpoint>> peers) {
    PG_VALIDATE_STATE(initialized_, "HostProxyRoute is not initialized");
    PG_VALIDATE_ARG(
        peers.size() == peer_capacity_,
        "host-proxy route peer snapshot size does not match capacity");

    // Validate and decode the complete snapshot before replacing any server.
    std::vector<std::string> server_names(peer_capacity_);
    std::vector<DevicePeerRoute> routes(peer_capacity_);
    for (uint32_t peer = 0; peer < peer_capacity_; ++peer) {
        PG_TRY(auto endpoint, findEndpoint(peers[peer]));
        if (!endpoint) continue;

        PG_TRY(server_names[peer], decodeServerName(*endpoint));
        auto& route = routes[peer];
        route.kind = DeviceRouteKind::HostProxy;
        route.region_size = peers[peer]->region_size;
        route.host_proxy.region_addr = peers[peer]->region_address;
    }
    PG_TRY(proxy_->setPeerServerNames(server_names));
    return routes;
}

PGResult<void> HostProxyRoute::quiesce() {
    PG_VALIDATE_STATE(initialized_, "HostProxyRoute is not initialized");
    return proxy_->waitUntilIdle();
}

PGResult<void> HostProxyRoute::shutdown() {
    if (shutdown_requested_) return {};
    if (!initialized_) {
        PG_TRY(proxy_->shutdown());
        shutdown_requested_ = true;
        return {};
    }
    PG_TRY(proxy_->waitUntilIdle(std::chrono::milliseconds(0)));
    PG_TRY(proxy_->shutdown());
    shutdown_requested_ = true;
    device_slots_ = nullptr;
    initialized_ = false;
    return {};
}

}  // namespace mooncake
