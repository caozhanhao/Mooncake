#include "device_comm/device_transfer/routes/p2p_route/p2p_route.h"

#include <cstring>

#include <transport/device/device_transport.h>

#include "gpu_runtime.h"

namespace mooncake {
namespace {

std::vector<uint8_t> encodeHandle(const std::vector<int32_t>& handle) {
    std::vector<uint8_t> metadata(handle.size() * sizeof(int32_t));
    if (!metadata.empty()) {
        std::memcpy(metadata.data(), handle.data(), metadata.size());
    }
    return metadata;
}

PGResult<std::vector<int32_t>> decodeHandle(const RouteEndpoint& endpoint) {
    PG_VALIDATE_ARG(endpoint.route_key == P2pRoute::kRouteKey,
                    "P2P route endpoint key does not match");
    PG_VALIDATE_ARG(endpoint.version == P2pRoute::kEndpointVersion,
                    "unsupported P2P route endpoint version");
    PG_VALIDATE_ARG(!endpoint.metadata.empty() &&
                        endpoint.metadata.size() % sizeof(int32_t) == 0,
                    "P2P route endpoint metadata is invalid");

    std::vector<int32_t> handle(endpoint.metadata.size() / sizeof(int32_t));
    std::memcpy(handle.data(), endpoint.metadata.data(),
                endpoint.metadata.size());
    return handle;
}

}  // namespace

P2pRoute::P2pRoute(device::P2pTransport& transport, void* local_region,
                   int device_index, uint32_t self_peer_index,
                   uint32_t peer_capacity)
    : transport_(transport),
      local_region_(local_region),
      device_index_(device_index),
      self_peer_index_(self_peer_index),
      peer_capacity_(peer_capacity) {}

std::string_view P2pRoute::routeKey() const noexcept { return kRouteKey; }

uint32_t P2pRoute::routeVersion() const noexcept { return kEndpointVersion; }

std::optional<RouteEndpoint> P2pRoute::localEndpoint() {
    const auto handle = localHandle();
    if (handle.empty()) return std::nullopt;
    return RouteEndpoint{
        .route_key = std::string(kRouteKey),
        .version = routeVersion(),
        .metadata = encodeHandle(handle),
    };
}

std::vector<int32_t> P2pRoute::localHandle() const {
    return transport_.exportIpcHandle(local_region_);
}

PGResult<std::vector<DevicePeerRoute>> P2pRoute::resolveRoutes(
    std::span<const std::optional<DeviceTransferEndpoint>> peers) {
    PG_VALIDATE_ARG(peers.size() == peer_capacity_,
                    "P2P route peer snapshot size does not match capacity");

    // Decode the complete snapshot before changing the imported mappings.
    std::vector<std::vector<int32_t>> handles(peer_capacity_);
    std::vector<int> active(peer_capacity_, 0);
    for (uint32_t peer = 0; peer < peer_capacity_; ++peer) {
        PG_TRY(auto endpoint, findEndpoint(peers[peer]));
        if (!endpoint) continue;
        PG_TRY(handles[peer], decodeHandle(*endpoint));
        active[peer] = 1;
    }

    PG_TRY(auto device_guard, GpuDeviceGuard::create(device_index_));
    transport_.importPeerHandles(local_region_, self_peer_index_,
                                 peer_capacity_, handles, active);

    std::vector<int32_t> available(peer_capacity_, 0);
    std::vector<void*> peer_bases(peer_capacity_, nullptr);
    PG_TRY_CUDA(cudaMemcpy(available.data(), transport_.availableTablePtr(),
                           available.size() * sizeof(int32_t),
                           cudaMemcpyDeviceToHost));
    PG_TRY_CUDA(cudaMemcpy(peer_bases.data(), transport_.peerPtrsTablePtr(),
                           peer_bases.size() * sizeof(void*),
                           cudaMemcpyDeviceToHost));

    std::vector<DevicePeerRoute> routes(peer_capacity_);
    for (uint32_t peer = 0; peer < peer_capacity_; ++peer) {
        if (!active[peer]) continue;

        uint64_t region_address = 0;
        if (peer == self_peer_index_) {
            region_address = reinterpret_cast<uint64_t>(local_region_);
        } else if (available[peer] && peer_bases[peer]) {
            region_address = reinterpret_cast<uint64_t>(peer_bases[peer]);
        }
        if (region_address == 0) continue;

        auto& route = routes[peer];
        route.kind = DeviceRouteKind::P2p;
        route.region_size = peers[peer]->region_size;
        route.p2p.region_addr = region_address;
    }
    return routes;
}

}  // namespace mooncake
