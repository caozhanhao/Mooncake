#ifndef MOONCAKE_PG_COLLECTIVE_RESOLVED_COLLECTIVE_VIEW_H
#define MOONCAKE_PG_COLLECTIVE_RESOLVED_COLLECTIVE_VIEW_H

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "collective/transport/peer_route.h"

namespace mooncake {

class DeviceLinkManager;
class LinkManager;
struct GroupView;

// Agent-local projection of one Coordinator-authoritative group view.
// Collective builders assign algorithm roles from this value without seeing
// global ranks or resolving transport resources themselves. Link managers own
// every resource referenced by peer_routes.
struct ResolvedCollectiveView {
    const PeerRoute* findRoute(InGroupRank peer) const {
        const auto& route = peer_routes[static_cast<size_t>(peer)];
        return route.has_value() ? &*route : nullptr;
    }

    uint64_t epoch = 0;
    std::vector<InGroupRank> active_order;
    std::optional<uint32_t> self_ordinal;
    std::vector<std::optional<PeerRoute>> peer_routes;
};

ResolvedCollectiveView resolveCollectiveView(const GroupView& view,
                                             InGroupRank self, DeviceId device,
                                             DeviceLinkManager& device_links,
                                             LinkManager& host_links);

}  // namespace mooncake

#endif  // MOONCAKE_PG_COLLECTIVE_RESOLVED_COLLECTIVE_VIEW_H
