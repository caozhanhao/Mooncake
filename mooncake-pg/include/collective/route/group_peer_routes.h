#ifndef MOONCAKE_PG_COLLECTIVE_ROUTE_GROUP_PEER_ROUTES_H
#define MOONCAKE_PG_COLLECTIVE_ROUTE_GROUP_PEER_ROUTES_H

#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

#include "collective/plan/active_group_ranks.h"
#include "collective/transport/peer_route.h"

namespace mooncake {

class DeviceLinkManager;
class LinkManager;
struct GroupView;

namespace device {
class P2pPeerMapping;
class RdmaTransport;
}  // namespace device

// Immutable routes selected for one local device and one authoritative group
// view. The table owns the process-level device resources referenced by its
// kernel-facing PeerRoute values; collective algorithms assign peer roles.
class GroupPeerRoutes {
   public:
    const PeerRoute* find(InGroupRank peer_in_group_rank) const {
        const auto& route = routes_[static_cast<size_t>(peer_in_group_rank)];
        return route.has_value() ? &*route : nullptr;
    }

   private:
    friend GroupPeerRoutes buildGroupPeerRoutes(
        const GroupView& view, const ActiveGroupRanks& active_ranks,
        InGroupRank self_in_group_rank, DeviceId device,
        DeviceLinkManager& device_links, LinkManager& host_links);

    std::vector<std::optional<PeerRoute>> routes_;
    std::vector<std::shared_ptr<device::P2pPeerMapping>> p2p_mappings_;
    std::shared_ptr<device::RdmaTransport> rdma_transport_;
};

GroupPeerRoutes buildGroupPeerRoutes(const GroupView& view,
                                     const ActiveGroupRanks& active_ranks,
                                     InGroupRank self_in_group_rank,
                                     DeviceId device,
                                     DeviceLinkManager& device_links,
                                     LinkManager& host_links);

}  // namespace mooncake

#endif  // MOONCAKE_PG_COLLECTIVE_ROUTE_GROUP_PEER_ROUTES_H
