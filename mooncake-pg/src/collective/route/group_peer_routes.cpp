#include "collective/route/group_peer_routes.h"

#include "control_plane/control_types.h"
#include "control_plane/device_link_manager.h"
#include "control_plane/link_manager.h"

namespace mooncake {

GroupPeerRoutes buildGroupPeerRoutes(const GroupView& view,
                                     const ActiveGroupRanks& active_ranks,
                                     InGroupRank self_in_group_rank,
                                     DeviceId device,
                                     DeviceLinkManager& device_links,
                                     LinkManager& host_links) {
    GroupPeerRoutes result;
    result.routes_.resize(view.rank_order.size());
    if (!active_ranks.self_ordinal.has_value()) return result;

    for (const auto peer_in_group_rank : active_ranks.ordered) {
        if (peer_in_group_rank == self_in_group_rank) continue;

        const auto peer_index = static_cast<size_t>(peer_in_group_rank);
        const auto global_peer = view.rank_order[peer_index];
        const auto& endpoint = view.members[global_peer].endpoint_v2;
        if (!endpoint.has_value()) continue;

        PeerRoute route;
        route.peer_in_group_rank = peer_in_group_rank;

        if (endpoint->device_p2p.has_value()) {
            auto* mapped_arena = device_links.resolveP2p(
                device, global_peer, endpoint->arena_generation);
            if (mapped_arena) {
                route.kind = PeerRouteKind::DevP2p;
                route.device_p2p.mapped_arena = mapped_arena;
                route.device_p2p.mapped_buffer =
                    static_cast<char*>(mapped_arena) +
                    endpoint->control_base_address -
                    endpoint->arena_base_address;
                result.routes_[peer_index] = route;
                continue;
            }
        }

        if (endpoint->device_rdma.has_value()) {
            auto local = device_links.resolveRdma(device, global_peer,
                                                  endpoint->arena_generation);
            if (local.has_value()) {
                route.kind = PeerRouteKind::DevRdma;
                route.device_rdma = DeviceRdmaPeerRoute{
                    .qp_contexts = local->qp_contexts,
                    .remote_keys = local->remote_keys,
                    .local_peer_index = local->local_peer_index,
                    .peer_index = local->peer_index,
                    .qps_per_peer = local->qps_per_peer,
                    .remote_arena_address = local->remote_arena_address,
                    .remote_buffer_address = local->remote_arena_address +
                                             endpoint->control_base_address -
                                             endpoint->arena_base_address,
                };
                result.routes_[peer_index] = route;
                continue;
            }
        }

        auto host = host_links.resolvePeerHandle(global_peer);
        if (!host.has_value()) continue;
        route.kind = PeerRouteKind::Host;
        route.host = HostPeerRoute{
            .link = *host,
            .remote_arena_address = endpoint->arena_base_address,
            .remote_buffer_address = endpoint->control_base_address,
        };
        result.routes_[peer_index] = route;
    }

    return result;
}

}  // namespace mooncake
