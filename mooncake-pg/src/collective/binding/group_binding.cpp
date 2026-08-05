#include "collective/binding/group_binding.h"

#include "control_plane/control_types.h"
#include "control_plane/device_link_manager.h"
#include "control_plane/link_manager.h"

namespace mooncake {

GroupParticipantOrder deriveGroupParticipantOrder(
    const GroupView& view, InGroupRank self_in_group_rank) {
    GroupParticipantOrder result;
    result.active_ranks.reserve(view.rank_order.size());
    for (size_t index = 0; index < view.rank_order.size(); ++index) {
        const auto global_rank = view.rank_order[index];
        if (!view.members[global_rank].isActive()) continue;
        if (static_cast<InGroupRank>(index) == self_in_group_rank) {
            result.self_ordinal =
                static_cast<uint32_t>(result.active_ranks.size());
            result.self_participating = true;
        }
        result.active_ranks.push_back(static_cast<InGroupRank>(index));
    }
    return result;
}

PGResult<ResolvedCollectivePeer> GroupPeerResolver::resolve(
    const GroupView& view, InGroupRank peer_in_group_rank) const {
    const auto global_peer = view.rank_order[peer_in_group_rank];
    const auto& endpoint = *view.members[global_peer].endpoint_v2;

    ResolvedCollectivePeer result;
    result.binding.peer_in_group_rank = peer_in_group_rank;

    // This priority is evaluated once while materializing an authoritative
    // GroupView. The executor receives the selected route and never falls back
    // locally after launch.
    if (endpoint.device_p2p.has_value()) {
        auto local = device_links_->resolveP2p(device_, global_peer,
                                               endpoint.arena_generation);
        if (local.has_value()) {
            result.binding.route = CollectiveRoute::DevP2p;
            result.binding.device_p2p.mapped_arena = local->mapped_arena_base;
            result.binding.device_p2p.mapped_buffer =
                static_cast<char*>(local->mapped_arena_base) +
                endpoint.control_base_address - endpoint.arena_base_address;
            result.keepalive = std::move(local->mapping);
            return result;
        }
    }

    if (endpoint.device_rdma.has_value()) {
        auto local = device_links_->resolveRdma(device_, global_peer,
                                                endpoint.arena_generation);
        if (local.has_value()) {
            result.binding.route = CollectiveRoute::DevRdma;
            result.binding.device_rdma = DeviceRdmaPeerBinding{
                .qp_contexts = local->qp_contexts,
                .remote_keys = local->remote_keys,
                .local_peer_index = local->local_peer_index,
                .peer_index = local->peer_index,
                .qps_per_peer = local->qps_per_peer,
                .remote_arena_address = local->remote_arena_address,
                .remote_buffer_address = local->remote_arena_address +
                                         endpoint.control_base_address -
                                         endpoint.arena_base_address,
            };
            result.keepalive = std::move(local->keepalive);
            return result;
        }
    }

    auto host = host_links_->resolvePeerHandle(global_peer);
    if (host.has_value()) {
        result.binding.route = CollectiveRoute::Host;
        result.binding.host = HostPeerBinding{
            .link = *host,
            .remote_arena_address = endpoint.arena_base_address,
            .remote_buffer_address = endpoint.control_base_address,
        };
        return result;
    }

    return makePGError(PGErrorCode::InvalidState,
                       "collective peer has no available local route");
}

}  // namespace mooncake
