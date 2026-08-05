#ifndef MOONCAKE_PG_COLLECTIVE_BINDING_GROUP_BINDING_H
#define MOONCAKE_PG_COLLECTIVE_BINDING_GROUP_BINDING_H

#include <cstdint>
#include <memory>
#include <vector>

#include "collective/transport/peer_binding.h"
#include "error_types.h"

namespace mooncake {

class DeviceLinkManager;
class LinkManager;
struct GroupView;

struct GroupParticipantOrder {
    std::vector<InGroupRank> active_ranks;
    uint32_t self_ordinal = 0;
    bool self_participating = false;
};

GroupParticipantOrder deriveGroupParticipantOrder(
    const GroupView& view, InGroupRank self_in_group_rank);

struct ResolvedCollectivePeer {
    CollectivePeerBinding binding;
    std::shared_ptr<void> keepalive;
};

// This is the only data-plane boundary that resolves GlobalRank and
// process-level link resources. Executors receive the returned InGroupRank
// binding and never inspect GroupView.
class GroupPeerResolver {
   public:
    GroupPeerResolver(DeviceLinkManager* device_links, LinkManager* host_links,
                      DeviceId device)
        : device_links_(device_links),
          host_links_(host_links),
          device_(device) {}

    PGResult<ResolvedCollectivePeer> resolve(
        const GroupView& view, InGroupRank peer_in_group_rank) const;

   private:
    DeviceLinkManager* device_links_ = nullptr;
    LinkManager* host_links_ = nullptr;
    DeviceId device_ = kInvalidDeviceId;
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_COLLECTIVE_BINDING_GROUP_BINDING_H
