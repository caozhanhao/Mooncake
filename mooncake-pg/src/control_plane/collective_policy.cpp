#include "control_plane/collective_policy.h"

#include <limits>

namespace mooncake {
namespace {

bool allActiveRanksPublishedV2(const GroupView& view) {
    if (view.status != GroupStatus::Ready) return false;
    for (const auto global_rank : view.rank_order) {
        const auto& member = view.members[global_rank];
        if (member.isActive() && !member.hasEndpointV2()) return false;
    }
    return true;
}

}  // namespace

bool hasRequiredCollectiveEndpoints(const GroupView& view,
                                    const GroupMember& member) {
    return member.hasEndpoint() && (view.collective_plans.allreduce_protocol !=
                                        AllReduceProtocol::Planned ||
                                    member.hasEndpointV2());
}

CollectivePlanSet CollectivePolicyBuilder::build(const GroupView& view) const {
    CollectivePlanSet plans;
    const bool endpoints_ready = allActiveRanksPublishedV2(view);
    const bool planned = view.collective_plans.allreduce_protocol ==
                             AllReduceProtocol::Planned ||
                         endpoints_ready;
    plans.allreduce_protocol =
        planned ? AllReduceProtocol::Planned : AllReduceProtocol::Legacy;
    if (endpoints_ready) {
        plans.allreduce_policies.push_back(AllReducePolicy{
            .max_message_bytes = std::numeric_limits<uint64_t>::max(),
        });
    }
    return plans;
}

}  // namespace mooncake
