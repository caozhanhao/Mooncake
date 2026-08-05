#ifndef MOONCAKE_PG_CONTROL_PLANE_COLLECTIVE_POLICY_H
#define MOONCAKE_PG_CONTROL_PLANE_COLLECTIVE_POLICY_H

#include "control_plane/control_types.h"

namespace mooncake {

// Activation must preserve the wire protocol already selected for the group.
bool hasRequiredCollectiveEndpoints(const GroupView& view,
                                    const GroupMember& member);

// Coordinator-side policy boundary. It emits one canonical algorithm policy
// per group and size bucket, never a per-rank executable plan.
class CollectivePolicyBuilder {
   public:
    CollectivePlanSet build(const GroupView& view) const;
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_CONTROL_PLANE_COLLECTIVE_POLICY_H
