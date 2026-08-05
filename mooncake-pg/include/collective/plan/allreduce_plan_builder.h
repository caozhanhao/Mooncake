#ifndef MOONCAKE_PG_COLLECTIVE_PLAN_ALLREDUCE_PLAN_BUILDER_H
#define MOONCAKE_PG_COLLECTIVE_PLAN_ALLREDUCE_PLAN_BUILDER_H

#include "collective/executor/allreduce_kernel_plan.cuh"
#include "collective/plan/active_group_ranks.h"
#include "collective/plan/collective_plan_registry.h"
#include "collective/plan/logical_plan.h"
#include "collective/route/group_peer_routes.h"

namespace mooncake {

class AllReducePlanBuilder final : public CollectivePlanBuilder {
   public:
    size_t kernelPlanBytes() const override;
    CollectivePlanBuildResult build(
        uint64_t view_epoch, const CollectivePlanSet& plans,
        const ActiveGroupRanks& active_ranks,
        const GroupPeerRoutes& peer_routes) const override;

   private:
    PGResult<void> buildAlgorithm(const FlatRingPlan& plan,
                                  const ActiveGroupRanks& active_ranks,
                                  const GroupPeerRoutes& peer_routes,
                                  AllReduceBucketKernelPlan& bucket,
                                  InGroupRank& unavailable_peer) const;
    PGResult<void> buildAlgorithm(const HierarchicalPlan& plan,
                                  const ActiveGroupRanks& active_ranks,
                                  const GroupPeerRoutes& peer_routes,
                                  AllReduceBucketKernelPlan& bucket,
                                  InGroupRank& unavailable_peer) const;
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_COLLECTIVE_PLAN_ALLREDUCE_PLAN_BUILDER_H
