#ifndef MOONCAKE_PG_COLLECTIVE_BINDING_ALLREDUCE_BINDING_H
#define MOONCAKE_PG_COLLECTIVE_BINDING_ALLREDUCE_BINDING_H

#include "collective/binding/collective_binding.h"
#include "collective/binding/group_binding.h"
#include "collective/executor/allreduce_kernel_plan.cuh"
#include "collective/plan.h"

namespace mooncake {

class AllReduceBindingMaterializer final
    : public CollectiveBindingMaterializer {
   public:
    AllReduceBindingMaterializer(DeviceLinkManager* device_links,
                                 LinkManager* host_links, DeviceId device,
                                 InGroupRank self_in_group_rank)
        : peer_resolver_(device_links, host_links, device),
          self_in_group_rank_(self_in_group_rank) {}

    size_t kernelPlanBytes() const override;
    MaterializedCollectiveBinding materialize(
        const GroupView& view) const override;

   private:
    PGResult<void> bind(const GroupView& view, const FlatRingPlan& plan,
                        const GroupParticipantOrder& participants,
                        AllReduceBucketKernelPlan& bucket,
                        std::vector<std::shared_ptr<void>>& resources,
                        InGroupRank& failed_peer) const;
    PGResult<void> bind(const GroupView& view, const HierarchicalPlan& plan,
                        const GroupParticipantOrder& participants,
                        AllReduceBucketKernelPlan& bucket,
                        std::vector<std::shared_ptr<void>>& resources,
                        InGroupRank& failed_peer) const;

    GroupPeerResolver peer_resolver_;
    InGroupRank self_in_group_rank_ = -1;
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_COLLECTIVE_BINDING_ALLREDUCE_BINDING_H
