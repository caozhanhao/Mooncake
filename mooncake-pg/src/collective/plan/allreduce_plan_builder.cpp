#include "collective/plan/allreduce_plan_builder.h"

#include <memory>
#include <utility>

#include "collective/runtime/control_block.cuh"

namespace mooncake {

size_t AllReducePlanBuilder::kernelPlanBytes() const {
    return sizeof(AllReduceKernelPlan);
}

PGResult<void> AllReducePlanBuilder::buildAlgorithm(
    const FlatRingPlan&, const ActiveGroupRanks& active_ranks,
    const GroupPeerRoutes& peer_routes, AllReduceBucketKernelPlan& bucket,
    InGroupRank& unavailable_peer) const {
    bucket.algorithm = AllReduceAlgorithm::FlatRing;
    const auto participant_count =
        static_cast<uint32_t>(active_ranks.ordered.size());
    const auto self_ordinal = *active_ranks.self_ordinal;
    bucket.flat_ring.participant_count = participant_count;
    bucket.flat_ring.self_ordinal = self_ordinal;
    if (participant_count == 1) return {};

    const auto predecessor =
        active_ranks.ordered[(self_ordinal + participant_count - 1) %
                             participant_count];
    const auto successor =
        active_ranks.ordered[(self_ordinal + 1) % participant_count];
    unavailable_peer = predecessor;
    const auto* predecessor_route = peer_routes.find(predecessor);
    if (!predecessor_route) {
        return makePGError(PGErrorCode::InvalidState,
                           "collective peer route is unavailable");
    }
    const auto* successor_route = predecessor_route;
    if (successor != predecessor) {
        unavailable_peer = successor;
        successor_route = peer_routes.find(successor);
        if (!successor_route) {
            return makePGError(PGErrorCode::InvalidState,
                               "collective peer route is unavailable");
        }
    }

    bucket.flat_ring.predecessor = *predecessor_route;
    bucket.flat_ring.successor = *successor_route;
    return {};
}

PGResult<void> AllReducePlanBuilder::buildAlgorithm(
    const HierarchicalPlan&, const ActiveGroupRanks&, const GroupPeerRoutes&,
    AllReduceBucketKernelPlan& bucket, InGroupRank&) const {
    bucket.algorithm = AllReduceAlgorithm::Hierarchical;
    return {};
}

CollectivePlanBuildResult AllReducePlanBuilder::build(
    uint64_t view_epoch, const CollectivePlanSet& plans,
    const ActiveGroupRanks& active_ranks,
    const GroupPeerRoutes& peer_routes) const {
    auto kernel_plan = std::make_shared<AllReduceKernelPlan>();
    kernel_plan->view_epoch = view_epoch;

    kernel_plan->self_participating =
        active_ranks.self_ordinal.has_value() ? 1 : 0;
    // Host admission rejects a new collective on an inactive rank, but CUDA
    // Graph replay has no host participation. Publishing an explicit local
    // identity is what lets a captured graph survive self deactivation without
    // executing another rank's ordinal or touching collective transport.
    if (!kernel_plan->self_participating) {
        return ReadyCollectivePlan{kernel_plan};
    }

    if (plans.allreduce_protocol == AllReduceProtocol::Legacy) {
        kernel_plan->error_code =
            static_cast<int32_t>(CollectiveProtocolError::Unsupported);
        return ReadyCollectivePlan{kernel_plan};
    }
    if (plans.allreduce_plans.empty()) {
        kernel_plan->error_code =
            static_cast<int32_t>(CollectiveProtocolError::InvalidPlan);
        return FailedCollectivePlan{
            kernel_plan,
            PGError{PGErrorCode::InvalidState,
                    "planned AllReduce has no executable policy"},
        };
    }

    kernel_plan->bucket_count =
        static_cast<uint32_t>(plans.allreduce_plans.size());
    for (uint32_t index = 0; index < kernel_plan->bucket_count; ++index) {
        const auto& logical = plans.allreduce_plans[index];
        auto& bucket = kernel_plan->buckets[index];
        bucket.max_message_bytes = logical.max_message_bytes;
        InGroupRank unavailable_peer = -1;
        auto algorithm_plan = std::visit(
            [&](const auto& algorithm) {
                return buildAlgorithm(algorithm, active_ranks, peer_routes,
                                      bucket, unavailable_peer);
            },
            logical.algorithm);
        if (!algorithm_plan.has_value()) {
            kernel_plan->error_code =
                static_cast<int32_t>(CollectiveProtocolError::InvalidPlan);
            kernel_plan->failed_peer = unavailable_peer;
            return FailedCollectivePlan{kernel_plan,
                                        std::move(algorithm_plan).error()};
        }
    }
    return ReadyCollectivePlan{kernel_plan};
}

}  // namespace mooncake
