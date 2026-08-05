#include "collective/binding/allreduce_binding.h"

#include <memory>
#include <utility>

#include "collective/runtime/control_block.cuh"
#include "control_plane/control_types.h"

namespace mooncake {

size_t AllReduceBindingMaterializer::kernelPlanBytes() const {
    return sizeof(AllReduceKernelPlan);
}

PGResult<void> AllReduceBindingMaterializer::bind(
    const GroupView& view, const FlatRingPlan&,
    const GroupParticipantOrder& participants,
    AllReduceBucketKernelPlan& bucket,
    std::vector<std::shared_ptr<void>>& resources,
    InGroupRank& failed_peer) const {
    bucket.algorithm = AllReduceAlgorithm::FlatRing;
    const auto participant_count =
        static_cast<uint32_t>(participants.active_ranks.size());
    const auto self_ordinal = participants.self_ordinal;
    bucket.flat_ring.participant_count = participant_count;
    bucket.flat_ring.self_ordinal = self_ordinal;
    if (participant_count == 1) return {};

    const auto predecessor =
        participants.active_ranks[(self_ordinal + participant_count - 1) %
                                  participant_count];
    const auto successor =
        participants.active_ranks[(self_ordinal + 1) % participant_count];
    failed_peer = predecessor;
    PG_TRY(auto predecessor_edge, peer_resolver_.resolve(view, predecessor));
    auto successor_edge = predecessor_edge;
    if (successor != predecessor) {
        failed_peer = successor;
        PG_TRY(successor_edge, peer_resolver_.resolve(view, successor));
    }

    bucket.flat_ring.predecessor = predecessor_edge.binding;
    bucket.flat_ring.successor = successor_edge.binding;
    if (predecessor_edge.keepalive) {
        resources.push_back(predecessor_edge.keepalive);
    }
    if (successor_edge.keepalive != predecessor_edge.keepalive) {
        resources.push_back(successor_edge.keepalive);
    }
    return {};
}

PGResult<void> AllReduceBindingMaterializer::bind(
    const GroupView&, const HierarchicalPlan&, const GroupParticipantOrder&,
    AllReduceBucketKernelPlan& bucket, std::vector<std::shared_ptr<void>>&,
    InGroupRank&) const {
    bucket.algorithm = AllReduceAlgorithm::Hierarchical;
    return {};
}

MaterializedCollectiveBinding AllReduceBindingMaterializer::materialize(
    const GroupView& view) const {
    auto kernel_plan = std::make_shared<AllReduceKernelPlan>();
    kernel_plan->view_epoch = view.epoch;

    std::vector<std::shared_ptr<void>> resources;
    const auto participants =
        deriveGroupParticipantOrder(view, self_in_group_rank_);
    kernel_plan->self_participating = participants.self_participating ? 1 : 0;
    // Host admission rejects a new collective on an inactive rank, but CUDA
    // Graph replay has no host participation. Publishing an explicit local
    // identity is what lets a captured graph survive self deactivation without
    // executing another rank's ordinal or touching collective transport.
    if (!kernel_plan->self_participating) {
        return ReadyCollectiveBinding{kernel_plan, {}};
    }

    if (view.collective_plans.allreduce_protocol == AllReduceProtocol::Legacy) {
        kernel_plan->error_code =
            static_cast<int32_t>(CollectiveProtocolError::Unsupported);
        return ReadyCollectiveBinding{kernel_plan, {}};
    }
    if (view.collective_plans.allreduce_plans.empty()) {
        kernel_plan->error_code =
            static_cast<int32_t>(CollectiveProtocolError::InvalidBinding);
        return FailedCollectiveBinding{
            kernel_plan,
            PGError{PGErrorCode::InvalidState,
                    "planned AllReduce has no executable policy"},
        };
    }

    kernel_plan->bucket_count =
        static_cast<uint32_t>(view.collective_plans.allreduce_plans.size());
    for (uint32_t index = 0; index < kernel_plan->bucket_count; ++index) {
        const auto& logical = view.collective_plans.allreduce_plans[index];
        auto& bucket = kernel_plan->buckets[index];
        bucket.max_message_bytes = logical.max_message_bytes;
        InGroupRank failed_peer = -1;
        auto bound = std::visit(
            [&](const auto& algorithm) {
                return bind(view, algorithm, participants, bucket, resources,
                            failed_peer);
            },
            logical.algorithm);
        if (!bound.has_value()) {
            kernel_plan->error_code =
                static_cast<int32_t>(CollectiveProtocolError::InvalidBinding);
            kernel_plan->failed_peer = failed_peer;
            return FailedCollectiveBinding{kernel_plan,
                                           std::move(bound).error()};
        }
    }
    return ReadyCollectiveBinding{kernel_plan, std::move(resources)};
}

}  // namespace mooncake
