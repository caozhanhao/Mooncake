#include "collective/allreduce/allreduce.h"

#include <limits>

#include "collective/runtime/control_block.cuh"

namespace mooncake {
namespace {

PGResult<FlatRingPlan> buildFlatRingPlan(const ResolvedCollectiveView& view) {
    FlatRingPlan ring;
    const auto participant_count =
        static_cast<uint32_t>(view.active_order.size());
    const auto self_ordinal = *view.self_ordinal;
    ring.participant_count = participant_count;
    ring.self_ordinal = self_ordinal;
    if (participant_count == 1) return ring;

    const auto predecessor =
        view.active_order[(self_ordinal + participant_count - 1) %
                          participant_count];
    const auto successor =
        view.active_order[(self_ordinal + 1) % participant_count];
    const auto* predecessor_route = view.findRoute(predecessor);
    if (!predecessor_route) {
        return makePGError(PGErrorCode::InvalidState,
                           "collective peer route is unavailable");
    }
    const auto* successor_route = predecessor_route;
    if (successor != predecessor) {
        successor_route = view.findRoute(successor);
        if (!successor_route) {
            return makePGError(PGErrorCode::InvalidState,
                               "collective peer route is unavailable");
        }
    }

    ring.predecessor = *predecessor_route;
    ring.successor = *successor_route;
    return ring;
}

}  // namespace

bool supportsPlannedAllReduce(DataType datatype, ReduceOp op) {
    return op == ReduceOp::Sum &&
           (datatype == DataType::Float16 || datatype == DataType::Bfloat16 ||
            datatype == DataType::Float32);
}

PGResult<AllReduceRequest> makeAllReduceRequest(const void* input, void* output,
                                                size_t element_count,
                                                DataType datatype,
                                                ReduceOp op) {
    PG_VALIDATE_ARG(supportsPlannedAllReduce(datatype, op),
                    "planned AllReduce signature is not supported");
    const uint64_t bytes_per_element = elementSize(datatype);
    PG_VALIDATE_ARG(element_count <= std::numeric_limits<uint64_t>::max() /
                                         bytes_per_element,
                    "planned AllReduce element count overflows uint64_t");
    return AllReduceRequest{
        .input = input,
        .output = output,
        .element_count = static_cast<uint64_t>(element_count),
        .datatype = datatype,
    };
}

PGResult<AllReducePlan> buildAllReducePlan(const CollectivePlanSet& plans,
                                           const ResolvedCollectiveView& view) {
    AllReducePlan plan;
    plan.view_epoch = view.epoch;
    plan.self_participating = view.self_ordinal.has_value() ? 1 : 0;

    // Host admission rejects an inactive rank, while graph replay has no host
    // participation. The published local identity keeps replay independent of
    // the membership used when the graph was captured.
    if (!plan.self_participating) return plan;

    if (plans.allreduce_protocol == AllReduceProtocol::Legacy) {
        plan.error_code =
            static_cast<int32_t>(CollectiveProtocolError::Unsupported);
        return plan;
    }
    if (plans.allreduce_policies.empty()) {
        return makePGError(PGErrorCode::InvalidState,
                           "planned AllReduce has no executable policy");
    }

    plan.bucket_count = static_cast<uint32_t>(plans.allreduce_policies.size());
    PG_TRY(auto ring, buildFlatRingPlan(view));
    for (uint32_t index = 0; index < plan.bucket_count; ++index) {
        const auto& policy = plans.allreduce_policies[index];
        auto& bucket = plan.buckets[index];
        bucket.max_message_bytes = policy.max_message_bytes;
        bucket.flat_ring = ring;
    }
    return plan;
}

void launchAllReduce(const AllReduceRequest& request, const AllReducePlan* plan,
                     const CollectiveKernelArgs& common, cudaStream_t stream) {
    launchAllReduceExecutor(
        AllReduceKernelArgs{
            .input = request.input,
            .output = request.output,
            .common = common,
            .plan = plan,
            .element_count = request.element_count,
            .datatype = request.datatype,
        },
        stream);
}

}  // namespace mooncake
