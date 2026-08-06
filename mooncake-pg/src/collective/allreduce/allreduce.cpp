#include "collective/allreduce/allreduce.h"

#include <limits>
#include <utility>

#include "collective/buffer/collective_buffer_pool.h"
#include "collective/plan/device_plan.h"
#include "collective/runtime/collective_channels.h"
#include "collective/runtime/collective_submission.h"
#include "collective/runtime/control_block.cuh"
#include "collective/runtime/runtime.h"

namespace mooncake {
namespace {

constexpr uint64_t kStagingBytes = 8 * kCollectiveMiB;
constexpr uint64_t kProtocolBytes = 4096;
constexpr uint64_t kBufferAlignment = 2 * kCollectiveMiB;
constexpr uint64_t kLayoutAlignment = 64;

constexpr uint64_t alignUp(uint64_t value, uint64_t alignment) {
    const auto remainder = value % alignment;
    return remainder == 0 ? value : value + alignment - remainder;
}

constexpr uint64_t kProtocolOffset =
    alignUp(kStagingBytes, kLayoutAlignment);
constexpr uint64_t kAllReduceBufferBytes =
    alignUp(kProtocolOffset + kProtocolBytes, kLayoutAlignment);

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

PGResult<std::unique_ptr<AllReduce>> AllReduce::create(
    CollectiveBufferPool& buffer_pool, CollectiveChannels& channels,
    TransferEngine* engine, DeviceId device, std::string te_location) {
    PG_TRY(auto plan, DevicePlan<AllReducePlan>::create(device));
    return std::unique_ptr<AllReduce>(new AllReduce(
        buffer_pool, channels, engine, device, std::move(te_location),
        std::move(plan)));
}

AllReduce::AllReduce(CollectiveBufferPool& buffer_pool,
                     CollectiveChannels& channels, TransferEngine* engine,
                     DeviceId device, std::string te_location,
                     std::unique_ptr<DevicePlan<AllReducePlan>> plan)
    : buffer_pool_(buffer_pool),
      channels_(channels),
      engine_(engine),
      device_(device),
      te_location_(std::move(te_location)),
      plan_(std::move(plan)) {}

AllReduce::~AllReduce() noexcept = default;

bool AllReduce::supports(DataType datatype, ReduceOp op) const {
    return protocol_ == AllReduceProtocol::Planned &&
           supportsPlannedAllReduce(datatype, op);
}

PGResult<void> AllReduce::apply(const CollectivePlanSet& plans,
                                const ResolvedCollectiveView& view) {
    protocol_ = plans.allreduce_protocol;
    auto built = buildAllReducePlan(plans, view);
    if (!built.has_value()) {
        AllReducePlan invalid;
        invalid.view_epoch = view.epoch;
        invalid.self_participating = view.self_ordinal.has_value() ? 1 : 0;
        invalid.error_code =
            static_cast<int32_t>(CollectiveProtocolError::InvalidPlan);
        plan_->publishInvalid(invalid);
        return makePGError(std::move(built).error());
    }
    plan_->publish(built.value());
    return {};
}

PGResult<std::shared_ptr<CollectiveSubmission>>
AllReduce::prepareSubmission(uint64_t timeout_device_ticks) {
    // Do not advance the wire-visible channel order until local buffer
    // acquisition succeeds. The buffer lease rolls back automatically if the
    // exact channel is unavailable.
    PG_TRY(auto buffer,
           buffer_pool_.acquire(device_, kAllReduceBufferBytes,
                                kBufferAlignment, te_location_, engine_));
    PG_TRY(auto channel, channels_.acquire());
    const auto kernel_resources = CollectiveKernelResources{
        .buffer =
            CollectiveKernelBuffer{
                .base = buffer.base(),
                .arena_offset = buffer.offset(),
                .staging_offset = 0,
                .staging_bytes = kStagingBytes,
                .protocol_offset = kProtocolOffset,
                .protocol_bytes = kProtocolBytes,
            },
        .peer_signals =
            CollectivePeerSignals{
                .base = channel.peer_signals_base,
                .offset = channel.peer_signals_offset,
            },
        .control = channel.device_control,
        .host_command = channel.device_command,
        .timeout_device_ticks = timeout_device_ticks,
    };
    return std::make_shared<CollectiveSubmission>(
        std::move(channel), std::move(buffer), kernel_resources);
}

PGResult<void> AllReduce::submit(CollectiveRuntime& runtime,
                                 const AllReduceRequest& request,
                                 cudaStream_t stream,
                                 int32_t* failed_ranks_hint,
                                 size_t failed_ranks_hint_count) {
    PG_VALIDATE_STATE(protocol_ == AllReduceProtocol::Planned,
                      "planned AllReduce is not enabled for this group");
    PG_TRY(auto plan, plan_->devicePlan());
    return runtime.submit(
        stream, failed_ranks_hint, failed_ranks_hint_count,
        [this, &runtime]() {
            return prepareSubmission(runtime.timeoutDeviceTicks());
        },
        [request, plan, stream](const CollectiveKernelArgs& common) {
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
        });
}

void AllReduce::retainPlanForProcessLifetime() {
    plan_->retainForProcessLifetime();
}

}  // namespace mooncake
