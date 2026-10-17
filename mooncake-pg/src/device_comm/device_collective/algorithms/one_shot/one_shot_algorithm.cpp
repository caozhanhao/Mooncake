#include "device_comm/device_collective/algorithms/one_shot/one_shot_algorithm.h"

#include <utility>

#include <glog/logging.h>

#include "device_comm/device_collective/device_control_update.h"
#include "device_comm/device_collective/protocols/ll/ll_protocol.h"
#include "device_comm/device_transfer/transfer_service.h"
#include "pg_utils.h"

namespace mooncake {

namespace {

struct OneShotEndpointMetadata {
    uint64_t buffer_offset = 0;
    uint64_t buffer_size = 0;
};

}  // namespace

OneShotAlgorithm::OneShotAlgorithm(RegionSlice packets,
                                   OneShotBufferLayout layout, int device_index,
                                   const LLProtocol& ll)
    : AlgorithmProvider(DeviceCollectiveAlgorithm::OneShot, kAlgorithmKey,
                        kEndpointVersion, ll,
                        {OpType::AllReduce, OpType::AllGather}),
      packets_(std::move(packets)),
      layout_(layout),
      device_index_(device_index) {}

PGResult<std::unique_ptr<OneShotAlgorithm>> OneShotAlgorithm::create(
    DeviceTransferService& transfer_service, uint32_t max_group_size,
    CollectiveRuntimeBindings collective, const LLProtocol& ll) {
    PG_VALIDATE_ARG(max_group_size > 0 && max_group_size <= kMaxNumRanks,
                    "One-shot group capacity is outside the supported range");
    PG_VALIDATE_ARG(ll.state() && collective.transfer_handle &&
                        collective.control_mailbox &&
                        collective.invocation_state,
                    "One-shot device bindings are incomplete");
    const OneShotBufferLayout layout{max_group_size};
    PG_TRY(auto packets, transfer_service.allocatePeerAccessible(
                             layout.packetBytes(), alignof(LLPacket)));
    const int device_index = transfer_service.deviceIndex();
    auto algorithm = std::unique_ptr<OneShotAlgorithm>(
        new OneShotAlgorithm(std::move(packets), layout, device_index, ll));
    PG_TRY(auto device_guard, GpuDeviceGuard::create(device_index));
    PG_TRY_CUDA(
        cudaMemset(algorithm->packets_.addr(), 0, algorithm->packets_.size()));
    PG_TRY_CUDA(cudaMalloc(reinterpret_cast<void**>(&algorithm->state_),
                           sizeof(OneShotDeviceState)));
    const OneShotDeviceState initial_state{
        .plan = {},
        .collective = collective,
        .ll = ll.state(),
    };
    PG_TRY_CUDA(cudaMemcpy(algorithm->state_, &initial_state,
                           sizeof(OneShotDeviceState), cudaMemcpyHostToDevice));
    return algorithm;
}

OneShotAlgorithm::~OneShotAlgorithm() noexcept {
    if (!state_) return;
    auto device_guard = GpuDeviceGuard::create(device_index_);
    if (!device_guard.has_value()) {
        LOG(ERROR) << "Failed to select CUDA device while releasing one-shot "
                      "state: "
                   << device_guard.error().message;
        return;
    }
    const auto result = cudaFree(state_);
    state_ = nullptr;
    if (result != cudaSuccess) {
        LOG(ERROR) << "Failed to free one-shot device state: "
                   << cudaGetErrorString(result);
    }
}

AlgorithmEndpoint OneShotAlgorithm::localEndpoint() const {
    return encodeEndpoint(OneShotEndpointMetadata{
        .buffer_offset = packets_.offset(), .buffer_size = packets_.size()});
}

PGResult<void> OneShotAlgorithm::updatePlan(
    const ResolvedGroupView& view, ControlUpdateBuilder& builder) const {
    PG_TRY(auto plan, buildPlan(view));
    return appendPlanUpdate(builder, plan);
}

PGResult<void> OneShotAlgorithm::invalidatePlan(
    ControlUpdateBuilder& builder) const {
    return appendPlanUpdate(builder, OneShotPlan{});
}

PGResult<OneShotPlan> OneShotAlgorithm::buildPlan(
    const ResolvedGroupView& view) const {
    if (view.self_active_index < 0) return OneShotPlan{};
    const auto max_group_size = layout_.max_group_size;
    const auto index = static_cast<size_t>(view.self_active_index);
    PG_VALIDATE_STATE(index < view.participants.size() &&
                          view.participants.size() <= max_group_size &&
                          max_group_size <= kMaxNumRanks,
                      "One-shot participants are outside the group");
    OneShotPlan plan{
        .status = DevicePlanStatus::Ready,
        .view_epoch = view.epoch,
        .layout = layout_,
        .packets = static_cast<LLPacket*>(packets_.addr()),
        .self_rank = view.participants[index].in_group_rank,
        .self_active_index = static_cast<uint32_t>(index),
        .participant_count = static_cast<uint32_t>(view.participants.size()),
    };
    uint32_t remote_index = 0;
    for (size_t peer = 0; peer < view.participants.size(); ++peer) {
        if (peer == index) continue;
        const auto& participant = view.participants[peer];
        const auto* endpoint = findEndpoint(participant.algorithms);
        PG_VALIDATE_STATE(endpoint,
                          "Active peer has no compatible one_shot endpoint");
        PG_TRY(auto metadata,
               decodeEndpoint<OneShotEndpointMetadata>(*endpoint));
        PG_VALIDATE_STATE(
            metadata.buffer_offset % alignof(LLPacket) == 0 &&
                metadata.buffer_size == layout_.packetBytes() &&
                !addOverflows(metadata.buffer_offset, metadata.buffer_size),
            "One-shot peer packet endpoint is invalid");
        auto& remote = plan.remote_peers[remote_index++];
        remote = participant.asPeer();
        remote.workspace_offset = metadata.buffer_offset;
    }
    return plan;
}

PGResult<void> OneShotAlgorithm::appendPlanUpdate(
    ControlUpdateBuilder& builder, const OneShotPlan& plan) const {
    return builder.copyBytes(&state_->plan, &plan, sizeof(plan));
}

PGResult<void> OneShotAlgorithm::enqueueImpl(const AllReduceRequest& request,
                                             cudaStream_t stream) const {
    PG_TRY_CUDA(launchOneShotAllReduceKernel(request, state_, stream));
    return {};
}

PGResult<void> OneShotAlgorithm::enqueueImpl(const AllGatherRequest& request,
                                             cudaStream_t stream) const {
    PG_TRY_CUDA(launchOneShotAllGatherKernel(request, state_, stream));
    return {};
}

}  // namespace mooncake
