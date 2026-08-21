#include "device_comm/device_collective/protocols/flat_ring_all_reduce/flat_ring_all_reduce.h"

#include <algorithm>
#include <array>
#include <exception>
#include <new>
#include <utility>
#include <vector>

#include <glog/logging.h>

#include "device_comm/device_collective/device_collective_resources.h"
#include "device_comm/device_transfer/transfer_service.h"

namespace mooncake {
namespace {

cudaError_t setPlanStatusAsync(DevicePlanStatus* device_status,
                               DevicePlanStatus status, cudaStream_t stream) {
    static_assert(sizeof(DevicePlanStatus) == 1);
    return cudaMemsetAsync(device_status, static_cast<int>(status),
                           sizeof(status), stream);
}

class StreamSyncGuard {
   public:
    explicit StreamSyncGuard(GpuStream& stream) : stream_(stream) {}

    ~StreamSyncGuard() noexcept {
        if (!active_) return;
        try {
            auto result = stream_.synchronize();
            if (!result.has_value()) {
                LOG(ERROR) << "Failed to drain Flat Ring control stream: "
                           << result.error().message;
            }
        } catch (const std::exception& error) {
            LOG(ERROR) << "Failed to drain Flat Ring control stream: "
                       << error.what();
        } catch (...) {
            LOG(ERROR) << "Failed to drain Flat Ring control stream";
        }
    }

    PGResult<void> finish() {
        auto result = stream_.synchronize();
        active_ = false;
        return result;
    }

   private:
    GpuStream& stream_;
    bool active_ = true;
};

FlatRingAllReducePlan bindProtocolMemory(
    FlatRingAllReducePlan plan, const DeviceCollectiveWorkspace& workspace,
    const DeviceCollectiveResources& resources,
    const DeviceArenaSlice* staging = nullptr) noexcept {
    const auto& buffer = workspace.buffer();
    plan.buffer_offset = buffer.offset();
    plan.buffer_size = kFlatRingBufferBytes;
    if (staging) {
        plan.staging_offset = staging->offset();
        plan.staging_size = kFlatRingStagingBytes;
    }
    plan.signal_offset = resources.signalOffset();
    plan.signal_count = resources.signalCount();
    return plan;
}

}  // namespace

struct FlatRingAllReduceProtocol::HostState {
    FlatRingAllReducePlan plan;
    const std::array<uint64_t, kMaxDeviceCollectiveChannels *
                                   kFlatRingPipelineSlots>
        initial_step_sequences = [] {
            std::array<uint64_t, kMaxDeviceCollectiveChannels *
                                     kFlatRingPipelineSlots>
                sequences{};
            sequences.fill(1);
            return sequences;
        }();
    const std::array<uint64_t, kMaxDeviceCollectiveChannels>
        initial_recv_buffer_ready_sequences = [] {
            std::array<uint64_t, kMaxDeviceCollectiveChannels> sequences{};
            sequences.fill(1);
            return sequences;
        }();
};

FlatRingAllReduceProtocol::FlatRingAllReduceProtocol(
    DeviceTransferService& transfer_service,
    DeviceCollectiveWorkspace& workspace,
    DeviceCollectiveResources& resources, GpuStream& control_stream,
    int device_index, InGroupRank self_rank, uint32_t max_group_size,
    FlatRingPersistentStateView state,
    FlatRingSignalLayout signal_layout) noexcept
    : transfer_service_(transfer_service),
      workspace_(workspace),
      resources_(resources),
      control_stream_(control_stream),
      device_index_(device_index),
      self_rank_(self_rank),
      max_group_size_(max_group_size),
      state_(state),
      signal_layout_(signal_layout) {}

PGResult<std::unique_ptr<FlatRingAllReduceProtocol>>
FlatRingAllReduceProtocol::create(
    DeviceTransferService& transfer_service,
    DeviceCollectiveWorkspace& workspace,
    DeviceCollectiveResources& resources, GpuStream& control_stream,
    int device_index, InGroupRank self_rank, uint32_t max_group_size) {
    PG_VALIDATE_ARG(max_group_size != 0,
                    "Flat Ring group capacity is zero");
    PG_VALIDATE_ARG(
        self_rank >= 0 && static_cast<uint32_t>(self_rank) < max_group_size,
        "Flat Ring self rank is outside the group");

    const auto state_layout = FlatRingPersistentStateLayout::make();
    const auto signal_layout = FlatRingSignalLayout::make(max_group_size);
    const auto common = resources.deviceView();
    PG_VALIDATE_ARG(workspace.buffer().addr(),
                    "Flat Ring buffer address is null");
    PG_VALIDATE_ARG(workspace.buffer().size() >= kFlatRingBufferBytes,
                    "Flat Ring buffer capacity is too small");
    PG_VALIDATE_ARG(resources.signals(),
                    "Flat Ring signal address is null");
    PG_VALIDATE_ARG(resources.signalCount() >= signal_layout.signal_count,
                    "Flat Ring signal capacity is too small");
    PG_VALIDATE_ARG(common.protocol_state_size >= state_layout.size,
                    "Flat Ring persistent-state capacity is too small");
    PG_VALIDATE_ARG(common.protocol_state,
                    "Flat Ring persistent-state address is null");

    auto protocol = std::unique_ptr<FlatRingAllReduceProtocol>(
        new FlatRingAllReduceProtocol(
            transfer_service, workspace, resources, control_stream,
            device_index, self_rank, max_group_size,
            state_layout.map(common.protocol_state), signal_layout));
    PG_TRY(protocol->initializeHostState());
    return protocol;
}

FlatRingAllReduceProtocol::~FlatRingAllReduceProtocol() noexcept {
    releaseHostState();
}

PGResult<void> FlatRingAllReduceProtocol::initializeHostState() {
    PG_TRY(auto device_guard, GpuDeviceGuard::create(device_index_));
    PG_TRY_CUDA(cudaHostAlloc(reinterpret_cast<void**>(&host_state_),
                              sizeof(HostState), cudaHostAllocPortable));
    std::construct_at(host_state_);
    return {};
}

void FlatRingAllReduceProtocol::releaseHostState() noexcept {
    ready_ = false;
    if (!host_state_) return;
    std::destroy_at(host_state_);
    const auto result = cudaFreeHost(host_state_);
    if (result != cudaSuccess) {
        LOG(ERROR) << "Failed to free Flat Ring host state: "
                   << cudaGetErrorString(result);
    }
    host_state_ = nullptr;
}

uint32_t FlatRingAllReduceProtocol::chooseChannelCount(
    size_t size, uint32_t participant_count) {
    PG_ASSERT(participant_count != 0,
              "cannot choose Flat Ring channels without participants");

    uint32_t channel_count = kMaxDeviceCollectiveChannels;
    while (channel_count > 1 &&
           size / participant_count / channel_count <
               kMinBytesPerChannelStep) {
        channel_count /= 2;
    }
    return channel_count;
}

PGResult<void> FlatRingAllReduceProtocol::publish(
    FlatRingAllReducePlan plan) {
    host_state_->plan = plan;
    ready_ = false;

    PG_TRY(auto device_guard, GpuDeviceGuard::create(device_index_));
    StreamSyncGuard sync_guard(control_stream_);

    // Make Graph replay reject the old image before changing any state it
    // references. The protocol is quiescent when its signal prefix is reset.
    PG_TRY_CUDA(setPlanStatusAsync(&state_.plan->status,
                                   DevicePlanStatus::Unavailable,
                                   control_stream_.get()));
    PG_TRY_CUDA(cudaMemsetAsync(
        resources_.signals(), 0,
        static_cast<size_t>(signal_layout_.signal_count) * sizeof(uint64_t),
        control_stream_.get()));
    PG_TRY_CUDA(cudaMemcpyAsync(
        state_.next_step_sequences,
        host_state_->initial_step_sequences.data(),
        host_state_->initial_step_sequences.size() * sizeof(uint64_t),
        cudaMemcpyHostToDevice, control_stream_.get()));
    PG_TRY_CUDA(cudaMemcpyAsync(
        state_.next_recv_buffer_ready_sequences,
        host_state_->initial_recv_buffer_ready_sequences.data(),
        host_state_->initial_recv_buffer_ready_sequences.size() *
            sizeof(uint64_t),
        cudaMemcpyHostToDevice, control_stream_.get()));
    PG_TRY_CUDA(cudaMemcpyAsync(
        &state_.plan->plan, &host_state_->plan, sizeof(host_state_->plan),
        cudaMemcpyHostToDevice, control_stream_.get()));
    PG_TRY_CUDA(setPlanStatusAsync(&state_.plan->status,
                                   DevicePlanStatus::Ready,
                                   control_stream_.get()));
    PG_TRY(sync_guard.finish());
    ready_ = true;
    return {};
}

PGResult<void> FlatRingAllReduceProtocol::invalidate() {
    ready_ = false;
    PG_TRY(auto device_guard, GpuDeviceGuard::create(device_index_));
    StreamSyncGuard sync_guard(control_stream_);
    PG_TRY_CUDA(setPlanStatusAsync(&state_.plan->status,
                                   DevicePlanStatus::Unavailable,
                                   control_stream_.get()));
    PG_TRY(sync_guard.finish());
    return {};
}

PGResult<void> FlatRingAllReduceProtocol::useLocalOnly() {
    return publish(bindProtocolMemory(
        FlatRingAllReducePlan{
            .self_rank = self_rank_,
            .self_active_index = 0,
            .participant_count = 1,
            .predecessor = {.in_group_rank = self_rank_},
            .successor = {.in_group_rank = self_rank_},
        },
        workspace_, resources_));
}

PGResult<void> FlatRingAllReduceProtocol::applyGroupView(
    const GroupView& view) {
    PG_ASSERT(view.max_group_size == static_cast<int32_t>(max_group_size_),
              "Flat Ring group capacity changed");

    std::vector<InGroupRank> participants;
    participants.reserve(view.rank_order.size());
    for (InGroupRank in_group_rank = 0;
         static_cast<size_t>(in_group_rank) < view.rank_order.size();
         ++in_group_rank) {
        const auto global_rank = view.rank_order[in_group_rank];
        const auto& member = view.members[global_rank];
        if (!member.isActive()) continue;

        PG_ASSERT(member.endpoint && member.endpoint->collective_v2,
                  "active Flat Ring peer ", in_group_rank,
                  " has no device collective endpoint");
        const auto& endpoint = *member.endpoint->collective_v2;
        PG_ASSERT(deviceCollectiveEndpointSupports(
                      endpoint, kFlatRingBufferBytes,
                      signal_layout_.signal_count),
                  "active Flat Ring peer ", in_group_rank,
                  " has insufficient resources: buffer_offset=",
                  endpoint.buffer_offset, ", buffer_size=",
                  endpoint.buffer_size, ", signal_offset=",
                  endpoint.signal_offset, ", signal_capacity=",
                  endpoint.signal_capacity);
        participants.push_back(in_group_rank);
    }

    const auto self =
        std::find(participants.begin(), participants.end(), self_rank_);
    if (self == participants.end()) return invalidate();

    const auto active_index =
        static_cast<size_t>(std::distance(participants.begin(), self));
    const auto participant_count = participants.size();
    const auto predecessor =
        participants[(active_index + participant_count - 1) %
                     participant_count];
    const auto successor = participants[(active_index + 1) % participant_count];
    auto targetFor = [&](InGroupRank in_group_rank) {
        const auto global_rank = view.rank_order[in_group_rank];
        const auto& endpoint =
            *view.members[global_rank].endpoint->collective_v2;
        return FlatRingPeerTarget{
            .global_rank = global_rank,
            .in_group_rank = in_group_rank,
            .buffer_offset = endpoint.buffer_offset,
            .signal_offset = endpoint.signal_offset,
        };
    };

    const auto predecessor_target = targetFor(predecessor);
    const auto successor_target = targetFor(successor);
    const DeviceArenaSlice* staging = nullptr;
    if (participant_count > 1) {
        PG_TRY(auto route_kind,
               transfer_service_.routeKind(successor_target.global_rank));
        switch (route_kind) {
            case DeviceRouteKind::P2p:
                break;
            case DeviceRouteKind::HostProxy:
                PG_TRY(staging, workspace_.ensureStaging());
                break;
            case DeviceRouteKind::Unreachable:
                return makePGError(
                    PGErrorCode::InvalidState,
                    "Flat Ring successor has no device transfer route");
            default:
                return makePGError(
                    PGErrorCode::NotSupported,
                    "Flat Ring successor route has no payload-write policy");
        }
    }

    FlatRingAllReducePlan plan{
        .self_rank = self_rank_,
        .self_active_index = static_cast<int32_t>(active_index),
        .participant_count = static_cast<uint32_t>(participant_count),
        .predecessor = predecessor_target,
        .successor = successor_target,
    };
    return publish(
        bindProtocolMemory(plan, workspace_, resources_, staging));
}

bool FlatRingAllReduceProtocol::ready() const noexcept { return ready_; }

FlatRingAllReduceDeviceResources
FlatRingAllReduceProtocol::deviceResources() const noexcept {
    return FlatRingAllReduceDeviceResources{
        .common = resources_.deviceView(),
        .state = state_,
        .signal_layout = signal_layout_,
    };
}

PGResult<void> FlatRingAllReduceProtocol::enqueue(
    const void* send_buffer, void* recv_buffer, size_t count, DataType datatype,
    ReduceOp op, cudaStream_t stream, int32_t* failed_ranks_hint) const {
    PG_VALIDATE_STATE(ready_, "Flat Ring AllReduce Plan is not ready");
    const uint32_t participant_count = host_state_->plan.participant_count;
    const size_t buffer_size = count * elementSize(datatype);
    const uint32_t channel_count =
        chooseChannelCount(buffer_size, participant_count);
    const FlatRingAllReduceKernelArgs request{
        .send_buffer = send_buffer,
        .recv_buffer = recv_buffer,
        .count = static_cast<uint64_t>(count),
        .datatype = datatype,
        .op = op,
        .failed_ranks_hint = failed_ranks_hint,
    };
    PG_TRY_CUDA(launchFlatRingAllReduceKernel(
        request, deviceResources(), channel_count, stream));
    return {};
}

}  // namespace mooncake
