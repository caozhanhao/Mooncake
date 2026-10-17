#include "device_comm/device_collective/protocols/simple/simple_protocol.h"

#include <string>
#include <utility>

#include <glog/logging.h>

#include "device_comm/device_collective/device_control_update.h"
#include "device_comm/device_collective/device_collective_workspace.h"
#include "device_comm/device_transfer/transfer_service.h"
#include "gpu_runtime.h"
#include "pg_utils.h"

namespace mooncake {

namespace {

struct SimpleEndpointMetadata {
    uint64_t signal_offset = 0;
    uint32_t signal_count = 0;
};

}  // namespace

SimpleProtocol::SimpleProtocol(RegionSlice signals,
                               SimplePipelineSignalLayout signal_layout,
                               DeviceTransferService& transfer_service,
                               InGroupRank self_rank) noexcept
    : ProtocolProvider(kProtocolKey, kEndpointVersion),
      signals_(std::move(signals)),
      signal_layout_(signal_layout),
      transfer_service_(transfer_service),
      self_rank_(self_rank),
      device_index_(transfer_service.deviceIndex()) {}

PGResult<std::unique_ptr<SimpleProtocol>> SimpleProtocol::create(
    DeviceTransferService& transfer_service, InGroupRank self_rank,
    uint32_t max_group_size) {
    PG_VALIDATE_ARG(max_group_size > 0 && max_group_size <= kMaxNumRanks,
                    "Simple group capacity is outside the supported range");
    PG_VALIDATE_ARG(
        self_rank >= 0 && static_cast<uint32_t>(self_rank) < max_group_size,
        "Simple self rank is outside the group");
    const auto signal_layout = SimplePipelineSignalLayout::make(max_group_size);
    const uint64_t signal_bytes =
        uint64_t{signal_layout.total_signal_count} * sizeof(uint64_t);
    PG_TRY(auto signals, transfer_service.allocatePeerAccessible(
                             signal_bytes, alignof(uint64_t)));
    auto protocol = std::unique_ptr<SimpleProtocol>(new SimpleProtocol(
        std::move(signals), signal_layout, transfer_service, self_rank));
    PG_TRY(auto device_guard, GpuDeviceGuard::create(protocol->device_index_));
    const size_t peer_progress_bytes = size_t{kMaxDeviceCollectiveChannels} *
                                       max_group_size *
                                       sizeof(SimplePeerProgress);
    PG_TRY_CUDA(cudaMalloc(reinterpret_cast<void**>(&protocol->peer_progress_),
                           peer_progress_bytes));
    PG_TRY_CUDA(cudaMemset(protocol->peer_progress_, 0, peer_progress_bytes));
    PG_TRY_CUDA(cudaMemset(protocol->signals_.addr(), 0, signal_bytes));
    PG_TRY_CUDA(cudaMalloc(reinterpret_cast<void**>(&protocol->state_),
                           sizeof(SimpleBindings)));
    const SimpleBindings initial_bindings{
        .transfer_handle = transfer_service.deviceHandle(),
        .self_rank = self_rank,
        .signals = static_cast<uint64_t*>(protocol->signals_.addr()),
        .signal_layout = signal_layout,
        .peer_progress = protocol->peer_progress_,
    };
    PG_TRY_CUDA(cudaMemcpy(protocol->state_, &initial_bindings,
                           sizeof(SimpleBindings), cudaMemcpyHostToDevice));
    return protocol;
}

SimpleProtocol::~SimpleProtocol() noexcept {
    if (!peer_progress_ && !state_) return;
    auto device_guard = GpuDeviceGuard::create(device_index_);
    if (!device_guard.has_value()) {
        LOG(ERROR) << "Failed to select CUDA device while releasing Simple "
                      "state: "
                   << device_guard.error().message;
        return;
    }
    for (void* allocation :
         {static_cast<void*>(state_), static_cast<void*>(peer_progress_)}) {
        if (!allocation) continue;
        const auto result = cudaFree(allocation);
        if (result != cudaSuccess) {
            LOG(ERROR) << "Failed to free Simple state: "
                       << cudaGetErrorString(result);
        }
    }
}

ProtocolEndpoint SimpleProtocol::localEndpoint() const {
    return encodeEndpoint(SimpleEndpointMetadata{
        .signal_offset = signals_.offset(),
        .signal_count = signal_layout_.total_signal_count});
}

PGResult<void> SimpleProtocol::updateState(
    const ResolvedGroupView& view, ControlUpdateBuilder& builder) const {
    PG_TRY(auto bindings, makeBindings(view));
    return appendStateUpdate(builder, bindings);
}

PGResult<void> SimpleProtocol::invalidateState(
    ControlUpdateBuilder& builder) const {
    return appendStateUpdate(builder, SimpleBindings{});
}

PGResult<SimpleBindings> SimpleProtocol::makeBindings(
    const ResolvedGroupView& view) const {
    SimpleBindings next{
        .transfer_handle = transfer_service_.deviceHandle(),
        .self_rank = self_rank_,
        .signals = static_cast<uint64_t*>(signals_.addr()),
        .signal_layout = signal_layout_,
        .peer_progress = peer_progress_,
    };
    if (view.self_active_index >= 0) {
        for (const auto& peer : view.participants) {
            if (peer.in_group_rank == next.self_rank) continue;
            PG_VALIDATE_STATE(peer.in_group_rank >= 0 &&
                                  static_cast<uint32_t>(peer.in_group_rank) <
                                      signal_layout_.max_group_size,
                              "Simple peer rank is outside the group");
            const auto* endpoint = findEndpoint(peer.protocols);
            PG_VALIDATE_STATE(endpoint,
                              "Active peer has no compatible simple endpoint");
            PG_TRY(auto metadata,
                   decodeEndpoint<SimpleEndpointMetadata>(*endpoint));
            const auto& workspace = peer.workspace;
            const uint64_t signal_bytes =
                uint64_t{metadata.signal_count} * sizeof(uint64_t);
            PG_VALIDATE_STATE(
                metadata.signal_count >= signal_layout_.total_signal_count &&
                    metadata.signal_offset % alignof(uint64_t) == 0 &&
                    !addOverflows(metadata.signal_offset, signal_bytes),
                "Simple peer signal endpoint is invalid");
            PG_VALIDATE_STATE(
                !rangesOverlap(workspace.buffer_offset, workspace.buffer_size,
                               metadata.signal_offset, signal_bytes),
                "Simple peer signals overlap the payload workspace");
            next.peer_bindings[peer.in_group_rank] = SimplePeer{
                .global_rank = peer.global_rank,
                .signal_offset = metadata.signal_offset,
            };
        }
    }
    return next;
}

PGResult<SimpleWorkspace> SimpleProtocol::bindWorkspace(
    DeviceCollectiveWorkspace& workspace, const CollectivePeer& send_peer,
    uint64_t bytes) const {
    PG_VALIDATE_STATE(bytes != 0 && bytes <= workspace.buffer().size(),
                      "Simple workspace capacity is invalid");
    SimpleWorkspace result{static_cast<char*>(workspace.buffer().addr()), bytes,
                           nullptr};
    PG_TRY(auto route, transfer_service_.routeInfo(send_peer.global_rank));
    PG_VALIDATE_STATE(route.reachable,
                      "transfer peer " + std::to_string(send_peer.global_rank) +
                          " has no device transfer route");
    if (!route.directly_addressable) {
        PG_TRY(auto staging, workspace.staging());
        result.staging = static_cast<char*>(staging->addr());
    }
    return result;
}

PGResult<void> SimpleProtocol::appendStateUpdate(
    ControlUpdateBuilder& builder, const SimpleBindings& bindings) const {
    PG_TRY(builder.copyBytes(state_, &bindings, sizeof(bindings)));
    PG_TRY(builder.fillU64(static_cast<uint64_t*>(signals_.addr()), 0,
                           signal_layout_.total_signal_count));
    return builder.fillBytes(peer_progress_, 0,
                             size_t{kMaxDeviceCollectiveChannels} *
                                 signal_layout_.max_group_size *
                                 sizeof(SimplePeerProgress));
}

}  // namespace mooncake
