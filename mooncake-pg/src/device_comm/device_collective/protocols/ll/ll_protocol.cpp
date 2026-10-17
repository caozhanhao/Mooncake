#include "device_comm/device_collective/protocols/ll/ll_protocol.h"

#include <utility>

#include <glog/logging.h>

#include "device_comm/device_collective/device_control_update.h"
#include "device_comm/device_transfer/transfer_service.h"
#include "gpu_runtime.h"
#include "pg_utils.h"

namespace mooncake {

namespace {

struct LLEndpointMetadata {
    uint64_t signal_offset = 0;
    uint32_t signal_count = 0;
};

}  // namespace

LLProtocol::LLProtocol(RegionSlice signals, InGroupRank self_rank,
                       uint32_t max_group_size,
                       DeviceTransferService& transfer_service) noexcept
    : ProtocolProvider(kProtocolKey, kEndpointVersion),
      signals_(std::move(signals)),
      max_group_size_(max_group_size),
      transfer_service_(transfer_service),
      self_rank_(self_rank),
      device_index_(transfer_service.deviceIndex()) {}

PGResult<std::unique_ptr<LLProtocol>> LLProtocol::create(
    DeviceTransferService& transfer_service, InGroupRank self_rank,
    uint32_t max_group_size) {
    PG_VALIDATE_ARG(max_group_size > 0 && max_group_size <= kMaxNumRanks,
                    "LL group capacity is outside the supported range");
    PG_VALIDATE_ARG(
        self_rank >= 0 && static_cast<uint32_t>(self_rank) < max_group_size,
        "LL self rank is outside the group");
    const uint64_t signal_bytes = uint64_t{kMaxDeviceCollectiveChannels} *
                                  kLLBarrierCount * max_group_size *
                                  sizeof(uint64_t);
    PG_TRY(auto signals, transfer_service.allocatePeerAccessible(
                             signal_bytes, alignof(uint64_t)));
    auto protocol = std::unique_ptr<LLProtocol>(new LLProtocol(
        std::move(signals), self_rank, max_group_size, transfer_service));
    PG_TRY(auto device_guard, GpuDeviceGuard::create(protocol->device_index_));
    PG_TRY_CUDA(cudaMalloc(reinterpret_cast<void**>(&protocol->state_),
                           sizeof(LLState)));
    const LLState initial_state{
        .bindings =
            {
                .transfer_handle = transfer_service.deviceHandle(),
                .self_rank = self_rank,
                .max_group_size = max_group_size,
                .signals = static_cast<uint64_t*>(protocol->signals_.addr()),
            },
    };
    PG_TRY_CUDA(cudaMemcpy(protocol->state_, &initial_state, sizeof(LLState),
                           cudaMemcpyHostToDevice));
    PG_TRY_CUDA(cudaMemset(protocol->signals_.addr(), 0, signal_bytes));
    return protocol;
}

LLProtocol::~LLProtocol() noexcept {
    if (!state_) return;
    auto device_guard = GpuDeviceGuard::create(device_index_);
    if (!device_guard.has_value()) {
        LOG(ERROR) << "Failed to select CUDA device while releasing LL "
                      "progress: "
                   << device_guard.error().message;
        return;
    }
    const auto result = cudaFree(state_);
    if (result != cudaSuccess) {
        LOG(ERROR) << "Failed to free LL progress: "
                   << cudaGetErrorString(result);
    }
}

ProtocolEndpoint LLProtocol::localEndpoint() const {
    return encodeEndpoint(LLEndpointMetadata{
        .signal_offset = signals_.offset(),
        .signal_count =
            static_cast<uint32_t>(signals_.size() / sizeof(uint64_t))});
}

PGResult<void> LLProtocol::updateState(const ResolvedGroupView& view,
                                       ControlUpdateBuilder& builder) const {
    PG_TRY(auto bindings, makeBindings(view));
    return appendStateUpdate(builder, bindings);
}

PGResult<void> LLProtocol::invalidateState(
    ControlUpdateBuilder& builder) const {
    return appendStateUpdate(builder, LLBindings{});
}

PGResult<LLBindings> LLProtocol::makeBindings(
    const ResolvedGroupView& view) const {
    LLBindings next{
        .transfer_handle = transfer_service_.deviceHandle(),
        .self_rank = self_rank_,
        .max_group_size = max_group_size_,
        .signals = static_cast<uint64_t*>(signals_.addr()),
    };
    if (view.self_active_index >= 0) {
        for (const auto& peer : view.participants) {
            if (peer.in_group_rank == next.self_rank) continue;
            PG_VALIDATE_STATE(
                peer.in_group_rank >= 0 &&
                    static_cast<uint32_t>(peer.in_group_rank) < max_group_size_,
                "LL peer rank is outside the group");
            PG_TRY(auto route, transfer_service_.routeInfo(peer.global_rank));
            PG_VALIDATE_STATE(
                route.directly_addressable && route.native_atomics,
                "LL requires direct memory access with native atomics");
            const auto* endpoint = findEndpoint(peer.protocols);
            PG_VALIDATE_STATE(endpoint,
                              "Active peer has no compatible ll endpoint");
            PG_TRY(auto metadata,
                   decodeEndpoint<LLEndpointMetadata>(*endpoint));
            const auto& workspace = peer.workspace;
            const uint64_t signal_bytes =
                uint64_t{metadata.signal_count} * sizeof(uint64_t);
            PG_VALIDATE_STATE(
                metadata.signal_offset % alignof(uint64_t) == 0 &&
                    signal_bytes == signals_.size() &&
                    !addOverflows(metadata.signal_offset, signal_bytes),
                "LL peer signal endpoint is invalid");
            PG_VALIDATE_STATE(
                !rangesOverlap(workspace.buffer_offset, workspace.buffer_size,
                               metadata.signal_offset, signal_bytes),
                "LL peer signals overlap the shared payload workspace");
            next.peer_bindings[peer.in_group_rank] = LLPeer{
                .global_rank = peer.global_rank,
                .signal_offset = metadata.signal_offset,
            };
        }
    }
    return next;
}

PGResult<void> LLProtocol::appendStateUpdate(ControlUpdateBuilder& builder,
                                             const LLBindings& bindings) const {
    // Publication preserves GPU progress; LLControl owns its reset.
    return builder.copyBytes(&state_->bindings, &bindings, sizeof(bindings));
}

}  // namespace mooncake
