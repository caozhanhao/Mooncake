#include "collective/group_collective_engine.h"

#include <utility>

#include "collective/buffer/collective_buffer_pool.h"
#include "collective/allreduce/allreduce.h"
#include "collective/resolved_collective_view.h"
#include "collective/runtime/collective_channels.h"
#include "collective/runtime/runtime.h"
#include "collective/transport/host_transfer_executor.h"
#include "control_plane/control_types.h"
#include "control_plane/device_link_manager.h"
#include "control_plane/link_manager.h"

namespace mooncake {

PGResult<std::unique_ptr<GroupCollectiveEngine>> GroupCollectiveEngine::create(
    CollectiveBufferPool& buffer_pool,
    HostTransferExecutor& host_transfer_executor,
    DeviceLinkManager& device_links, LinkManager& host_links,
    TransferEngine* transfer_engine, DeviceId device,
    InGroupRank self_in_group_rank, std::string te_location,
    size_t collective_timeout_us,
    std::function<PGResult<void>(InGroupRank)> report_failure) {
    PG_TRY(host_transfer_executor.initialize(transfer_engine, &host_links));

    auto result =
        std::unique_ptr<GroupCollectiveEngine>(new GroupCollectiveEngine(
            device_links, host_links, device, self_in_group_rank));
    PG_TRY(result->channels_,
           CollectiveChannels::create(&buffer_pool, &host_transfer_executor,
                                      device, te_location, transfer_engine));

    const auto arena = buffer_pool.arena(device);
    device_links.bindCollectiveArena(arena);
    PG_TRY(result->allreduce_,
           AllReduce::create(buffer_pool, *result->channels_, transfer_engine,
                             device, te_location));
    PG_TRY(result->runtime_,
           CollectiveRuntime::create(device, collective_timeout_us,
                                     std::move(report_failure)));

    result->endpoint_ = GroupEndpointV2{
        .device = arena.device,
        .arena_generation = arena.generation,
        .arena_base_address =
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(arena.base)),
        .arena_bytes = arena.bytes,
        .control_base_address = static_cast<uint64_t>(
            reinterpret_cast<uintptr_t>(result->channels_->peerControlBase())),
        .control_layout = result->channels_->layout(),
    };
    if (!arena.p2p_handle.empty()) {
        result->endpoint_.device_p2p = P2pArenaDescriptor{
            .opaque_handle = arena.p2p_handle,
        };
    }
    device_links.populateEndpoint(result->endpoint_);
    return result;
}

GroupCollectiveEngine::~GroupCollectiveEngine() noexcept {
    if (!closed_) (void)close(!runtime_);
}

bool GroupCollectiveEngine::shouldUsePlannedAllReduce(DataType datatype,
                                                      ReduceOp op) const {
    return allreduce_->supports(datatype, op);
}

PGResult<void> GroupCollectiveEngine::applyView(const GroupView& view) {
    const auto resolved = resolveCollectiveView(
        view, self_in_group_rank_, device_, device_links_, host_links_);
    channels_->resetOrder();
    return allreduce_->apply(view.collective_plans, resolved);
}

PGResult<void> GroupCollectiveEngine::allReduce(
    const AllReduceRequest& request, cudaStream_t stream,
    int32_t* failed_ranks_hint, size_t failed_ranks_hint_count) {
    return allreduce_->submit(*runtime_, request, stream, failed_ranks_hint,
                              failed_ranks_hint_count);
}

bool GroupCollectiveEngine::stop(std::chrono::milliseconds timeout) {
    if (!runtime_) return true;
    runtime_->stopAccepting();
    const bool resources_safe = runtime_->drain(timeout);
    if (!resources_safe) {
        allreduce_->retainPlanForProcessLifetime();
        device_links_.retainForProcessLifetime();
    }
    runtime_.reset();
    return resources_safe;
}

bool GroupCollectiveEngine::close(bool resources_safe) {
    if (closed_) return false;
    closed_ = true;
    if (!resources_safe) {
        if (allreduce_) allreduce_->retainPlanForProcessLifetime();
        device_links_.retainForProcessLifetime();
    }
    runtime_.reset();
    allreduce_.reset();
    const bool released = !channels_ || channels_->close(resources_safe);
    channels_.reset();
    return released;
}

}  // namespace mooncake
