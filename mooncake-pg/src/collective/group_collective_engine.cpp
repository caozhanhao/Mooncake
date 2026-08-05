#include "collective/group_collective_engine.h"

#include <utility>

#include "collective/buffer/collective_buffer_pool.h"
#include "collective/allreduce/allreduce.h"
#include "collective/plan/device_plan.h"
#include "collective/resolved_collective_view.h"
#include "collective/runtime/collective_lane_pool.h"
#include "collective/runtime/runtime.h"
#include "collective/runtime/control_block.cuh"
#include "collective/runtime/control_pool.h"
#include "collective/transport/host_transfer_executor.h"
#include "control_plane/control_types.h"
#include "control_plane/device_link_manager.h"
#include "control_plane/link_manager.h"

namespace mooncake {

PGResult<std::unique_ptr<GroupCollectiveEngine>> GroupCollectiveEngine::create(
    CollectiveBufferPool& buffer_pool, CollectiveControlPool& control_pool,
    HostTransferExecutor& host_transfer_executor,
    DeviceLinkManager& device_links, LinkManager& host_links,
    TransferEngine* transfer_engine, DeviceId device,
    InGroupRank self_in_group_rank, std::string te_location,
    size_t collective_timeout_us,
    std::function<PGResult<void>(InGroupRank)> report_failure) {
    PG_TRY(control_pool.initialize());
    PG_TRY(host_transfer_executor.initialize(transfer_engine, &host_links));

    auto result =
        std::unique_ptr<GroupCollectiveEngine>(new GroupCollectiveEngine(
            device_links, host_links, device, self_in_group_rank));
    PG_TRY(result->lanes_,
           CollectiveLanePool::create(&buffer_pool, device, te_location,
                                      transfer_engine));

    const auto arena = buffer_pool.arena(device);
    device_links.bindCollectiveArena(arena);
    PG_TRY(result->allreduce_plan_, DevicePlan<AllReducePlan>::create(device));
    PG_TRY(result->runtime_,
           CollectiveRuntime::create(
               &buffer_pool, &control_pool, &host_transfer_executor,
               result->lanes_.get(), te_location, transfer_engine, device,
               collective_timeout_us, std::move(report_failure)));

    result->endpoint_ = GroupEndpointV2{
        .device = arena.device,
        .arena_generation = arena.generation,
        .arena_base_address =
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(arena.base)),
        .arena_bytes = arena.bytes,
        .control_base_address = static_cast<uint64_t>(
            reinterpret_cast<uintptr_t>(result->lanes_->controlBase())),
        .control_layout = result->lanes_->layout(),
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
    return allreduce_protocol_ == AllReduceProtocol::Planned &&
           supportsPlannedAllReduce(datatype, op);
}

PGResult<void> GroupCollectiveEngine::applyView(const GroupView& view) {
    const auto resolved = resolveCollectiveView(
        view, self_in_group_rank_, device_, device_links_, host_links_);
    auto built = buildAllReducePlan(view.collective_plans, resolved);

    allreduce_protocol_ = view.collective_plans.allreduce_protocol;
    view_epoch_ = view.epoch;
    if (!built.has_value()) {
        AllReducePlan invalid;
        invalid.view_epoch = view.epoch;
        invalid.self_participating = resolved.self_ordinal.has_value() ? 1 : 0;
        invalid.error_code =
            static_cast<int32_t>(CollectiveProtocolError::InvalidPlan);
        allreduce_plan_->publishInvalid(invalid);
        return makePGError(std::move(built).error());
    }
    allreduce_plan_->publish(built.value());
    return {};
}

PGResult<void> GroupCollectiveEngine::allReduce(
    const AllReduceRequest& request, cudaStream_t stream,
    int32_t* failed_ranks_hint, size_t failed_ranks_hint_count) {
    PG_VALIDATE_STATE(allreduce_protocol_ == AllReduceProtocol::Planned,
                      "planned AllReduce is not enabled for this group");
    PG_TRY(auto plan, allreduce_plan_->devicePlan());
    return runtime_->submit(view_epoch_, stream, failed_ranks_hint,
                            failed_ranks_hint_count,
                            [&](const CollectiveKernelArgs& common) {
                                launchAllReduce(request, plan, common, stream);
                            });
}

bool GroupCollectiveEngine::stop(std::chrono::milliseconds timeout) {
    if (!runtime_) return true;
    runtime_->stopAccepting();
    const bool resources_safe = runtime_->drain(timeout);
    if (!resources_safe) {
        allreduce_plan_->retainForProcessLifetime();
        device_links_.retainForProcessLifetime();
    }
    runtime_.reset();
    return resources_safe;
}

bool GroupCollectiveEngine::close(bool resources_safe) {
    if (closed_) return false;
    closed_ = true;
    if (!resources_safe) {
        if (allreduce_plan_) allreduce_plan_->retainForProcessLifetime();
        device_links_.retainForProcessLifetime();
    }
    runtime_.reset();
    allreduce_plan_.reset();
    const bool released = !lanes_ || lanes_->close(resources_safe);
    lanes_.reset();
    return released;
}

}  // namespace mooncake
