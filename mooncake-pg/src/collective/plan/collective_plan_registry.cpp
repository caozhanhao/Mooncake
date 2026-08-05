#include "collective/plan/collective_plan_registry.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <memory>
#include <optional>
#include <utility>

#include <glog/logging.h>

#include <cuda_alike.h>

#include "collective/plan/active_group_ranks.h"
#include "collective/route/group_peer_routes.h"
#include "control_plane/control_types.h"
#include "control_plane/device_link_manager.h"
#include "control_plane/link_manager.h"
#include "gpu_runtime.h"

namespace mooncake {
namespace {

struct CudaHostMemoryDeleter {
    void operator()(void* memory) const noexcept { (void)cudaFreeHost(memory); }
};

using CudaHostMemory = std::unique_ptr<void, CudaHostMemoryDeleter>;

}  // namespace

PGResult<CollectivePlanHandle> CollectivePlanPublisher::handle() const {
    if (!ready_) {
        return makePGError(PGErrorCode::InvalidState,
                           "collective plan is not ready");
    }
    return CollectivePlanHandle{
        .plan = plan_device_,
        .lane_sequences = lane_sequences_device_,
    };
}

CollectivePlanRegistry::~CollectivePlanRegistry() noexcept {
    if (retained_) return;
    for (auto& publisher : publishers_) release(*publisher);
}

PGResult<void> CollectivePlanRegistry::allocate(
    CollectivePlanPublisher& publisher) {
    const GpuDeviceGuard guard(device_);

    void* plan_host = nullptr;
    void* plan_device = nullptr;
    PG_TRY_CUDA(cudaHostAlloc(&plan_host, publisher.plan_bytes_,
                              cudaHostAllocMapped | cudaHostAllocPortable));
    CudaHostMemory plan(plan_host);
    PG_TRY_CUDA(cudaHostGetDevicePointer(&plan_device, plan.get(), 0));

    void* sequences_host = nullptr;
    void* sequences_device = nullptr;
    PG_TRY_CUDA(cudaHostAlloc(&sequences_host,
                              control_lane_count_ * sizeof(uint64_t),
                              cudaHostAllocMapped | cudaHostAllocPortable));
    CudaHostMemory sequences(sequences_host);
    PG_TRY_CUDA(
        cudaHostGetDevicePointer(&sequences_device, sequences.get(), 0));

    std::memset(plan.get(), 0, publisher.plan_bytes_);
    std::fill_n(static_cast<uint64_t*>(sequences.get()), control_lane_count_,
                uint64_t{0});

    publisher.plan_host_ = plan.release();
    publisher.plan_device_ = plan_device;
    publisher.lane_sequences_host_ =
        static_cast<uint64_t*>(sequences.release());
    publisher.lane_sequences_device_ = static_cast<uint64_t*>(sequences_device);
    return {};
}

void CollectivePlanRegistry::publish(CollectivePlanPublisher& publisher,
                                     const void* kernel_plan) {
    std::memcpy(publisher.plan_host_, kernel_plan, publisher.plan_bytes_);
}

PGResult<CollectivePlanPublisher*> CollectivePlanRegistry::addBuilder(
    std::unique_ptr<CollectivePlanBuilder> builder) {
    const auto plan_bytes = builder->kernelPlanBytes();
    PG_ASSERT(plan_bytes != 0,
              "collective plan builder has an empty kernel plan");
    auto publisher = std::unique_ptr<CollectivePlanPublisher>(
        new CollectivePlanPublisher(std::move(builder), plan_bytes));
    PG_TRY(allocate(*publisher));
    auto* result = publisher.get();

    publishers_.push_back(std::move(publisher));
    return result;
}

PGResult<void> CollectivePlanRegistry::apply(const GroupView& view) {
    const auto active_ranks = deriveActiveGroupRanks(view, self_in_group_rank_);
    const auto peer_routes =
        buildGroupPeerRoutes(view, active_ranks, self_in_group_rank_, device_,
                             device_links_, host_links_);
    std::optional<PGError> first_error;
    for (auto& owned : publishers_) {
        auto& publisher = *owned;
        auto built = publisher.builder_->build(
            view.epoch, view.collective_plans, active_ranks, peer_routes);
        // Agent effects may rebuild the same authoritative view after a
        // rank-state or link update. Only a new epoch starts a new wire-token
        // domain; resetting on a same-epoch apply could reuse an old token and
        // would happen at different times on different ranks.
        if (publisher.view_epoch_ != view.epoch) {
            std::fill_n(publisher.lane_sequences_host_, control_lane_count_,
                        uint64_t{0});
        }
        if (auto* ready = std::get_if<ReadyCollectivePlan>(&built)) {
            publish(publisher, ready->kernel_plan.get());
            publisher.ready_ = true;
        } else {
            auto& failed = std::get<FailedCollectivePlan>(built);
            publish(publisher, failed.kernel_plan.get());
            publisher.ready_ = false;
            if (!first_error) first_error = std::move(failed.error);
        }
        publisher.view_epoch_ = view.epoch;
    }
    if (first_error) return makePGError(std::move(*first_error));
    return {};
}

void CollectivePlanRegistry::retainForProcessLifetime() { retained_ = true; }

void CollectivePlanRegistry::release(
    CollectivePlanPublisher& publisher) noexcept {
    try {
        const GpuDeviceGuard guard(device_);
        if (publisher.lane_sequences_host_)
            cudaFreeHost(publisher.lane_sequences_host_);
        if (publisher.plan_host_) cudaFreeHost(publisher.plan_host_);
    } catch (const std::exception& error) {
        LOG(WARNING) << "Collective plan cleanup failed: " << error.what();
    } catch (...) {
        LOG(WARNING) << "Collective plan cleanup failed";
    }
    publisher.lane_sequences_host_ = nullptr;
    publisher.lane_sequences_device_ = nullptr;
    publisher.plan_host_ = nullptr;
    publisher.plan_device_ = nullptr;
}

}  // namespace mooncake
