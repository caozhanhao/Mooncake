#include "collective/plan/collective_plan_registry.h"

#include <algorithm>
#include <atomic>
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

struct RetainedPlanRoutes {
    std::mutex mutex;
    std::vector<std::shared_ptr<const GroupPeerRoutes>> routes;
};

RetainedPlanRoutes& retainedPlans() {
    static auto* retained = new RetainedPlanRoutes;
    return *retained;
}

struct CudaHostMemoryDeleter {
    void operator()(void* memory) const noexcept { (void)cudaFreeHost(memory); }
};

using CudaHostMemory = std::unique_ptr<void, CudaHostMemoryDeleter>;

}  // namespace

PGResult<CollectivePlanHandle> CollectivePlanPublisher::handle(
    uint64_t view_epoch) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!ready_ || view_epoch_ != view_epoch) {
        return makePGError(PGErrorCode::InvalidState,
                           "collective plan does not match GroupView");
    }
    return CollectivePlanHandle{
        .slots = slots_device_,
        .lane_sequences = lane_sequences_device_,
        .active_slot = active_slot_device_,
    };
}

CollectivePlanRegistry::~CollectivePlanRegistry() noexcept {
    if (retained_) return;
    for (auto& publisher : publishers_) release(*publisher);
}

PGResult<void> CollectivePlanRegistry::allocate(
    CollectivePlanPublisher& publisher) {
    const GpuDeviceGuard guard(device_);
    const auto slot_bytes = kCollectivePlanSlots * publisher.slot_bytes_;

    void* slots_host = nullptr;
    void* slots_device = nullptr;
    PG_TRY_CUDA(cudaHostAlloc(&slots_host, slot_bytes,
                              cudaHostAllocMapped | cudaHostAllocPortable));
    CudaHostMemory slots(slots_host);
    PG_TRY_CUDA(cudaHostGetDevicePointer(&slots_device, slots.get(), 0));

    void* sequences_host = nullptr;
    void* sequences_device = nullptr;
    PG_TRY_CUDA(cudaHostAlloc(&sequences_host,
                              control_lane_count_ * sizeof(uint64_t),
                              cudaHostAllocMapped | cudaHostAllocPortable));
    CudaHostMemory sequences(sequences_host);
    PG_TRY_CUDA(
        cudaHostGetDevicePointer(&sequences_device, sequences.get(), 0));

    void* active_slot_host = nullptr;
    void* active_slot_device = nullptr;
    PG_TRY_CUDA(cudaHostAlloc(&active_slot_host, sizeof(uint32_t),
                              cudaHostAllocMapped | cudaHostAllocPortable));
    CudaHostMemory active_slot(active_slot_host);
    PG_TRY_CUDA(
        cudaHostGetDevicePointer(&active_slot_device, active_slot.get(), 0));

    std::memset(slots.get(), 0, slot_bytes);
    std::fill_n(static_cast<uint64_t*>(sequences.get()), control_lane_count_,
                uint64_t{0});
    *static_cast<uint32_t*>(active_slot.get()) = 0;

    publisher.slots_host_ = slots.release();
    publisher.slots_device_ = slots_device;
    publisher.lane_sequences_host_ =
        static_cast<uint64_t*>(sequences.release());
    publisher.lane_sequences_device_ = static_cast<uint64_t*>(sequences_device);
    publisher.active_slot_host_ = static_cast<uint32_t*>(active_slot.release());
    publisher.active_slot_device_ = static_cast<uint32_t*>(active_slot_device);
    return {};
}

void CollectivePlanRegistry::publish(
    CollectivePlanPublisher& publisher, const void* kernel_plan,
    std::shared_ptr<const GroupPeerRoutes> peer_routes) {
    const uint32_t active =
        std::atomic_ref<uint32_t>(*publisher.active_slot_host_)
            .load(std::memory_order_relaxed);
    const uint32_t next = (active + 1) % kCollectivePlanSlots;
    auto* target = static_cast<char*>(publisher.slots_host_) +
                   next * publisher.slot_bytes_;
    std::memcpy(target, kernel_plan, publisher.slot_bytes_);
    publisher.slot_routes_[next] = std::move(peer_routes);
    std::atomic_ref<uint32_t>(*publisher.active_slot_host_)
        .store(next, std::memory_order_release);
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

    std::lock_guard<std::mutex> lock(mutex_);
    publishers_.push_back(std::move(publisher));
    return result;
}

PGResult<void> CollectivePlanRegistry::apply(const GroupView& view) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto active_ranks = deriveActiveGroupRanks(view, self_in_group_rank_);
    std::shared_ptr<const GroupPeerRoutes> peer_routes =
        std::make_shared<GroupPeerRoutes>(
            buildGroupPeerRoutes(view, active_ranks, self_in_group_rank_,
                                 device_, device_links_, host_links_));
    std::optional<PGError> first_error;
    for (auto& owned : publishers_) {
        auto& publisher = *owned;
        std::lock_guard<std::mutex> publisher_lock(publisher.mutex_);
        auto built = publisher.builder_->build(
            view.epoch, view.collective_plans, active_ranks, *peer_routes);
        // Agent effects may rebuild the same authoritative view after a
        // rank-state or link update. Only a new epoch starts a new wire-token
        // domain; resetting on a same-epoch apply could reuse an old token and
        // would happen at different times on different ranks.
        if (publisher.view_epoch_ != view.epoch) {
            std::fill_n(publisher.lane_sequences_host_, control_lane_count_,
                        uint64_t{0});
        }
        if (auto* ready = std::get_if<ReadyCollectivePlan>(&built)) {
            publish(publisher, ready->kernel_plan.get(), peer_routes);
            publisher.ready_ = true;
        } else {
            auto& failed = std::get<FailedCollectivePlan>(built);
            publish(publisher, failed.kernel_plan.get(), nullptr);
            publisher.ready_ = false;
            if (!first_error) first_error = std::move(failed.error);
        }
        publisher.view_epoch_ = view.epoch;
    }
    if (first_error) return makePGError(std::move(*first_error));
    return {};
}

void CollectivePlanRegistry::retainForProcessLifetime() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (retained_) return;
    auto& retained = retainedPlans();
    std::lock_guard<std::mutex> retained_lock(retained.mutex);
    for (const auto& publisher : publishers_) {
        std::lock_guard<std::mutex> publisher_lock(publisher->mutex_);
        for (const auto& routes : publisher->slot_routes_) {
            if (routes) retained.routes.push_back(routes);
        }
    }
    retained_ = true;
}

void CollectivePlanRegistry::release(
    CollectivePlanPublisher& publisher) noexcept {
    try {
        const GpuDeviceGuard guard(device_);
        if (publisher.lane_sequences_host_)
            cudaFreeHost(publisher.lane_sequences_host_);
        if (publisher.active_slot_host_)
            cudaFreeHost(publisher.active_slot_host_);
        if (publisher.slots_host_) cudaFreeHost(publisher.slots_host_);
    } catch (const std::exception& error) {
        LOG(WARNING) << "Collective plan cleanup failed: " << error.what();
    } catch (...) {
        LOG(WARNING) << "Collective plan cleanup failed";
    }
    publisher.lane_sequences_host_ = nullptr;
    publisher.lane_sequences_device_ = nullptr;
    publisher.active_slot_host_ = nullptr;
    publisher.active_slot_device_ = nullptr;
    publisher.slots_host_ = nullptr;
    publisher.slots_device_ = nullptr;
}

}  // namespace mooncake
