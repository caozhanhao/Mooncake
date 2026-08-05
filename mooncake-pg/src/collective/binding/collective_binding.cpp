#include "collective/binding/collective_binding.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <exception>
#include <memory>
#include <optional>
#include <utility>

#include <glog/logging.h>

#include <cuda_alike.h>

#include "control_plane/control_types.h"
#include "gpu_runtime.h"

namespace mooncake {
namespace {

struct RetainedBindingResources {
    std::mutex mutex;
    std::vector<std::shared_ptr<void>> resources;
};

RetainedBindingResources& retainedBindings() {
    static auto* retained = new RetainedBindingResources;
    return *retained;
}

struct CudaHostMemoryDeleter {
    void operator()(void* memory) const noexcept { (void)cudaFreeHost(memory); }
};

using CudaHostMemory = std::unique_ptr<void, CudaHostMemoryDeleter>;

}  // namespace

GroupCollectiveBindings::~GroupCollectiveBindings() noexcept {
    if (retained_) return;
    for (auto& binding : bindings_) release(*binding);
}

PGResult<void> GroupCollectiveBindings::allocate(BindingEntry& entry) {
    const GpuDeviceGuard guard(device_);
    const auto slot_bytes = kCollectiveBindingSlots * entry.slot_bytes;

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

    entry.slots_host = slots.release();
    entry.slots_device = slots_device;
    entry.lane_sequences_host = static_cast<uint64_t*>(sequences.release());
    entry.lane_sequences_device = static_cast<uint64_t*>(sequences_device);
    entry.active_slot_host = static_cast<uint32_t*>(active_slot.release());
    entry.active_slot_device = static_cast<uint32_t*>(active_slot_device);
    return {};
}

void GroupCollectiveBindings::publish(
    BindingEntry& entry, const void* kernel_plan,
    std::vector<std::shared_ptr<void>> resources) {
    const uint32_t active = std::atomic_ref<uint32_t>(*entry.active_slot_host)
                                .load(std::memory_order_relaxed);
    const uint32_t next = (active + 1) % kCollectiveBindingSlots;
    auto* target =
        static_cast<char*>(entry.slots_host) + next * entry.slot_bytes;
    std::memcpy(target, kernel_plan, entry.slot_bytes);
    entry.slot_resources[next] = std::move(resources);
    std::atomic_ref<uint32_t>(*entry.active_slot_host)
        .store(next, std::memory_order_release);
}

PGResult<CollectiveBindingId> GroupCollectiveBindings::add(
    std::unique_ptr<CollectiveBindingMaterializer> materializer) {
    const auto plan_bytes = materializer->kernelPlanBytes();
    PG_ASSERT(plan_bytes != 0,
              "collective binding materializer has an empty kernel plan");
    auto entry = std::make_unique<BindingEntry>();
    entry->slot_bytes = plan_bytes;
    entry->materializer = std::move(materializer);
    PG_TRY(allocate(*entry));

    std::lock_guard<std::mutex> lock(mutex_);
    bindings_.push_back(std::move(entry));
    return static_cast<CollectiveBindingId>(bindings_.size() - 1);
}

PGResult<void> GroupCollectiveBindings::apply(const GroupView& view) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::optional<PGError> first_error;
    for (auto& owned : bindings_) {
        auto& entry = *owned;
        auto materialized = entry.materializer->materialize(view);
        // Agent effects may rematerialize the same authoritative view after a
        // rank-state or link update. Only a new epoch starts a new wire-token
        // domain; resetting on a same-epoch apply could reuse an old token and
        // would happen at different times on different ranks.
        if (entry.view_epoch != view.epoch) {
            std::fill_n(entry.lane_sequences_host, control_lane_count_,
                        uint64_t{0});
        }
        if (auto* ready = std::get_if<ReadyCollectiveBinding>(&materialized)) {
            publish(entry, ready->kernel_plan.get(),
                    std::move(ready->resources));
            entry.ready = true;
        } else {
            auto& failed = std::get<FailedCollectiveBinding>(materialized);
            publish(entry, failed.kernel_plan.get(), {});
            entry.ready = false;
            if (!first_error) first_error = std::move(failed.error);
        }
        entry.view_epoch = view.epoch;
    }
    if (first_error) return makePGError(std::move(*first_error));
    return {};
}

PGResult<void> GroupCollectiveBindings::requireReady(
    CollectiveBindingId binding_id, uint64_t view_epoch) const {
    std::lock_guard<std::mutex> lock(mutex_);
    PG_ASSERT(binding_id < bindings_.size(),
              "collective binding id is invalid");
    const auto& entry = *bindings_[binding_id];
    if (!entry.ready || entry.view_epoch != view_epoch) {
        return makePGError(PGErrorCode::InvalidState,
                           "collective binding does not match GroupView");
    }
    return {};
}

CollectiveBindingView GroupCollectiveBindings::deviceView(
    CollectiveBindingId binding_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    PG_ASSERT(binding_id < bindings_.size(),
              "collective binding id is invalid");
    const auto& entry = *bindings_[binding_id];
    return CollectiveBindingView{
        .slots = entry.slots_device,
        .lane_sequences = entry.lane_sequences_device,
        .active_slot = entry.active_slot_device,
    };
}

bool GroupCollectiveBindings::readyForRecovery(
    CollectiveBindingId binding_id, uint64_t failed_view_epoch) const {
    std::lock_guard<std::mutex> lock(mutex_);
    PG_ASSERT(binding_id < bindings_.size(),
              "collective binding id is invalid");
    const auto& entry = *bindings_[binding_id];
    return entry.ready && entry.view_epoch > failed_view_epoch;
}

void GroupCollectiveBindings::retainForProcessLifetime() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (retained_) return;
    auto& retained = retainedBindings();
    std::lock_guard<std::mutex> retained_lock(retained.mutex);
    for (const auto& entry : bindings_) {
        for (const auto& resources : entry->slot_resources) {
            retained.resources.insert(retained.resources.end(),
                                      resources.begin(), resources.end());
        }
    }
    retained_ = true;
}

void GroupCollectiveBindings::release(BindingEntry& entry) noexcept {
    try {
        const GpuDeviceGuard guard(device_);
        if (entry.lane_sequences_host) cudaFreeHost(entry.lane_sequences_host);
        if (entry.active_slot_host) cudaFreeHost(entry.active_slot_host);
        if (entry.slots_host) cudaFreeHost(entry.slots_host);
    } catch (const std::exception& error) {
        LOG(WARNING) << "Collective binding cleanup failed: " << error.what();
    } catch (...) {
        LOG(WARNING) << "Collective binding cleanup failed";
    }
    entry.lane_sequences_host = nullptr;
    entry.lane_sequences_device = nullptr;
    entry.active_slot_host = nullptr;
    entry.active_slot_device = nullptr;
    entry.slots_host = nullptr;
    entry.slots_device = nullptr;
}

}  // namespace mooncake
