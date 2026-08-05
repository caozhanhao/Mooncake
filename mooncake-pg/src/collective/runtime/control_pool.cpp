#include "collective/runtime/control_pool.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <memory>

#include <glog/logging.h>

#include <cuda_alike.h>

namespace mooncake {
namespace {

struct CudaHostMemoryDeleter {
    void operator()(void* memory) const noexcept { (void)cudaFreeHost(memory); }
};

using CudaHostMemory = std::unique_ptr<void, CudaHostMemoryDeleter>;

}  // namespace

CollectiveControlPool::~CollectiveControlPool() noexcept {
    try {
        shutdown();
    } catch (const std::exception& error) {
        LOG(WARNING) << "Collective control cleanup failed: " << error.what();
    } catch (...) {
        LOG(WARNING) << "Collective control cleanup failed";
    }
}

PGResult<void> CollectiveControlPool::initialize(uint32_t control_count) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (initialized_.load(std::memory_order_acquire)) return {};
    PG_VALIDATE_ARG(control_count != 0, "collective control pool is empty");

    const auto bytes = control_count * sizeof(CollectiveControlBlock);
    void* host = nullptr;
    PG_TRY_CUDA(cudaHostAlloc(&host, bytes,
                              cudaHostAllocMapped | cudaHostAllocPortable));
    CudaHostMemory host_memory(host);
    void* device = nullptr;
    PG_TRY_CUDA(cudaHostGetDevicePointer(&device, host_memory.get(), 0));

    controls_.resize(control_count);
    std::memset(host_memory.get(), 0, bytes);
    host_controls_ =
        static_cast<CollectiveControlBlock*>(host_memory.release());
    device_controls_ = static_cast<CollectiveControlBlock*>(device);
    initialized_.store(true, std::memory_order_release);
    return {};
}

PGResult<CollectiveControlLease> CollectiveControlPool::acquire() {
    PG_VALIDATE_STATE(initialized_.load(std::memory_order_acquire),
                      "collective control pool is not initialized");
    std::lock_guard<std::mutex> lock(mutex_);
    for (uint32_t index = 0; index < controls_.size(); ++index) {
        auto& state = controls_[index];
        if (state != SlotState::Free) continue;
        state = SlotState::Acquired;
        host_controls_[index] = CollectiveControlBlock{};
        return CollectiveControlLease{index, host_controls_ + index,
                                      device_controls_ + index};
    }
    return makePGError(PGErrorCode::ResourceBusy,
                       "collective control pool is exhausted");
}

void CollectiveControlPool::release(const CollectiveControlLease& control) {
    std::lock_guard<std::mutex> lock(mutex_);
    PG_ASSERT(control.index < controls_.size(),
              "collective control index is invalid");
    auto& state = controls_[control.index];
    PG_ASSERT(state == SlotState::Acquired &&
                  control.host == host_controls_ + control.index &&
                  control.device == device_controls_ + control.index,
              "collective control lease is invalid");
    state = SlotState::Free;
}

void CollectiveControlPool::abandon(const CollectiveControlLease& control) {
    std::lock_guard<std::mutex> lock(mutex_);
    PG_ASSERT(control.index < controls_.size(),
              "collective control index is invalid");
    auto& state = controls_[control.index];
    PG_ASSERT(state == SlotState::Acquired &&
                  control.host == host_controls_ + control.index &&
                  control.device == device_controls_ + control.index,
              "collective control lease is invalid");
    state = SlotState::Abandoned;
}

void CollectiveControlPool::shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_.exchange(false, std::memory_order_acq_rel)) return;
    const bool safe_to_free =
        std::all_of(controls_.begin(), controls_.end(),
                    [](const auto state) { return state == SlotState::Free; });
    if (safe_to_free) {
        (void)cudaFreeHost(host_controls_);
    } else {
        LOG(WARNING) << "Retaining collective controls because "
                        "asynchronous resources may still reference them";
    }
    host_controls_ = nullptr;
    device_controls_ = nullptr;
    controls_.clear();
}

}  // namespace mooncake
