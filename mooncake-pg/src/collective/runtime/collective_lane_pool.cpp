#include "collective/runtime/collective_lane_pool.h"

#include <algorithm>
#include <utility>

#include <cuda_alike.h>

#include "gpu_runtime.h"

namespace mooncake {
namespace {

constexpr uint64_t kSignalBytes = 4096;
constexpr uint64_t kControlAlignment = 64;

uint64_t alignUp(uint64_t value, uint64_t alignment) {
    const auto remainder = value % alignment;
    return remainder == 0 ? value : value + alignment - remainder;
}

PGResult<CollectiveControlLayout> buildControlLayout(uint32_t lane_count) {
    PG_VALIDATE_ARG(lane_count != 0, "collective communicator needs a lane");
    CollectiveControlLayout layout;
    layout.version = 1;
    layout.alignment = kControlAlignment;
    layout.lane_count = lane_count;
    layout.lanes.reserve(lane_count);
    uint64_t offset = 0;
    for (uint32_t lane = 0; lane < lane_count; ++lane) {
        layout.lanes.push_back(
            CollectiveControlLaneLayout{{offset, kSignalBytes}});
        offset += kSignalBytes;
    }
    layout.total_bytes = alignUp(offset, kControlAlignment);
    return layout;
}

}  // namespace

PGResult<std::unique_ptr<CollectiveLanePool>> CollectiveLanePool::create(
    CollectiveBufferPool* buffers, DeviceId device,
    const std::string& te_location, TransferEngine* engine,
    uint32_t lane_count) {
    PG_TRY(auto layout, buildControlLayout(lane_count));
    PG_TRY(auto control,
           buffers->acquire(device, layout.total_bytes, layout.alignment,
                            te_location, engine));
    auto pool = std::unique_ptr<CollectiveLanePool>(
        new CollectiveLanePool(buffers, std::move(control), std::move(layout)));
    const GpuDeviceGuard guard(device);
    PG_TRY_CUDA(cudaMemset(pool->control_->base(), 0, pool->control_->bytes()));
    pool->may_be_in_use_ = true;
    return pool;
}

CollectiveLanePool::~CollectiveLanePool() noexcept {
    (void)close(!may_be_in_use_);
}

PGResult<CollectiveLaneLease> CollectiveLanePool::acquire(
    uint32_t preferred_lane) {
    std::lock_guard<std::mutex> lock(mutex_);
    PG_VALIDATE_STATE(!closed_, "collective lane pool is closed");
    for (uint32_t offset = 0; offset < lanes_.size(); ++offset) {
        const auto index = (preferred_lane + offset) % lanes_.size();
        auto& lane = lanes_[index];
        if (lane.in_use || !lane.reusable) continue;
        lane.in_use = true;
        return CollectiveLaneLease{index};
    }
    return makePGError(PGErrorCode::ResourceBusy, "collective lanes are busy");
}

bool CollectiveLanePool::release(const CollectiveLaneLease& lane,
                                 bool resource_idle) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_) return false;
    PG_ASSERT(lane.index < lanes_.size(), "collective lane index is invalid");
    auto& state = lanes_[lane.index];
    PG_ASSERT(state.in_use, "collective lane was released twice");
    state.in_use = false;
    state.reusable = resource_idle;
    return resource_idle;
}

bool CollectiveLanePool::close(bool resource_idle) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_) return false;
    closed_ = true;
    const bool lanes_idle = std::all_of(
        lanes_.begin(), lanes_.end(),
        [](const auto& lane) { return !lane.in_use && lane.reusable; });
    return buffers_->release(*control_, resource_idle && lanes_idle);
}

}  // namespace mooncake
