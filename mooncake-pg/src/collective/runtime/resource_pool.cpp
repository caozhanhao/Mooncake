#include "collective/runtime/resource_pool.h"

#include <atomic>

namespace mooncake {
namespace {

constexpr uint64_t kStagingBytes = 8 * kCollectiveMiB;
constexpr uint64_t kProtocolBytes = 4096;
constexpr uint64_t kBufferAlignment = 2 * kCollectiveMiB;
constexpr uint64_t kLayoutAlignment = 64;

uint64_t alignUp(uint64_t value, uint64_t alignment) {
    const auto remainder = value % alignment;
    return remainder == 0 ? value : value + alignment - remainder;
}

const CollectiveBufferLayout& collectiveBufferLayout() {
    static const CollectiveBufferLayout layout{
        .staging = {0, kStagingBytes},
        .protocol = {alignUp(kStagingBytes, kLayoutAlignment), kProtocolBytes},
    };
    return layout;
}

uint64_t collectiveBufferBytes() {
    const auto& layout = collectiveBufferLayout();
    return alignUp(layout.protocol.offset + layout.protocol.bytes,
                   kLayoutAlignment);
}

}  // namespace

CollectiveResourceLease::~CollectiveResourceLease() noexcept {
    (void)release(!submitted_);
}

CollectiveResourceLease::CollectiveResourceLease(
    CollectiveResourceLease&& other) noexcept {
    moveFrom(std::move(other));
}

CollectiveResourceLease& CollectiveResourceLease::operator=(
    CollectiveResourceLease&& other) noexcept {
    if (this != &other) {
        (void)release(!submitted_);
        moveFrom(std::move(other));
    }
    return *this;
}

bool CollectiveResourceLease::release(bool resource_idle) noexcept {
    if (!pool_) return true;
    auto* pool = std::exchange(pool_, nullptr);
    return pool->release(*this, resource_idle);
}

bool CollectiveResourceLease::retire() noexcept {
    if (!pool_) return true;
    const bool resource_idle =
        std::atomic_ref<uint32_t>(control.host->resource_idle)
            .load(std::memory_order_acquire) != 0;
    return release(resource_idle);
}

void CollectiveResourceLease::moveFrom(
    CollectiveResourceLease&& other) noexcept {
    lane = other.lane;
    buffer = std::move(other.buffer);
    control = other.control;
    host_command = other.host_command;
    pool_ = std::exchange(other.pool_, nullptr);
    submitted_ = std::exchange(other.submitted_, false);
    has_lane_ = std::exchange(other.has_lane_, false);
}

PGResult<CollectiveResourceLease> CollectiveResourcePool::acquire(
    uint32_t preferred_lane) {
    CollectiveResourceLease resources(this);
    PG_TRY(resources.lane, lanes_->acquire(preferred_lane));
    resources.has_lane_ = true;

    auto control = control_pool_->tryAcquire();
    if (!control.has_value()) {
        return makePGError(PGErrorCode::ResourceBusy,
                           "collective control block is busy");
    }
    resources.control = *control;

    PG_TRY(resources.buffer,
           buffer_pool_->acquire(device_, collectiveBufferBytes(),
                                 kBufferAlignment, te_location_, engine_));

    auto host_command = host_proxy_->tryAcquireCommand(resources.control.host);
    if (!host_command.has_value()) {
        return makePGError(PGErrorCode::ResourceBusy,
                           "collective host-transfer command is busy");
    }
    resources.host_command = *host_command;
    return resources;
}

const CollectiveBufferLayout& CollectiveResourcePool::bufferLayout() {
    return collectiveBufferLayout();
}

bool CollectiveResourcePool::release(CollectiveResourceLease& resources,
                                     bool resource_idle) noexcept {
    bool reusable = resource_idle;
    if (resources.host_command.host) {
        reusable =
            host_proxy_->releaseCommand(resources.host_command, reusable);
        resources.host_command = {};
    }
    if (resources.buffer) {
        reusable = buffer_pool_->release(*resources.buffer, reusable);
        resources.buffer.reset();
    }
    if (resources.has_lane_) {
        reusable = lanes_->release(resources.lane, reusable);
        resources.has_lane_ = false;
    }
    if (resources.control.host) {
        reusable = control_pool_->release(resources.control, reusable);
        resources.control = {};
    }
    resources.submitted_ = false;
    return reusable;
}

}  // namespace mooncake
