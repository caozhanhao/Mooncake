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
    if (submitted_) {
        abandon();
    } else {
        (void)release();
    }
}

CollectiveResourceLease::CollectiveResourceLease(
    CollectiveResourceLease&& other) noexcept {
    moveFrom(std::move(other));
}

CollectiveResourceLease& CollectiveResourceLease::operator=(
    CollectiveResourceLease&& other) noexcept {
    if (this != &other) {
        if (submitted_) {
            abandon();
        } else {
            (void)release();
        }
        moveFrom(std::move(other));
    }
    return *this;
}

bool CollectiveResourceLease::release() noexcept {
    if (!pool_) return true;
    auto* pool = std::exchange(pool_, nullptr);
    return pool->release(*this);
}

void CollectiveResourceLease::abandon() noexcept {
    if (!pool_) return;
    auto* pool = std::exchange(pool_, nullptr);
    pool->abandon(*this);
}

bool CollectiveResourceLease::retire() noexcept {
    if (!pool_) return true;
    const bool resources_idle =
        std::atomic_ref<uint32_t>(control.host->transport_idle)
            .load(std::memory_order_acquire) != 0;
    if (resources_idle) return release();
    abandon();
    return false;
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

    PG_TRY(resources.control, control_pool_->acquire());

    PG_TRY(resources.buffer,
           buffer_pool_->acquire(device_, collectiveBufferBytes(),
                                 kBufferAlignment, te_location_, engine_));

    PG_TRY(resources.host_command,
           host_executor_->acquireCommand(resources.control.host));
    return resources;
}

const CollectiveBufferLayout& CollectiveResourcePool::bufferLayout() {
    return collectiveBufferLayout();
}

bool CollectiveResourcePool::release(
    CollectiveResourceLease& resources) noexcept {
    if (resources.host_command.host) {
        if (!host_executor_->releaseCommand(resources.host_command)) {
            resources.host_command = {};
            abandon(resources);
            return false;
        }
        resources.host_command = {};
    }
    if (resources.buffer) {
        buffer_pool_->release(*resources.buffer);
        resources.buffer.reset();
    }
    if (resources.has_lane_) {
        lanes_->release(resources.lane);
        resources.has_lane_ = false;
    }
    if (resources.control.host) {
        control_pool_->release(resources.control);
        resources.control = {};
    }
    resources.submitted_ = false;
    return true;
}

void CollectiveResourcePool::abandon(
    CollectiveResourceLease& resources) noexcept {
    if (resources.host_command.host) {
        host_executor_->abandonCommand(resources.host_command);
        resources.host_command = {};
    }
    if (resources.buffer) {
        buffer_pool_->abandon(*resources.buffer);
        resources.buffer.reset();
    }
    if (resources.has_lane_) {
        lanes_->abandon(resources.lane);
        resources.has_lane_ = false;
    }
    if (resources.control.host) {
        control_pool_->abandon(resources.control);
        resources.control = {};
    }
    resources.submitted_ = false;
}

}  // namespace mooncake
