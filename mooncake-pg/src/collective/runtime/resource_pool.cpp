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
        std::atomic_ref<uint32_t>(channel.host_control->transport_idle)
            .load(std::memory_order_acquire) != 0;
    if (resources_idle) return release();
    abandon();
    return false;
}

void CollectiveResourceLease::moveFrom(
    CollectiveResourceLease&& other) noexcept {
    channel = other.channel;
    buffer = std::move(other.buffer);
    pool_ = std::exchange(other.pool_, nullptr);
    submitted_ = std::exchange(other.submitted_, false);
    has_channel_ = std::exchange(other.has_channel_, false);
}

PGResult<CollectiveResourceLease> CollectiveResourcePool::acquire(
    uint32_t channel_index) {
    CollectiveResourceLease resources(this);
    PG_TRY(resources.channel, channels_->acquire(channel_index));
    resources.has_channel_ = true;

    PG_TRY(resources.buffer,
           buffer_pool_->acquire(device_, collectiveBufferBytes(),
                                 kBufferAlignment, te_location_, engine_));
    return resources;
}

const CollectiveBufferLayout& CollectiveResourcePool::bufferLayout() {
    return collectiveBufferLayout();
}

bool CollectiveResourcePool::release(
    CollectiveResourceLease& resources) noexcept {
    if (resources.buffer) {
        buffer_pool_->release(*resources.buffer);
        resources.buffer.reset();
    }
    if (resources.has_channel_) {
        channels_->release(resources.channel);
        resources.has_channel_ = false;
    }
    resources.submitted_ = false;
    return true;
}

void CollectiveResourcePool::abandon(
    CollectiveResourceLease& resources) noexcept {
    if (resources.buffer) {
        buffer_pool_->abandon(*resources.buffer);
        resources.buffer.reset();
    }
    if (resources.has_channel_) {
        channels_->abandon(resources.channel);
        resources.has_channel_ = false;
    }
    resources.submitted_ = false;
}

}  // namespace mooncake
