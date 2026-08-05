#include "collective/runtime/resource_pool.h"

#include "pg_utils.h"

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

PGResult<CollectiveResourceLease> CollectiveResourcePool::acquire(
    uint32_t preferred_lane) {
    PG_TRY(auto lane, lanes_->acquire(preferred_lane));
    auto lane_rollback =
        makeScopeExit([&]() noexcept { (void)lanes_->release(lane, true); });
    auto control = control_pool_->tryAcquire();
    if (!control.has_value()) {
        return makePGError(PGErrorCode::ResourceBusy,
                           "collective control block is busy");
    }
    auto control_rollback = makeScopeExit(
        [&]() noexcept { (void)control_pool_->release(*control, true); });
    PG_TRY(auto buffer,
           buffer_pool_->acquire(device_, collectiveBufferBytes(),
                                 kBufferAlignment, te_location_, engine_));
    auto buffer_rollback = makeScopeExit(
        [&]() noexcept { (void)buffer_pool_->release(*buffer, true); });

    auto host_command = host_proxy_->tryAcquireCommand(control->host);
    if (!host_command.has_value()) {
        return makePGError(PGErrorCode::ResourceBusy,
                           "collective host-transfer command is busy");
    }
    lane_rollback.dismiss();
    control_rollback.dismiss();
    buffer_rollback.dismiss();
    return CollectiveResourceLease{std::move(lane), std::move(buffer), *control,
                                   *host_command};
}

const CollectiveBufferLayout& CollectiveResourcePool::bufferLayout() {
    return collectiveBufferLayout();
}

bool CollectiveResourcePool::release(const CollectiveResourceLease& resources,
                                     bool resource_idle) {
    bool reusable =
        host_proxy_->releaseCommand(resources.host_command, resource_idle);
    reusable = buffer_pool_->release(*resources.buffer, reusable);
    reusable = lanes_->release(resources.lane, reusable);
    return control_pool_->release(resources.control, reusable);
}

}  // namespace mooncake
