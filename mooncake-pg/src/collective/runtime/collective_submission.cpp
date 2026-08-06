#include "collective/runtime/collective_submission.h"

#include <atomic>

namespace mooncake {
namespace {

bool commandSettled(HostTransferCommand& command) {
    const auto state = static_cast<HostTransferCommandState>(
        std::atomic_ref<uint32_t>(command.state)
            .load(std::memory_order_acquire));
    return state == HostTransferCommandState::Idle ||
           state == HostTransferCommandState::Completed ||
           state == HostTransferCommandState::Failed;
}

}  // namespace

CollectiveSubmission::~CollectiveSubmission() noexcept {
    if (!channels_) return;
    if (submitted_) {
        abandon();
    } else {
        release();
    }
}

CollectiveKernelArgs CollectiveSubmission::kernelArgs(
    uint64_t failure_target_id) const {
    return CollectiveKernelArgs{
        .resources = kernel_resources_,
        .invocation_sequence = channel_.invocation_sequence,
        .failure_target_id = failure_target_id,
    };
}

bool CollectiveSubmission::retire() noexcept {
    if (!channels_) return true;
    const bool transport_idle =
        std::atomic_ref<uint32_t>(channel_.host_control->transport_idle)
            .load(std::memory_order_acquire) != 0;
    if (transport_idle && commandSettled(*channel_.host_command)) {
        release();
        return true;
    }
    abandon();
    return false;
}

void CollectiveSubmission::release() noexcept {
    buffer_pool_->release(*buffer_);
    buffer_.reset();
    channels_->release(channel_);
    channels_ = nullptr;
    buffer_pool_ = nullptr;
}

void CollectiveSubmission::abandon() noexcept {
    buffer_pool_->abandon(*buffer_);
    buffer_.reset();
    channels_->abandon(channel_);
    channels_ = nullptr;
    buffer_pool_ = nullptr;
}

}  // namespace mooncake
