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
    if (submitted_) {
        buffer_.abandon();
        channel_.abandon();
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
    if (!channel_.host_control) return true;
    const bool transport_idle =
        std::atomic_ref<uint32_t>(channel_.host_control->transport_idle)
            .load(std::memory_order_acquire) != 0;
    if (transport_idle && commandSettled(*channel_.host_command)) {
        buffer_.release();
        channel_.release();
        return true;
    }
    buffer_.abandon();
    channel_.abandon();
    return false;
}

}  // namespace mooncake
