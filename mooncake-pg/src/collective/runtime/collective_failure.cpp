#include "collective/runtime/collective_failure.h"

#include <algorithm>
#include <atomic>

#include <glog/logging.h>

namespace mooncake {

std::optional<CollectiveFailureHandler::Claim> CollectiveFailureHandler::claim(
    const std::shared_ptr<CollectiveResourceLease>& resources,
    const std::vector<CollectiveFailureTarget>& targets) {
    auto& failure = resources->control.host->failure;
    auto state = std::atomic_ref<uint32_t>(failure.state);
    uint32_t expected = static_cast<uint32_t>(CollectiveFailureState::Pending);
    if (!state.compare_exchange_strong(
            expected, static_cast<uint32_t>(CollectiveFailureState::Handling),
            std::memory_order_acq_rel, std::memory_order_acquire)) {
        return std::nullopt;
    }

    const auto target =
        std::find_if(targets.begin(), targets.end(), [&](const auto& target) {
            return target.failure_target_id == failure.failure_target_id;
        });
    PG_ASSERT(target != targets.end(),
              "collective failure report has no tracked target");
    return Claim{
        .resources = resources,
        .target = *target,
        .failed_peer = failure.failed_peer,
    };
}

void CollectiveFailureHandler::handle(const Claim& failure) {
    if (failure.target.failed_ranks_hint && failure.failed_peer >= 0 &&
        static_cast<size_t>(failure.failed_peer) <
            failure.target.failed_ranks_hint_count) {
        failure.target.failed_ranks_hint[failure.failed_peer] = 1;
    }

    if (report_callback_ && failure.failed_peer >= 0) {
        auto result = report_callback_(failure.failed_peer);
        if (!result.has_value()) {
            LOG(WARNING) << "Collective failure report failed for peer "
                         << failure.failed_peer << ": "
                         << result.error().message;
        }
    }

    auto& report = failure.resources->control.host->failure;
    std::atomic_ref<uint32_t>(report.state)
        .store(static_cast<uint32_t>(CollectiveFailureState::Acknowledged),
               std::memory_order_release);
}

}  // namespace mooncake
