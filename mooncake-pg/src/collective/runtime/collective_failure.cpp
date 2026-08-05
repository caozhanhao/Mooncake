#include "collective/runtime/collective_failure.h"

#include <algorithm>
#include <atomic>
#include <utility>

#include <glog/logging.h>

namespace mooncake {

std::optional<CollectiveFailureHandler::Claim> CollectiveFailureHandler::claim(
    const std::vector<std::shared_ptr<TrackedCollective>>& collectives) {
    for (const auto& collective : collectives) {
        std::lock_guard<std::mutex> lock(collective->mutex);
        auto& failure = collective->resources.control.host->failure;
        auto state = std::atomic_ref<uint32_t>(failure.state);
        uint32_t expected =
            static_cast<uint32_t>(CollectiveFailureState::Pending);
        if (!state.compare_exchange_strong(
                expected,
                static_cast<uint32_t>(CollectiveFailureState::Handling),
                std::memory_order_acq_rel, std::memory_order_acquire)) {
            continue;
        }

        const auto target = std::find_if(
            collective->failure_targets.begin(),
            collective->failure_targets.end(), [&](const auto& candidate) {
                return candidate.failure_cookie == failure.failure_cookie;
            });
        PG_ASSERT(target != collective->failure_targets.end(),
                  "collective failure cookie has no tracked target");
        return Claim{
            .collective = collective,
            .target = *target,
            .failed_peer = failure.failed_peer,
        };
    }
    return std::nullopt;
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

    auto& report = failure.collective->resources.control.host->failure;
    std::atomic_ref<uint32_t>(report.state)
        .store(static_cast<uint32_t>(CollectiveFailureState::Acknowledged),
               std::memory_order_release);
}

}  // namespace mooncake
