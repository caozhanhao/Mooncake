#include "collective/runtime/collective_failure.h"

#include <algorithm>
#include <atomic>
#include <utility>

#include <glog/logging.h>

#include "collective/binding/collective_binding.h"

namespace mooncake {

std::optional<CollectiveFailureHandler::Claim> CollectiveFailureHandler::claim(
    const std::vector<std::shared_ptr<TrackedCollective>>& collectives) {
    for (const auto& collective : collectives) {
        std::lock_guard<std::mutex> lock(collective->mutex);
        auto& gate = collective->resources.control.host->failure_gate;
        auto state = std::atomic_ref<uint32_t>(gate.state);
        uint32_t expected =
            static_cast<uint32_t>(CollectiveFailureGateState::FailurePending);
        if (!state.compare_exchange_strong(
                expected,
                static_cast<uint32_t>(CollectiveFailureGateState::Recovering),
                std::memory_order_acq_rel, std::memory_order_acquire)) {
            continue;
        }

        const auto target = std::find_if(
            collective->failure_targets.begin(),
            collective->failure_targets.end(), [&](const auto& candidate) {
                return candidate.failure_cookie == gate.failure_cookie;
            });
        PG_ASSERT(target != collective->failure_targets.end(),
                  "collective failure cookie has no tracked target");
        return Claim{
            .collective = collective,
            .target = *target,
            .view_epoch = gate.failure_view_epoch,
            .failed_peer = gate.failed_peer,
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

    bool recovery_synchronized = !recovery_callback_ || failure.failed_peer < 0;
    if (recovery_callback_ && failure.failed_peer >= 0) {
        auto result = recovery_callback_(failure.failed_peer);
        if (result.has_value()) {
            recovery_synchronized = true;
        } else {
            LOG(WARNING) << "Collective failure recovery failed for peer "
                         << failure.failed_peer << ": "
                         << result.error().message;
        }
    }

    const auto& resources = failure.collective->resources;
    const bool can_retry = recovery_synchronized &&
                           bindings_->readyForRecovery(
                               failure.target.binding_id, failure.view_epoch) &&
                           resource_pool_->readyForRetry(resources);
    auto& gate = resources.control.host->failure_gate;
    std::atomic_ref<uint32_t>(gate.state)
        .store(static_cast<uint32_t>(can_retry
                                         ? CollectiveFailureGateState::Open
                                         : CollectiveFailureGateState::Closed),
               std::memory_order_release);
}

}  // namespace mooncake
