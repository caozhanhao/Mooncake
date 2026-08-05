#ifndef MOONCAKE_PG_COLLECTIVE_RUNTIME_COLLECTIVE_FAILURE_H
#define MOONCAKE_PG_COLLECTIVE_RUNTIME_COLLECTIVE_FAILURE_H

#include <functional>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "collective/runtime/tracked_collective.h"
#include "error_types.h"

namespace mooncake {

class CollectiveProgressEngine;
class GroupCollectiveBindings;

using CollectiveFailureRecoveryCallback =
    std::function<PGResult<void>(InGroupRank failed_peer)>;

// Owns the CPU side of the failure gate. Completion polling only supplies the
// collectives to inspect; this class correlates failure evidence, synchronizes
// membership recovery and decides whether the parked executor may retry.
class CollectiveFailureHandler {
   public:
    CollectiveFailureHandler(
        CollectiveResourcePool* resource_pool,
        GroupCollectiveBindings* bindings,
        CollectiveFailureRecoveryCallback recovery_callback)
        : resource_pool_(resource_pool),
          bindings_(bindings),
          recovery_callback_(std::move(recovery_callback)) {}

   private:
    friend class CollectiveProgressEngine;

    struct Claim {
        std::shared_ptr<TrackedCollective> collective;
        CollectiveFailureTarget target;
        uint64_t view_epoch = 0;
        InGroupRank failed_peer = -1;
    };

    static std::optional<Claim> claim(
        const std::vector<std::shared_ptr<TrackedCollective>>& collectives);
    void handle(const Claim& failure);

    CollectiveResourcePool* resource_pool_ = nullptr;
    GroupCollectiveBindings* bindings_ = nullptr;
    CollectiveFailureRecoveryCallback recovery_callback_;
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_COLLECTIVE_RUNTIME_COLLECTIVE_FAILURE_H
