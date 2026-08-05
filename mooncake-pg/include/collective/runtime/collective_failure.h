#ifndef MOONCAKE_PG_COLLECTIVE_RUNTIME_COLLECTIVE_FAILURE_H
#define MOONCAKE_PG_COLLECTIVE_RUNTIME_COLLECTIVE_FAILURE_H

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "collective/runtime/resource_pool.h"
#include "error_types.h"

namespace mooncake {

class CollectiveHostProgress;

using CollectiveFailureReportCallback =
    std::function<PGResult<void>(InGroupRank failed_peer)>;

struct CollectiveFailureTarget {
    uint64_t failure_target_id = 0;
    int32_t* failed_ranks_hint = nullptr;
    size_t failed_ranks_hint_count = 0;
};

// Owns the CPU side of the device-to-host failure handshake. It correlates
// failure evidence with the caller's failed-ranks hint, reports it to the
// control plane, and acknowledges the failed invocation. When configured, the
// report callback also fences the authoritative view. It never waits for ranks
// that completed the collective without observing a failure.
class CollectiveFailureHandler {
   public:
    explicit CollectiveFailureHandler(
        CollectiveFailureReportCallback report_callback)
        : report_callback_(std::move(report_callback)) {}

   private:
    friend class CollectiveHostProgress;

    struct Claim {
        std::shared_ptr<CollectiveResourceLease> resources;
        CollectiveFailureTarget target;
        InGroupRank failed_peer = -1;
    };

    static std::optional<Claim> claim(
        const std::shared_ptr<CollectiveResourceLease>& resources,
        const std::vector<CollectiveFailureTarget>& targets);
    void handle(const Claim& failure);

    CollectiveFailureReportCallback report_callback_;
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_COLLECTIVE_RUNTIME_COLLECTIVE_FAILURE_H
