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

using CollectiveFailureReportCallback =
    std::function<PGResult<void>(InGroupRank failed_peer)>;

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
    friend class CollectiveProgressEngine;

    struct Claim {
        std::shared_ptr<TrackedCollective> collective;
        CollectiveFailureTarget target;
        InGroupRank failed_peer = -1;
    };

    static std::optional<Claim> claim(
        const std::vector<std::shared_ptr<TrackedCollective>>& collectives);
    void handle(const Claim& failure);

    CollectiveFailureReportCallback report_callback_;
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_COLLECTIVE_RUNTIME_COLLECTIVE_FAILURE_H
