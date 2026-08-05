#ifndef MOONCAKE_PG_COLLECTIVE_RUNTIME_TRACKED_COLLECTIVE_H
#define MOONCAKE_PG_COLLECTIVE_RUNTIME_TRACKED_COLLECTIVE_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include <cuda_alike.h>

#include "collective/runtime/resource_pool.h"

namespace mooncake {

struct CollectiveFailureTarget {
    uint64_t failure_target_id = 0;
    int32_t* failed_ranks_hint = nullptr;
    size_t failed_ranks_hint_count = 0;
};

// The host-side progress record for one resource lease. An eager collective
// has one failure target and a completion event. A captured collective has no
// completion event and may contain multiple graph nodes that reuse the same
// pinned resources in stream order.
struct TrackedCollective {
    explicit TrackedCollective(CollectiveResourceLease value)
        : resources(std::move(value)) {}
    ~TrackedCollective() noexcept {
        if (completion) (void)cudaEventDestroy(completion);
    }

    CollectiveResourceLease resources;
    cudaEvent_t completion = nullptr;
    std::vector<CollectiveFailureTarget> failure_targets;
    std::mutex mutex;
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_COLLECTIVE_RUNTIME_TRACKED_COLLECTIVE_H
