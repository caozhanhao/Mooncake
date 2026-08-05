#ifndef MOONCAKE_PG_COLLECTIVE_PLAN_PLAN_HANDLE_CUH
#define MOONCAKE_PG_COLLECTIVE_PLAN_PLAN_HANDLE_CUH

#include <cstdint>

#include "collective/types.h"

namespace mooncake {

// Stable kernel reference to the published collective plan. View application
// overwrites the mapped plan only at the collective-quiescent boundary assumed
// by this slice. Future control-plane quiescing will enforce that boundary, so
// every eager invocation and graph replay can read the current plan through
// the same captured address. Per-lane sequences provide the view-local wire
// tokens.
struct CollectivePlanHandle {
    const void* plan = nullptr;
    uint64_t* lane_sequences = nullptr;
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_COLLECTIVE_PLAN_PLAN_HANDLE_CUH
