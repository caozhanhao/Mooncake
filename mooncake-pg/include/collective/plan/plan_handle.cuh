#ifndef MOONCAKE_PG_COLLECTIVE_PLAN_PLAN_HANDLE_CUH
#define MOONCAKE_PG_COLLECTIVE_PLAN_PLAN_HANDLE_CUH

#include <cstdint>

#include "collective/types.h"

namespace mooncake {

inline constexpr uint32_t kCollectivePlanSlots = 2;

// Stable kernel reference to the published collective plans. Membership,
// algorithm and peer routes live in the typed plan slots selected through
// active_slot on every invocation. Per-lane sequences enter the same view-local
// token domain when a new active slot is published.
struct CollectivePlanHandle {
    const void* slots = nullptr;
    uint64_t* lane_sequences = nullptr;
    const uint32_t* active_slot = nullptr;
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_COLLECTIVE_PLAN_PLAN_HANDLE_CUH
