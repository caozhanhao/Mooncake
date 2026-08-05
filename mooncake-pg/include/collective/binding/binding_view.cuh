#ifndef MOONCAKE_PG_COLLECTIVE_BINDING_BINDING_VIEW_CUH
#define MOONCAKE_PG_COLLECTIVE_BINDING_BINDING_VIEW_CUH

#include <cstdint>

#include "collective/types.h"

namespace mooncake {

inline constexpr uint32_t kCollectiveBindingSlots = 2;

// Captured executors retain this stable envelope. Membership, algorithm and
// peer bindings live in the two typed kernel-plan slots behind it and are
// selected through the mapped active-slot word on every attempt. Per-lane
// sequences intentionally live in the same envelope: a new view resets them
// before its active-slot release, so all ranks enter the new wire-token domain
// together.
struct CollectiveBindingView {
    const void* slots = nullptr;
    uint64_t* lane_sequences = nullptr;
    const uint32_t* active_slot = nullptr;
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_COLLECTIVE_BINDING_BINDING_VIEW_CUH
