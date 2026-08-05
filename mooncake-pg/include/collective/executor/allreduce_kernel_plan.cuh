#ifndef MOONCAKE_PG_COLLECTIVE_EXECUTOR_ALLREDUCE_KERNEL_PLAN_CUH
#define MOONCAKE_PG_COLLECTIVE_EXECUTOR_ALLREDUCE_KERNEL_PLAN_CUH

#include <cstdint>

#include "collective/transport/peer_binding.h"
#include "collective/types.h"

namespace mooncake {

enum class AllReduceAlgorithm : uint8_t {
    FlatRing = 0,
    Hierarchical,
};

struct FlatRingKernelPlan {
    uint32_t participant_count = 0;
    uint32_t self_ordinal = 0;
    CollectivePeerBinding predecessor;
    CollectivePeerBinding successor;
};

// The topology ABI intentionally remains empty in the first vertical slice.
// Keeping the alternative in the dispatcher makes the missing boundary
// visible without guessing a hierarchical schedule.
struct HierarchicalKernelPlan {};

struct AllReduceBucketKernelPlan {
    uint64_t max_message_bytes = ~uint64_t{0};
    AllReduceAlgorithm algorithm = AllReduceAlgorithm::FlatRing;
    FlatRingKernelPlan flat_ring;
    HierarchicalKernelPlan hierarchical;
};

inline constexpr uint32_t kMaxAllReduceSizeBuckets = 8;

// Replay-time plan selected through CollectiveBindingView.
// Captured kernel arguments retain its stable envelope, never a GroupView or
// frozen participant set.
struct alignas(64) AllReduceKernelPlan {
    uint64_t view_epoch = 0;
    uint32_t self_participating = 0;
    uint32_t bucket_count = 0;
    int32_t error_code = 0;
    InGroupRank failed_peer = -1;
    AllReduceBucketKernelPlan buckets[kMaxAllReduceSizeBuckets];
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_COLLECTIVE_EXECUTOR_ALLREDUCE_KERNEL_PLAN_CUH
