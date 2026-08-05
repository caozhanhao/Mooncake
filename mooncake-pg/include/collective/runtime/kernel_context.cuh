#ifndef MOONCAKE_PG_COLLECTIVE_RUNTIME_KERNEL_CONTEXT_CUH
#define MOONCAKE_PG_COLLECTIVE_RUNTIME_KERNEL_CONTEXT_CUH

#include <cstdint>

#include "collective/plan/plan_handle.cuh"
#include "collective/transport/kernel_resources.cuh"

namespace mooncake {

// Common runtime-to-executor ABI. Collective-specific kernel arguments embed
// this context instead of copying its fields into another parallel shape.
struct CollectiveKernelContext {
    CollectiveKernelResources resources;
    uint32_t lane_index = 0;
    uint64_t failure_target_id = 0;
    CollectivePlanHandle plan;
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_COLLECTIVE_RUNTIME_KERNEL_CONTEXT_CUH
