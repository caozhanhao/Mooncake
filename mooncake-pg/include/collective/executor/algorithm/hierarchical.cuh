#ifndef MOONCAKE_PG_COLLECTIVE_EXECUTOR_ALGORITHM_HIERARCHICAL_CUH
#define MOONCAKE_PG_COLLECTIVE_EXECUTOR_ALGORITHM_HIERARCHICAL_CUH

#include "collective/executor/allreduce_device.cuh"

namespace mooncake::hierarchical_allreduce {

// The alternative deliberately lives inside the same stable executor
// envelope as Flat Ring. A later Coordinator view may replace a failed
// hierarchical policy with a Ring policy without recapturing the graph. The
// kernel never makes that fallback decision from local timeout evidence.
inline __device__ bool run(const AllReduceExecutorArgs& args,
                           const HierarchicalKernelPlan&, const void*, uint64_t,
                           uint64_t, uint32_t) {
    setCollectiveError(
        args, static_cast<int32_t>(CollectiveProtocolError::Unsupported), -1);
    __syncthreads();
    return false;
}

}  // namespace mooncake::hierarchical_allreduce

#endif  // MOONCAKE_PG_COLLECTIVE_EXECUTOR_ALGORITHM_HIERARCHICAL_CUH
