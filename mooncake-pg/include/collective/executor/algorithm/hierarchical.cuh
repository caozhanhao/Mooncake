#ifndef MOONCAKE_PG_COLLECTIVE_EXECUTOR_ALGORITHM_HIERARCHICAL_CUH
#define MOONCAKE_PG_COLLECTIVE_EXECUTOR_ALGORITHM_HIERARCHICAL_CUH

#include "collective/executor/allreduce_device.cuh"

namespace mooncake::hierarchical_allreduce {

// The alternative deliberately lives inside the same stable executor envelope
// as Flat Ring. A later application invocation may observe a Coordinator plan
// that replaces hierarchical AllReduce with Ring without graph recapture. The
// failed invocation itself never retries or chooses a fallback algorithm.
inline __device__ bool run(const AllReduceKernelArgs& args,
                           const HierarchicalKernelPlan&, const void*, uint64_t,
                           uint64_t) {
    setCollectiveError(
        args, static_cast<int32_t>(CollectiveProtocolError::Unsupported), -1);
    __syncthreads();
    return false;
}

}  // namespace mooncake::hierarchical_allreduce

#endif  // MOONCAKE_PG_COLLECTIVE_EXECUTOR_ALGORITHM_HIERARCHICAL_CUH
