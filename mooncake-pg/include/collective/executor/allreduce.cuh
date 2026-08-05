#ifndef MOONCAKE_PG_COLLECTIVE_EXECUTOR_ALLREDUCE_CUH
#define MOONCAKE_PG_COLLECTIVE_EXECUTOR_ALLREDUCE_CUH

#include <cstdint>

#include <cuda_alike.h>

#include "collective/executor/allreduce_kernel_plan.cuh"
#include "collective/runtime/kernel_context.cuh"
#include "comm_types.h"

namespace mooncake {

struct AllReduceExecutorArgs {
    const void* input = nullptr;
    void* output = nullptr;
    // Every invocation snapshots input before its first attempt. A retry must
    // never consume an output partially committed by an earlier attempt.
    void* retry_input = nullptr;
    CollectiveKernelContext context;

    uint64_t element_count = 0;
    DataType datatype = DataType::Float16;
};

void launchAllReduceExecutor(const AllReduceExecutorArgs& args,
                             cudaStream_t stream);

}  // namespace mooncake

#endif  // MOONCAKE_PG_COLLECTIVE_EXECUTOR_ALLREDUCE_CUH
