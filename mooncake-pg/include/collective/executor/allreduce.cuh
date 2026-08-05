#ifndef MOONCAKE_PG_COLLECTIVE_EXECUTOR_ALLREDUCE_CUH
#define MOONCAKE_PG_COLLECTIVE_EXECUTOR_ALLREDUCE_CUH

#include <cstdint>

#include <cuda_alike.h>

#include "collective/executor/allreduce_kernel_plan.cuh"
#include "collective/runtime/kernel_args.cuh"
#include "comm_types.h"

namespace mooncake {

struct AllReduceKernelArgs {
    const void* input = nullptr;
    void* output = nullptr;
    CollectiveKernelArgs common;

    uint64_t element_count = 0;
    DataType datatype = DataType::Float16;
};

void launchAllReduceExecutor(const AllReduceKernelArgs& args,
                             cudaStream_t stream);

}  // namespace mooncake

#endif  // MOONCAKE_PG_COLLECTIVE_EXECUTOR_ALLREDUCE_CUH
