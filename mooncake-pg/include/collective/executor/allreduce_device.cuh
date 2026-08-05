#ifndef MOONCAKE_PG_COLLECTIVE_EXECUTOR_ALLREDUCE_DEVICE_CUH
#define MOONCAKE_PG_COLLECTIVE_EXECUTOR_ALLREDUCE_DEVICE_CUH

#include "collective/executor/allreduce.cuh"
#include "collective/runtime/device_protocol.cuh"

namespace mooncake {

inline __device__ uint32_t allReduceElementBytes(DataType datatype) {
    return datatype == DataType::Float32 ? 4 : 2;
}

inline __device__ void setCollectiveError(const AllReduceKernelArgs& args,
                                          int32_t error_code,
                                          InGroupRank failed_peer) {
    setCollectiveError(args.common.resources, error_code, failed_peer);
}

inline __device__ bool waitForCollectiveToken(const uint64_t* address,
                                              uint64_t expected,
                                              const AllReduceKernelArgs& args,
                                              InGroupRank failed_peer) {
    return waitForCollectiveToken(address, expected, args.common.resources,
                                  failed_peer);
}

}  // namespace mooncake

#endif  // MOONCAKE_PG_COLLECTIVE_EXECUTOR_ALLREDUCE_DEVICE_CUH
