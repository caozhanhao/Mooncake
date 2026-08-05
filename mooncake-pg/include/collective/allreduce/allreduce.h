#ifndef MOONCAKE_PG_COLLECTIVE_ALLREDUCE_H
#define MOONCAKE_PG_COLLECTIVE_ALLREDUCE_H

#include <cstddef>
#include <cstdint>

#include <cuda_alike.h>

#include "collective/device_context.cuh"
#include "collective/plan/logical_plan.h"
#include "collective/resolved_collective_view.h"
#include "collective/transport/peer_route.h"
#include "comm_types.h"
#include "error_types.h"

namespace mooncake {

struct FlatRingDevicePlan {
    uint32_t participant_count = 0;
    uint32_t self_ordinal = 0;
    PeerRoute predecessor;
    PeerRoute successor;
};

struct AllReduceBucketDevicePlan {
    uint64_t max_message_bytes = ~uint64_t{0};
    FlatRingDevicePlan flat_ring;
};

inline constexpr uint32_t kMaxAllReduceSizeBuckets = 8;

// Captured kernels retain the address of this mapped value, not a membership
// snapshot. A later quiescent view application can therefore replace roles
// and routes without graph recapture.
struct alignas(64) AllReduceDevicePlan {
    uint64_t view_epoch = 0;
    uint32_t self_participating = 0;
    uint32_t bucket_count = 0;
    int32_t error_code = 0;
    InGroupRank failed_peer = -1;
    AllReduceBucketDevicePlan buckets[kMaxAllReduceSizeBuckets];
};

struct AllReduceRequest {
    const void* input = nullptr;
    void* output = nullptr;
    uint64_t element_count = 0;
    DataType datatype = DataType::Float16;
};

bool supportsPlannedAllReduce(DataType datatype, ReduceOp op);

PGResult<AllReduceRequest> makeAllReduceRequest(const void* input, void* output,
                                                size_t element_count,
                                                DataType datatype, ReduceOp op);

PGResult<AllReduceDevicePlan> buildAllReduceDevicePlan(
    const CollectivePlanSet& plans, const ResolvedCollectiveView& view);

struct AllReduceKernelArgs {
    const void* input = nullptr;
    void* output = nullptr;
    CollectiveKernelArgs common;
    MappedPlanHandle<AllReduceDevicePlan> plan;
    uint64_t element_count = 0;
    DataType datatype = DataType::Float16;
};

void launchAllReduce(const AllReduceRequest& request,
                     MappedPlanHandle<AllReduceDevicePlan> plan,
                     const CollectiveKernelArgs& common, cudaStream_t stream);
void launchAllReduceExecutor(const AllReduceKernelArgs& args,
                             cudaStream_t stream);

}  // namespace mooncake

#endif  // MOONCAKE_PG_COLLECTIVE_ALLREDUCE_H
