#ifndef MOONCAKE_PG_COLLECTIVE_ALLREDUCE_H
#define MOONCAKE_PG_COLLECTIVE_ALLREDUCE_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include <cuda_alike.h>

#include "collective/device_context.cuh"
#include "collective/plan/logical_plan.h"
#include "collective/resolved_collective_view.h"
#include "collective/transport/peer_route.h"
#include "comm_types.h"
#include "error_types.h"

namespace mooncake {

class CollectiveBufferPool;
class CollectiveChannels;
class CollectiveSubmission;
class TransferEngine;

struct FlatRingPlan {
    uint32_t participant_count = 0;
    uint32_t self_ordinal = 0;
    PeerRoute predecessor;
    PeerRoute successor;
};

struct AllReducePlanBucket {
    uint64_t max_message_bytes = ~uint64_t{0};
    FlatRingPlan flat_ring;
};

inline constexpr uint32_t kMaxAllReduceSizeBuckets = 8;

// Captured kernels retain a stable pointer to this value, not a membership
// snapshot. A later quiescent view application can therefore replace roles and
// routes without graph recapture.
struct alignas(64) AllReducePlan {
    uint64_t view_epoch = 0;
    uint32_t self_participating = 0;
    uint32_t bucket_count = 0;
    int32_t error_code = 0;
    InGroupRank failed_peer = -1;
    AllReducePlanBucket buckets[kMaxAllReduceSizeBuckets];
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

PGResult<AllReducePlan> buildAllReducePlan(const CollectivePlanSet& plans,
                                           const ResolvedCollectiveView& view);

// The current bulk AllReduce protocol chooses and lays out its own registered
// transfer buffer. Runtime receives only the finished submission and remains
// independent of operation-specific memory requirements.
PGResult<std::shared_ptr<CollectiveSubmission>> prepareAllReduceSubmission(
    CollectiveBufferPool& buffer_pool, CollectiveChannels& channels,
    uint32_t channel_index, DeviceId device, const std::string& te_location,
    TransferEngine* engine, uint64_t timeout_device_ticks);

struct AllReduceKernelArgs {
    const void* input = nullptr;
    void* output = nullptr;
    CollectiveKernelArgs common;
    const AllReducePlan* plan = nullptr;
    uint64_t element_count = 0;
    DataType datatype = DataType::Float16;
};

void launchAllReduce(const AllReduceRequest& request, const AllReducePlan* plan,
                     const CollectiveKernelArgs& common, cudaStream_t stream);
void launchAllReduceExecutor(const AllReduceKernelArgs& args,
                             cudaStream_t stream);

}  // namespace mooncake

#endif  // MOONCAKE_PG_COLLECTIVE_ALLREDUCE_H
