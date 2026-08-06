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
class CollectiveRuntime;
class CollectiveSubmission;
class TransferEngine;
template <typename Plan>
class DevicePlan;

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

// Group-scoped owner of the typed AllReduce feature. It contains the complete
// policy-to-plan, submission-preparation and launch path without introducing a
// virtual collective registry.
class AllReduce {
   public:
    static PGResult<std::unique_ptr<AllReduce>> create(
        CollectiveBufferPool& buffer_pool, CollectiveChannels& channels,
        TransferEngine* engine, DeviceId device, std::string te_location);
    ~AllReduce() noexcept;

    bool supports(DataType datatype, ReduceOp op) const;
    PGResult<void> apply(const CollectivePlanSet& plans,
                         const ResolvedCollectiveView& view);
    PGResult<void> submit(CollectiveRuntime& runtime,
                          const AllReduceRequest& request,
                          cudaStream_t stream, int32_t* failed_ranks_hint,
                          size_t failed_ranks_hint_count);
    void retainPlanForProcessLifetime();

    AllReduce(const AllReduce&) = delete;
    AllReduce& operator=(const AllReduce&) = delete;

   private:
    AllReduce(CollectiveBufferPool& buffer_pool,
              CollectiveChannels& channels, TransferEngine* engine,
              DeviceId device, std::string te_location,
              std::unique_ptr<DevicePlan<AllReducePlan>> plan);

    PGResult<std::shared_ptr<CollectiveSubmission>> prepareSubmission(
        uint64_t timeout_device_ticks);

    CollectiveBufferPool& buffer_pool_;
    CollectiveChannels& channels_;
    TransferEngine* engine_ = nullptr;
    DeviceId device_ = kInvalidDeviceId;
    std::string te_location_;
    std::unique_ptr<DevicePlan<AllReducePlan>> plan_;
    AllReduceProtocol protocol_ = AllReduceProtocol::Legacy;
};

struct AllReduceKernelArgs {
    const void* input = nullptr;
    void* output = nullptr;
    CollectiveKernelArgs common;
    const AllReducePlan* plan = nullptr;
    uint64_t element_count = 0;
    DataType datatype = DataType::Float16;
};

void launchAllReduceExecutor(const AllReduceKernelArgs& args,
                             cudaStream_t stream);

}  // namespace mooncake

#endif  // MOONCAKE_PG_COLLECTIVE_ALLREDUCE_H
