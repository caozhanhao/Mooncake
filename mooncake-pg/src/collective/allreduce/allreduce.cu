#include "collective/allreduce/allreduce.h"

#include <cstdint>

#include "collective/allreduce/flat_ring.cuh"

namespace mooncake {
namespace {

using namespace mooncake::device;

__global__ void allReduceExecutorKernel(AllReduceKernelArgs args) {
    __shared__ AllReduceBucketDevicePlan device_plan;
    __shared__ uint64_t view_epoch;
    __shared__ uint64_t collective_sequence;
    __shared__ int32_t self_participating;
    __shared__ int32_t plan_error;
    __shared__ InGroupRank plan_failed_peer;

    const auto& common = args.common;
    const uint32_t bytes_per_element =
        flat_ring::allReduceElementBytes(args.datatype);

    if (!prepareCollectiveInvocation(common)) {
        reportCollectiveFailureAndWait(common);
        return;
    }

    // Every launch reads the current Coordinator plan through a stable handle.
    // A failed launch never changes plan or runs again. A later eager call or
    // CUDA Graph replay is a new application invocation and can observe the
    // plan published by sync-after-failure without graph recapture.
    if (threadIdx.x == 0) {
        const auto& published = *args.plan.value;
        view_epoch = published.view_epoch;
        // Sequence identifies this invocation on its physical lane. The
        // counter is shared by every collective operation using that lane.
        collective_sequence = atomicAdd(
            reinterpret_cast<unsigned long long*>(common.invocation_sequence),
            1ULL);
        self_participating = static_cast<int32_t>(published.self_participating);
        plan_error = published.error_code;
        plan_failed_peer = published.failed_peer;
        if (plan_error == 0 && published.bucket_count != 0) {
            const uint64_t message_bytes =
                args.element_count * bytes_per_element;
            uint32_t bucket_index = 0;
            while (bucket_index + 1 < published.bucket_count &&
                   message_bytes >
                       published.buckets[bucket_index].max_message_bytes) {
                ++bucket_index;
            }
            device_plan = published.buckets[bucket_index];
        } else if (plan_error == 0) {
            plan_error =
                static_cast<int32_t>(CollectiveProtocolError::Unsupported);
        }
    }
    __syncthreads();

    bool success = true;
    // Membership is Coordinator-owned. An inactive captured rank is a local
    // identity for this invocation; the kernel does not invent another role.
    if (self_participating == 0) {
        copyCollectiveBytes(args.output, args.input,
                            args.element_count * bytes_per_element);
    } else {
        success = plan_error == 0;
        if (!success) {
            flat_ring::setCollectiveError(args, plan_error, plan_failed_peer);
        } else {
            success = flat_ring::run(args, device_plan.flat_ring, args.input,
                                     view_epoch, collective_sequence);
        }
    }
    __syncthreads();

    // Failure may leave output partially written. PG reports it and finishes
    // this invocation; the application owns any later retry.
    if (!success) reportCollectiveFailureAndWait(common);
}

}  // namespace

void launchAllReduceExecutor(const AllReduceKernelArgs& args,
                             cudaStream_t stream) {
    allReduceExecutorKernel<<<1, 256, 0, stream>>>(args);
}

}  // namespace mooncake
