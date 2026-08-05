#include "collective/executor/allreduce.cuh"

#include <cstdint>

#include "collective/executor/algorithm/flat_ring.cuh"
#include "collective/executor/algorithm/hierarchical.cuh"
#include "collective/executor/allreduce_device.cuh"

namespace mooncake {
namespace {

using namespace mooncake::device;

__global__ void allReduceExecutorKernel(AllReduceExecutorArgs args) {
    __shared__ AllReduceBucketKernelPlan kernel_plan;
    __shared__ uint64_t view_epoch;
    __shared__ uint64_t collective_sequence;
    __shared__ int32_t self_participating;
    __shared__ int32_t plan_error;
    __shared__ InGroupRank plan_failed_peer;

    const auto& context = args.context;
    const uint32_t bytes_per_element = allReduceElementBytes(args.datatype);

    if (!prepareCollectiveInvocation(context)) {
        reportCollectiveFailureAndWait(context);
        return;
    }

    // Every launch reads the active Coordinator plan through a stable handle.
    // A failed launch never changes plan or runs again. A later eager call or
    // CUDA Graph replay is a new application invocation and can observe the
    // plan published by sync-after-failure without graph recapture.
    if (threadIdx.x == 0) {
        const uint32_t slot_index = static_cast<uint32_t>(mc_ld_acquire(
            reinterpret_cast<const int*>(context.plan.active_slot)));
        const auto* plans =
            static_cast<const AllReduceKernelPlan*>(context.plan.slots);
        const auto& published = plans[slot_index];
        view_epoch = published.view_epoch;
        // Sequence identifies this invocation on its physical lane. The
        // mapped counter is reset before a new plan is published, so the
        // active-slot acquire above establishes its view-local domain.
        collective_sequence = atomicAdd_system(
            reinterpret_cast<unsigned long long*>(context.plan.lane_sequences +
                                                  context.lane_index),
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
            kernel_plan = published.buckets[bucket_index];
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
            setCollectiveError(args, plan_error, plan_failed_peer);
        } else {
            switch (kernel_plan.algorithm) {
                case AllReduceAlgorithm::FlatRing:
                    success =
                        flat_ring::run(args, kernel_plan.flat_ring, args.input,
                                       view_epoch, collective_sequence);
                    break;
                case AllReduceAlgorithm::Hierarchical:
                    success = hierarchical_allreduce::run(
                        args, kernel_plan.hierarchical, args.input, view_epoch,
                        collective_sequence);
                    break;
            }
        }
    }
    __syncthreads();

    // Failure may leave output partially written. PG reports it and finishes
    // this invocation; the application owns any later retry.
    if (!success) reportCollectiveFailureAndWait(context);
}

}  // namespace

void launchAllReduceExecutor(const AllReduceExecutorArgs& args,
                             cudaStream_t stream) {
    allReduceExecutorKernel<<<1, 256, 0, stream>>>(args);
}

}  // namespace mooncake
