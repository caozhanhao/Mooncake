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
    __shared__ uint32_t retry_attempt;
    __shared__ int32_t sequence_initialized;
    __shared__ int32_t self_participating;
    __shared__ int32_t plan_error;
    __shared__ InGroupRank plan_failed_peer;

    const auto& context = args.context;
    const auto& resources = context.resources;
    auto* control = resources.control;
    const uint32_t bytes_per_element = allReduceElementBytes(args.datatype);

    if (!enterCollectiveFailureGate(context)) return;

    // This snapshot is part of the common invocation protocol, not a graph
    // special case. An eager call can also fail after partially committing
    // output and then be retried against a new Coordinator view.
    copyCollectiveBytes(args.retry_input, args.input,
                        args.element_count * bytes_per_element);
    __syncthreads();

    if (threadIdx.x == 0) {
        sequence_initialized = 0;
        retry_attempt = 0;
    }
    __syncthreads();

    // This is a recovery envelope, not a persistent kernel. A successful call
    // runs one attempt and exits. Only a failed attempt parks at the mapped
    // host gate while the CPU reports failure and applies an authoritative
    // GroupView. The next attempt reloads algorithm, membership and bindings
    // through the stable binding, so graph replay does not capture them.
    while (true) {
        if (threadIdx.x == 0) {
            const uint32_t slot_index = static_cast<uint32_t>(mc_ld_acquire(
                reinterpret_cast<const int*>(context.binding.active_slot)));
            const auto* plans =
                static_cast<const AllReduceKernelPlan*>(context.binding.slots);
            const auto& published = plans[slot_index];
            view_epoch = published.view_epoch;
            // Sequence identifies this invocation on its physical lane. The
            // mapped counter is reset before a new plan is published, so the
            // active-slot acquire above establishes its view-local domain.
            // A retry keeps the value already acquired here and advances its
            // separate attempt domain after CPU authorizes recovery.
            if (!sequence_initialized) {
                collective_sequence = atomicAdd_system(
                    reinterpret_cast<unsigned long long*>(
                        context.binding.lane_sequences + context.lane_index),
                    1ULL);
                sequence_initialized = 1;
            }
            self_participating =
                static_cast<int32_t>(published.self_participating);
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
            control->failed_peer = -1;
            control->first_error_code = 0;
            control->resource_idle = 1;
        }
        __syncthreads();

        bool success = true;
        // Membership is Coordinator-owned. An inactive captured rank is a
        // local identity for this replay; the kernel does not invent another
        // role. Restoring the invocation snapshot also removes any partial
        // output written by an attempt that failed before self deactivation.
        if (self_participating == 0) {
            copyCollectiveBytes(args.output, args.retry_input,
                                args.element_count * bytes_per_element);
        } else {
            success = plan_error == 0;
            if (!success) {
                setCollectiveError(args, plan_error, plan_failed_peer);
            } else {
                switch (kernel_plan.algorithm) {
                    case AllReduceAlgorithm::FlatRing:
                        success = flat_ring::run(
                            args, kernel_plan.flat_ring, args.retry_input,
                            view_epoch, collective_sequence, retry_attempt);
                        break;
                    case AllReduceAlgorithm::Hierarchical:
                        success = hierarchical_allreduce::run(
                            args, kernel_plan.hierarchical, args.retry_input,
                            view_epoch, collective_sequence, retry_attempt);
                        break;
                }
            }
        }
        __syncthreads();

        if (success ||
            !publishAndWaitForCollectiveRecovery(context, view_epoch)) {
            return;
        }
        if (threadIdx.x == 0) ++retry_attempt;
        __syncthreads();
    }
}

}  // namespace

void launchAllReduceExecutor(const AllReduceExecutorArgs& args,
                             cudaStream_t stream) {
    allReduceExecutorKernel<<<1, 256, 0, stream>>>(args);
}

}  // namespace mooncake
