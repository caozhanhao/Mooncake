#ifndef MOONCAKE_PG_DEVICE_COMM_DEVICE_COLLECTIVE_DEVICE_COLLECTIVE_KERNEL_CUH
#define MOONCAKE_PG_DEVICE_COMM_DEVICE_COLLECTIVE_DEVICE_COLLECTIVE_KERNEL_CUH

#include <cooperative_groups.h>
#include <transport/device/device_ops.cuh>

#include "device_comm/device_collective/device_collective_types.cuh"

namespace mooncake {
namespace collective_device {

namespace cg = cooperative_groups;

[[nodiscard]] __device__ __forceinline__ bool invocationFailed(
    const DeviceCollectiveKernelResources& resources, uint32_t* shared_result,
    cg::thread_block block) {
    bool failed = false;
    if (block.thread_rank() == 0) {
        auto* const latched = reinterpret_cast<unsigned int*>(
            &resources.invocation->failure_latched);
        failed = atomicAdd(latched, 0u) != 0;
        *shared_result = failed ? 1 : 0;
    }
    block.sync();
    failed = *shared_result != 0;
    block.sync();
    return failed;
}

// Called by thread 0 of the CTA that performs the final arrived_channels
// increment. Failed invocations wait here until the host replaces shared
// protocol state.
__device__ __forceinline__ void finalizeInvocation(
    const DeviceCollectiveKernelResources& resources) {
    auto* const invocation = resources.invocation;
    auto* const failed =
        reinterpret_cast<unsigned int*>(&invocation->failure_latched);
    auto* const arrived =
        reinterpret_cast<unsigned int*>(&invocation->arrived_channels);

    if (atomicAdd(failed, 0u) != 0u) {
        auto* const mailbox = resources.recovery;
        const uint64_t generation =
            device::mc_ld_acquire_u64(&mailbox->failure_generation) + 1;

        mailbox->failed_rank = invocation->failed_rank;
        mailbox->failed_hint_address = invocation->failed_hint_address;
        __threadfence_system();
        device::mc_st_release_u64(&mailbox->failure_generation, generation);
        while (device::mc_ld_acquire_u64(&mailbox->ready_generation) <
               generation) {
        }
    }

    atomicExch(failed, 0u);
    __threadfence();
    atomicExch(arrived, 0u);
}

// Completes this channel after success, a failure detected by another channel,
// or a new failure detected by this channel. Only the detecting channel
// supplies a failed rank; the first detector records the failure metadata.
__device__ __forceinline__ void completeChannel(
    const DeviceCollectiveKernelResources& resources, cg::thread_block block,
    InGroupRank detected_failed_rank = kInvalidInGroupRank,
    int32_t* failed_ranks_hint = nullptr) {
    // No thread may publish channel completion while another thread in the CTA
    // can still access the current Plan or protocol buffers.
    block.sync();
    if (block.thread_rank() == 0) {
        auto* const invocation = resources.invocation;
        auto* const failed =
            reinterpret_cast<unsigned int*>(&invocation->failure_latched);
        auto* const arrived =
            reinterpret_cast<unsigned int*>(&invocation->arrived_channels);

        if (detected_failed_rank != kInvalidInGroupRank &&
            atomicCAS(failed, 0u, 1u) == 0u) {
            invocation->failed_rank = detected_failed_rank;
            invocation->failed_hint_address =
                reinterpret_cast<uint64_t>(failed_ranks_hint);
        }

        __threadfence();
        const uint32_t previous = atomicAdd(arrived, 1u);
        if (previous + 1 == gridDim.x) {
            finalizeInvocation(resources);
        }
    }
    block.sync();
}

}  // namespace collective_device
}  // namespace mooncake

#endif  // MOONCAKE_PG_DEVICE_COMM_DEVICE_COLLECTIVE_DEVICE_COLLECTIVE_KERNEL_CUH
