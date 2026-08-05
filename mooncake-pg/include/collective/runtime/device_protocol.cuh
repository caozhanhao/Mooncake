#ifndef MOONCAKE_PG_COLLECTIVE_RUNTIME_DEVICE_PROTOCOL_CUH
#define MOONCAKE_PG_COLLECTIVE_RUNTIME_DEVICE_PROTOCOL_CUH

#include "collective/runtime/kernel_args.cuh"
#include "transport/device/device_ops.cuh"

namespace mooncake {

using namespace mooncake::device;

inline __device__ bool collectiveTimedOut(uint64_t start,
                                          uint64_t timeout_ticks) {
    return timeout_ticks != 0 && clock64() - start >= timeout_ticks;
}

inline __device__ void setCollectiveError(
    const CollectiveKernelResources& resources, int32_t error_code,
    InGroupRank failed_peer) {
    if (threadIdx.x != 0) return;
    auto& control = *resources.control;
    if (control.first_error_code == 0) {
        control.first_error_code = error_code;
        control.failed_peer = failed_peer;
    }
}

inline __device__ bool waitForCollectiveToken(
    const uint64_t* address, uint64_t expected,
    const CollectiveKernelResources& resources, InGroupRank failed_peer) {
    __shared__ int success;
    if (threadIdx.x == 0) {
        const uint64_t start = clock64();
        while (mc_ld_acquire_u64(address) != expected &&
               !collectiveTimedOut(start, resources.timeout_device_ticks)) {
        }
        success = mc_ld_acquire_u64(address) == expected ? 1 : 0;
    }
    __syncthreads();
    if (!success) {
        setCollectiveError(
            resources, static_cast<int32_t>(CollectiveProtocolError::Timeout),
            failed_peer);
        __syncthreads();
    }
    return success != 0;
}

inline __device__ void copyCollectiveBytes(void* destination,
                                           const void* source, uint64_t bytes,
                                           uint32_t worker_index,
                                           uint32_t worker_count) {
    auto* output = static_cast<char*>(destination);
    const auto* input = static_cast<const char*>(source);
    for (uint64_t index = worker_index; index < bytes; index += worker_count) {
        output[index] = input[index];
    }
}

inline __device__ void copyCollectiveBytes(void* destination,
                                           const void* source, uint64_t bytes) {
    copyCollectiveBytes(destination, source, bytes, threadIdx.x, blockDim.x);
}

// A graph-pinned invocation may reuse its control block after a preceding
// failure. It can start only when the preceding transport no longer references
// the pooled data-plane resources. Resource availability is a lifetime rule,
// not an authorization to retry the failed invocation.
inline __device__ bool prepareCollectiveInvocation(
    const CollectiveKernelArgs& common) {
    __shared__ int resource_available;
    if (threadIdx.x == 0) {
        const auto& resources = common.resources;
        auto& control = *resources.control;
        auto& failure = control.failure;
        if (static_cast<CollectiveFailureState>(
                mc_ld_acquire(reinterpret_cast<const int*>(&failure.state))) ==
            CollectiveFailureState::Acknowledged) {
            mc_st_release_u32(
                &failure.state,
                static_cast<uint32_t>(CollectiveFailureState::Idle));
        }
        const bool asynchronous_resource_idle =
            mc_ld_acquire(
                reinterpret_cast<const int*>(&control.resource_idle)) != 0;
        const auto host_command_state =
            static_cast<HostTransferCommandState>(mc_ld_acquire(
                reinterpret_cast<const int*>(&resources.host_command->state)));
        const bool host_command_reusable =
            host_command_state == HostTransferCommandState::Idle ||
            host_command_state == HostTransferCommandState::Completed ||
            host_command_state == HostTransferCommandState::Failed;
        resource_available =
            asynchronous_resource_idle && host_command_reusable ? 1 : 0;
        if (resource_available) {
            control.first_error_code = 0;
            control.failed_peer = -1;
        } else {
            control.first_error_code = failure.error_code;
            control.failed_peer = failure.failed_peer;
        }
    }
    __syncthreads();
    return resource_available != 0;
}

inline __device__ void reportCollectiveFailureAndWait(
    const CollectiveKernelArgs& common) {
    if (threadIdx.x == 0) {
        const auto& resources = common.resources;
        auto& control = *resources.control;
        auto& failure = control.failure;
        failure.error_code = control.first_error_code;
        failure.failed_peer = control.failed_peer;
        failure.failure_target_id = common.failure_target_id;
        mc_st_release_u32(&failure.state, static_cast<uint32_t>(
                                              CollectiveFailureState::Pending));
        while (static_cast<CollectiveFailureState>(mc_ld_acquire(
                   reinterpret_cast<const int*>(&failure.state))) !=
               CollectiveFailureState::Acknowledged) {
        }
    }
    __syncthreads();
}

}  // namespace mooncake

#endif  // MOONCAKE_PG_COLLECTIVE_RUNTIME_DEVICE_PROTOCOL_CUH
