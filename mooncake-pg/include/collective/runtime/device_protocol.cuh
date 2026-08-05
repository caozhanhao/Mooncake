#ifndef MOONCAKE_PG_COLLECTIVE_RUNTIME_DEVICE_PROTOCOL_CUH
#define MOONCAKE_PG_COLLECTIVE_RUNTIME_DEVICE_PROTOCOL_CUH

#include "collective/runtime/kernel_context.cuh"
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

// Failure recovery is part of every invocation, including eager invocation.
// CUDA Graph only changes the lane lifetime; it does not select a different
// device protocol. The CPU may publish a new binding while this kernel is
// parked, but only the Coordinator decides whether its plan changes the
// algorithm or membership.
inline __device__ bool publishAndWaitForCollectiveRecovery(
    const CollectiveKernelContext& context, uint64_t failure_view_epoch) {
    __shared__ int retry;
    if (threadIdx.x == 0) {
        const auto& resources = context.resources;
        auto& control = *resources.control;
        auto& gate = control.failure_gate;
        gate.error_code = control.first_error_code;
        gate.failed_peer = control.failed_peer;
        gate.failure_cookie = context.failure_cookie;
        gate.failure_view_epoch = failure_view_epoch;
        mc_st_release_u32(
            &gate.state,
            static_cast<uint32_t>(CollectiveFailureGateState::FailurePending));
        while (true) {
            const auto state = static_cast<CollectiveFailureGateState>(
                mc_ld_acquire(reinterpret_cast<const int*>(&gate.state)));
            if (state == CollectiveFailureGateState::Open ||
                state == CollectiveFailureGateState::Closed) {
                retry = state == CollectiveFailureGateState::Open ? 1 : 0;
                break;
            }
        }
    }
    __syncthreads();
    return retry != 0;
}

// A graph-pinned lane outlives one kernel invocation. If its preceding attempt
// could not prove the underlying transport idle, the gate remains closed
// across replay. The next invocation may ask CPU progress to recover again,
// but it must not touch the snapshot or transport arena until CPU explicitly
// reopens the gate.
inline __device__ bool enterCollectiveFailureGate(
    const CollectiveKernelContext& context) {
    __shared__ int gate_closed;
    __shared__ uint64_t failure_view_epoch;
    if (threadIdx.x == 0) {
        const auto& resources = context.resources;
        auto& control = *resources.control;
        auto& gate = control.failure_gate;
        gate_closed = static_cast<CollectiveFailureGateState>(mc_ld_acquire(
                          reinterpret_cast<const int*>(&gate.state))) ==
                              CollectiveFailureGateState::Closed
                          ? 1
                          : 0;
        if (gate_closed) {
            control.first_error_code = gate.error_code;
            control.failed_peer = gate.failed_peer;
            failure_view_epoch = gate.failure_view_epoch;
        }
    }
    __syncthreads();
    if (!gate_closed) return true;
    return publishAndWaitForCollectiveRecovery(context, failure_view_epoch);
}

}  // namespace mooncake

#endif  // MOONCAKE_PG_COLLECTIVE_RUNTIME_DEVICE_PROTOCOL_CUH
