#ifndef MOONCAKE_PG_DEVICE_COMM_DEVICE_COLLECTIVE_ALGORITHMS_RING_ALGORITHM_CUH
#define MOONCAKE_PG_DEVICE_COMM_DEVICE_COLLECTIVE_ALGORITHMS_RING_ALGORITHM_CUH

#include <cstdint>

#include <cuda_alike.h>

#include "common_types.h"
#include "device_comm/device_collective/protocols/simple/simple_types.cuh"

namespace mooncake {

// An oversized chunk reduces pipeline overlap, so a cap is applied here.
inline constexpr uint64_t kMaxRingChunkBytes = 512ull * 1024;  // 512 KiB

// Ring neighbors, active-index mapping and workspace bindings for one group
// view. The runtime publishes this plan through a device control update.
struct RingPlan {
    DevicePlanStatus status = DevicePlanStatus::Unavailable;
    uint64_t view_epoch = kInvalidViewEpoch;

    SimpleWorkspace workspace;

    int32_t self_active_index = -1;
    uint32_t participant_count = 0;
    // Active index -> stable group slot, used only for operand addressing.
    InGroupRank in_group_ranks[kMaxNumRanks] = {};
    // Predecessor first, successor last. A two-rank Ring has one remote peer.
    CollectivePeer remote_peers[2] = {};
    uint32_t peer_count = 0;

    [[nodiscard]] __device__ __forceinline__ const CollectivePeer& predecessor()
        const {
        return remote_peers[0];
    }

    [[nodiscard]] __device__ __forceinline__ const CollectivePeer& successor()
        const {
        return remote_peers[peer_count - 1];
    }

    [[nodiscard]] __device__ __forceinline__ RemotePeerList
    remotePeers() const {
        return {remote_peers, peer_count};
    }
};

// Local device state owned by RingAlgorithm and shared by its kernels.
struct alignas(256) RingDeviceState {
    RingPlan plan;
    CollectiveRuntimeBindings collective;
    // Borrowed Simple bindings; progress is shared with other Simple
    // algorithms.
    SimpleBindings* simple = nullptr;
};

cudaError_t launchRingAllReduceKernel(const AllReduceRequest& request,
                                      RingDeviceState* state,
                                      cudaStream_t stream);
cudaError_t launchRingAllGatherKernel(const AllGatherRequest& request,
                                      RingDeviceState* state,
                                      cudaStream_t stream);

}  // namespace mooncake

#endif  // MOONCAKE_PG_DEVICE_COMM_DEVICE_COLLECTIVE_ALGORITHMS_RING_ALGORITHM_CUH
