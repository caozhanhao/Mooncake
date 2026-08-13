#ifndef MOONCAKE_PG_DEVICE_COMM_DEVICE_COLLECTIVE_DEVICE_COLLECTIVE_TYPES_CUH
#define MOONCAKE_PG_DEVICE_COMM_DEVICE_COLLECTIVE_DEVICE_COLLECTIVE_TYPES_CUH

#include <cstddef>
#include <cstdint>

#include <cuda_alike.h>

#include "common_types.h"
#include "device_comm/device_transfer/transfer_types.cuh"

namespace mooncake {

// This algorithm can bind at most one channel CTA to each transfer-service
// lane. The Service owns lane capacity; the collective still chooses how many
// channels to use for an invocation.
inline constexpr uint32_t kMaxDeviceCollectiveChannels = kTransferLaneCount;
inline constexpr size_t kDeviceCollectiveTransferBufferSize = 16ull << 20;
inline constexpr size_t kDeviceCollectiveWorkspaceSize =
    2 * kDeviceCollectiveTransferBufferSize;

enum class DeviceCollectivePlanStatus : uint32_t {
    Unavailable = 0,
    Ready = 1,
};

struct DeviceAllReducePlanImage {
    DeviceCollectivePlanStatus status =
        DeviceCollectivePlanStatus::Unavailable;

    InGroupRank self_rank = kInvalidInGroupRank;
    int32_t self_active_index = -1;
    uint32_t participant_count = 0;

    InGroupRank predecessor_rank = kInvalidInGroupRank;
    InGroupRank successor_rank = kInvalidInGroupRank;
};

// The device publishes a new failure generation only after every active
// channel CTA has stopped touching the old Plan, peer map, and shared buffers.
// Host publishes the matching ready generation after recovery has applied a
// usable or unavailable Plan.
struct alignas(64) DeviceCollectiveRecoveryMailbox {
    uint64_t failure_generation = 0;
    uint64_t ready_generation = 0;

    // The failure data below is valid only while failure_generation is newer
    // than ready_generation.
    InGroupRank failed_rank = 0;
    uint64_t failed_hint_address = 0;
};

// Device-only rendezvous reused by serialized invocations. Each channel CTA
// retires exactly once. The last CTA either resets it for the next invocation
// or publishes the already-latched failure and parks for host recovery.
struct alignas(64) DeviceCollectiveInvocationState {
    uint32_t arrived_channels = 0;
    uint32_t failure_latched = 0;
};

// Bound view of one [channel][peer] table in a Runtime control slice. The two
// offsets locate the table relative to the local Service region and to a
// peer's control slice respectively. `max_group_size` is the number of entries
// in each channel row.
struct DeviceCollectiveSignalTable {
    uint64_t* slots = nullptr;
    uint64_t local_region_offset = 0;
    uint64_t control_offset = 0;
    uint32_t max_group_size = 0;

    __device__ __forceinline__ uint64_t
    localSlotOffset(uint32_t channel_index, InGroupRank peer_rank) const {
        return local_region_offset +
               slotIndex(channel_index, peer_rank) * sizeof(uint64_t);
    }

    __device__ __forceinline__ uint64_t
    remoteSlotOffset(uint32_t channel_index, InGroupRank peer_rank) const {
        return control_offset +
               slotIndex(channel_index, peer_rank) * sizeof(uint64_t);
    }

   private:
    // The row layout is an implementation detail of the bound table, not part
    // of the collective kernel's addressing logic.
    __device__ __forceinline__ uint64_t slotIndex(uint32_t channel_index,
                                                  InGroupRank peer_rank) const {
        return static_cast<uint64_t>(channel_index) * max_group_size +
               static_cast<uint32_t>(peer_rank);
    }
};

// A group rank resolves to one process-wide transfer-service peer plus the
// communicator-specific remote control range inside that peer's region.
struct DeviceCollectivePeerBinding {
    uint32_t peer_idx = UINT32_MAX;
    uint64_t remote_control_offset = 0;
};

struct DeviceCollectivePeerTable {
    DeviceCollectivePeerBinding* entries = nullptr;
};

struct DeviceCollectiveTransferBuffer {
    void* addr = nullptr;
    uint64_t region_offset = 0;
    uint64_t size = 0;
};

// Fully bound resources consumed by operation-specific kernels. Kernels use
// these typed addresses and tables without knowing how the Runtime packed its
// control slice. Transfer-service resources are shared by the device; peers
// and control state belong to this Runtime's DeviceArena slice.
struct DeviceCollectiveKernelResources {
    const DeviceTransferHandle* transfer_handle = nullptr;
    DeviceCollectivePeerTable peers;
    DeviceCollectiveTransferBuffer send_buffer;
    DeviceCollectiveTransferBuffer recv_buffer;
    uint64_t timeout_ticks = 0;

    DeviceAllReducePlanImage* all_reduce_plan = nullptr;
    uint64_t* next_step_sequences = nullptr;
    DeviceCollectiveInvocationState* invocation = nullptr;
    DeviceCollectiveRecoveryMailbox* recovery = nullptr;

    DeviceCollectiveSignalTable signal_slots;
    DeviceCollectiveSignalTable consumed_ack_slots;
};

struct DeviceAllReduceKernelArgs {
    const void* send_buffer = nullptr;
    void* recv_buffer = nullptr;
    uint64_t count = 0;
    DataType datatype = DataType::Float32;
    uint32_t channel_count = 1;
    int32_t* failed_ranks_hint = nullptr;
};

// Implemented in device_collective.cu so ordinary host C++ never sees kernel
// launch syntax.
cudaError_t launchDeviceAllReduceKernel(
    const DeviceAllReduceKernelArgs& request,
    const DeviceCollectiveKernelResources& resources, cudaStream_t stream);

}  // namespace mooncake

#endif  // MOONCAKE_PG_DEVICE_COMM_DEVICE_COLLECTIVE_DEVICE_COLLECTIVE_TYPES_CUH
