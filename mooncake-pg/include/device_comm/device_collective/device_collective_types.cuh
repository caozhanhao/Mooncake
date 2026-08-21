#ifndef MOONCAKE_PG_DEVICE_COMM_DEVICE_COLLECTIVE_DEVICE_COLLECTIVE_TYPES_CUH
#define MOONCAKE_PG_DEVICE_COMM_DEVICE_COLLECTIVE_DEVICE_COLLECTIVE_TYPES_CUH

#include <cstddef>
#include <cstdint>

#include <cuda_alike.h>

#include "common_types.h"
#include "device_comm/device_transfer/transfer_types.cuh"

namespace mooncake {

// Communicator-wide capacities. Their internal layouts belong to the selected
// protocol, not to the common collective runtime.
inline constexpr uint32_t kMaxDeviceCollectiveChannels = 32;
static_assert(kMaxDeviceCollectiveChannels <= kTransferLaneCount);
inline constexpr size_t kDeviceCollectiveBufferCapacity = 16ull << 20;

inline constexpr bool isDeviceAllReduceCombinationSupported(
    DataType datatype, ReduceOp op) noexcept {
    switch (op) {
        case ReduceOp::Sum:
        case ReduceOp::Product:
        case ReduceOp::Min:
        case ReduceOp::Max:
            break;
        default:
            return false;
    }
    switch (datatype) {
        case DataType::Float16:
            return op == ReduceOp::Sum;
        case DataType::Uint8:
        case DataType::Int8:
        case DataType::Int16:
        case DataType::Int32:
        case DataType::Int64:
        case DataType::Bfloat16:
        case DataType::Float32:
        case DataType::Float64:
        case DataType::Bool:
            return true;
        default:
            return false;
    }
}

enum class DevicePlanStatus : uint8_t {
    Unavailable = 0,
    Ready = 1,
};

template <typename Plan>
struct DevicePlanSlot {
    DevicePlanStatus status = DevicePlanStatus::Unavailable;
    Plan plan{};
};

// The device publishes a failure generation only after every active channel
// CTA has stopped touching the old protocol state and shared buffer.
struct alignas(64) DeviceCollectiveRecoveryMailbox {
    uint64_t failure_generation = 0;
    uint64_t ready_generation = 0;

    InGroupRank failed_rank = 0;
    uint64_t failed_hint_address = 0;
};

// Mutable GPU-side state for one collective launch.
// The first failing CTA latches failure metadata here; the last arriving
// CTA publishes a snapshot to the host recovery mailbox.
struct alignas(64) DeviceCollectiveInvocationState {
    uint32_t arrived_channels = 0;
    uint32_t failure_latched = 0;
    InGroupRank failed_rank = kInvalidInGroupRank;
    uint64_t failed_hint_address = 0;
};

// Protocol-independent control passed to a collective kernel. Payload and
// signal arena bindings belong to the selected protocol's device Plan.
struct DeviceCollectiveKernelResources {
    const DeviceTransferHandle* transfer_handle = nullptr;

    void* protocol_state = nullptr;
    uint64_t protocol_state_size = 0;

    DeviceCollectiveInvocationState* invocation = nullptr;
    DeviceCollectiveRecoveryMailbox* recovery = nullptr;
    uint64_t timeout_ticks = 0;
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_DEVICE_COMM_DEVICE_COLLECTIVE_DEVICE_COLLECTIVE_TYPES_CUH
