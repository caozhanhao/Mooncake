#ifndef MOONCAKE_PG_DEVICE_COMM_DEVICE_COLLECTIVE_PROTOCOLS_FLAT_RING_ALL_REDUCE_H
#define MOONCAKE_PG_DEVICE_COMM_DEVICE_COLLECTIVE_PROTOCOLS_FLAT_RING_ALL_REDUCE_H

#include <cstddef>
#include <cstdint>
#include <memory>

#include "control_plane/control_types.h"
#include "device_comm/device_collective/protocols/flat_ring_all_reduce/flat_ring_all_reduce_types.cuh"
#include "error_types.h"
#include "gpu_runtime.h"

namespace mooncake {

class DeviceCollectiveResources;
class DeviceCollectiveWorkspace;
class DeviceTransferService;

// Owns every host-side AllReduce- and Flat-Ring-specific decision. The common
// runtime supplies ordering and recovery lifecycle only.
class FlatRingAllReduceProtocol {
   public:
    static PGResult<std::unique_ptr<FlatRingAllReduceProtocol>> create(
        DeviceTransferService& transfer_service,
        DeviceCollectiveWorkspace& workspace,
        DeviceCollectiveResources& resources, GpuStream& control_stream,
        int device_index, InGroupRank self_rank, uint32_t max_group_size);

    ~FlatRingAllReduceProtocol() noexcept;

    FlatRingAllReduceProtocol(const FlatRingAllReduceProtocol&) = delete;
    FlatRingAllReduceProtocol& operator=(const FlatRingAllReduceProtocol&) =
        delete;

    PGResult<void> useLocalOnly();
    PGResult<void> applyGroupView(const GroupView& view);
    PGResult<void> invalidate();

    [[nodiscard]] bool ready() const noexcept;

    PGResult<void> enqueue(const void* send_buffer, void* recv_buffer,
                           size_t count, DataType datatype, ReduceOp op,
                           cudaStream_t stream,
                           int32_t* failed_ranks_hint) const;

   private:
    static constexpr size_t kMinBytesPerChannelStep = 256ull << 10;

    struct HostState;

    FlatRingAllReduceProtocol(
        DeviceTransferService& transfer_service,
        DeviceCollectiveWorkspace& workspace,
        DeviceCollectiveResources& resources, GpuStream& control_stream,
        int device_index, InGroupRank self_rank, uint32_t max_group_size,
        FlatRingPersistentStateView state,
        FlatRingSignalLayout signal_layout) noexcept;

    PGResult<void> initializeHostState();
    void releaseHostState() noexcept;
    PGResult<void> publish(FlatRingAllReducePlan plan);
    [[nodiscard]] FlatRingAllReduceDeviceResources deviceResources()
        const noexcept;
    [[nodiscard]] static uint32_t chooseChannelCount(
        size_t size, uint32_t participant_count);

    DeviceTransferService& transfer_service_;
    DeviceCollectiveWorkspace& workspace_;
    DeviceCollectiveResources& resources_;
    GpuStream& control_stream_;
    int device_index_ = -1;
    InGroupRank self_rank_ = kInvalidInGroupRank;
    uint32_t max_group_size_ = 0;
    FlatRingPersistentStateView state_;
    FlatRingSignalLayout signal_layout_;
    HostState* host_state_ = nullptr;
    bool ready_ = false;
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_DEVICE_COMM_DEVICE_COLLECTIVE_PROTOCOLS_FLAT_RING_ALL_REDUCE_H
