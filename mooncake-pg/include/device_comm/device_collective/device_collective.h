#ifndef MOONCAKE_PG_DEVICE_COMM_DEVICE_COLLECTIVE_DEVICE_COLLECTIVE_H
#define MOONCAKE_PG_DEVICE_COMM_DEVICE_COLLECTIVE_DEVICE_COLLECTIVE_H

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

#include "common_types.h"
#include "control_plane/control_types.h"
#include "device_comm/device_arena.h"
#include "device_comm/device_transfer/transfer_service.h"
#include "device_comm/device_collective/device_collective_recovery.h"
#include "error_types.h"
#include "device_comm/device_collective/device_collective_types.cuh"
#include "gpu_runtime.h"

namespace mooncake {

class StrongStream;

class DeviceCollectiveRuntime {
   public:
    using RecoveryHandler = std::function<PGResult<void>(InGroupRank)>;

    static PGResult<std::unique_ptr<DeviceCollectiveRuntime>> create(
        DeviceTransferService& transfer_service, DeviceArena& arena,
        const DeviceArenaSlice& workspace, StrongStream& strong_stream,
        int device_index, GlobalRank self_rank, uint32_t max_group_size,
        size_t collective_timeout_us);

    ~DeviceCollectiveRuntime() noexcept;

    DeviceCollectiveRuntime(const DeviceCollectiveRuntime&) = delete;
    DeviceCollectiveRuntime& operator=(const DeviceCollectiveRuntime&) = delete;

    [[nodiscard]] const DeviceCollectiveEndpoint& localEndpoint() const;

    PGResult<void> useLocalOnly(InGroupRank self_rank);
    PGResult<void> materializeGroupView(const GroupView& view);

    PGResult<void> enableRecovery(DeviceCollectiveRecoveryWorker& worker,
                                  RecoveryHandler handler);

    PGResult<void> enqueueAllReduce(const void* send_buffer, void* recv_buffer,
                                    size_t count, DataType datatype,
                                    ReduceOp op,
                                    cudaStream_t user_stream_handle,
                                    int32_t* failed_ranks_hint);

    PGResult<void> shutdown(std::chrono::milliseconds eager_timeout);

   private:
    friend class MooncakeCommunicator;

    // Host-side packing recipe for one Runtime's control slice. Only make()
    // and bind() understand byte offsets; device code receives the bound
    // resources instead.
    struct ControlSliceLayout {
        uint64_t size = 0;
        uint64_t peers_offset = 0;
        uint64_t plan_offset = 0;
        uint64_t next_step_sequences_offset = 0;
        uint64_t invocation_offset = 0;
        uint64_t signal_slots_offset = 0;
        uint64_t consumed_ack_slots_offset = 0;
        uint32_t max_group_size = 0;

        static ControlSliceLayout make(uint32_t max_group_size);

        [[nodiscard]] DeviceCollectiveKernelResources bind(
            void* control_addr, uint64_t control_region_offset,
            const DeviceTransferHandle* transfer_handle,
            DeviceCollectiveTransferBuffer send_buffer,
            DeviceCollectiveTransferBuffer recv_buffer,
            uint64_t timeout_ticks) const noexcept;
    };

    struct HostControl;

    DeviceCollectiveRuntime(DeviceTransferService& transfer_service,
                            int device_index, GlobalRank self_rank,
                            uint64_t timeout_ticks, ControlSliceLayout layout,
                            DeviceArenaSlice control_slice,
                            DeviceCollectiveKernelResources kernel_resources,
                            StrongStream& strong_stream,
                            DeviceCollectiveEndpoint endpoint,
                            GpuStream control_stream, GpuEvent handoff_event);

    PGResult<void> initializeHostControl();
    PGResult<void> publishPlan(DeviceAllReducePlanImage plan,
                               bool reset_protocol_state);
    PGResult<void> attachGraphUse(const GpuCaptureInfo& capture);
    PGResult<void> recoverFailure(uint64_t generation);
    void releaseHostControl() noexcept;

    DeviceTransferService& transfer_service_;
    int device_index_ = -1;
    GlobalRank self_rank_ = kInvalidGlobalRank;
    uint64_t timeout_ticks_ = 0;
    ControlSliceLayout layout_;
    DeviceArenaSlice control_slice_;
    StrongStream& strong_stream_;
    DeviceCollectiveEndpoint endpoint_;
    DeviceCollectiveKernelResources kernel_resources_;
    HostControl* host_control_ = nullptr;
    DeviceAllReducePlanImage host_plan_;
    RecoveryHandler recovery_handler_;
    DeviceCollectiveRecoveryWorker* recovery_worker_ = nullptr;

    GpuStream control_stream_;
    GpuEvent handoff_event_;
    mutable std::mutex mutex_;
    std::atomic<size_t> live_graph_uses_{0};
    bool shutdown_requested_ = false;
    bool shutdown_complete_ = false;
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_DEVICE_COMM_DEVICE_COLLECTIVE_DEVICE_COLLECTIVE_H
