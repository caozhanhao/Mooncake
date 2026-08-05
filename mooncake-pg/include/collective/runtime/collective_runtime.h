#ifndef MOONCAKE_PG_COLLECTIVE_RUNTIME_COLLECTIVE_RUNTIME_H
#define MOONCAKE_PG_COLLECTIVE_RUNTIME_COLLECTIVE_RUNTIME_H

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>

#include <cuda_alike.h>

#include "collective/runtime/collective_invocation.h"
#include "collective/runtime/collective_lane_pool.h"
#include "collective/runtime/collective_progress.h"
#include "collective/runtime/resource_pool.h"
#include "error_types.h"

namespace mooncake {

// Host control and kernel-launch boundaries:
//
//   GroupView -> BindingMaterializer -> CollectiveBinding
//                                           |
//   API -> collective dispatch -------------+-> BindingView
//              |                                  |
//              `-> Invocation --------------------+-> GroupCollectiveRuntime
//                                                         |             |
//                                                         v             v
//                                                   ResourcePool ProgressEngine
//                                                                      `->
//                                                                      FailureHandler
//
//   context -> operation executor -> algorithm -> transport route
//
// Materializers may inspect GroupView and process link managers. Everything
// below CollectiveKernelContext is group-local and sees only published kernel
// plans, raw kernel resources and InGroupRank peer bindings.

// One runtime per communicator and shared by all collective plugins. The first
// vertical slice registers only AllReduce; the admission, lane, graph and
// failure-reporting boundaries are intentionally collective-neutral.
class GroupCollectiveRuntime {
   public:
    static PGResult<std::unique_ptr<GroupCollectiveRuntime>> create(
        CollectiveBufferPool* buffer_pool, CollectiveControlPool* control_pool,
        CollectiveHostTransferProxy* host_transfer_proxy,
        CollectiveLanePool* lanes, std::string te_location,
        TransferEngine* engine, DeviceId device, size_t collective_timeout_us,
        CollectiveFailureReportCallback failure_report_callback);
    ~GroupCollectiveRuntime() noexcept;

    PGResult<void> execute(CollectiveInvocation& invocation,
                           CollectiveBindingView binding_view,
                           uint64_t view_epoch, cudaStream_t stream,
                           int32_t* failed_ranks_hint,
                           size_t failed_ranks_hint_count);

    void stopAccepting();
    bool drain(std::chrono::milliseconds timeout);

    GroupCollectiveRuntime(const GroupCollectiveRuntime&) = delete;
    GroupCollectiveRuntime& operator=(const GroupCollectiveRuntime&) = delete;

   private:
    GroupCollectiveRuntime(CollectiveBufferPool* buffer_pool,
                           CollectiveControlPool* control_pool,
                           CollectiveHostTransferProxy* host_transfer_proxy,
                           CollectiveLanePool* lanes, std::string te_location,
                           TransferEngine* engine, DeviceId device,
                           uint64_t timeout_device_ticks)
        : lanes_(lanes),
          resource_pool_(buffer_pool, control_pool, lanes_, host_transfer_proxy,
                         device, std::move(te_location), engine),
          device_(device),
          timeout_device_ticks_(timeout_device_ticks) {}

    CollectiveKernelContext makeKernelContext(
        const CollectiveResourceLease& resources,
        CollectiveBindingView binding_view, uint64_t failure_cookie) const;
    PGResult<std::shared_ptr<TrackedCollective>> acquireTrackedCollective(
        std::optional<uint64_t> capture_id, cudaStream_t capture_stream);
    void trackCollective(const std::shared_ptr<TrackedCollective>& collective,
                         CollectiveFailureTarget target,
                         cudaEvent_t completion);
    void rollbackEmptyCapturedCollective(uint64_t capture_id) noexcept;

    struct CapturedCollective {
        cudaStream_t capture_stream = nullptr;
        std::shared_ptr<TrackedCollective> collective;
    };

    // MooncakeCommunicator owns this group-scoped collaborator and destroys
    // the runtime before it.
    CollectiveLanePool* lanes_ = nullptr;
    CollectiveResourcePool resource_pool_;
    std::unique_ptr<CollectiveProgressEngine> progress_;
    DeviceId device_ = kInvalidDeviceId;
    uint64_t timeout_device_ticks_ = 0;

    std::mutex admission_mutex_;
    uint32_t next_lane_ = 0;
    std::optional<uint64_t> lane_view_epoch_;
    uint64_t next_failure_cookie_ = 1;
    std::unordered_map<uint64_t, CapturedCollective> captured_collectives_;
    std::atomic<bool> accepting_{true};
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_COLLECTIVE_RUNTIME_COLLECTIVE_RUNTIME_H
