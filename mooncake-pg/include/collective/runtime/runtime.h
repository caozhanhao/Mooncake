#ifndef MOONCAKE_PG_COLLECTIVE_RUNTIME_RUNTIME_H
#define MOONCAKE_PG_COLLECTIVE_RUNTIME_RUNTIME_H

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>

#include <cuda_alike.h>

#include "collective/runtime/collective_lane_pool.h"
#include "collective/runtime/collective_monitor.h"
#include "collective/runtime/resource_pool.h"
#include "error_types.h"

namespace mooncake {

// Communicator-scoped submission runtime. It owns invocation resources,
// graph retention, eager completion and failure monitoring; operation code owns
// validation, its typed plan and its kernel launcher.
class CollectiveRuntime {
   public:
    static PGResult<std::unique_ptr<CollectiveRuntime>> create(
        CollectiveBufferPool* buffer_pool, CollectiveControlPool* control_pool,
        HostTransferExecutor* host_transfer_executor, CollectiveLanePool* lanes,
        std::string te_location, TransferEngine* engine, DeviceId device,
        size_t collective_timeout_us,
        CollectiveFailureReportCallback failure_report_callback);
    ~CollectiveRuntime() noexcept;

    PGResult<void> submit(
        uint64_t view_epoch, cudaStream_t stream, int32_t* failed_ranks_hint,
        size_t failed_ranks_hint_count,
        const std::function<void(const CollectiveKernelArgs&)>& launch);

    void stopAccepting();
    bool drain(std::chrono::milliseconds timeout);

    CollectiveRuntime(const CollectiveRuntime&) = delete;
    CollectiveRuntime& operator=(const CollectiveRuntime&) = delete;

   private:
    CollectiveRuntime(CollectiveBufferPool* buffer_pool,
                      CollectiveControlPool* control_pool,
                      HostTransferExecutor* host_transfer_executor,
                      CollectiveLanePool* lanes, std::string te_location,
                      TransferEngine* engine, DeviceId device,
                      uint64_t timeout_device_ticks)
        : lanes_(lanes),
          resource_pool_(buffer_pool, control_pool, lanes_,
                         host_transfer_executor, device, std::move(te_location),
                         engine),
          device_(device),
          timeout_device_ticks_(timeout_device_ticks) {}

    CollectiveKernelArgs makeKernelArgs(
        const CollectiveResourceLease& resources,
        uint64_t failure_target_id) const;

    struct GraphResources {
        // Reusing one resource set is safe only while captured nodes remain
        // ordered by the same stream. Multi-stream graph capture is deferred.
        cudaStream_t bound_stream = nullptr;
        std::shared_ptr<CollectiveResourceLease> resources;
    };

    // GroupCollectiveEngine owns the runtime and destroys it before lanes_.
    CollectiveLanePool* lanes_ = nullptr;
    CollectiveResourcePool resource_pool_;
    std::unique_ptr<CollectiveMonitor> monitor_;
    DeviceId device_ = kInvalidDeviceId;
    uint64_t timeout_device_ticks_ = 0;

    std::mutex admission_mutex_;
    uint32_t next_lane_ = 0;
    std::optional<uint64_t> lane_view_epoch_;
    uint64_t next_failure_target_id_ = 1;
    std::unordered_map<uint64_t, GraphResources> graph_resources_;
    std::atomic<bool> accepting_{true};
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_COLLECTIVE_RUNTIME_RUNTIME_H
