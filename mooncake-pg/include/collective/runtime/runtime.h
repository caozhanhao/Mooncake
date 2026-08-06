#ifndef MOONCAKE_PG_COLLECTIVE_RUNTIME_RUNTIME_H
#define MOONCAKE_PG_COLLECTIVE_RUNTIME_RUNTIME_H

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>

#include <cuda_alike.h>

#include "collective/runtime/collective_monitor.h"
#include "collective/runtime/collective_submission.h"
#include "error_types.h"

namespace mooncake {

// Communicator-scoped submission runtime. Operation code prepares resources;
// this class serializes admission, reuses a submission while graph nodes are
// captured, and gives the submitted resources to completion/failure tracking.
class CollectiveRuntime {
   public:
    static PGResult<std::unique_ptr<CollectiveRuntime>> create(
        DeviceId device, size_t collective_timeout_us,
        CollectiveFailureReportCallback failure_report_callback);
    ~CollectiveRuntime() noexcept;

    PGResult<void> submit(
        cudaStream_t stream, int32_t* failed_ranks_hint,
        size_t failed_ranks_hint_count,
        const std::function<PGResult<std::shared_ptr<CollectiveSubmission>>()>&
            prepare,
        const std::function<void(const CollectiveKernelArgs&)>& launch);

    uint64_t timeoutDeviceTicks() const { return timeout_device_ticks_; }

    void stopAccepting();
    bool drain(std::chrono::milliseconds timeout);

    CollectiveRuntime(const CollectiveRuntime&) = delete;
    CollectiveRuntime& operator=(const CollectiveRuntime&) = delete;

   private:
    struct CapturedSubmission {
        // Sharing is valid only for nodes ordered by the same capture stream.
        // Replay never consults this host-side binding; the graph user object
        // owns the submission itself.
        cudaStream_t capture_stream = nullptr;
        std::weak_ptr<CollectiveSubmission> submission;
    };

    CollectiveRuntime(DeviceId device, uint64_t timeout_device_ticks)
        : device_(device),
          timeout_device_ticks_(timeout_device_ticks) {}

    PGResult<std::shared_ptr<CollectiveSubmission>> acquireSubmission(
        const GraphCaptureState& capture, cudaStream_t stream,
        const std::function<
            PGResult<std::shared_ptr<CollectiveSubmission>>()>& prepare);

    std::unique_ptr<CollectiveMonitor> monitor_;
    DeviceId device_ = kInvalidDeviceId;
    uint64_t timeout_device_ticks_ = 0;

    std::mutex admission_mutex_;
    std::unordered_map<uint64_t, CapturedSubmission> captured_submissions_;
    uint64_t next_failure_target_id_ = 1;
    std::atomic<bool> accepting_{true};
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_COLLECTIVE_RUNTIME_RUNTIME_H
