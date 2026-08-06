#ifndef MOONCAKE_PG_COLLECTIVE_RUNTIME_COLLECTIVE_MONITOR_H
#define MOONCAKE_PG_COLLECTIVE_RUNTIME_COLLECTIVE_MONITOR_H

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

#include <cuda_alike.h>

#include "collective/runtime/collective_submission.h"
#include "error_types.h"
#include "gpu_runtime.h"

namespace mooncake {

using CollectiveFailureReportCallback =
    std::function<PGResult<void>(InGroupRank failed_peer)>;

struct CollectiveFailureTarget {
    uint64_t failure_target_id = 0;
    int32_t* failed_ranks_hint = nullptr;
    size_t failed_ranks_hint_count = 0;
};

// Stream-ordered evidence that one eager submission has completed. The monitor
// owns this value until the event permits submission retirement.
struct StreamCompletion {
    explicit StreamCompletion(std::shared_ptr<CollectiveSubmission> value)
        : submission(std::move(value)) {}
    ~StreamCompletion() noexcept;

    StreamCompletion(const StreamCompletion&) = delete;
    StreamCompletion& operator=(const StreamCompletion&) = delete;

    std::shared_ptr<CollectiveSubmission> submission;
    cudaEvent_t completion = nullptr;
};

// Communicator-scoped host loop. It observes device failure reports and owns
// submission retirement for both stream completions and captured GPU Graphs.
// It does not progress collective data.
class CollectiveMonitor {
   public:
    static PGResult<std::unique_ptr<CollectiveMonitor>> create(
        DeviceId device, CollectiveFailureReportCallback report_failure);
    ~CollectiveMonitor() noexcept;

    void registerFailureTarget(
        std::shared_ptr<CollectiveSubmission> submission,
        CollectiveFailureTarget target);
    void unregisterFailureTarget(
        const std::shared_ptr<CollectiveSubmission>& submission,
        uint64_t failure_target_id);
    // Adopt retirement responsibility for a captured submission. The graph
    // keeps it alive; its user-object callback only queues retirement.
    PGResult<void> adoptGraphSubmission(
        const GraphCaptureState& capture,
        std::shared_ptr<CollectiveSubmission> submission);
    void markCompletionUnproven();
    void adoptStreamCompletion(std::unique_ptr<StreamCompletion> completion);

    // The timeout bounds eager completion only. Captured submissions keep the
    // communicator alive until every graph and executable releases them.
    bool drain(std::chrono::milliseconds eager_timeout);
    void stop() noexcept;

    CollectiveMonitor(const CollectiveMonitor&) = delete;
    CollectiveMonitor& operator=(const CollectiveMonitor&) = delete;

   private:
    struct FailureSource {
        std::shared_ptr<CollectiveSubmission> submission;
        std::vector<CollectiveFailureTarget> targets;
    };

    struct FailureClaim {
        std::shared_ptr<CollectiveSubmission> submission;
        CollectiveFailureTarget target;
        InGroupRank failed_peer = -1;
    };

    struct GraphReleaseQueue;
    struct GraphReleasePayload;

    CollectiveMonitor(DeviceId device,
                      CollectiveFailureReportCallback report_failure)
        : device_(device), report_failure_(std::move(report_failure)) {}

    static void CUDART_CB graphSubmissionReleased(void* payload);
    bool retireCompletedStream();
    bool retireReleasedGraphSubmission();
    void retireSubmission(
        const std::shared_ptr<CollectiveSubmission>& submission);
    static std::optional<FailureClaim> claimFailure(
        const FailureSource& source);
    void handleFailure(const FailureClaim& failure);
    bool handleOneFailure();
    void monitorLoop() noexcept;
    void removeFailureSourceLocked(
        const std::shared_ptr<CollectiveSubmission>& submission);

    DeviceId device_ = kInvalidDeviceId;
    CollectiveFailureReportCallback report_failure_;

    mutable std::mutex mutex_;
    std::condition_variable drain_cv_;
    std::vector<FailureSource> failure_sources_;
    std::vector<std::unique_ptr<StreamCompletion>> stream_completions_;
    size_t graph_submission_count_ = 0;
    std::shared_ptr<GraphReleaseQueue> graph_release_queue_;
    bool has_unproven_completion_ = false;
    std::thread thread_;
    std::atomic<bool> stopping_{false};
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_COLLECTIVE_RUNTIME_COLLECTIVE_MONITOR_H
