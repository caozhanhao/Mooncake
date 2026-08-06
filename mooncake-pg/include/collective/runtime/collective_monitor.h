#ifndef MOONCAKE_PG_COLLECTIVE_RUNTIME_COLLECTIVE_MONITOR_H
#define MOONCAKE_PG_COLLECTIVE_RUNTIME_COLLECTIVE_MONITOR_H

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cuda_alike.h>

#include "collective/runtime/resource_pool.h"
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

// Stream-ordered evidence that one eager use of the resources has completed.
// The monitor owns this value until the event permits resource retirement.
struct StreamCompletion {
    explicit StreamCompletion(std::shared_ptr<CollectiveResourceLease> value)
        : resources(std::move(value)) {}
    ~StreamCompletion() noexcept;

    StreamCompletion(const StreamCompletion&) = delete;
    StreamCompletion& operator=(const StreamCompletion&) = delete;

    std::shared_ptr<CollectiveResourceLease> resources;
    cudaEvent_t completion = nullptr;
};

// Communicator-scoped host loop. It observes device failure reports and owns
// resource retirement for both stream completions and captured CUDA Graphs. It
// does not progress collective data.
class CollectiveMonitor {
   public:
    static PGResult<std::unique_ptr<CollectiveMonitor>> create(
        DeviceId device, CollectiveFailureReportCallback report_failure);
    ~CollectiveMonitor() noexcept;

    void registerFailureTarget(
        std::shared_ptr<CollectiveResourceLease> resources,
        CollectiveFailureTarget target);
    void unregisterFailureTarget(
        const std::shared_ptr<CollectiveResourceLease>& resources,
        uint64_t failure_target_id);
    PGResult<std::shared_ptr<CollectiveResourceLease>> findGraphResources(
        uint64_t graph_id, cudaStream_t capture_stream) const;
    PGResult<void> retainGraphResources(
        const GraphCaptureState& capture, cudaStream_t capture_stream,
        std::shared_ptr<CollectiveResourceLease> resources);
    void markCompletionUnproven();
    void retainStreamCompletion(std::unique_ptr<StreamCompletion> completion);

    bool drain(std::chrono::milliseconds timeout);
    void stop() noexcept;

    CollectiveMonitor(const CollectiveMonitor&) = delete;
    CollectiveMonitor& operator=(const CollectiveMonitor&) = delete;

   private:
    struct FailureSource {
        std::shared_ptr<CollectiveResourceLease> resources;
        std::vector<CollectiveFailureTarget> targets;
    };

    struct FailureClaim {
        std::shared_ptr<CollectiveResourceLease> resources;
        CollectiveFailureTarget target;
        InGroupRank failed_peer = -1;
    };

    struct CapturedResources {
        // Reusing one resource set is safe only while captured nodes remain
        // ordered by this stream. Multi-stream capture is deferred.
        cudaStream_t capture_stream = nullptr;
        std::shared_ptr<CollectiveResourceLease> resources;
    };

    struct GraphReleaseQueue;
    struct GraphReleasePayload;

    CollectiveMonitor(DeviceId device,
                      CollectiveFailureReportCallback report_failure)
        : device_(device), report_failure_(std::move(report_failure)) {}

    static void CUDART_CB graphResourcesReleased(void* payload);
    bool retireCompletedStream();
    bool retireReleasedGraph();
    void retireResources(
        const std::shared_ptr<CollectiveResourceLease>& resources);
    void retireGraphResourcesAtShutdown() noexcept;
    static std::optional<FailureClaim> claimFailure(
        const FailureSource& source);
    void handleFailure(const FailureClaim& failure);
    bool handleOneFailure();
    void monitorLoop() noexcept;
    void removeFailureSourceLocked(
        const std::shared_ptr<CollectiveResourceLease>& resources);

    DeviceId device_ = kInvalidDeviceId;
    CollectiveFailureReportCallback report_failure_;

    mutable std::mutex mutex_;
    std::vector<FailureSource> failure_sources_;
    std::vector<std::unique_ptr<StreamCompletion>> stream_completions_;
    std::unordered_map<uint64_t, CapturedResources> graph_resources_;
    std::shared_ptr<GraphReleaseQueue> graph_release_queue_;
    bool has_unproven_completion_ = false;
    std::thread thread_;
    std::atomic<bool> stopping_{false};
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_COLLECTIVE_RUNTIME_COLLECTIVE_MONITOR_H
