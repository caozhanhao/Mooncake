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
#include <utility>
#include <vector>

#include <cuda_alike.h>

#include "collective/runtime/resource_pool.h"
#include "error_types.h"

namespace mooncake {

using CollectiveFailureReportCallback =
    std::function<PGResult<void>(InGroupRank failed_peer)>;

struct CollectiveFailureTarget {
    uint64_t failure_target_id = 0;
    int32_t* failed_ranks_hint = nullptr;
    size_t failed_ranks_hint_count = 0;
};

// One launched eager invocation. The monitor owns this value until its
// completion event proves that the resource lease can be retired.
struct EagerSubmission {
    explicit EagerSubmission(std::shared_ptr<CollectiveResourceLease> value)
        : resources(std::move(value)) {}
    ~EagerSubmission() noexcept;

    EagerSubmission(const EagerSubmission&) = delete;
    EagerSubmission& operator=(const EagerSubmission&) = delete;

    std::shared_ptr<CollectiveResourceLease> resources;
    cudaEvent_t completion = nullptr;
};

// Communicator-scoped host loop. It observes device failure reports for eager
// and captured resources, and owns eager submissions until completion. It does
// not progress collective data or own graph-retention policy.
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
    void stopObserving(
        const std::shared_ptr<CollectiveResourceLease>& resources);
    void markCompletionUnproven();
    void submit(std::unique_ptr<EagerSubmission> submission);

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

    CollectiveMonitor(DeviceId device,
                      CollectiveFailureReportCallback report_failure)
        : device_(device), report_failure_(std::move(report_failure)) {}

    bool retireCompletedSubmission();
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
    std::vector<std::unique_ptr<EagerSubmission>> eager_submissions_;
    bool has_unproven_completion_ = false;
    std::thread thread_;
    std::atomic<bool> stopping_{false};
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_COLLECTIVE_RUNTIME_COLLECTIVE_MONITOR_H
