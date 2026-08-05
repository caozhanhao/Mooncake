#ifndef MOONCAKE_PG_COLLECTIVE_RUNTIME_HOST_PROGRESS_H
#define MOONCAKE_PG_COLLECTIVE_RUNTIME_HOST_PROGRESS_H

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include <cuda_alike.h>

#include "collective/runtime/collective_failure.h"
#include "error_types.h"

namespace mooncake {

// One launched eager invocation. Host progress owns this value until its
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
class CollectiveHostProgress {
   public:
    static PGResult<std::unique_ptr<CollectiveHostProgress>> create(
        DeviceId device,
        std::unique_ptr<CollectiveFailureHandler> failure_handler);
    ~CollectiveHostProgress() noexcept;

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

    CollectiveHostProgress(const CollectiveHostProgress&) = delete;
    CollectiveHostProgress& operator=(const CollectiveHostProgress&) = delete;

   private:
    struct FailureSource {
        std::shared_ptr<CollectiveResourceLease> resources;
        std::vector<CollectiveFailureTarget> targets;
    };

    CollectiveHostProgress(
        DeviceId device,
        std::unique_ptr<CollectiveFailureHandler> failure_handler)
        : device_(device), failure_handler_(std::move(failure_handler)) {}

    bool retireCompletedSubmission();
    bool handleOneFailure();
    void progressLoop() noexcept;
    void removeFailureSourceLocked(
        const std::shared_ptr<CollectiveResourceLease>& resources);

    DeviceId device_ = kInvalidDeviceId;
    std::unique_ptr<CollectiveFailureHandler> failure_handler_;

    mutable std::mutex mutex_;
    std::vector<FailureSource> failure_sources_;
    std::vector<std::unique_ptr<EagerSubmission>> eager_submissions_;
    bool has_unproven_completion_ = false;
    std::thread thread_;
    std::atomic<bool> stopping_{false};
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_COLLECTIVE_RUNTIME_HOST_PROGRESS_H
