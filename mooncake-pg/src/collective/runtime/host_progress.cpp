#include "collective/runtime/host_progress.h"

#include <algorithm>
#include <atomic>
#include <string>
#include <system_error>
#include <thread>
#include <utility>

#include <glog/logging.h>

namespace mooncake {

EagerSubmission::~EagerSubmission() noexcept {
    if (completion) (void)cudaEventDestroy(completion);
}

PGResult<std::unique_ptr<CollectiveHostProgress>>
CollectiveHostProgress::create(DeviceId device,
                               CollectiveFailureReportCallback report_failure) {
    auto progress = std::unique_ptr<CollectiveHostProgress>(
        new CollectiveHostProgress(device, std::move(report_failure)));
    try {
        progress->thread_ =
            std::thread(&CollectiveHostProgress::progressLoop, progress.get());
    } catch (const std::system_error& error) {
        return makePGError(
            PGErrorCode::SystemError,
            std::string("failed to start collective host progress: ") +
                error.what());
    }
    return progress;
}

CollectiveHostProgress::~CollectiveHostProgress() noexcept { stop(); }

void CollectiveHostProgress::registerFailureTarget(
    std::shared_ptr<CollectiveResourceLease> resources,
    CollectiveFailureTarget target) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto source =
        std::find_if(failure_sources_.begin(), failure_sources_.end(),
                     [&](const auto& candidate) {
                         return candidate.resources == resources;
                     });
    if (source != failure_sources_.end()) {
        source->targets.push_back(target);
        return;
    }
    failure_sources_.push_back(FailureSource{
        .resources = std::move(resources),
        .targets = {target},
    });
}

void CollectiveHostProgress::removeFailureSourceLocked(
    const std::shared_ptr<CollectiveResourceLease>& resources) {
    failure_sources_.erase(
        std::remove_if(
            failure_sources_.begin(), failure_sources_.end(),
            [&](const auto& source) { return source.resources == resources; }),
        failure_sources_.end());
}

void CollectiveHostProgress::unregisterFailureTarget(
    const std::shared_ptr<CollectiveResourceLease>& resources,
    uint64_t failure_target_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto source =
        std::find_if(failure_sources_.begin(), failure_sources_.end(),
                     [&](const auto& candidate) {
                         return candidate.resources == resources;
                     });
    if (source == failure_sources_.end()) return;
    source->targets.erase(
        std::remove_if(source->targets.begin(), source->targets.end(),
                       [&](const auto& target) {
                           return target.failure_target_id == failure_target_id;
                       }),
        source->targets.end());
    if (source->targets.empty()) failure_sources_.erase(source);
}

void CollectiveHostProgress::stopObserving(
    const std::shared_ptr<CollectiveResourceLease>& resources) {
    std::lock_guard<std::mutex> lock(mutex_);
    removeFailureSourceLocked(resources);
}

void CollectiveHostProgress::markCompletionUnproven() {
    std::lock_guard<std::mutex> lock(mutex_);
    has_unproven_completion_ = true;
}

void CollectiveHostProgress::submit(
    std::unique_ptr<EagerSubmission> submission) {
    std::lock_guard<std::mutex> lock(mutex_);
    eager_submissions_.push_back(std::move(submission));
}

bool CollectiveHostProgress::retireCompletedSubmission() {
    std::unique_ptr<EagerSubmission> completed;
    cudaError_t query = cudaErrorNotReady;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (size_t index = 0; index < eager_submissions_.size(); ++index) {
            query = cudaEventQuery(eager_submissions_[index]->completion);
            if (query == cudaErrorNotReady) continue;

            completed = std::move(eager_submissions_[index]);
            eager_submissions_[index] = std::move(eager_submissions_.back());
            eager_submissions_.pop_back();
            if (query == cudaSuccess) {
                removeFailureSourceLocked(completed->resources);
            } else {
                has_unproven_completion_ = true;
            }
            break;
        }
    }
    if (!completed) return false;

    if (query != cudaSuccess) {
        // Completion is unproven. Keep observing failure reports and retain
        // the submitted lease until shutdown rather than releasing memory
        // that the device may still reference.
        LOG(WARNING) << "Collective completion query failed: "
                     << cudaGetErrorString(query);
        return true;
    }
    if (!completed->resources->retire()) {
        LOG(WARNING) << "Retained collective resources after completion";
    }
    return true;
}

std::optional<CollectiveHostProgress::FailureClaim>
CollectiveHostProgress::claimFailure(const FailureSource& source) {
    auto& failure = source.resources->control.host->failure;
    auto state = std::atomic_ref<uint32_t>(failure.state);
    uint32_t expected = static_cast<uint32_t>(CollectiveFailureState::Pending);
    if (!state.compare_exchange_strong(
            expected, static_cast<uint32_t>(CollectiveFailureState::Handling),
            std::memory_order_acq_rel, std::memory_order_acquire)) {
        return std::nullopt;
    }

    const auto target = std::find_if(
        source.targets.begin(), source.targets.end(), [&](const auto& value) {
            return value.failure_target_id == failure.failure_target_id;
        });
    PG_ASSERT(target != source.targets.end(),
              "collective failure report has no tracked target");
    return FailureClaim{
        .resources = source.resources,
        .target = *target,
        .failed_peer = failure.failed_peer,
    };
}

void CollectiveHostProgress::handleFailure(const FailureClaim& failure) {
    if (failure.target.failed_ranks_hint && failure.failed_peer >= 0 &&
        static_cast<size_t>(failure.failed_peer) <
            failure.target.failed_ranks_hint_count) {
        failure.target.failed_ranks_hint[failure.failed_peer] = 1;
    }

    if (report_failure_ && failure.failed_peer >= 0) {
        auto result = report_failure_(failure.failed_peer);
        if (!result.has_value()) {
            LOG(WARNING) << "Collective failure report failed for peer "
                         << failure.failed_peer << ": "
                         << result.error().message;
        }
    }

    auto& report = failure.resources->control.host->failure;
    std::atomic_ref<uint32_t>(report.state)
        .store(static_cast<uint32_t>(CollectiveFailureState::Acknowledged),
               std::memory_order_release);
}

bool CollectiveHostProgress::handleOneFailure() {
    std::optional<FailureClaim> failure;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& source : failure_sources_) {
            failure = claimFailure(source);
            if (failure.has_value()) break;
        }
    }
    if (!failure.has_value()) return false;
    handleFailure(*failure);
    return true;
}

void CollectiveHostProgress::progressLoop() noexcept {
    const auto select = cudaSetDevice(device_);
    if (select != cudaSuccess) {
        LOG(ERROR) << "Collective host progress cannot select CUDA device "
                   << device_ << ": " << cudaGetErrorString(select);
        return;
    }
    while (!stopping_.load(std::memory_order_acquire)) {
        bool progressed = handleOneFailure();
        progressed |= retireCompletedSubmission();
        if (!progressed) {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
    }
}

bool CollectiveHostProgress::drain(std::chrono::milliseconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (true) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (has_unproven_completion_) return false;
            if (eager_submissions_.empty()) return true;
        }
        if (std::chrono::steady_clock::now() >= deadline) return false;
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
}

void CollectiveHostProgress::stop() noexcept {
    if (stopping_.exchange(true, std::memory_order_acq_rel)) return;
    if (thread_.joinable()) thread_.join();
}

}  // namespace mooncake
