#include "collective/runtime/collective_monitor.h"

#include <algorithm>
#include <atomic>
#include <string>
#include <system_error>
#include <thread>
#include <utility>

#include <glog/logging.h>

namespace mooncake {

struct CollectiveMonitor::GraphReleaseQueue {
    std::mutex mutex;
    std::vector<uint64_t> released;
    bool closed = false;
};

struct CollectiveMonitor::GraphReleasePayload {
    std::shared_ptr<GraphReleaseQueue> queue;
    uint64_t graph_id = 0;
    bool notify = false;
};

StreamCompletion::~StreamCompletion() noexcept {
    if (completion) (void)cudaEventDestroy(completion);
}

PGResult<std::unique_ptr<CollectiveMonitor>> CollectiveMonitor::create(
    DeviceId device, CollectiveFailureReportCallback report_failure) {
    auto monitor = std::unique_ptr<CollectiveMonitor>(
        new CollectiveMonitor(device, std::move(report_failure)));
    monitor->graph_release_queue_ = std::make_shared<GraphReleaseQueue>();
    try {
        monitor->thread_ =
            std::thread(&CollectiveMonitor::monitorLoop, monitor.get());
    } catch (const std::system_error& error) {
        return makePGError(
            PGErrorCode::SystemError,
            std::string("failed to start collective monitor: ") + error.what());
    }
    return monitor;
}

CollectiveMonitor::~CollectiveMonitor() noexcept { stop(); }

void CollectiveMonitor::registerFailureTarget(
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

void CollectiveMonitor::removeFailureSourceLocked(
    const std::shared_ptr<CollectiveResourceLease>& resources) {
    failure_sources_.erase(
        std::remove_if(
            failure_sources_.begin(), failure_sources_.end(),
            [&](const auto& source) { return source.resources == resources; }),
        failure_sources_.end());
}

void CollectiveMonitor::unregisterFailureTarget(
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

PGResult<std::shared_ptr<CollectiveResourceLease>>
CollectiveMonitor::findGraphResources(uint64_t graph_id,
                                      cudaStream_t capture_stream) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto found = graph_resources_.find(graph_id);
    if (found == graph_resources_.end()) {
        return std::shared_ptr<CollectiveResourceLease>{};
    }
    PG_VALIDATE_STATE(
        found->second.capture_stream == capture_stream,
        "multi-stream collective CUDA Graph capture is unsupported");
    return found->second.resources;
}

PGResult<void> CollectiveMonitor::retainGraphResources(
    const GraphCaptureState& capture, cudaStream_t capture_stream,
    std::shared_ptr<CollectiveResourceLease> resources) {
    // This callback proves that no graph or executable can use the resources
    // again. It does not represent completion of any individual replay.
    auto payload = std::make_unique<GraphReleasePayload>(GraphReleasePayload{
        .queue = graph_release_queue_,
        .graph_id = capture.id,
    });
    auto* payload_ptr = payload.get();
    PG_TRY(auto user_object,
           GpuGraphUserObject::create(payload_ptr, graphResourcesReleased));
    (void)payload.release();
    PG_TRY(user_object.moveTo(capture.graph));
    payload_ptr->notify = true;

    std::lock_guard<std::mutex> lock(mutex_);
    graph_resources_.emplace(
        capture.id, CapturedResources{.capture_stream = capture_stream,
                                      .resources = std::move(resources)});
    return {};
}

void CollectiveMonitor::markCompletionUnproven() {
    std::lock_guard<std::mutex> lock(mutex_);
    has_unproven_completion_ = true;
}

void CollectiveMonitor::retainStreamCompletion(
    std::unique_ptr<StreamCompletion> completion) {
    std::lock_guard<std::mutex> lock(mutex_);
    stream_completions_.push_back(std::move(completion));
}

void CUDART_CB CollectiveMonitor::graphResourcesReleased(void* opaque) {
    auto payload = std::unique_ptr<GraphReleasePayload>(
        static_cast<GraphReleasePayload*>(opaque));
    if (!payload->notify) return;

    const auto queue = payload->queue;
    std::lock_guard<std::mutex> lock(queue->mutex);
    if (queue->closed) return;
    queue->released.push_back(payload->graph_id);
}

void CollectiveMonitor::retireResources(
    const std::shared_ptr<CollectiveResourceLease>& resources) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        removeFailureSourceLocked(resources);
    }
    if (!resources->retire()) {
        LOG(WARNING) << "Retained collective resources after retirement";
    }
}

bool CollectiveMonitor::retireCompletedStream() {
    std::unique_ptr<StreamCompletion> completed;
    cudaError_t query = cudaErrorNotReady;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (size_t index = 0; index < stream_completions_.size(); ++index) {
            query = cudaEventQuery(stream_completions_[index]->completion);
            if (query == cudaErrorNotReady) continue;

            completed = std::move(stream_completions_[index]);
            stream_completions_[index] =
                std::move(stream_completions_.back());
            stream_completions_.pop_back();
            if (query != cudaSuccess) has_unproven_completion_ = true;
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
    retireResources(completed->resources);
    return true;
}

bool CollectiveMonitor::retireReleasedGraph() {
    uint64_t graph_id = 0;
    {
        std::lock_guard<std::mutex> lock(graph_release_queue_->mutex);
        if (graph_release_queue_->released.empty()) return false;
        graph_id = graph_release_queue_->released.back();
        graph_release_queue_->released.pop_back();
    }

    std::shared_ptr<CollectiveResourceLease> resources;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto found = graph_resources_.find(graph_id);
        if (found == graph_resources_.end()) return true;
        resources = std::move(found->second.resources);
        graph_resources_.erase(found);
    }
    retireResources(resources);
    return true;
}

std::optional<CollectiveMonitor::FailureClaim> CollectiveMonitor::claimFailure(
    const FailureSource& source) {
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

void CollectiveMonitor::handleFailure(const FailureClaim& failure) {
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

bool CollectiveMonitor::handleOneFailure() {
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

void CollectiveMonitor::monitorLoop() noexcept {
    const auto select = cudaSetDevice(device_);
    if (select != cudaSuccess) {
        LOG(ERROR) << "Collective monitor cannot select CUDA device " << device_
                   << ": " << cudaGetErrorString(select);
        return;
    }
    while (!stopping_.load(std::memory_order_acquire)) {
        bool progressed = handleOneFailure();
        progressed |= retireReleasedGraph();
        progressed |= retireCompletedStream();
        if (!progressed) {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
    }
}

bool CollectiveMonitor::drain(std::chrono::milliseconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (true) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (has_unproven_completion_) return false;
            if (stream_completions_.empty()) return true;
        }
        if (std::chrono::steady_clock::now() >= deadline) return false;
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
}

void CollectiveMonitor::retireGraphResourcesAtShutdown() noexcept {
    {
        std::lock_guard<std::mutex> lock(graph_release_queue_->mutex);
        graph_release_queue_->closed = true;
        graph_release_queue_->released.clear();
    }

    while (true) {
        std::shared_ptr<CollectiveResourceLease> resources;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (graph_resources_.empty()) return;
            auto graph = graph_resources_.begin();
            resources = std::move(graph->second.resources);
            graph_resources_.erase(graph);
        }
        retireResources(resources);
    }
}

void CollectiveMonitor::stop() noexcept {
    if (stopping_.exchange(true, std::memory_order_acq_rel)) return;
    if (thread_.joinable()) thread_.join();
    retireGraphResourcesAtShutdown();
}

}  // namespace mooncake
