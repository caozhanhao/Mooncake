#include "collective/runtime/collective_runtime.h"

#include <atomic>
#include <limits>
#include <memory>
#include <string>
#include <utility>

#include "gpu_runtime.h"

namespace mooncake {
namespace {

PGResult<void> cudaFailure(cudaError_t error, const char* operation) {
    return makePGError(
        PGErrorCode::SystemError,
        std::string(operation) + " failed: " + cudaGetErrorString(error));
}

}  // namespace

PGResult<std::unique_ptr<GroupCollectiveRuntime>>
GroupCollectiveRuntime::create(
    CollectiveBufferPool* buffer_pool, CollectiveControlPool* control_pool,
    CollectiveHostTransferProxy* host_transfer_proxy, CollectiveLanePool* lanes,
    std::string te_location, TransferEngine* engine, DeviceId device,
    size_t collective_timeout_us,
    CollectiveFailureReportCallback failure_report_callback) {
    const GpuDeviceGuard guard(device);
    int clock_rate_khz = 0;
    PG_TRY_CUDA(
        cudaDeviceGetAttribute(&clock_rate_khz, cudaDevAttrClockRate, device));
    const auto clock_rate = static_cast<uint64_t>(clock_rate_khz);
    const uint64_t timeout_ticks =
        collective_timeout_us >
                std::numeric_limits<uint64_t>::max() / clock_rate
            ? std::numeric_limits<uint64_t>::max()
            : collective_timeout_us * clock_rate / 1000ULL;

    auto runtime =
        std::unique_ptr<GroupCollectiveRuntime>(new GroupCollectiveRuntime(
            buffer_pool, control_pool, host_transfer_proxy, lanes,
            std::move(te_location), engine, device, timeout_ticks));
    auto failure_handler = std::make_unique<CollectiveFailureHandler>(
        std::move(failure_report_callback));
    PG_TRY(runtime->host_progress_,
           CollectiveHostProgress::create(device, std::move(failure_handler)));
    return runtime;
}

GroupCollectiveRuntime::~GroupCollectiveRuntime() noexcept {
    stopAccepting();
    if (host_progress_) {
        (void)host_progress_->drain(std::chrono::milliseconds(100));
        host_progress_->stop();
    }
    for (const auto& [_, graph] : graph_resources_) {
        host_progress_->stopObserving(graph.resources);
        (void)graph.resources->retire();
    }
}

CollectiveKernelArgs GroupCollectiveRuntime::makeKernelArgs(
    const CollectiveResourceLease& resources, CollectivePlanHandle plan,
    uint64_t failure_target_id) const {
    const auto& layout = CollectiveResourcePool::bufferLayout();
    const auto& control_layout = lanes_->layout().lanes[resources.lane.index];
    return CollectiveKernelArgs{
        .resources =
            CollectiveKernelResources{
                .buffer =
                    CollectiveKernelBuffer{
                        .base = resources.buffer->base(),
                        .arena_offset = resources.buffer->offset(),
                        .staging_offset = layout.staging.offset,
                        .staging_bytes = layout.staging.bytes,
                        .protocol_offset = layout.protocol.offset,
                        .protocol_bytes = layout.protocol.bytes,
                    },
                .peer_control =
                    CollectivePeerControl{
                        .buffer = lanes_->controlBase(),
                        .signals_offset = control_layout.signals.offset,
                    },
                .control = resources.control.device,
                .host_command = resources.host_command.device,
                .timeout_device_ticks = timeout_device_ticks_,
            },
        .lane_index = resources.lane.index,
        .failure_target_id = failure_target_id,
        .plan = plan,
    };
}

PGResult<void> GroupCollectiveRuntime::execute(CollectiveInvocation& invocation,
                                               CollectivePlanHandle plan,
                                               uint64_t view_epoch,
                                               cudaStream_t stream,
                                               int32_t* failed_ranks_hint,
                                               size_t failed_ranks_hint_count) {
    PG_VALIDATE_STATE(accepting_.load(std::memory_order_acquire),
                      "collective runtime is stopping");
    PG_VALIDATE_ARG(failed_ranks_hint, "failed-ranks hint is null");

    const GpuDeviceGuard guard(device_);
    PG_TRY(auto capture, queryGraphCapture(stream));
    std::lock_guard<std::mutex> admission(admission_mutex_);
    PG_VALIDATE_STATE(accepting_.load(std::memory_order_acquire),
                      "collective runtime is stopping");
    if (!lane_view_epoch_ || *lane_view_epoch_ != view_epoch) {
        lane_view_epoch_ = view_epoch;
        next_lane_ = 0;
    }

    std::shared_ptr<CollectiveResourceLease> resources;
    bool new_graph_resources = false;
    if (capture.active) {
        if (const auto found = graph_resources_.find(capture.id);
            found != graph_resources_.end()) {
            PG_VALIDATE_STATE(
                found->second.bound_stream == stream,
                "multi-stream collective CUDA Graph capture is unsupported");
            resources = found->second.resources;
        } else {
            PG_TRY(auto lease, resource_pool_.acquire(next_lane_));
            next_lane_ = (lease.lane.index + 1) % lanes_->layout().lane_count;
            resources =
                std::make_shared<CollectiveResourceLease>(std::move(lease));
            graph_resources_.emplace(
                capture.id,
                GraphResources{.bound_stream = stream, .resources = resources});
            new_graph_resources = true;
        }
    } else {
        PG_TRY(auto lease, resource_pool_.acquire(next_lane_));
        next_lane_ = (lease.lane.index + 1) % lanes_->layout().lane_count;
        resources = std::make_shared<CollectiveResourceLease>(std::move(lease));
    }

    std::unique_ptr<EagerSubmission> eager_submission;
    if (!capture.active) {
        eager_submission = std::make_unique<EagerSubmission>(resources);
        PG_TRY_CUDA(cudaEventCreateWithFlags(&eager_submission->completion,
                                             cudaEventDisableTiming));
    }

    const uint64_t failure_target_id = next_failure_target_id_++;
    host_progress_->registerFailureTarget(
        resources, CollectiveFailureTarget{
                       .failure_target_id = failure_target_id,
                       .failed_ranks_hint = failed_ranks_hint,
                       .failed_ranks_hint_count = failed_ranks_hint_count,
                   });

    const auto common = makeKernelArgs(*resources, plan, failure_target_id);
    (void)cudaGetLastError();
    invocation.launch(common, stream);
    const auto launch_error = cudaGetLastError();
    if (launch_error != cudaSuccess) {
        host_progress_->unregisterFailureTarget(resources, failure_target_id);
        if (new_graph_resources) graph_resources_.erase(capture.id);
        return cudaFailure(launch_error, "collective kernel launch");
    }
    resources->markSubmitted();

    if (capture.active) return {};

    const auto record_error =
        cudaEventRecord(eager_submission->completion, stream);
    if (record_error != cudaSuccess) {
        // Host progress continues to observe a possible device failure. With
        // no completion evidence, the submitted resources are retained and
        // will be quarantined when the runtime is destroyed.
        host_progress_->markCompletionUnproven();
        return cudaFailure(record_error, "collective completion event record");
    }
    host_progress_->submit(std::move(eager_submission));
    return {};
}

void GroupCollectiveRuntime::stopAccepting() {
    accepting_.store(false, std::memory_order_release);
}

bool GroupCollectiveRuntime::drain(std::chrono::milliseconds timeout) {
    stopAccepting();
    std::lock_guard<std::mutex> admission(admission_mutex_);
    return !host_progress_ || host_progress_->drain(timeout);
}

}  // namespace mooncake
