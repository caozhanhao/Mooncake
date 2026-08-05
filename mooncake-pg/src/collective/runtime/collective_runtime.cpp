#include "collective/runtime/collective_runtime.h"

#include <limits>
#include <utility>

#include "gpu_runtime.h"
#include "pg_utils.h"

namespace mooncake {
namespace {

PGResult<std::optional<uint64_t>> activeCaptureId(cudaStream_t stream) {
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    PG_TRY_CUDA(cudaStreamIsCapturing(stream, &status));
    if (status == cudaStreamCaptureStatusNone) return std::nullopt;
    PG_VALIDATE_STATE(status == cudaStreamCaptureStatusActive,
                      "collective CUDA Graph capture is invalidated");

    unsigned long long capture_id = 0;
    PG_TRY_CUDA(cudaStreamGetCaptureInfo(stream, &status, &capture_id));
    PG_VALIDATE_STATE(status == cudaStreamCaptureStatusActive,
                      "collective CUDA Graph capture is not active");
    return std::optional<uint64_t>{static_cast<uint64_t>(capture_id)};
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
    PG_TRY(runtime->progress_,
           CollectiveProgressEngine::create(&runtime->resource_pool_, device,
                                            std::move(failure_handler)));
    return runtime;
}

GroupCollectiveRuntime::~GroupCollectiveRuntime() noexcept {
    stopAccepting();
    if (progress_) {
        (void)progress_->drain(std::chrono::milliseconds(100));
        progress_->stop();
    }
    for (const auto& [_, captured] : captured_collectives_) {
        progress_->untrack(captured.collective);
        const bool resource_idle =
            std::atomic_ref<uint32_t>(
                captured.collective->resources.control.host->resource_idle)
                .load(std::memory_order_acquire) != 0;
        (void)resource_pool_.release(captured.collective->resources,
                                     resource_idle);
    }
}

PGResult<std::shared_ptr<TrackedCollective>>
GroupCollectiveRuntime::acquireTrackedCollective(
    std::optional<uint64_t> capture_id, cudaStream_t capture_stream) {
    if (capture_id) {
        if (const auto found = captured_collectives_.find(*capture_id);
            found != captured_collectives_.end()) {
            PG_VALIDATE_STATE(found->second.capture_stream == capture_stream,
                              "capture id is already bound to another stream");
            return found->second.collective;
        }
    }

    PG_TRY(auto resources, resource_pool_.acquire(next_lane_));
    next_lane_ = (resources.lane.index + 1) % lanes_->layout().lane_count;
    auto collective = std::make_shared<TrackedCollective>(std::move(resources));
    if (capture_id) {
        captured_collectives_.emplace(
            *capture_id, CapturedCollective{.capture_stream = capture_stream,
                                            .collective = collective});
    }
    return collective;
}

void GroupCollectiveRuntime::trackCollective(
    const std::shared_ptr<TrackedCollective>& collective,
    CollectiveFailureTarget target, cudaEvent_t completion) {
    bool first_target = false;
    {
        std::lock_guard<std::mutex> lock(collective->mutex);
        first_target = collective->failure_targets.empty();
        if (completion) collective->completion = completion;
        collective->failure_targets.push_back(target);
    }
    if (first_target) progress_->track(collective);
}

void GroupCollectiveRuntime::rollbackEmptyCapturedCollective(
    uint64_t capture_id) noexcept {
    const auto captured = captured_collectives_.find(capture_id);
    if (captured == captured_collectives_.end()) return;
    const auto& collective = captured->second.collective;
    {
        std::lock_guard<std::mutex> lock(collective->mutex);
        if (!collective->failure_targets.empty()) return;
    }
    (void)resource_pool_.release(collective->resources, true);
    captured_collectives_.erase(captured);
}

CollectiveKernelContext GroupCollectiveRuntime::makeKernelContext(
    const CollectiveResourceLease& resources, CollectiveBinding binding,
    uint64_t failure_target_id) const {
    const auto& layout = CollectiveResourcePool::bufferLayout();
    const auto& control_layout = lanes_->layout().lanes[resources.lane.index];
    return CollectiveKernelContext{
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
        .binding = binding,
    };
}

PGResult<void> GroupCollectiveRuntime::execute(CollectiveInvocation& invocation,
                                               CollectiveBinding binding,
                                               uint64_t view_epoch,
                                               cudaStream_t stream,
                                               int32_t* failed_ranks_hint,
                                               size_t failed_ranks_hint_count) {
    PG_VALIDATE_STATE(accepting_.load(std::memory_order_acquire),
                      "collective runtime is stopping");
    PG_VALIDATE_ARG(failed_ranks_hint, "failed-ranks hint is null");

    const GpuDeviceGuard guard(device_);
    PG_TRY(auto capture_id, activeCaptureId(stream));
    std::lock_guard<std::mutex> admission(admission_mutex_);
    PG_VALIDATE_STATE(accepting_.load(std::memory_order_acquire),
                      "collective runtime is stopping");
    if (!lane_view_epoch_ || *lane_view_epoch_ != view_epoch) {
        lane_view_epoch_ = view_epoch;
        next_lane_ = 0;
    }

    PG_TRY(auto collective, acquireTrackedCollective(capture_id, stream));

    const uint64_t failure_target_id = next_failure_target_id_++;
    const auto context =
        makeKernelContext(collective->resources, binding, failure_target_id);
    CollectiveFailureTarget failure_target{
        .failure_target_id = failure_target_id,
        .failed_ranks_hint = failed_ranks_hint,
        .failed_ranks_hint_count = failed_ranks_hint_count,
    };

    bool resource_idle = true;
    cudaEvent_t completion = nullptr;
    auto collective_rollback = makeScopeExit([&]() noexcept {
        if (completion) {
            (void)cudaEventDestroy(completion);
        }
        if (capture_id) {
            rollbackEmptyCapturedCollective(*capture_id);
        } else {
            (void)resource_pool_.release(collective->resources, resource_idle);
        }
    });
    if (!capture_id) {
        PG_TRY_CUDA(
            cudaEventCreateWithFlags(&completion, cudaEventDisableTiming));
    }

    (void)cudaGetLastError();
    invocation.launch(context, stream);
    PG_TRY_CUDA(cudaGetLastError());
    resource_idle = false;

    if (!capture_id) {
        PG_TRY_CUDA(cudaEventRecord(completion, stream));
    }
    trackCollective(collective, failure_target, completion);
    completion = nullptr;
    collective_rollback.dismiss();
    return {};
}

void GroupCollectiveRuntime::stopAccepting() {
    accepting_.store(false, std::memory_order_release);
}

bool GroupCollectiveRuntime::drain(std::chrono::milliseconds timeout) {
    stopAccepting();
    std::lock_guard<std::mutex> admission(admission_mutex_);
    return !progress_ || progress_->drain(timeout);
}

}  // namespace mooncake
