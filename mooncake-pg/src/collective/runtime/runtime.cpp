#include "collective/runtime/runtime.h"

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

PGResult<std::unique_ptr<CollectiveRuntime>> CollectiveRuntime::create(
    DeviceId device, size_t collective_timeout_us,
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

    auto runtime = std::unique_ptr<CollectiveRuntime>(
        new CollectiveRuntime(device, timeout_ticks));
    PG_TRY(runtime->monitor_, CollectiveMonitor::create(
                                  device, std::move(failure_report_callback)));
    return runtime;
}

CollectiveRuntime::~CollectiveRuntime() noexcept {
    stopAccepting();
    if (monitor_) {
        (void)monitor_->drain(std::chrono::milliseconds(100));
        monitor_->stop();
    }
}

PGResult<void> CollectiveRuntime::submit(
    cudaStream_t stream, int32_t* failed_ranks_hint,
    size_t failed_ranks_hint_count,
    const std::function<PGResult<std::shared_ptr<CollectiveSubmission>>()>&
        prepare,
    const std::function<void(const CollectiveKernelArgs&)>& launch) {
    PG_VALIDATE_STATE(accepting_.load(std::memory_order_acquire),
                      "collective runtime is stopping");
    PG_VALIDATE_ARG(failed_ranks_hint, "failed-ranks hint is null");

    const GpuDeviceGuard guard(device_);
    PG_TRY(auto capture, queryGraphCapture(stream));
    std::lock_guard<std::mutex> admission(admission_mutex_);
    PG_VALIDATE_STATE(accepting_.load(std::memory_order_acquire),
                      "collective runtime is stopping");

    PG_TRY(auto submission,
           monitor_->acquireSubmission(capture, stream, prepare));

    std::unique_ptr<StreamCompletion> stream_completion;
    if (!capture.active) {
        stream_completion = std::make_unique<StreamCompletion>(submission);
        PG_TRY_CUDA(cudaEventCreateWithFlags(&stream_completion->completion,
                                             cudaEventDisableTiming));
    }

    const uint64_t failure_target_id = next_failure_target_id_++;
    monitor_->registerFailureTarget(
        submission, CollectiveFailureTarget{
                        .failure_target_id = failure_target_id,
                        .failed_ranks_hint = failed_ranks_hint,
                        .failed_ranks_hint_count = failed_ranks_hint_count,
                    });

    const auto common = submission->kernelArgs(failure_target_id);
    (void)cudaGetLastError();
    launch(common);
    const auto launch_error = cudaGetLastError();
    if (launch_error != cudaSuccess) {
        monitor_->unregisterFailureTarget(submission, failure_target_id);
        return cudaFailure(launch_error, "collective kernel launch");
    }
    submission->markSubmitted();

    if (stream_completion) {
        const auto record_error =
            cudaEventRecord(stream_completion->completion, stream);
        if (record_error != cudaSuccess) {
            // The monitor continues to observe a possible device failure. With
            // no completion evidence, the submission is retained and will be
            // quarantined when the runtime is destroyed.
            monitor_->markCompletionUnproven();
            return cudaFailure(record_error,
                               "collective completion event record");
        }
        monitor_->retainStreamCompletion(std::move(stream_completion));
    }
    return {};
}

void CollectiveRuntime::stopAccepting() {
    accepting_.store(false, std::memory_order_release);
}

bool CollectiveRuntime::drain(std::chrono::milliseconds timeout) {
    stopAccepting();
    std::lock_guard<std::mutex> admission(admission_mutex_);
    return !monitor_ || monitor_->drain(timeout);
}

}  // namespace mooncake
