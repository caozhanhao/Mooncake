#include "collective/runtime/collective_progress.h"

#include <algorithm>
#include <atomic>
#include <string>
#include <system_error>
#include <thread>
#include <utility>

#include <glog/logging.h>

namespace mooncake {

PGResult<std::unique_ptr<CollectiveProgressEngine>>
CollectiveProgressEngine::create(
    CollectiveResourcePool* resource_pool, DeviceId device,
    std::unique_ptr<CollectiveFailureHandler> failure_handler) {
    auto progress =
        std::unique_ptr<CollectiveProgressEngine>(new CollectiveProgressEngine(
            resource_pool, device, std::move(failure_handler)));
    try {
        progress->thread_ = std::thread(&CollectiveProgressEngine::progressLoop,
                                        progress.get());
    } catch (const std::system_error& error) {
        return makePGError(
            PGErrorCode::SystemError,
            std::string("failed to start collective progress: ") +
                error.what());
    }
    return progress;
}

CollectiveProgressEngine::~CollectiveProgressEngine() noexcept { stop(); }

void CollectiveProgressEngine::track(
    std::shared_ptr<TrackedCollective> collective) {
    std::lock_guard<std::mutex> lock(mutex_);
    collectives_.push_back(std::move(collective));
}

void CollectiveProgressEngine::untrack(
    const std::shared_ptr<TrackedCollective>& collective) {
    std::lock_guard<std::mutex> lock(mutex_);
    collectives_.erase(
        std::remove(collectives_.begin(), collectives_.end(), collective),
        collectives_.end());
}

bool CollectiveProgressEngine::retireCompletedCollective() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (size_t index = 0; index < collectives_.size(); ++index) {
        auto collective = collectives_[index];
        std::lock_guard<std::mutex> collective_lock(collective->mutex);
        if (!collective->completion) continue;
        const auto query = cudaEventQuery(collective->completion);
        if (query == cudaErrorNotReady) continue;

        const bool resource_idle =
            query == cudaSuccess &&
            std::atomic_ref<uint32_t>(
                collective->resources.control.host->resource_idle)
                    .load(std::memory_order_acquire) != 0;
        if (!resource_pool_->release(collective->resources, resource_idle)) {
            LOG(WARNING) << "Retained collective resources after "
                            "completion";
        }
        if (query == cudaSuccess) {
            const auto destroy = cudaEventDestroy(collective->completion);
            if (destroy != cudaSuccess) {
                LOG(WARNING) << "Failed to destroy collective completion "
                                "event: "
                             << cudaGetErrorString(destroy);
            }
        } else {
            LOG(WARNING) << "Collective completion query failed: "
                         << cudaGetErrorString(query);
        }
        collectives_[index] = std::move(collectives_.back());
        collectives_.pop_back();
        return true;
    }
    return false;
}

bool CollectiveProgressEngine::handleOneFailure() {
    std::optional<CollectiveFailureHandler::Claim> failure;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        failure = failure_handler_->claim(collectives_);
    }
    if (!failure.has_value()) return false;
    failure_handler_->handle(*failure);
    return true;
}

void CollectiveProgressEngine::progressLoop() noexcept {
    const auto select = cudaSetDevice(device_);
    if (select != cudaSuccess) {
        LOG(ERROR) << "Collective progress cannot select CUDA device "
                   << device_ << ": " << cudaGetErrorString(select);
        return;
    }
    while (!stopping_.load(std::memory_order_acquire)) {
        bool progressed = handleOneFailure();
        progressed |= retireCompletedCollective();
        if (!progressed) {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
    }
}

bool CollectiveProgressEngine::drain(std::chrono::milliseconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (true) {
        bool has_eager_collective = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (const auto& collective : collectives_) {
                std::lock_guard<std::mutex> collective_lock(collective->mutex);
                has_eager_collective |= collective->completion != nullptr;
            }
        }
        if (!has_eager_collective) return true;
        if (std::chrono::steady_clock::now() >= deadline) return false;
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
}

void CollectiveProgressEngine::stop() noexcept {
    if (stopping_.exchange(true, std::memory_order_acq_rel)) return;
    if (thread_.joinable()) thread_.join();
}

}  // namespace mooncake
