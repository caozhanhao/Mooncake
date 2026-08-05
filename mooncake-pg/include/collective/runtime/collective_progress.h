#ifndef MOONCAKE_PG_COLLECTIVE_RUNTIME_COLLECTIVE_PROGRESS_H
#define MOONCAKE_PG_COLLECTIVE_RUNTIME_COLLECTIVE_PROGRESS_H

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

#include "collective/runtime/tracked_collective.h"
#include "collective/runtime/collective_failure.h"
#include "error_types.h"

namespace mooncake {

// Host progress is collective-independent: it observes failure reports and
// completion events, and retires eager collectives after their CUDA completion
// event. Captured collectives remain tracked until the runtime removes them.
class CollectiveProgressEngine {
   public:
    static PGResult<std::unique_ptr<CollectiveProgressEngine>> create(
        CollectiveResourcePool* resource_pool, DeviceId device,
        std::unique_ptr<CollectiveFailureHandler> failure_handler);
    ~CollectiveProgressEngine() noexcept;

    void track(std::shared_ptr<TrackedCollective> collective);
    void untrack(const std::shared_ptr<TrackedCollective>& collective);

    bool drain(std::chrono::milliseconds timeout);
    void stop() noexcept;

    CollectiveProgressEngine(const CollectiveProgressEngine&) = delete;
    CollectiveProgressEngine& operator=(const CollectiveProgressEngine&) =
        delete;

   private:
    CollectiveProgressEngine(
        CollectiveResourcePool* resource_pool, DeviceId device,
        std::unique_ptr<CollectiveFailureHandler> failure_handler)
        : resource_pool_(resource_pool),
          device_(device),
          failure_handler_(std::move(failure_handler)) {}

    bool retireCompletedCollective();
    bool handleOneFailure();
    void progressLoop() noexcept;

    CollectiveResourcePool* resource_pool_ = nullptr;
    DeviceId device_ = kInvalidDeviceId;
    std::unique_ptr<CollectiveFailureHandler> failure_handler_;

    mutable std::mutex mutex_;
    std::vector<std::shared_ptr<TrackedCollective>> collectives_;
    std::thread thread_;
    std::atomic<bool> stopping_{false};
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_COLLECTIVE_RUNTIME_COLLECTIVE_PROGRESS_H
