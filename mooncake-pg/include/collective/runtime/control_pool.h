#ifndef MOONCAKE_PG_COLLECTIVE_RUNTIME_CONTROL_POOL_H
#define MOONCAKE_PG_COLLECTIVE_RUNTIME_CONTROL_POOL_H

#include <atomic>
#include <cstdint>
#include <mutex>
#include <optional>
#include <vector>

#include "collective/runtime/control_block.cuh"
#include "error_types.h"

namespace mooncake {

struct CollectiveControlLease {
    uint32_t index = 0;
    CollectiveControlBlock* host = nullptr;
    CollectiveControlBlock* device = nullptr;
};

// Process-level pool of mapped control blocks. These blocks carry the
// common failure and asynchronous-resource protocol; transport-specific
// commands are owned by their transport service.
class CollectiveControlPool {
   public:
    CollectiveControlPool() = default;
    ~CollectiveControlPool() noexcept;

    PGResult<void> initialize(uint32_t control_count = 128);
    std::optional<CollectiveControlLease> tryAcquire();
    bool release(const CollectiveControlLease& control, bool resource_idle);
    void shutdown();

    CollectiveControlPool(const CollectiveControlPool&) = delete;
    CollectiveControlPool& operator=(const CollectiveControlPool&) = delete;

   private:
    struct ControlState {
        bool in_use = false;
        bool reusable = true;
    };

    CollectiveControlBlock* host_controls_ = nullptr;
    CollectiveControlBlock* device_controls_ = nullptr;
    std::vector<ControlState> controls_;
    std::mutex mutex_;
    std::atomic<bool> initialized_{false};
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_COLLECTIVE_RUNTIME_CONTROL_POOL_H
