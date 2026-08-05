#ifndef MOONCAKE_PG_COLLECTIVE_RUNTIME_COLLECTIVE_LANE_POOL_H
#define MOONCAKE_PG_COLLECTIVE_RUNTIME_COLLECTIVE_LANE_POOL_H

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "collective/buffer/collective_buffer_pool.h"
#include "collective/endpoint.h"
#include "error_types.h"

namespace mooncake {

struct CollectiveLaneLease {
    uint32_t index = 0;
};

// Communicator-scoped collective lanes. Each lane reserves one stable control
// span used for peer wire tokens; invocation buffers remain process-pooled.
class CollectiveLanePool {
   public:
    static PGResult<std::unique_ptr<CollectiveLanePool>> create(
        CollectiveBufferPool* buffers, DeviceId device,
        const std::string& te_location, TransferEngine* engine,
        uint32_t lane_count = 3);
    ~CollectiveLanePool() noexcept;

    const CollectiveControlLayout& layout() const { return layout_; }
    void* controlBase() const { return control_->base(); }

    PGResult<CollectiveLaneLease> acquire(uint32_t preferred_lane);
    bool release(const CollectiveLaneLease& lane, bool resource_idle);
    bool close(bool resource_idle);

    CollectiveLanePool(const CollectiveLanePool&) = delete;
    CollectiveLanePool& operator=(const CollectiveLanePool&) = delete;

   private:
    struct LaneState {
        bool in_use = false;
        bool reusable = true;
    };

    CollectiveLanePool(CollectiveBufferPool* buffers,
                       std::unique_ptr<CollectiveBufferLease> control,
                       CollectiveControlLayout layout)
        : buffers_(buffers),
          control_(std::move(control)),
          layout_(std::move(layout)),
          lanes_(layout_.lane_count) {}

    CollectiveBufferPool* buffers_ = nullptr;
    std::unique_ptr<CollectiveBufferLease> control_;
    CollectiveControlLayout layout_;

    std::mutex mutex_;
    std::vector<LaneState> lanes_;
    bool may_be_in_use_ = false;
    bool closed_ = false;
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_COLLECTIVE_RUNTIME_COLLECTIVE_LANE_POOL_H
