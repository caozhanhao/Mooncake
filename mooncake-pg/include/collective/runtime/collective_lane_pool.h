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

// Communicator-scoped collective lanes. Each lane owns one local invocation
// sequence and one stable span for peer wire tokens; invocation buffers remain
// process-pooled.
class CollectiveLanePool {
   public:
    static PGResult<std::unique_ptr<CollectiveLanePool>> create(
        CollectiveBufferPool* buffers, DeviceId device,
        const std::string& te_location, TransferEngine* engine,
        uint32_t lane_count = 3);
    ~CollectiveLanePool() noexcept;

    const CollectiveControlLayout& layout() const { return layout_; }
    void* controlBase() const { return control_->base(); }
    uint64_t* invocationSequence(const CollectiveLaneLease& lane) const;

    PGResult<CollectiveLaneLease> acquire(uint32_t preferred_lane);
    void release(const CollectiveLaneLease& lane);
    void abandon(const CollectiveLaneLease& lane);
    bool close(bool resources_safe);

    CollectiveLanePool(const CollectiveLanePool&) = delete;
    CollectiveLanePool& operator=(const CollectiveLanePool&) = delete;

   private:
    enum class LaneState : uint8_t {
        Free = 0,
        Acquired,
        Abandoned,
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
    bool closed_ = false;
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_COLLECTIVE_RUNTIME_COLLECTIVE_LANE_POOL_H
