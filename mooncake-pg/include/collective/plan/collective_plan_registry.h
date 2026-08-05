#ifndef MOONCAKE_PG_COLLECTIVE_PLAN_COLLECTIVE_PLAN_REGISTRY_H
#define MOONCAKE_PG_COLLECTIVE_PLAN_COLLECTIVE_PLAN_REGISTRY_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <utility>
#include <variant>
#include <vector>

#include "collective/plan/plan_handle.cuh"
#include "error_types.h"

namespace mooncake {

class DeviceLinkManager;
class GroupPeerRoutes;
class LinkManager;
struct ActiveGroupRanks;
struct CollectivePlanSet;
struct GroupView;

// The concrete builder owns this short-lived typed host value until the
// generic publisher copies it into a plan slot. Link and runtime resources are
// not type-erased here.
struct ReadyCollectivePlan {
    std::shared_ptr<const void> kernel_plan;
};

struct FailedCollectivePlan {
    // Failed views are still published so a captured executor cannot continue
    // using the preceding view's routes.
    std::shared_ptr<const void> kernel_plan;
    PGError error;
};

using CollectivePlanBuildResult =
    std::variant<ReadyCollectivePlan, FailedCollectivePlan>;

// Collective plugins translate one logical policy into their typed kernel ABI.
// Membership and peer routes have already been projected from GroupView.
class CollectivePlanBuilder {
   public:
    virtual ~CollectivePlanBuilder() = default;

    virtual size_t kernelPlanBytes() const = 0;
    virtual CollectivePlanBuildResult build(
        uint64_t view_epoch, const CollectivePlanSet& plans,
        const ActiveGroupRanks& active_ranks,
        const GroupPeerRoutes& peer_routes) const = 0;
};

// Owns one communicator-scoped, graph-stable publication for one collective.
class CollectivePlanPublisher {
   public:
    PGResult<CollectivePlanHandle> handle(uint64_t view_epoch) const;

    CollectivePlanPublisher(const CollectivePlanPublisher&) = delete;
    CollectivePlanPublisher& operator=(const CollectivePlanPublisher&) = delete;

   private:
    friend class CollectivePlanRegistry;

    CollectivePlanPublisher(std::unique_ptr<CollectivePlanBuilder> builder,
                            size_t slot_bytes)
        : builder_(std::move(builder)), slot_bytes_(slot_bytes) {}

    std::unique_ptr<CollectivePlanBuilder> builder_;
    size_t slot_bytes_ = 0;
    void* slots_host_ = nullptr;
    void* slots_device_ = nullptr;
    uint64_t* lane_sequences_host_ = nullptr;
    uint64_t* lane_sequences_device_ = nullptr;
    uint32_t* active_slot_host_ = nullptr;
    uint32_t* active_slot_device_ = nullptr;

    bool ready_ = false;
    uint64_t view_epoch_ = 0;
    std::array<std::shared_ptr<const GroupPeerRoutes>, kCollectivePlanSlots>
        slot_routes_;
    mutable std::mutex mutex_;
};

class CollectivePlanRegistry {
   public:
    CollectivePlanRegistry(DeviceLinkManager& device_links,
                           LinkManager& host_links, DeviceId device,
                           InGroupRank self_in_group_rank,
                           uint32_t control_lane_count)
        : device_links_(device_links),
          host_links_(host_links),
          device_(device),
          self_in_group_rank_(self_in_group_rank),
          control_lane_count_(control_lane_count) {}
    ~CollectivePlanRegistry() noexcept;

    PGResult<CollectivePlanPublisher*> addBuilder(
        std::unique_ptr<CollectivePlanBuilder> builder);
    PGResult<void> apply(const GroupView& view);

    void retainForProcessLifetime();

    CollectivePlanRegistry(const CollectivePlanRegistry&) = delete;
    CollectivePlanRegistry& operator=(const CollectivePlanRegistry&) = delete;

   private:
    PGResult<void> allocate(CollectivePlanPublisher& publisher);
    void publish(CollectivePlanPublisher& publisher, const void* kernel_plan,
                 std::shared_ptr<const GroupPeerRoutes> peer_routes);
    void release(CollectivePlanPublisher& publisher) noexcept;

    DeviceLinkManager& device_links_;
    LinkManager& host_links_;
    DeviceId device_ = kInvalidDeviceId;
    InGroupRank self_in_group_rank_ = -1;
    uint32_t control_lane_count_ = 0;
    std::vector<std::unique_ptr<CollectivePlanPublisher>> publishers_;
    std::mutex mutex_;
    bool retained_ = false;
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_COLLECTIVE_PLAN_COLLECTIVE_PLAN_REGISTRY_H
