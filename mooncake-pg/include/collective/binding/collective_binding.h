#ifndef MOONCAKE_PG_COLLECTIVE_BINDING_COLLECTIVE_BINDING_H
#define MOONCAKE_PG_COLLECTIVE_BINDING_COLLECTIVE_BINDING_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <variant>
#include <vector>

#include "collective/binding/binding_view.cuh"
#include "error_types.h"

namespace mooncake {

struct GroupView;

struct ReadyCollectiveBinding {
    std::shared_ptr<const void> kernel_plan;
    std::vector<std::shared_ptr<void>> resources;
};

struct FailedCollectiveBinding {
    // Failed views are still published so a captured executor cannot continue
    // using the preceding view's routes.
    std::shared_ptr<const void> kernel_plan;
    PGError error;
};

using MaterializedCollectiveBinding =
    std::variant<ReadyCollectiveBinding, FailedCollectiveBinding>;

// Collective plugins own only the translation from logical GroupView policy
// to their typed executor ABI. Stable publication and the resources referenced
// by each published kernel-plan slot are group-scoped and shared by every
// plugin.
class CollectiveBindingMaterializer {
   public:
    virtual ~CollectiveBindingMaterializer() = default;

    virtual size_t kernelPlanBytes() const = 0;
    virtual MaterializedCollectiveBinding materialize(
        const GroupView& view) const = 0;
};

class GroupCollectiveBindings {
   public:
    GroupCollectiveBindings(DeviceId device, uint32_t control_lane_count)
        : device_(device), control_lane_count_(control_lane_count) {}
    ~GroupCollectiveBindings() noexcept;

    PGResult<CollectiveBindingId> add(
        std::unique_ptr<CollectiveBindingMaterializer> materializer);
    PGResult<void> apply(const GroupView& view);

    PGResult<void> requireReady(CollectiveBindingId binding_id,
                                uint64_t view_epoch) const;
    CollectiveBindingView deviceView(CollectiveBindingId binding_id) const;
    // A parked executor must never retry the exact view that failed. CPU
    // progress reopens its stable gate only after a newer binding is ready.
    bool readyForRecovery(CollectiveBindingId binding_id,
                          uint64_t failed_view_epoch) const;

    void retainForProcessLifetime();

    GroupCollectiveBindings(const GroupCollectiveBindings&) = delete;
    GroupCollectiveBindings& operator=(const GroupCollectiveBindings&) = delete;

   private:
    struct BindingEntry {
        std::unique_ptr<CollectiveBindingMaterializer> materializer;
        size_t slot_bytes = 0;
        void* slots_host = nullptr;
        void* slots_device = nullptr;
        uint64_t* lane_sequences_host = nullptr;
        uint64_t* lane_sequences_device = nullptr;
        uint32_t* active_slot_host = nullptr;
        uint32_t* active_slot_device = nullptr;

        bool ready = false;
        uint64_t view_epoch = 0;
        std::array<std::vector<std::shared_ptr<void>>, kCollectiveBindingSlots>
            slot_resources;
    };

    PGResult<void> allocate(BindingEntry& entry);
    void publish(BindingEntry& entry, const void* kernel_plan,
                 std::vector<std::shared_ptr<void>> resources);
    void release(BindingEntry& entry) noexcept;

    DeviceId device_ = kInvalidDeviceId;
    uint32_t control_lane_count_ = 0;
    std::vector<std::unique_ptr<BindingEntry>> bindings_;
    mutable std::mutex mutex_;
    bool retained_ = false;
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_COLLECTIVE_BINDING_COLLECTIVE_BINDING_H
