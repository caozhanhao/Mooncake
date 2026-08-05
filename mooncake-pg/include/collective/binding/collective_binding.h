#ifndef MOONCAKE_PG_COLLECTIVE_BINDING_COLLECTIVE_BINDING_H
#define MOONCAKE_PG_COLLECTIVE_BINDING_COLLECTIVE_BINDING_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <utility>
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

// One communicator-scoped binding for one collective implementation. It owns
// the stable device publication read by every invocation of that collective.
class CollectiveBinding {
   public:
    PGResult<CollectiveBindingView> deviceView(uint64_t view_epoch) const;

    CollectiveBinding(const CollectiveBinding&) = delete;
    CollectiveBinding& operator=(const CollectiveBinding&) = delete;

   private:
    friend class GroupCollectiveBindings;

    CollectiveBinding(
        std::unique_ptr<CollectiveBindingMaterializer> materializer,
        size_t slot_bytes)
        : materializer_(std::move(materializer)), slot_bytes_(slot_bytes) {}

    std::unique_ptr<CollectiveBindingMaterializer> materializer_;
    size_t slot_bytes_ = 0;
    void* slots_host_ = nullptr;
    void* slots_device_ = nullptr;
    uint64_t* lane_sequences_host_ = nullptr;
    uint64_t* lane_sequences_device_ = nullptr;
    uint32_t* active_slot_host_ = nullptr;
    uint32_t* active_slot_device_ = nullptr;

    bool ready_ = false;
    uint64_t view_epoch_ = 0;
    std::array<std::vector<std::shared_ptr<void>>, kCollectiveBindingSlots>
        slot_resources_;
    mutable std::mutex mutex_;
};

class GroupCollectiveBindings {
   public:
    GroupCollectiveBindings(DeviceId device, uint32_t control_lane_count)
        : device_(device), control_lane_count_(control_lane_count) {}
    ~GroupCollectiveBindings() noexcept;

    PGResult<CollectiveBinding*> add(
        std::unique_ptr<CollectiveBindingMaterializer> materializer);
    PGResult<void> apply(const GroupView& view);

    void retainForProcessLifetime();

    GroupCollectiveBindings(const GroupCollectiveBindings&) = delete;
    GroupCollectiveBindings& operator=(const GroupCollectiveBindings&) = delete;

   private:
    PGResult<void> allocate(CollectiveBinding& binding);
    void publish(CollectiveBinding& binding, const void* kernel_plan,
                 std::vector<std::shared_ptr<void>> resources);
    void release(CollectiveBinding& binding) noexcept;

    DeviceId device_ = kInvalidDeviceId;
    uint32_t control_lane_count_ = 0;
    std::vector<std::unique_ptr<CollectiveBinding>> bindings_;
    std::mutex mutex_;
    bool retained_ = false;
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_COLLECTIVE_BINDING_COLLECTIVE_BINDING_H
