#ifndef MOONCAKE_PG_COLLECTIVE_RUNTIME_RESOURCE_POOL_H
#define MOONCAKE_PG_COLLECTIVE_RUNTIME_RESOURCE_POOL_H

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "collective/buffer/collective_buffer_pool.h"
#include "collective/runtime/control_pool.h"
#include "collective/runtime/collective_lane_pool.h"
#include "collective/transport/host_transfer_proxy.h"
#include "error_types.h"

namespace mooncake {

class CollectiveResourcePool;

// Every collective receives the same opaque staging region. Algorithms own
// its internal layout and tile larger tensors through it. The protocol tail is
// reserved for transport-visible tokens and dynamic peer-buffer exchange.
struct CollectiveBufferLayout {
    BufferSpan staging;
    BufferSpan protocol;
};

// MooncakePGContext (process/device lifetime)
//   |- CollectiveBufferPool       shared registered arena
//   |- CollectiveControlPool        shared failure/resource controls
//   `- CollectiveHostTransferProxy optional host-transfer commands
//
// GroupCollectiveRuntime (communicator lifetime)
//   |- CollectiveLanePool         stable wire-control addresses + lane owners
//   |- eager invocation           temporary CollectiveResourceLease
//   `- graph_resources[id]        same lease, retained until destruction
//
// The lease owns the composed resources. Before submission, destruction is a
// normal acquisition rollback. After markSubmitted(), destruction
// conservatively retains the resources unless its lifetime owner calls
// retire(); the lease then checks the shared idle state before pool reuse.
class CollectiveResourceLease {
   public:
    ~CollectiveResourceLease() noexcept;

    CollectiveResourceLease(const CollectiveResourceLease&) = delete;
    CollectiveResourceLease& operator=(const CollectiveResourceLease&) = delete;
    CollectiveResourceLease(CollectiveResourceLease&& other) noexcept;
    CollectiveResourceLease& operator=(
        CollectiveResourceLease&& other) noexcept;

    void markSubmitted() noexcept { submitted_ = true; }
    // Retires a completed submission. Resources return to their pools only
    // when the shared control block proves that asynchronous transport is no
    // longer using them; otherwise the pools quarantine them.
    bool retire() noexcept;

    CollectiveLaneLease lane;
    std::unique_ptr<CollectiveBufferLease> buffer;
    CollectiveControlLease control;
    HostTransferCommandLease host_command;

   private:
    friend class CollectiveResourcePool;

    explicit CollectiveResourceLease(CollectiveResourcePool* pool)
        : pool_(pool) {}
    bool release(bool resource_idle) noexcept;
    void moveFrom(CollectiveResourceLease&& other) noexcept;

    CollectiveResourcePool* pool_ = nullptr;
    bool submitted_ = false;
    bool has_lane_ = false;
};

// Eager calls lease and graphs pin the same data-plane resource. Every lease
// includes a host command because a later authoritative plan may move a
// captured collective between device and host routes. Device-only attempts do
// not touch that command.
class CollectiveResourcePool {
   public:
    CollectiveResourcePool(CollectiveBufferPool* buffer_pool,
                           CollectiveControlPool* control_pool,
                           CollectiveLanePool* lanes,
                           CollectiveHostTransferProxy* host_proxy,
                           DeviceId device, std::string te_location,
                           TransferEngine* engine)
        : buffer_pool_(buffer_pool),
          control_pool_(control_pool),
          lanes_(lanes),
          host_proxy_(host_proxy),
          device_(device),
          te_location_(std::move(te_location)),
          engine_(engine) {}

    PGResult<CollectiveResourceLease> acquire(uint32_t preferred_lane);
    static const CollectiveBufferLayout& bufferLayout();

   private:
    friend class CollectiveResourceLease;

    bool release(CollectiveResourceLease& resources,
                 bool resource_idle) noexcept;

    CollectiveBufferPool* buffer_pool_ = nullptr;
    CollectiveControlPool* control_pool_ = nullptr;
    CollectiveLanePool* lanes_ = nullptr;
    CollectiveHostTransferProxy* host_proxy_ = nullptr;
    DeviceId device_ = kInvalidDeviceId;
    std::string te_location_;
    TransferEngine* engine_ = nullptr;
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_COLLECTIVE_RUNTIME_RESOURCE_POOL_H
