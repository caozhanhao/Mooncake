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
//   `- captured_collectives[id]   same lease, retained until destruction
//
// The pool only composes these resources. It does not know whether the
// caller will return the lease after one invocation or retain it for a graph.
struct CollectiveResourceLease {
    CollectiveLaneLease lane;
    std::unique_ptr<CollectiveBufferLease> buffer;
    CollectiveControlLease control;
    HostTransferCommandLease host_command;
};

// Eager calls lease and graphs pin the same data-plane resource. Every lease
// includes a host command because a later authoritative binding may move a
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

    PGResult<CollectiveResourceLease> tryAcquire(uint32_t preferred_lane);
    static const CollectiveBufferLayout& bufferLayout();
    bool release(const CollectiveResourceLease& resources, bool resource_idle);

   private:
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
