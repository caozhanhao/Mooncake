#ifndef MOONCAKE_PG_COLLECTIVE_RUNTIME_RESOURCE_POOL_H
#define MOONCAKE_PG_COLLECTIVE_RUNTIME_RESOURCE_POOL_H

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "collective/buffer/collective_buffer_pool.h"
#include "collective/runtime/collective_channels.h"
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

// MooncakePGContext owns the shared registered buffer arena. The communicator
// owns its fixed control channels; only the large transfer buffer is leased per
// invocation.
//
// GroupCollectiveEngine
//   |- CollectiveChannels         stable wire/Host control addresses
//   `- CollectiveRuntime
//       |- eager invocation       temporary CollectiveResourceLease
//       `- captured graph         same lease, retired by its CUDA User Object
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
    // Retires a completed submission. An idle bundle returns to its pools;
    // otherwise every component is explicitly abandoned.
    bool retire() noexcept;

    CollectiveChannel channel;
    std::unique_ptr<CollectiveBufferLease> buffer;

   private:
    friend class CollectiveResourcePool;

    explicit CollectiveResourceLease(CollectiveResourcePool* pool)
        : pool_(pool) {}
    bool release() noexcept;
    void abandon() noexcept;
    void moveFrom(CollectiveResourceLease&& other) noexcept;

    CollectiveResourcePool* pool_ = nullptr;
    bool submitted_ = false;
    bool has_channel_ = false;
};

// Temporary bridge until operation code owns its bulk-buffer preparation.
// Channels are communicator-owned; this pool leases only the current bulk
// transfer buffer and composes it with an exact channel.
class CollectiveResourcePool {
   public:
    CollectiveResourcePool(CollectiveBufferPool* buffer_pool,
                           CollectiveChannels* channels, DeviceId device,
                           std::string te_location, TransferEngine* engine)
        : buffer_pool_(buffer_pool),
          channels_(channels),
          device_(device),
          te_location_(std::move(te_location)),
          engine_(engine) {}

    PGResult<CollectiveResourceLease> acquire(uint32_t channel_index);
    static const CollectiveBufferLayout& bufferLayout();

   private:
    friend class CollectiveResourceLease;

    bool release(CollectiveResourceLease& resources) noexcept;
    void abandon(CollectiveResourceLease& resources) noexcept;

    CollectiveBufferPool* buffer_pool_ = nullptr;
    CollectiveChannels* channels_ = nullptr;
    DeviceId device_ = kInvalidDeviceId;
    std::string te_location_;
    TransferEngine* engine_ = nullptr;
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_COLLECTIVE_RUNTIME_RESOURCE_POOL_H
