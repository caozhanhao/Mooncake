#ifndef MOONCAKE_PG_COLLECTIVE_BUFFER_POOL_H
#define MOONCAKE_PG_COLLECTIVE_BUFFER_POOL_H

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "collective/types.h"
#include "error_types.h"

namespace mooncake {

class TransferEngine;
class CollectiveBufferPool;
namespace device {
class P2pTransport;
}

inline constexpr uint64_t kCollectiveMiB = 1024ULL * 1024ULL;

struct CollectiveBufferPoolConfig {
    uint64_t arena_bytes = 512 * kCollectiveMiB;
};

// Process-level registration facts. The pool does not turn these facts into a
// group endpoint; a communicator combines them with its own control span.
struct CollectiveArenaView {
    DeviceId device = kInvalidDeviceId;
    uint64_t generation = 0;
    void* base = nullptr;
    uint64_t bytes = 0;
    std::vector<int32_t> p2p_handle;
};

// Move-only ownership of one byte range from the process/device arena. Normal
// destruction returns it to the pool; abandon() is the explicit asynchronous
// safety escape hatch.
class CollectiveBufferLease {
   public:
    ~CollectiveBufferLease() noexcept;

    CollectiveBufferLease(const CollectiveBufferLease&) = delete;
    CollectiveBufferLease& operator=(const CollectiveBufferLease&) = delete;
    CollectiveBufferLease(CollectiveBufferLease&& other) noexcept;
    CollectiveBufferLease& operator=(CollectiveBufferLease&& other) noexcept;

    void* base() const { return base_; }
    uint64_t offset() const { return offset_; }
    uint64_t bytes() const { return bytes_; }

    void release() noexcept;
    // Permanently removes an asynchronously referenced range from reuse. Its
    // registered arena is retained for process lifetime.
    void abandon() noexcept;

   private:
    friend class CollectiveBufferPool;
    CollectiveBufferLease(CollectiveBufferPool* pool, DeviceId device,
                          void* base, uint64_t offset, uint64_t bytes)
        : pool_(pool),
          device_(device),
          base_(base),
          offset_(offset),
          bytes_(bytes) {}

    void moveFrom(CollectiveBufferLease&& other) noexcept;
    void clear() noexcept;

    CollectiveBufferPool* pool_ = nullptr;
    DeviceId device_ = kInvalidDeviceId;
    void* base_ = nullptr;
    uint64_t offset_ = 0;
    uint64_t bytes_ = 0;
};

// Process-level owner of per-device registered memory. It knows only arenas
// and byte ranges; group policy, endpoint publication and collective lanes are
// owned above this layer.
class CollectiveBufferPool {
   public:
    CollectiveBufferPool() = default;
    ~CollectiveBufferPool() noexcept;

    PGResult<CollectiveBufferLease> acquire(
        DeviceId device, uint64_t bytes, uint64_t alignment,
        const std::string& te_location, TransferEngine* engine,
        const CollectiveBufferPoolConfig& config = {});
    CollectiveArenaView arena(DeviceId device) const;
    void shutdown();

    CollectiveBufferPool(const CollectiveBufferPool&) = delete;
    CollectiveBufferPool& operator=(const CollectiveBufferPool&) = delete;

   private:
    friend class CollectiveBufferLease;

    struct FreeBlock {
        uint64_t offset = 0;
        uint64_t bytes = 0;
    };

    struct RegisteredArena {
        ~RegisteredArena() noexcept;

        uint64_t generation = 0;
        DeviceId device = kInvalidDeviceId;
        TransferEngine* engine = nullptr;
        void* base = nullptr;
        uint64_t bytes = 0;
        uint64_t active_allocations = 0;
        bool has_abandoned_allocation = false;
        std::unique_ptr<device::P2pTransport> p2p_transport;
        std::vector<int32_t> p2p_handle;
        std::vector<FreeBlock> free_blocks;
    };

    PGResult<RegisteredArena*> getOrCreateArena(
        DeviceId device, const std::string& te_location, TransferEngine* engine,
        const CollectiveBufferPoolConfig& config);
    static PGResult<uint64_t> allocateBlock(RegisteredArena& arena,
                                            uint64_t bytes, uint64_t alignment);
    static void releaseBlock(RegisteredArena& arena, uint64_t offset,
                             uint64_t bytes);
    static void releaseArena(RegisteredArena& arena) noexcept;
    void release(CollectiveBufferLease& lease);
    void abandon(CollectiveBufferLease& lease);

    mutable std::mutex mutex_;
    bool shutdown_ = false;
    uint64_t next_arena_generation_ = 1;
    std::unordered_map<DeviceId, std::unique_ptr<RegisteredArena>> arenas_;
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_COLLECTIVE_BUFFER_POOL_H
