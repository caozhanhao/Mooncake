#include "collective/buffer/collective_buffer_pool.h"

#include <algorithm>
#include <exception>
#include <utility>

#include <glog/logging.h>

#include <cuda_alike.h>
#include <transfer_engine.h>
#include <transport/device/device_transport.h>

#include "gpu_runtime.h"
#include "pg_utils.h"

namespace mooncake {
namespace {

uint64_t alignUp(uint64_t value, uint64_t alignment) {
    const auto remainder = value % alignment;
    return remainder == 0 ? value : value + alignment - remainder;
}

void* byteOffset(void* base, uint64_t offset) {
    return static_cast<char*>(base) + offset;
}

}  // namespace

CollectiveBufferPool::~CollectiveBufferPool() noexcept {
    try {
        shutdown();
    } catch (const std::exception& error) {
        LOG(WARNING) << "Collective buffer pool shutdown failed: "
                     << error.what();
    } catch (...) {
        LOG(WARNING) << "Collective buffer pool shutdown failed";
    }
}

PGResult<CollectiveBufferPool::RegisteredArena*>
CollectiveBufferPool::getOrCreateArena(
    DeviceId device, const std::string& te_location, TransferEngine* engine,
    const CollectiveBufferPoolConfig& config) {
    if (auto found = arenas_.find(device); found != arenas_.end()) {
        return found->second.get();
    }
    PG_VALIDATE_ARG(device >= 0 && engine && !te_location.empty() &&
                        config.arena_bytes != 0,
                    "invalid collective buffer pool configuration");

    const GpuDeviceGuard guard(device);
    auto arena = std::make_unique<RegisteredArena>();
    arena->generation = next_arena_generation_++;
    arena->device = device;
    arena->engine = engine;
    arena->bytes = config.arena_bytes;
    arena->p2p_transport = device::createP2pDeviceTransport(1);
    if (arena->p2p_transport) {
        arena->base = arena->p2p_transport->allocateBuffer(arena->bytes);
    }
    if (!arena->base) {
        arena->p2p_transport.reset();
        PG_TRY_CUDA(cudaMalloc(&arena->base, arena->bytes));
    }
    bool registered = false;
    auto allocation_rollback = makeScopeExit([&]() noexcept {
        if (registered) {
            (void)engine->unregisterLocalMemory(arena->base);
        }
        if (arena->p2p_transport) {
            arena->p2p_transport->freeBuffer(arena->base);
        } else {
            (void)cudaFree(arena->base);
        }
    });
    PG_TRY_CUDA(cudaMemset(arena->base, 0, arena->bytes));
    if (arena->p2p_transport) {
        arena->p2p_handle = arena->p2p_transport->exportIpcHandle(arena->base);
    }
    if (engine->registerLocalMemory(arena->base, arena->bytes, te_location) !=
        0) {
        return makePGError(PGErrorCode::TransferEngineError,
                           "failed to register collective buffer pool");
    }
    registered = true;
    arena->free_blocks.push_back({0, arena->bytes});
    const auto position = arenas_.emplace(device, std::move(arena)).first;
    allocation_rollback.dismiss();
    return position->second.get();
}

PGResult<uint64_t> CollectiveBufferPool::allocateBlock(RegisteredArena& arena,
                                                       uint64_t bytes,
                                                       uint64_t alignment) {
    for (size_t index = 0; index < arena.free_blocks.size(); ++index) {
        const auto block = arena.free_blocks[index];
        const auto aligned = alignUp(block.offset, alignment);
        const auto prefix = aligned - block.offset;
        if (prefix > block.bytes || bytes > block.bytes - prefix) continue;
        const auto suffix = block.bytes - prefix - bytes;
        arena.free_blocks.reserve(arena.free_blocks.size() + 1);
        arena.free_blocks.erase(arena.free_blocks.begin() + index);
        auto insert_at = index;
        if (prefix) {
            arena.free_blocks.insert(arena.free_blocks.begin() + insert_at,
                                     {block.offset, prefix});
            ++insert_at;
        }
        if (suffix) {
            arena.free_blocks.insert(arena.free_blocks.begin() + insert_at,
                                     {aligned + bytes, suffix});
        }
        return aligned;
    }
    return makePGError(PGErrorCode::ResourceBusy,
                       "collective buffer pool is exhausted");
}

void CollectiveBufferPool::releaseBlock(RegisteredArena& arena, uint64_t offset,
                                        uint64_t bytes) {
    arena.free_blocks.push_back({offset, bytes});
    std::sort(arena.free_blocks.begin(), arena.free_blocks.end(),
              [](const auto& left, const auto& right) {
                  return left.offset < right.offset;
              });
    size_t merged = 0;
    for (const auto block : arena.free_blocks) {
        if (merged && arena.free_blocks[merged - 1].offset +
                              arena.free_blocks[merged - 1].bytes ==
                          block.offset) {
            arena.free_blocks[merged - 1].bytes += block.bytes;
        } else {
            arena.free_blocks[merged++] = block;
        }
    }
    arena.free_blocks.resize(merged);
}

void CollectiveBufferPool::releaseArena(RegisteredArena& arena) noexcept {
    if (!arena.base) return;
    try {
        const GpuDeviceGuard guard(arena.device);
        if (arena.engine && arena.engine->unregisterLocalMemory(arena.base)) {
            LOG(WARNING) << "Retaining collective buffer pool after "
                            "unregister failure, device="
                         << arena.device;
            (void)arena.p2p_transport.release();
            return;
        }
        if (arena.p2p_transport) {
            arena.p2p_transport->freeBuffer(arena.base);
        } else {
            (void)cudaFree(arena.base);
        }
        arena.base = nullptr;
    } catch (...) {
        LOG(WARNING) << "Retaining collective buffer pool after cleanup "
                        "failure";
        (void)arena.p2p_transport.release();
    }
}

PGResult<std::unique_ptr<CollectiveBufferLease>>
CollectiveBufferPool::tryAcquire(DeviceId device, uint64_t bytes,
                                 uint64_t alignment,
                                 const std::string& te_location,
                                 TransferEngine* engine,
                                 const CollectiveBufferPoolConfig& config) {
    PG_VALIDATE_ARG(bytes != 0 && alignment != 0,
                    "invalid collective buffer allocation");
    auto lease =
        std::unique_ptr<CollectiveBufferLease>(new CollectiveBufferLease);
    std::lock_guard<std::mutex> lock(mutex_);
    PG_VALIDATE_STATE(!shutdown_, "collective buffer pool is shut down");
    PG_TRY(auto arena, getOrCreateArena(device, te_location, engine, config));
    PG_TRY(auto offset, allocateBlock(*arena, bytes, alignment));
    ++arena->active_allocations;
    lease->device_ = device;
    lease->base_ = byteOffset(arena->base, offset);
    lease->offset_ = offset;
    lease->bytes_ = bytes;
    return lease;
}

bool CollectiveBufferPool::release(CollectiveBufferLease& lease,
                                   bool resource_idle) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto arena = arenas_.find(lease.device_);
    if (arena == arenas_.end()) return false;
    --arena->second->active_allocations;
    if (!resource_idle) {
        arena->second->has_retained_allocation = true;
        return false;
    }
    releaseBlock(*arena->second, lease.offset_, lease.bytes_);
    return true;
}

CollectiveArenaView CollectiveBufferPool::arena(DeviceId device) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto found = arenas_.find(device);
    PG_ASSERT(found != arenas_.end(),
              "collective arena must exist after buffer acquisition");
    const auto& arena = *found->second;
    return CollectiveArenaView{arena.device, arena.generation, arena.base,
                               arena.bytes, arena.p2p_handle};
}

void CollectiveBufferPool::shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (shutdown_) return;
    shutdown_ = true;
    for (auto& [device, arena] : arenas_) {
        if (arena->active_allocations != 0 || arena->has_retained_allocation) {
            LOG(WARNING) << "Retaining collective buffer pool for OS cleanup, "
                            "device="
                         << device;
            (void)arena.release();
        } else {
            releaseArena(*arena);
        }
    }
    arenas_.clear();
}

}  // namespace mooncake
