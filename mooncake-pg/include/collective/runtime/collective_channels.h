#ifndef MOONCAKE_PG_COLLECTIVE_RUNTIME_COLLECTIVE_CHANNELS_H
#define MOONCAKE_PG_COLLECTIVE_RUNTIME_COLLECTIVE_CHANNELS_H

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "collective/buffer/collective_buffer_pool.h"
#include "collective/endpoint.h"
#include "collective/runtime/control_block.cuh"
#include "collective/transport/host_transfer_command.cuh"
#include "error_types.h"

namespace mooncake {

class HostTransferExecutor;
class CollectiveChannels;

// Move-only exclusive use of one fixed communicator channel. The channel's
// peer-visible signals, local failure control and Host transfer command keep
// stable addresses until communicator teardown. Normal destruction releases
// the use; abandon() permanently quarantines it.
struct CollectiveChannelLease {
    ~CollectiveChannelLease() noexcept;

    CollectiveChannelLease(const CollectiveChannelLease&) = delete;
    CollectiveChannelLease& operator=(const CollectiveChannelLease&) = delete;
    CollectiveChannelLease(CollectiveChannelLease&& other) noexcept;
    CollectiveChannelLease& operator=(CollectiveChannelLease&& other) noexcept;

    void release() noexcept;
    void abandon() noexcept;

    uint64_t* invocation_sequence = nullptr;
    void* peer_signals_base = nullptr;
    uint64_t peer_signals_offset = 0;
    CollectiveControlBlock* host_control = nullptr;
    CollectiveControlBlock* device_control = nullptr;
    HostTransferCommand* host_command = nullptr;
    HostTransferCommand* device_command = nullptr;

   private:
    friend class CollectiveChannels;
    CollectiveChannelLease() = default;

    void moveFrom(CollectiveChannelLease&& other) noexcept;
    void clear() noexcept;

    CollectiveChannels* owner_ = nullptr;
    uint32_t index_ = 0;
};

class CollectiveChannels {
   public:
    static PGResult<std::unique_ptr<CollectiveChannels>> create(
        CollectiveBufferPool* buffers, HostTransferExecutor* host_executor,
        DeviceId device, const std::string& te_location,
        TransferEngine* engine, uint32_t channel_count = 3);
    ~CollectiveChannels() noexcept;

    const CollectiveControlLayout& layout() const { return layout_; }
    void* peerControlBase() const { return peer_control_.base(); }

    // Channel identity is wire-visible. Acquisition selects only the next
    // channel in collective order and never falls back to a locally free one.
    PGResult<CollectiveChannelLease> acquire();
    // View application is collective-quiescent. Resetting here gives every
    // rank the same channel order for the replacement view.
    void resetOrder();
    bool close(bool resources_safe);

    CollectiveChannels(const CollectiveChannels&) = delete;
    CollectiveChannels& operator=(const CollectiveChannels&) = delete;

   private:
    friend struct CollectiveChannelLease;

    enum class ChannelState : uint8_t {
        Free = 0,
        Acquired,
        Abandoned,
    };

    CollectiveChannels(HostTransferExecutor* host_executor,
                       CollectiveBufferLease peer_control,
                       CollectiveControlLayout layout, void* host_memory,
                       CollectiveControlBlock* host_controls,
                       CollectiveControlBlock* device_controls,
                       HostTransferCommand* host_commands,
                       HostTransferCommand* device_commands)
        : host_executor_(host_executor),
          peer_control_(std::move(peer_control)),
          layout_(std::move(layout)),
          host_memory_(host_memory),
          host_controls_(host_controls),
          device_controls_(device_controls),
          host_commands_(host_commands),
          device_commands_(device_commands),
          states_(layout_.channel_count) {}

    HostTransferExecutor* host_executor_ = nullptr;
    CollectiveBufferLease peer_control_;
    CollectiveControlLayout layout_;

    void* host_memory_ = nullptr;
    CollectiveControlBlock* host_controls_ = nullptr;
    CollectiveControlBlock* device_controls_ = nullptr;
    HostTransferCommand* host_commands_ = nullptr;
    HostTransferCommand* device_commands_ = nullptr;

    std::mutex mutex_;
    std::vector<ChannelState> states_;
    uint32_t next_channel_index_ = 0;
    bool closed_ = false;

    void release(uint32_t channel_index);
    void abandon(uint32_t channel_index);
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_COLLECTIVE_RUNTIME_COLLECTIVE_CHANNELS_H
