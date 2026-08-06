#ifndef MOONCAKE_PG_COLLECTIVE_RUNTIME_COLLECTIVE_CHANNELS_H
#define MOONCAKE_PG_COLLECTIVE_RUNTIME_COLLECTIVE_CHANNELS_H

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "collective/buffer/collective_buffer_pool.h"
#include "collective/endpoint.h"
#include "collective/runtime/control_block.cuh"
#include "collective/transport/host_transfer_command.cuh"
#include "error_types.h"

namespace mooncake {

class HostTransferExecutor;

// One fixed communicator channel. Its peer-visible signals, local failure
// control and Host transfer command keep stable addresses until communicator
// teardown. Acquiring a channel grants exclusive use; it allocates no memory.
struct CollectiveChannel {
    uint32_t index = 0;
    uint64_t* invocation_sequence = nullptr;
    void* peer_signals_base = nullptr;
    uint64_t peer_signals_offset = 0;
    CollectiveControlBlock* host_control = nullptr;
    CollectiveControlBlock* device_control = nullptr;
    HostTransferCommand* host_command = nullptr;
    HostTransferCommand* device_command = nullptr;
};

class CollectiveChannels {
   public:
    static PGResult<std::unique_ptr<CollectiveChannels>> create(
        CollectiveBufferPool* buffers, HostTransferExecutor* host_executor,
        DeviceId device, const std::string& te_location,
        TransferEngine* engine, uint32_t channel_count = 3);
    ~CollectiveChannels() noexcept;

    const CollectiveControlLayout& layout() const { return layout_; }
    void* peerControlBase() const { return peer_control_->base(); }

    // Channel identity is wire-visible. Callers must request the exact channel
    // selected by the matching collective order on every rank.
    PGResult<CollectiveChannel> acquire(uint32_t channel_index);
    void release(const CollectiveChannel& channel);
    void abandon(const CollectiveChannel& channel);
    bool close(bool resources_safe);

    CollectiveChannels(const CollectiveChannels&) = delete;
    CollectiveChannels& operator=(const CollectiveChannels&) = delete;

   private:
    enum class ChannelState : uint8_t {
        Free = 0,
        Acquired,
        Abandoned,
    };

    CollectiveChannels(CollectiveBufferPool* buffers,
                       HostTransferExecutor* host_executor,
                       std::unique_ptr<CollectiveBufferLease> peer_control,
                       CollectiveControlLayout layout, void* host_memory,
                       CollectiveControlBlock* host_controls,
                       CollectiveControlBlock* device_controls,
                       HostTransferCommand* host_commands,
                       HostTransferCommand* device_commands)
        : buffers_(buffers),
          host_executor_(host_executor),
          peer_control_(std::move(peer_control)),
          layout_(std::move(layout)),
          host_memory_(host_memory),
          host_controls_(host_controls),
          device_controls_(device_controls),
          host_commands_(host_commands),
          device_commands_(device_commands),
          states_(layout_.channel_count) {}

    CollectiveBufferPool* buffers_ = nullptr;
    HostTransferExecutor* host_executor_ = nullptr;
    std::unique_ptr<CollectiveBufferLease> peer_control_;
    CollectiveControlLayout layout_;

    void* host_memory_ = nullptr;
    CollectiveControlBlock* host_controls_ = nullptr;
    CollectiveControlBlock* device_controls_ = nullptr;
    HostTransferCommand* host_commands_ = nullptr;
    HostTransferCommand* device_commands_ = nullptr;

    std::mutex mutex_;
    std::vector<ChannelState> states_;
    bool commands_registered_ = true;
    bool closed_ = false;
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_COLLECTIVE_RUNTIME_COLLECTIVE_CHANNELS_H
