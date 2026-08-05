#ifndef MOONCAKE_PG_COLLECTIVE_TRANSPORT_HOST_TRANSFER_EXECUTOR_H
#define MOONCAKE_PG_COLLECTIVE_TRANSPORT_HOST_TRANSFER_EXECUTOR_H

#include <atomic>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

#include "collective/transport/host_transfer_command.cuh"
#include "error_types.h"

namespace mooncake {

class LinkManager;
class TransferEngine;
struct CollectiveControlBlock;
enum class CollectiveProtocolError : int32_t;

struct HostTransferCommandLease {
    uint32_t index = 0;
    HostTransferCommand* host = nullptr;
    HostTransferCommand* device = nullptr;
};

// Process-level executor for device-produced Host transfer commands. It owns
// only command storage; the control block passed at acquisition remains owned
// by the common runtime resource pool.
class HostTransferExecutor {
   public:
    HostTransferExecutor() = default;
    ~HostTransferExecutor() noexcept;

    PGResult<void> initialize(TransferEngine* engine, LinkManager* links,
                              uint32_t command_count = 128);
    PGResult<HostTransferCommandLease> acquireCommand(
        CollectiveControlBlock* control);
    bool releaseCommand(const HostTransferCommandLease& command);
    void abandonCommand(const HostTransferCommandLease& command);
    void shutdown();

    bool initialized() const {
        return initialized_.load(std::memory_order_acquire);
    }

    HostTransferExecutor(const HostTransferExecutor&) = delete;
    HostTransferExecutor& operator=(const HostTransferExecutor&) = delete;

   private:
    enum class SlotState : uint8_t {
        Free = 0,
        Acquired,
        Abandoned,
    };

    struct CommandSlot {
        SlotState state = SlotState::Free;
        CollectiveControlBlock* control = nullptr;
    };

    struct ActiveTransfer {
        enum class Phase : uint8_t { Data = 0, Signal };
        uint32_t command_index = 0;
        Phase phase = Phase::Data;
        uint64_t batch_id = 0;
    };

    bool beginCommand(uint32_t command_index);
    bool submitPhase(uint32_t command_index, ActiveTransfer::Phase phase,
                     ActiveTransfer& active);
    bool advanceTransfer(ActiveTransfer& active);
    void failCommand(uint32_t command_index, CollectiveProtocolError error);
    void runLoop();

    TransferEngine* engine_ = nullptr;
    LinkManager* links_ = nullptr;
    HostTransferCommand* host_commands_ = nullptr;
    HostTransferCommand* device_commands_ = nullptr;
    uint32_t command_count_ = 0;

    std::mutex command_mutex_;
    std::vector<CommandSlot> commands_;
    std::vector<ActiveTransfer> active_transfers_;

    std::thread thread_;
    std::atomic<bool> initialized_{false};
    std::atomic<bool> stopping_{false};
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_COLLECTIVE_TRANSPORT_HOST_TRANSFER_EXECUTOR_H
