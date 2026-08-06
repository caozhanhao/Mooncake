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

// Process-level executor for device-produced Host transfer commands.
// Communicators own stable command/control regions and register them here for
// polling; the executor owns neither region.
class HostTransferExecutor {
   public:
    HostTransferExecutor() = default;
    ~HostTransferExecutor() noexcept;

    PGResult<void> initialize(TransferEngine* engine, LinkManager* links);
    PGResult<void> registerCommands(HostTransferCommand* commands,
                                    CollectiveControlBlock* controls,
                                    uint32_t command_count);
    bool unregisterCommands(HostTransferCommand* commands);
    void shutdown();

    bool initialized() const {
        return initialized_.load(std::memory_order_acquire);
    }

    HostTransferExecutor(const HostTransferExecutor&) = delete;
    HostTransferExecutor& operator=(const HostTransferExecutor&) = delete;

   private:
    struct CommandRegion {
        HostTransferCommand* commands = nullptr;
        CollectiveControlBlock* controls = nullptr;
        uint32_t command_count = 0;
    };

    struct ActiveTransfer {
        enum class Phase : uint8_t { Data = 0, Signal };
        HostTransferCommand* command = nullptr;
        CollectiveControlBlock* control = nullptr;
        Phase phase = Phase::Data;
        uint64_t batch_id = 0;
    };

    bool beginCommand(HostTransferCommand& command,
                      CollectiveControlBlock& control);
    bool submitPhase(HostTransferCommand& command,
                     CollectiveControlBlock& control,
                     ActiveTransfer::Phase phase, ActiveTransfer& active);
    bool advanceTransfer(ActiveTransfer& active);
    static void failCommand(HostTransferCommand& command,
                            CollectiveControlBlock& control,
                            CollectiveProtocolError error);
    void runLoop();

    TransferEngine* engine_ = nullptr;
    LinkManager* links_ = nullptr;
    std::mutex command_mutex_;
    std::vector<CommandRegion> command_regions_;
    std::vector<ActiveTransfer> active_transfers_;

    std::thread thread_;
    std::atomic<bool> initialized_{false};
    std::atomic<bool> stopping_{false};
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_COLLECTIVE_TRANSPORT_HOST_TRANSFER_EXECUTOR_H
