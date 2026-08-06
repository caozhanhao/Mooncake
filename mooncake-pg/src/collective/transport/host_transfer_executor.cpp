#include "collective/transport/host_transfer_executor.h"

#include <algorithm>
#include <chrono>
#include <exception>

#include <glog/logging.h>

#include <cuda_alike.h>
#include <transfer_engine.h>

#include "collective/runtime/control_block.cuh"
#include "control_plane/link_manager.h"

namespace mooncake {
namespace {

uint32_t loadState(const HostTransferCommand& command) {
    return std::atomic_ref<uint32_t>(const_cast<uint32_t&>(command.state))
        .load(std::memory_order_acquire);
}

void storeState(HostTransferCommand& command, HostTransferCommandState state) {
    std::atomic_ref<uint32_t>(command.state)
        .store(static_cast<uint32_t>(state), std::memory_order_release);
}

}  // namespace

HostTransferExecutor::~HostTransferExecutor() noexcept {
    try {
        shutdown();
    } catch (const std::exception& error) {
        LOG(WARNING) << "Host transfer executor shutdown failed: "
                     << error.what();
    } catch (...) {
        LOG(WARNING) << "Host transfer executor shutdown failed";
    }
}

PGResult<void> HostTransferExecutor::initialize(TransferEngine* engine,
                                                LinkManager* links) {
    std::lock_guard<std::mutex> lock(command_mutex_);
    if (initialized()) return {};

    try {
        engine_ = engine;
        links_ = links;
        stopping_.store(false, std::memory_order_release);
        thread_ = std::thread(&HostTransferExecutor::runLoop, this);
        initialized_.store(true, std::memory_order_release);
    } catch (const std::exception& exception) {
        engine_ = nullptr;
        links_ = nullptr;
        return makePGError(
            PGErrorCode::SystemError,
            std::string("failed to start host transfer executor: ") +
                exception.what());
    }
    return {};
}

PGResult<void> HostTransferExecutor::registerCommands(
    HostTransferCommand* commands, CollectiveControlBlock* controls,
    uint32_t command_count) {
    if (!initialized() || stopping_.load(std::memory_order_acquire)) {
        return makePGError(PGErrorCode::InvalidState,
                           "host transfer executor is not running");
    }
    std::lock_guard<std::mutex> lock(command_mutex_);
    const auto duplicate =
        std::find_if(command_regions_.begin(), command_regions_.end(),
                     [&](const auto& region) {
                         return region.commands == commands;
                     });
    if (duplicate != command_regions_.end()) {
        return makePGError(PGErrorCode::InvalidState,
                           "host transfer commands are already registered");
    }
    command_regions_.push_back({commands, controls, command_count});
    return {};
}

bool HostTransferExecutor::unregisterCommands(HostTransferCommand* commands) {
    std::lock_guard<std::mutex> lock(command_mutex_);
    const auto region =
        std::find_if(command_regions_.begin(), command_regions_.end(),
                     [&](const auto& candidate) {
                         return candidate.commands == commands;
                     });
    if (region == command_regions_.end()) return false;
    const auto active =
        std::find_if(active_transfers_.begin(), active_transfers_.end(),
                     [&](const auto& transfer) {
                         for (uint32_t index = 0;
                              index < region->command_count; ++index) {
                             if (transfer.command == region->commands + index) {
                                 return true;
                             }
                         }
                         return false;
                     });
    if (active != active_transfers_.end()) return false;
    for (uint32_t index = 0; index < region->command_count; ++index) {
        const auto state = static_cast<HostTransferCommandState>(
            loadState(region->commands[index]));
        if (state == HostTransferCommandState::Ready ||
            state == HostTransferCommandState::RunningData ||
            state == HostTransferCommandState::RunningSignal) {
            return false;
        }
    }
    command_regions_.erase(region);
    return true;
}

bool HostTransferExecutor::submitPhase(HostTransferCommand& command,
                                       CollectiveControlBlock& control,
                                       ActiveTransfer::Phase phase,
                                       ActiveTransfer& active) {
    const auto segment = links_->resolvePeer(command.peer_host_link);
    if (!segment.has_value()) {
        failCommand(command, control, CollectiveProtocolError::InvalidRoute);
        return false;
    }

    void* source = nullptr;
    uint64_t target = 0;
    size_t bytes = 0;
    if (phase == ActiveTransfer::Phase::Data) {
        source = reinterpret_cast<void*>(command.source_address);
        target = command.target_address;
        bytes = static_cast<size_t>(command.bytes);
    } else {
        source = reinterpret_cast<void*>(command.signal_source_address);
        target = command.signal_target_address;
        bytes = sizeof(uint64_t);
    }
    if (!source || target == 0 || bytes == 0) {
        failCommand(command, control, CollectiveProtocolError::InvalidRoute);
        return false;
    }

    const auto batch = engine_->allocateBatchID(1);
    if (batch == INVALID_BATCH_ID) {
        failCommand(command, control, CollectiveProtocolError::Transport);
        return false;
    }
    std::atomic_ref<uint32_t>(control.transport_idle)
        .store(0, std::memory_order_release);
    const auto submit = engine_->submitTransfer(
        batch, {TransferRequest{.opcode = TransferRequest::WRITE,
                                .source = source,
                                .target_id = *segment,
                                .target_offset = target,
                                .length = bytes}});
    if (!submit.ok()) {
        const auto released = engine_->freeBatchID(batch);
        if (released.ok()) {
            std::atomic_ref<uint32_t>(control.transport_idle)
                .store(1, std::memory_order_release);
        }
        failCommand(command, control, CollectiveProtocolError::Transport);
        return false;
    }

    active = ActiveTransfer{&command, &control, phase,
                            static_cast<uint64_t>(batch)};
    storeState(command, phase == ActiveTransfer::Phase::Data
                            ? HostTransferCommandState::RunningData
                            : HostTransferCommandState::RunningSignal);
    return true;
}

bool HostTransferExecutor::beginCommand(HostTransferCommand& command,
                                        CollectiveControlBlock& control) {
    uint32_t expected = static_cast<uint32_t>(HostTransferCommandState::Ready);
    if (!std::atomic_ref<uint32_t>(command.state)
             .compare_exchange_strong(
                 expected,
                 static_cast<uint32_t>(HostTransferCommandState::RunningData),
                 std::memory_order_acq_rel, std::memory_order_acquire)) {
        return false;
    }

    ActiveTransfer active;
    const auto kind = static_cast<HostTransferCommandKind>(command.kind);
    if (kind == HostTransferCommandKind::PutAndSignal && command.bytes != 0) {
        if (submitPhase(command, control, ActiveTransfer::Phase::Data,
                        active)) {
            active_transfers_.push_back(active);
        }
    } else if (kind == HostTransferCommandKind::PutAndSignal ||
               kind == HostTransferCommandKind::Signal) {
        if (submitPhase(command, control, ActiveTransfer::Phase::Signal,
                        active)) {
            active_transfers_.push_back(active);
        }
    } else {
        failCommand(command, control, CollectiveProtocolError::InvalidRoute);
    }
    return true;
}

bool HostTransferExecutor::advanceTransfer(ActiveTransfer& active) {
    auto& command = *active.command;
    auto& control = *active.control;
    TransferStatus status{};
    const auto query = engine_->getTransferStatus(active.batch_id, 0, status);
    if (!query.ok()) {
        failCommand(command, control, CollectiveProtocolError::Transport);
        return true;
    }
    if (status.s == TransferStatusEnum::WAITING ||
        status.s == TransferStatusEnum::PENDING) {
        return false;
    }
    if (status.s != TransferStatusEnum::COMPLETED) {
        failCommand(command, control,
                    status.s == TransferStatusEnum::TIMEOUT
                        ? CollectiveProtocolError::Timeout
                        : CollectiveProtocolError::Transport);
        return true;
    }
    const auto released = engine_->freeBatchID(active.batch_id);
    if (!released.ok()) {
        failCommand(command, control, CollectiveProtocolError::Transport);
        return true;
    }
    std::atomic_ref<uint32_t>(control.transport_idle)
        .store(1, std::memory_order_release);

    if (active.phase == ActiveTransfer::Phase::Data) {
        return !submitPhase(command, control, ActiveTransfer::Phase::Signal,
                            active);
    }
    storeState(command, HostTransferCommandState::Completed);
    return true;
}

void HostTransferExecutor::failCommand(HostTransferCommand& command,
                                       CollectiveControlBlock& control,
                                       CollectiveProtocolError error) {
    std::atomic_ref<int32_t> first_error(control.first_error_code);
    int32_t expected = 0;
    first_error.compare_exchange_strong(expected, static_cast<int32_t>(error),
                                        std::memory_order_acq_rel,
                                        std::memory_order_acquire);
    std::atomic_ref<InGroupRank>(control.failed_peer)
        .store(command.peer_in_group_rank, std::memory_order_release);
    storeState(command, HostTransferCommandState::Failed);
}

void HostTransferExecutor::runLoop() {
    while (!stopping_.load(std::memory_order_acquire)) {
        bool progressed = false;
        {
            std::lock_guard<std::mutex> lock(command_mutex_);
            for (auto& region : command_regions_) {
                for (uint32_t index = 0; index < region.command_count; ++index) {
                    if (loadState(region.commands[index]) ==
                        static_cast<uint32_t>(HostTransferCommandState::Ready)) {
                        progressed |= beginCommand(region.commands[index],
                                                   region.controls[index]);
                    }
                }
            }
            for (size_t index = 0; index < active_transfers_.size();) {
                if (advanceTransfer(active_transfers_[index])) {
                    active_transfers_[index] = active_transfers_.back();
                    active_transfers_.pop_back();
                    progressed = true;
                } else {
                    ++index;
                }
            }
        }
        if (!progressed) {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
    }
}

void HostTransferExecutor::shutdown() {
    {
        std::lock_guard<std::mutex> lock(command_mutex_);
        if (!initialized_.exchange(false, std::memory_order_acq_rel)) return;
        stopping_.store(true, std::memory_order_release);
    }
    if (thread_.joinable()) thread_.join();

    std::lock_guard<std::mutex> lock(command_mutex_);
    if (!active_transfers_.empty()) {
        LOG(WARNING) << "Host transfer executor stopped with active transfers";
    }
    command_regions_.clear();
    active_transfers_.clear();
    engine_ = nullptr;
    links_ = nullptr;
}

}  // namespace mooncake
