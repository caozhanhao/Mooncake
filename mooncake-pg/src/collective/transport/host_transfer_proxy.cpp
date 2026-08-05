#include "collective/transport/host_transfer_proxy.h"

#include <chrono>
#include <cstring>
#include <exception>
#include <memory>

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

struct CudaHostMemoryDeleter {
    void operator()(void* memory) const noexcept { (void)cudaFreeHost(memory); }
};

using CudaHostMemory = std::unique_ptr<void, CudaHostMemoryDeleter>;

}  // namespace

CollectiveHostTransferProxy::~CollectiveHostTransferProxy() noexcept {
    try {
        shutdown();
    } catch (const std::exception& error) {
        LOG(WARNING) << "Collective host proxy shutdown failed: "
                     << error.what();
    } catch (...) {
        LOG(WARNING) << "Collective host proxy shutdown failed";
    }
}

PGResult<void> CollectiveHostTransferProxy::initialize(TransferEngine* engine,
                                                       LinkManager* links,
                                                       uint32_t command_count) {
    std::lock_guard<std::mutex> lock(command_mutex_);
    if (initialized()) return {};

    const auto bytes = command_count * sizeof(HostTransferCommand);
    void* host = nullptr;
    PG_TRY_CUDA(cudaHostAlloc(&host, bytes,
                              cudaHostAllocMapped | cudaHostAllocPortable));
    CudaHostMemory host_memory(host);
    void* device = nullptr;
    PG_TRY_CUDA(cudaHostGetDevicePointer(&device, host_memory.get(), 0));

    try {
        commands_.resize(command_count);
        active_transfers_.reserve(command_count);
        std::memset(host_memory.get(), 0, bytes);
        engine_ = engine;
        links_ = links;
        host_commands_ = static_cast<HostTransferCommand*>(host_memory.get());
        device_commands_ = static_cast<HostTransferCommand*>(device);
        command_count_ = command_count;
        stopping_.store(false, std::memory_order_release);
        progress_thread_ =
            std::thread(&CollectiveHostTransferProxy::progressLoop, this);
        initialized_.store(true, std::memory_order_release);
    } catch (const std::exception& exception) {
        commands_.clear();
        active_transfers_.clear();
        engine_ = nullptr;
        links_ = nullptr;
        host_commands_ = nullptr;
        device_commands_ = nullptr;
        command_count_ = 0;
        return makePGError(
            PGErrorCode::SystemError,
            std::string("failed to start collective host proxy: ") +
                exception.what());
    }
    (void)host_memory.release();
    return {};
}

std::optional<HostTransferCommandLease>
CollectiveHostTransferProxy::tryAcquireCommand(
    CollectiveControlBlock* control) {
    if (!initialized() || stopping_.load(std::memory_order_acquire)) {
        return std::nullopt;
    }
    std::lock_guard<std::mutex> lock(command_mutex_);
    for (uint32_t index = 0; index < command_count_; ++index) {
        auto& state = commands_[index];
        if (state.in_use || !state.reusable) continue;
        state.in_use = true;
        state.control = control;
        host_commands_[index] = HostTransferCommand{};
        return HostTransferCommandLease{index, host_commands_ + index,
                                        device_commands_ + index};
    }
    return std::nullopt;
}

bool CollectiveHostTransferProxy::commandReusableLocked(
    const HostTransferCommandLease& lease, bool resource_idle) {
    auto state = static_cast<HostTransferCommandState>(loadState(*lease.host));
    if (resource_idle && (state == HostTransferCommandState::Completed ||
                          state == HostTransferCommandState::Failed)) {
        storeState(*lease.host, HostTransferCommandState::Idle);
        state = HostTransferCommandState::Idle;
    }
    return resource_idle && state == HostTransferCommandState::Idle;
}

bool CollectiveHostTransferProxy::commandReusable(
    const HostTransferCommandLease& command, bool resource_idle) {
    std::lock_guard<std::mutex> lock(command_mutex_);
    PG_ASSERT(command.index < commands_.size() &&
                  commands_[command.index].in_use &&
                  command.host == host_commands_ + command.index &&
                  command.device == device_commands_ + command.index,
              "collective host command lease is invalid");
    return commandReusableLocked(command, resource_idle);
}

bool CollectiveHostTransferProxy::releaseCommand(
    const HostTransferCommandLease& command, bool resource_idle) {
    std::lock_guard<std::mutex> lock(command_mutex_);
    PG_ASSERT(command.index < commands_.size() &&
                  commands_[command.index].in_use &&
                  command.host == host_commands_ + command.index &&
                  command.device == device_commands_ + command.index,
              "collective host command lease is invalid");
    auto& state = commands_[command.index];
    state.reusable = commandReusableLocked(command, resource_idle);
    state.in_use = false;
    if (state.reusable) state.control = nullptr;
    return state.reusable;
}

bool CollectiveHostTransferProxy::submitPhase(uint32_t command_index,
                                              ActiveTransfer::Phase phase,
                                              ActiveTransfer& active) {
    auto& command = host_commands_[command_index];
    auto& control = *commands_[command_index].control;
    const auto segment = links_->resolvePeer(command.peer_host_link);
    if (!segment.has_value()) {
        failCommand(command_index, CollectiveProtocolError::InvalidBinding);
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
        failCommand(command_index, CollectiveProtocolError::InvalidBinding);
        return false;
    }

    const auto batch = engine_->allocateBatchID(1);
    if (batch == INVALID_BATCH_ID) {
        failCommand(command_index, CollectiveProtocolError::Transport);
        return false;
    }
    std::atomic_ref<uint32_t>(control.resource_idle)
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
            std::atomic_ref<uint32_t>(control.resource_idle)
                .store(1, std::memory_order_release);
        }
        failCommand(command_index, CollectiveProtocolError::Transport);
        return false;
    }

    active = ActiveTransfer{command_index, phase, static_cast<uint64_t>(batch)};
    storeState(command, phase == ActiveTransfer::Phase::Data
                            ? HostTransferCommandState::RunningData
                            : HostTransferCommandState::RunningSignal);
    return true;
}

bool CollectiveHostTransferProxy::beginCommand(uint32_t command_index) {
    auto& command = host_commands_[command_index];
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
        if (submitPhase(command_index, ActiveTransfer::Phase::Data, active)) {
            active_transfers_.push_back(active);
        }
    } else if (kind == HostTransferCommandKind::PutAndSignal ||
               kind == HostTransferCommandKind::Signal) {
        if (submitPhase(command_index, ActiveTransfer::Phase::Signal, active)) {
            active_transfers_.push_back(active);
        }
    } else {
        failCommand(command_index, CollectiveProtocolError::InvalidBinding);
    }
    return true;
}

bool CollectiveHostTransferProxy::advanceTransfer(ActiveTransfer& active) {
    auto& command = host_commands_[active.command_index];
    auto& control = *commands_[active.command_index].control;
    TransferStatus status{};
    const auto query = engine_->getTransferStatus(active.batch_id, 0, status);
    if (!query.ok()) {
        failCommand(active.command_index, CollectiveProtocolError::Transport);
        return true;
    }
    if (status.s == TransferStatusEnum::WAITING ||
        status.s == TransferStatusEnum::PENDING) {
        return false;
    }
    if (status.s != TransferStatusEnum::COMPLETED) {
        failCommand(active.command_index,
                    status.s == TransferStatusEnum::TIMEOUT
                        ? CollectiveProtocolError::Timeout
                        : CollectiveProtocolError::Transport);
        return true;
    }
    const auto released = engine_->freeBatchID(active.batch_id);
    if (!released.ok()) {
        failCommand(active.command_index, CollectiveProtocolError::Transport);
        return true;
    }
    std::atomic_ref<uint32_t>(control.resource_idle)
        .store(1, std::memory_order_release);

    if (active.phase == ActiveTransfer::Phase::Data) {
        return !submitPhase(active.command_index, ActiveTransfer::Phase::Signal,
                            active);
    }
    storeState(command, HostTransferCommandState::Completed);
    return true;
}

void CollectiveHostTransferProxy::failCommand(uint32_t command_index,
                                              CollectiveProtocolError error) {
    auto& command = host_commands_[command_index];
    auto& control = *commands_[command_index].control;
    std::atomic_ref<int32_t> first_error(control.first_error_code);
    int32_t expected = 0;
    first_error.compare_exchange_strong(expected, static_cast<int32_t>(error),
                                        std::memory_order_acq_rel,
                                        std::memory_order_acquire);
    std::atomic_ref<InGroupRank>(control.failed_peer)
        .store(command.peer_in_group_rank, std::memory_order_release);
    storeState(command, HostTransferCommandState::Failed);
}

void CollectiveHostTransferProxy::progressLoop() {
    while (!stopping_.load(std::memory_order_acquire)) {
        bool progressed = false;
        for (uint32_t index = 0; index < command_count_; ++index) {
            if (loadState(host_commands_[index]) ==
                static_cast<uint32_t>(HostTransferCommandState::Ready)) {
                progressed |= beginCommand(index);
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
        if (!progressed) {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
    }
}

void CollectiveHostTransferProxy::shutdown() {
    {
        std::lock_guard<std::mutex> lock(command_mutex_);
        if (!initialized_.exchange(false, std::memory_order_acq_rel)) return;
        stopping_.store(true, std::memory_order_release);
    }
    if (progress_thread_.joinable()) progress_thread_.join();

    bool safe_to_free = active_transfers_.empty();
    {
        std::lock_guard<std::mutex> lock(command_mutex_);
        for (const auto& command : commands_) {
            safe_to_free &= !command.in_use && command.reusable;
        }
    }
    if (safe_to_free && host_commands_) {
        (void)cudaFreeHost(host_commands_);
    } else if (host_commands_) {
        LOG(WARNING) << "Retaining collective host commands because "
                        "resource-idle is unproven";
    }
    host_commands_ = nullptr;
    device_commands_ = nullptr;
    command_count_ = 0;
    commands_.clear();
    active_transfers_.clear();
    engine_ = nullptr;
    links_ = nullptr;
}

}  // namespace mooncake
