#include "collective/runtime/collective_channels.h"

#include <algorithm>
#include <atomic>
#include <memory>
#include <utility>

#include <cuda_alike.h>

#include "collective/transport/host_transfer_executor.h"
#include "gpu_runtime.h"

namespace mooncake {
namespace {

constexpr uint64_t kSignalBytes = 4096;
constexpr uint64_t kControlAlignment = 64;

uint64_t alignUp(uint64_t value, uint64_t alignment) {
    const auto remainder = value % alignment;
    return remainder == 0 ? value : value + alignment - remainder;
}

PGResult<CollectiveControlLayout> buildControlLayout(uint32_t channel_count) {
    PG_VALIDATE_ARG(channel_count != 0,
                    "collective communicator needs a channel");
    CollectiveControlLayout layout;
    layout.version = 1;
    layout.alignment = kControlAlignment;
    layout.channel_count = channel_count;
    layout.channels.reserve(channel_count);
    // The local prefix contains one invocation sequence per channel. Peers
    // address only the signal spans that follow it.
    uint64_t offset =
        alignUp(channel_count * sizeof(uint64_t), kControlAlignment);
    for (uint32_t channel = 0; channel < channel_count; ++channel) {
        layout.channels.push_back(
            CollectiveControlChannelLayout{{offset, kSignalBytes}});
        offset += kSignalBytes;
    }
    layout.total_bytes = alignUp(offset, kControlAlignment);
    return layout;
}

struct HostMemoryDeleter {
    void operator()(void* memory) const noexcept {
        if (memory) (void)cudaFreeHost(memory);
    }
};

using HostMemory = std::unique_ptr<void, HostMemoryDeleter>;

PGResult<void> cudaError(cudaError_t error, const char* operation) {
    if (error == cudaSuccess) return {};
    return makePGError(PGErrorCode::SystemError,
                       std::string(operation) + " failed: " +
                           cudaGetErrorString(error));
}

void resetHostCommand(HostTransferCommand& command) {
    command.kind =
        static_cast<uint32_t>(HostTransferCommandKind::PutAndSignal);
    command.peer_host_link = kInvalidHostLinkHandle;
    command.peer_in_group_rank = -1;
    command.source_address = 0;
    command.target_address = 0;
    command.bytes = 0;
    command.signal_source_address = 0;
    command.signal_target_address = 0;
    std::atomic_ref<uint32_t>(command.state)
        .store(static_cast<uint32_t>(HostTransferCommandState::Idle),
               std::memory_order_release);
}

}  // namespace

PGResult<std::unique_ptr<CollectiveChannels>> CollectiveChannels::create(
    CollectiveBufferPool* buffers, HostTransferExecutor* host_executor,
    DeviceId device, const std::string& te_location, TransferEngine* engine,
    uint32_t channel_count) {
    PG_VALIDATE_ARG(buffers && host_executor,
                    "collective channel dependencies are null");
    PG_TRY(auto layout, buildControlLayout(channel_count));

    const uint64_t controls_bytes =
        channel_count * sizeof(CollectiveControlBlock);
    const uint64_t commands_offset =
        alignUp(controls_bytes, alignof(HostTransferCommand));
    const uint64_t host_bytes =
        commands_offset + channel_count * sizeof(HostTransferCommand);

    void* host = nullptr;
    PG_TRY_CUDA(cudaHostAlloc(&host, host_bytes,
                              cudaHostAllocMapped | cudaHostAllocPortable));
    HostMemory host_memory(host);
    void* device_memory = nullptr;
    PG_TRY_CUDA(cudaHostGetDevicePointer(&device_memory, host_memory.get(), 0));

    auto* host_controls = static_cast<CollectiveControlBlock*>(host);
    auto* device_controls = static_cast<CollectiveControlBlock*>(device_memory);
    auto* host_commands = reinterpret_cast<HostTransferCommand*>(
        static_cast<char*>(host) + commands_offset);
    auto* device_commands = reinterpret_cast<HostTransferCommand*>(
        static_cast<char*>(device_memory) + commands_offset);
    for (uint32_t index = 0; index < channel_count; ++index) {
        host_controls[index] = CollectiveControlBlock{};
        host_commands[index] = HostTransferCommand{};
    }

    PG_TRY(auto peer_control,
           buffers->acquire(device, layout.total_bytes, layout.alignment,
                            te_location, engine));
    const GpuDeviceGuard guard(device);
    auto initialized = cudaError(
        cudaMemset(peer_control->base(), 0, peer_control->bytes()),
        "cudaMemset collective peer control");
    if (!initialized.has_value()) {
        buffers->release(*peer_control);
        return makePGError(std::move(initialized).error());
    }

    auto registered = host_executor->registerCommands(
        host_commands, host_controls, channel_count);
    if (!registered.has_value()) {
        buffers->release(*peer_control);
        return makePGError(std::move(registered).error());
    }

    return std::unique_ptr<CollectiveChannels>(new CollectiveChannels(
        buffers, host_executor, std::move(peer_control), std::move(layout),
        host_memory.release(), host_controls, device_controls, host_commands,
        device_commands));
}

CollectiveChannels::~CollectiveChannels() noexcept {
    if (!closed_) (void)close(false);
}

PGResult<CollectiveChannel> CollectiveChannels::acquire() {
    std::lock_guard<std::mutex> lock(mutex_);
    PG_VALIDATE_STATE(!closed_, "collective channels are closed");
    const auto channel_index = next_channel_index_;
    auto& state = states_[channel_index];
    if (state != ChannelState::Free) {
        return makePGError(PGErrorCode::ResourceBusy,
                           "collective channel is busy");
    }
    state = ChannelState::Acquired;
    next_channel_index_ = (channel_index + 1) % states_.size();
    host_controls_[channel_index] = CollectiveControlBlock{};
    resetHostCommand(host_commands_[channel_index]);
    return CollectiveChannel{
        .index = channel_index,
        .invocation_sequence =
            static_cast<uint64_t*>(peer_control_->base()) + channel_index,
        .peer_signals_base = peer_control_->base(),
        .peer_signals_offset = layout_.channels[channel_index].signals.offset,
        .host_control = host_controls_ + channel_index,
        .device_control = device_controls_ + channel_index,
        .host_command = host_commands_ + channel_index,
        .device_command = device_commands_ + channel_index,
    };
}

void CollectiveChannels::resetOrder() {
    std::lock_guard<std::mutex> lock(mutex_);
    PG_ASSERT(!closed_, "collective channels are closed");
    next_channel_index_ = 0;
}

void CollectiveChannels::release(const CollectiveChannel& channel) {
    std::lock_guard<std::mutex> lock(mutex_);
    PG_ASSERT(!closed_, "collective channels are closed");
    PG_ASSERT(channel.index < states_.size(),
              "collective channel index is invalid");
    auto& state = states_[channel.index];
    PG_ASSERT(state == ChannelState::Acquired,
              "collective channel was released twice");
    state = ChannelState::Free;
}

void CollectiveChannels::abandon(const CollectiveChannel& channel) {
    std::lock_guard<std::mutex> lock(mutex_);
    PG_ASSERT(!closed_, "collective channels are closed");
    PG_ASSERT(channel.index < states_.size(),
              "collective channel index is invalid");
    auto& state = states_[channel.index];
    PG_ASSERT(state == ChannelState::Acquired,
              "collective channel was abandoned twice");
    state = ChannelState::Abandoned;
}

bool CollectiveChannels::close(bool resources_safe) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_) return false;
    closed_ = true;

    const bool channels_idle =
        std::all_of(states_.begin(), states_.end(), [](const auto state) {
            return state == ChannelState::Free;
        });
    bool released = resources_safe && channels_idle;
    if (released) {
        released = host_executor_->unregisterCommands(host_commands_);
    }

    if (released) {
        buffers_->release(*peer_control_);
        (void)cudaFreeHost(host_memory_);
    } else {
        buffers_->abandon(*peer_control_);
    }
    peer_control_.reset();
    host_memory_ = nullptr;
    host_controls_ = nullptr;
    device_controls_ = nullptr;
    host_commands_ = nullptr;
    device_commands_ = nullptr;
    return released;
}

}  // namespace mooncake
