#include "device_comm/device_collective/device_collective_resources.h"

#include <algorithm>
#include <utility>

#include "gpu_runtime.h"
#include "pg_utils.h"

namespace mooncake {
namespace {

PGResult<uint64_t> reserveBytes(uint64_t& cursor, uint64_t size,
                                uint64_t alignment) {
    PG_VALIDATE_ARG(alignment != 0, "resource alignment is zero");
    const uint64_t padding = alignmentPadding(cursor, alignment);
    PG_VALIDATE_ARG(!addOverflows(cursor, padding),
                    "device collective resource layout overflows");
    cursor += padding;
    const uint64_t offset = cursor;
    PG_VALIDATE_ARG(!addOverflows(cursor, size),
                    "device collective resource layout overflows");
    cursor += size;
    return offset;
}

template <typename Item>
Item* itemAt(void* base, uint64_t offset) noexcept {
    return reinterpret_cast<Item*>(static_cast<char*>(base) + offset);
}

bool rangesOverlap(uint64_t first_offset, uint64_t first_size,
                   uint64_t second_offset, uint64_t second_size) noexcept {
    if (addOverflows(first_offset, first_size) ||
        addOverflows(second_offset, second_size)) {
        return true;
    }
    return first_offset < second_offset + second_size &&
           second_offset < first_offset + first_size;
}

}  // namespace

DeviceCollectiveWorkspace::DeviceCollectiveWorkspace(
    DeviceArena& arena, DeviceArenaSlice buffer, size_t alignment) noexcept
    : arena_(arena),
      buffer_(std::move(buffer)),
      alignment_(alignment) {}

PGResult<std::unique_ptr<DeviceCollectiveWorkspace>>
DeviceCollectiveWorkspace::create(DeviceArena& arena, size_t buffer_size,
                                  size_t alignment) {
    PG_TRY(auto buffer, arena.allocate(buffer_size, alignment));
    return std::unique_ptr<DeviceCollectiveWorkspace>(
        new DeviceCollectiveWorkspace(arena, std::move(buffer), alignment));
}

const DeviceArenaSlice& DeviceCollectiveWorkspace::buffer() const noexcept {
    return buffer_;
}

PGResult<const DeviceArenaSlice*> DeviceCollectiveWorkspace::ensureStaging() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!staging_) {
        PG_TRY(auto staging, arena_.allocate(buffer_.size(), alignment_));
        staging_.emplace(std::move(staging));
    }
    return &*staging_;
}

PGResult<DeviceCollectiveResources::StateLayout>
DeviceCollectiveResources::StateLayout::make(
    uint32_t signal_count, uint64_t protocol_state_size,
    uint64_t protocol_state_alignment) {
    PG_VALIDATE_ARG(signal_count != 0,
                    "device collective signal capacity is zero");
    PG_VALIDATE_ARG(protocol_state_size != 0,
                    "device collective protocol-state capacity is zero");
    PG_VALIDATE_ARG(protocol_state_alignment != 0,
                    "device collective protocol-state alignment is zero");
    PG_VALIDATE_ARG(
        (protocol_state_alignment & (protocol_state_alignment - 1)) == 0,
        "device collective protocol-state alignment is not a power of two");

    StateLayout layout;
    layout.alignment =
        std::max(kMinimumAlignment, protocol_state_alignment);
    uint64_t cursor = 0;
    PG_TRY(layout.invocation_offset,
           reserveBytes(cursor, sizeof(DeviceCollectiveInvocationState),
                        alignof(DeviceCollectiveInvocationState)));
    PG_TRY(layout.signals_offset,
           reserveBytes(cursor,
                        static_cast<uint64_t>(signal_count) * sizeof(uint64_t),
                        alignof(uint64_t)));
    PG_TRY(layout.protocol_state_offset,
           reserveBytes(cursor, protocol_state_size,
                        protocol_state_alignment));

    const uint64_t tail_padding = alignmentPadding(cursor, layout.alignment);
    PG_VALIDATE_ARG(!addOverflows(cursor, tail_padding),
                    "device collective resource layout overflows");
    layout.size = cursor + tail_padding;
    return layout;
}

DeviceCollectiveResources::DeviceCollectiveResources(
    DeviceArenaSlice state_slice,
    DeviceCollectiveKernelResources device_view, uint64_t* signals,
    uint64_t signal_offset, uint32_t signal_count) noexcept
    : state_slice_(std::move(state_slice)),
      device_view_(device_view),
      signals_(signals),
      signal_offset_(signal_offset),
      signal_count_(signal_count) {}

PGResult<std::unique_ptr<DeviceCollectiveResources>>
DeviceCollectiveResources::create(
    DeviceArena& arena, const DeviceTransferHandle* transfer_handle,
    uint64_t timeout_ticks, uint32_t signal_count,
    uint64_t protocol_state_size, uint64_t protocol_state_alignment) {
    PG_VALIDATE_ARG(transfer_handle,
                    "device collective transfer handle is null");

    PG_TRY(auto layout,
           StateLayout::make(signal_count, protocol_state_size,
                             protocol_state_alignment));
    PG_TRY(auto state_slice,
           arena.allocate(layout.size, layout.alignment));
    {
        PG_TRY(auto device_guard,
               GpuDeviceGuard::create(arena.deviceIndex()));
        PG_TRY_CUDA(cudaMemset(state_slice.addr(), 0, state_slice.size()));
    }

    auto* const state_base = static_cast<char*>(state_slice.addr());
    auto* const signals =
        itemAt<uint64_t>(state_base, layout.signals_offset);
    const uint64_t signal_offset =
        state_slice.offset() + layout.signals_offset;
    DeviceCollectiveKernelResources device_view{
        .transfer_handle = transfer_handle,
        .protocol_state = state_base + layout.protocol_state_offset,
        .protocol_state_size = protocol_state_size,
        .invocation = itemAt<DeviceCollectiveInvocationState>(
            state_base, layout.invocation_offset),
        .timeout_ticks = timeout_ticks,
    };

    return std::unique_ptr<DeviceCollectiveResources>(
        new DeviceCollectiveResources(std::move(state_slice), device_view,
                                      signals, signal_offset, signal_count));
}

DeviceCollectiveKernelResources DeviceCollectiveResources::deviceView()
    const noexcept {
    return device_view_;
}

uint64_t* DeviceCollectiveResources::signals() const noexcept {
    return signals_;
}

uint64_t DeviceCollectiveResources::signalOffset() const noexcept {
    return signal_offset_;
}

uint32_t DeviceCollectiveResources::signalCount() const noexcept {
    return signal_count_;
}

void DeviceCollectiveResources::setRecoveryMailbox(
    DeviceCollectiveRecoveryMailbox* mailbox) noexcept {
    device_view_.recovery = mailbox;
}

bool deviceCollectiveEndpointSupports(
    const DeviceCollectiveEndpoint& endpoint, uint64_t required_buffer_size,
    uint32_t required_signal_count) noexcept {
    if (endpoint.buffer_size < required_buffer_size ||
        endpoint.signal_capacity < required_signal_count) {
        return false;
    }
    const uint64_t signal_bytes =
        static_cast<uint64_t>(endpoint.signal_capacity) * sizeof(uint64_t);
    if (addOverflows(endpoint.buffer_offset, endpoint.buffer_size) ||
        addOverflows(endpoint.signal_offset, signal_bytes)) {
        return false;
    }
    return !rangesOverlap(endpoint.buffer_offset, endpoint.buffer_size,
                          endpoint.signal_offset, signal_bytes);
}

}  // namespace mooncake
