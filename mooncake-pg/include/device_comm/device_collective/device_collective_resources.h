#ifndef MOONCAKE_PG_DEVICE_COMM_DEVICE_COLLECTIVE_DEVICE_COLLECTIVE_RESOURCES_H
#define MOONCAKE_PG_DEVICE_COMM_DEVICE_COLLECTIVE_DEVICE_COLLECTIVE_RESOURCES_H

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>

#include "control_plane/control_types.h"
#include "device_comm/device_arena.h"
#include "device_comm/device_collective/device_collective_types.cuh"
#include "error_types.h"

namespace mooncake {

// Context-wide payload storage shared by device collective protocols. The
// buffer is always present and is the only payload region published to peers.
// Staging is allocated only when a non-direct route first needs it; it remains
// local, registered and stable until context shutdown.
//
// StrongStream serializes protocol kernels that borrow these buffers, so one
// pair can be shared by every communicator in the context.
class DeviceCollectiveWorkspace {
   public:
    static PGResult<std::unique_ptr<DeviceCollectiveWorkspace>> create(
        DeviceArena& arena, size_t buffer_size, size_t alignment);

    ~DeviceCollectiveWorkspace() noexcept = default;

    DeviceCollectiveWorkspace(const DeviceCollectiveWorkspace&) = delete;
    DeviceCollectiveWorkspace& operator=(const DeviceCollectiveWorkspace&) =
        delete;

    [[nodiscard]] const DeviceArenaSlice& buffer() const noexcept;

    // Allocates at most once. The returned slice has the same size and
    // alignment as buffer() and is never published in an endpoint.
    PGResult<const DeviceArenaSlice*> ensureStaging();

   private:
    DeviceCollectiveWorkspace(DeviceArena& arena, DeviceArenaSlice buffer,
                              size_t alignment) noexcept;

    DeviceArena& arena_;
    DeviceArenaSlice buffer_;
    size_t alignment_ = 0;
    std::optional<DeviceArenaSlice> staging_;
    std::mutex mutex_;
};

[[nodiscard]] bool deviceCollectiveEndpointSupports(
    const DeviceCollectiveEndpoint& endpoint, uint64_t required_buffer_size,
    uint32_t required_signal_count) noexcept;

// Owns the communicator-local signal/protocol-state allocation and exposes one
// flat kernel view. Creation fixes every address and capacity; ordinary
// launches and CUDA Graph replay perform no resource allocation.
class DeviceCollectiveResources {
   public:
    static PGResult<std::unique_ptr<DeviceCollectiveResources>> create(
        DeviceArena& arena, const DeviceTransferHandle* transfer_handle,
        uint64_t timeout_ticks, uint32_t signal_count,
        uint64_t protocol_state_size,
        uint64_t protocol_state_alignment);

    ~DeviceCollectiveResources() noexcept = default;

    DeviceCollectiveResources(const DeviceCollectiveResources&) = delete;
    DeviceCollectiveResources& operator=(const DeviceCollectiveResources&) =
        delete;

    [[nodiscard]] DeviceCollectiveKernelResources deviceView() const noexcept;
    [[nodiscard]] uint64_t* signals() const noexcept;
    [[nodiscard]] uint64_t signalOffset() const noexcept;
    [[nodiscard]] uint32_t signalCount() const noexcept;

    void setRecoveryMailbox(
        DeviceCollectiveRecoveryMailbox* mailbox) noexcept;

   private:
    struct StateLayout {
        static constexpr uint64_t kMinimumAlignment = 256;

        uint64_t size = 0;
        uint64_t alignment = kMinimumAlignment;
        uint64_t invocation_offset = 0;
        uint64_t signals_offset = 0;
        uint64_t protocol_state_offset = 0;

        static PGResult<StateLayout> make(
            uint32_t signal_count, uint64_t protocol_state_size,
            uint64_t protocol_state_alignment);
    };

    DeviceCollectiveResources(
        DeviceArenaSlice state_slice,
        DeviceCollectiveKernelResources device_view, uint64_t* signals,
        uint64_t signal_offset, uint32_t signal_count) noexcept;

    DeviceArenaSlice state_slice_;
    DeviceCollectiveKernelResources device_view_;
    uint64_t* signals_ = nullptr;
    uint64_t signal_offset_ = 0;
    uint32_t signal_count_ = 0;
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_DEVICE_COMM_DEVICE_COLLECTIVE_DEVICE_COLLECTIVE_RESOURCES_H
