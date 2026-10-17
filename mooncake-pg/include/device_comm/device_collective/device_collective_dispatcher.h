#ifndef MOONCAKE_PG_DEVICE_COMM_DEVICE_COLLECTIVE_DEVICE_COLLECTIVE_DISPATCHER_H
#define MOONCAKE_PG_DEVICE_COMM_DEVICE_COLLECTIVE_DEVICE_COLLECTIVE_DISPATCHER_H

#include <cstdint>
#include <memory>
#include <vector>

#include "control_plane/control_types.h"
#include "device_comm/device_collective/algorithms/algorithm_provider.h"
#include "device_comm/device_collective/device_collective_types.cuh"
#include "device_comm/device_collective/protocols/protocol_provider.h"
#include "device_comm/device_collective/resolved_group_view.h"
#include "error_types.h"

namespace mooncake {

class DeviceCollectiveWorkspace;
class DeviceTransferService;
class ControlUpdateBuilder;

// Owns the collective algorithms, their shared protocols and the installed
// group view. Selects implementations using that view's algorithm policy.
class DeviceCollectiveDispatcher {
   public:
    static PGResult<std::unique_ptr<DeviceCollectiveDispatcher>> create(
        DeviceTransferService& transfer_service,
        DeviceCollectiveWorkspace& workspace, int device_index,
        InGroupRank self_rank, uint32_t max_group_size,
        CollectiveRuntimeBindings collective);

    ~DeviceCollectiveDispatcher() noexcept;
    DeviceCollectiveDispatcher(const DeviceCollectiveDispatcher&) = delete;
    DeviceCollectiveDispatcher& operator=(const DeviceCollectiveDispatcher&) =
        delete;

    // Fill protocol and algorithm endpoints.
    void fillLocalEndpoint(DeviceGroupEndpoint& endpoint) const;

    // Install the group view and update protocol state and plans through the
    // builder. Every call produces a complete update, even for the same view.
    PGResult<void> installGroupView(ResolvedGroupView view,
                                    ControlUpdateBuilder& builder);

    // Apply the installed view again through the builder during recovery.
    PGResult<void> reinstallGroupView(ControlUpdateBuilder& builder) const;

    // Discard the installed view and mark plans unavailable through the
    // builder.
    PGResult<void> invalidateGroupView(ControlUpdateBuilder& builder);

    PGResult<const AlgorithmProvider*> selectAlgorithm(
        const DeviceCollectiveRequest& request) const;

   private:
    DeviceCollectiveDispatcher() = default;

    template <typename Protocol, typename... Args>
    Protocol* tryLoadProtocol(Args&&... args);
    template <typename Algorithm, typename... Args>
    void tryLoadAlgorithm(Args&&... args);

    PGResult<void> appendStateUpdate(const ResolvedGroupView& view,
                                     ControlUpdateBuilder& builder) const;
    PGResult<const AlgorithmProvider*> findAlgorithm(
        DeviceCollectiveAlgorithm type) const;

    // Destruction runs in reverse order: algorithms borrow protocol resources.
    std::vector<std::unique_ptr<ProtocolProvider>> protocols_;
    std::vector<std::unique_ptr<AlgorithmProvider>> algorithms_;
    ResolvedGroupView view_;
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_DEVICE_COMM_DEVICE_COLLECTIVE_DEVICE_COLLECTIVE_DISPATCHER_H
