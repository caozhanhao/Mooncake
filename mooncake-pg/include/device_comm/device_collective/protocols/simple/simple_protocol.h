#ifndef MOONCAKE_PG_DEVICE_COMM_DEVICE_COLLECTIVE_PROTOCOLS_SIMPLE_PROTOCOL_H
#define MOONCAKE_PG_DEVICE_COMM_DEVICE_COLLECTIVE_PROTOCOLS_SIMPLE_PROTOCOL_H

#include <cstddef>
#include <cstdint>
#include <memory>

#include "control_plane/control_types.h"
#include "device_comm/device_collective/resolved_group_view.h"
#include "device_comm/device_collective/protocols/protocol_provider.h"
#include "device_comm/device_collective/protocols/simple/simple_types.cuh"
#include "device_comm/device_transfer/transfer_region.h"
#include "error_types.h"

namespace mooncake {

class ControlUpdateBuilder;
class DeviceTransferService;
class DeviceCollectiveWorkspace;

// One communicator's Simple signals and local [channel][peer] progress.
class SimpleProtocol final : public ProtocolProvider {
   public:
    static constexpr std::string_view kProtocolKey = "simple";
    static constexpr uint32_t kEndpointVersion = 1;

    static PGResult<std::unique_ptr<SimpleProtocol>> create(
        DeviceTransferService& transfer_service, InGroupRank self_rank,
        uint32_t max_group_size);

    ~SimpleProtocol() noexcept override;
    SimpleProtocol(const SimpleProtocol&) = delete;
    SimpleProtocol& operator=(const SimpleProtocol&) = delete;

    [[nodiscard]] SimpleBindings* state() const noexcept { return state_; }
    [[nodiscard]] ProtocolEndpoint localEndpoint() const override;

    PGResult<void> updateState(const ResolvedGroupView& view,
                               ControlUpdateBuilder& builder) const override;
    PGResult<void> invalidateState(
        ControlUpdateBuilder& builder) const override;

    PGResult<SimpleWorkspace> bindWorkspace(
        DeviceCollectiveWorkspace& workspace, const CollectivePeer& send_peer,
        uint64_t bytes) const;

   private:
    SimpleProtocol(RegionSlice signals,
                   SimplePipelineSignalLayout signal_layout,
                   DeviceTransferService& transfer_service,
                   InGroupRank self_rank) noexcept;

    PGResult<SimpleBindings> makeBindings(const ResolvedGroupView& view) const;
    PGResult<void> appendStateUpdate(ControlUpdateBuilder& builder,
                                     const SimpleBindings& bindings) const;

    RegionSlice signals_;
    SimplePipelineSignalLayout signal_layout_;
    SimplePeerProgress* peer_progress_ = nullptr;
    SimpleBindings* state_ = nullptr;
    DeviceTransferService& transfer_service_;
    InGroupRank self_rank_;
    int device_index_;
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_DEVICE_COMM_DEVICE_COLLECTIVE_PROTOCOLS_SIMPLE_PROTOCOL_H
