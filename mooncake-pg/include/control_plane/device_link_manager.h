#ifndef MOONCAKE_PG_CONTROL_PLANE_DEVICE_LINK_MANAGER_H
#define MOONCAKE_PG_CONTROL_PLANE_DEVICE_LINK_MANAGER_H

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

#include "control_plane/control_types.h"

namespace mooncake {

struct CollectiveArenaView;
namespace device {
class P2pPeerMapping;
}

struct DeviceP2pHandle {
    std::shared_ptr<device::P2pPeerMapping> mapping;
    void* mapped_arena_base = nullptr;
};

// The indices are opaque slots owned by TE's device RDMA implementation. They
// are never used for group rank arithmetic.
struct DeviceRdmaHandle {
    std::shared_ptr<void> keepalive;
    void* qp_contexts = nullptr;
    const uint32_t* remote_keys = nullptr;
    int32_t local_peer_index = -1;
    int32_t peer_index = -1;
    int32_t qps_per_peer = 0;
    uint64_t remote_arena_address = 0;
};

// Process-level owner of device mappings and QPs. Concrete observations are
// keyed by (local device, global peer), while AgentStateMachine receives one
// process-level Device reachability contribution aggregated across devices.
// Groups retain immutable handles selected for their exact local device.
class DeviceLinkManager {
   public:
    DeviceLinkManager() = default;
    ~DeviceLinkManager() noexcept;

    void init(GlobalRank rank, int max_world_size);
    void bindCollectiveArena(const CollectiveArenaView& arena);
    void populateEndpoint(GroupEndpointV2& endpoint) const;
    void observeGroupView(const GroupView& view,
                          const std::vector<uint64_t>& rank_epochs);

    std::optional<DeviceP2pHandle> resolveP2p(DeviceId source_device,
                                              GlobalRank peer,
                                              uint64_t arena_generation) const;
    std::optional<DeviceRdmaHandle> resolveRdma(
        DeviceId source_device, GlobalRank peer,
        uint64_t arena_generation) const;

    void disconnect(GlobalRank peer);
    void clear();
    void shutdown() noexcept;

    using EventCallback = std::function<void(PeerLinkUpdate)>;
    void setEventCallback(EventCallback callback);

    DeviceLinkManager(const DeviceLinkManager&) = delete;
    DeviceLinkManager& operator=(const DeviceLinkManager&) = delete;

   private:
    struct PeerObservation {
        uint64_t p2p_target_rank_epoch = 0;
        uint64_t rdma_target_rank_epoch = 0;
        uint64_t p2p_arena_generation = 0;
        uint64_t rdma_arena_generation = 0;
        std::shared_ptr<device::P2pPeerMapping> p2p_mapping;
        uint64_t remote_rdma_arena_address = 0;
        bool rdma_available = false;
    };

    struct DeviceArenaState;

    bool rankInRange(GlobalRank peer) const {
        return 0 <= peer && peer < max_world_size_;
    }
    void connectRdmaPeers(DeviceArenaState& arena,
                          std::vector<PeerObservation>& observations,
                          const GroupView& view,
                          const std::vector<uint64_t>& rank_epochs,
                          std::vector<GlobalRank>& changed_peers);
    PeerLinkUpdate makeLinkUpdate(GlobalRank peer) const;
    void emit(PeerLinkUpdate update);

    GlobalRank rank_ = kInvalidGlobalRank;
    int max_world_size_ = 0;
    std::unordered_map<DeviceId, std::vector<PeerObservation>> device_peers_;
    std::vector<uint64_t> target_rank_epochs_;
    mutable std::mutex peers_mutex_;
    std::unordered_map<DeviceId, std::shared_ptr<DeviceArenaState>> arenas_;

    EventCallback event_callback_;
    mutable std::mutex event_callback_mutex_;
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_CONTROL_PLANE_DEVICE_LINK_MANAGER_H
