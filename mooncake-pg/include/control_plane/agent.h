#ifndef MOONCAKE_PG_AGENT_H
#define MOONCAKE_PG_AGENT_H

#include <cstdint>
#include <optional>
#include <unordered_map>

#include "rpc.h"

namespace mooncake {

// AgentStateMachine - Pure state machine for the control-plane client.
class AgentStateMachine {
   public:
    AgentStateMachine(GlobalRank rank, int max_world_size);

    void registerGroup(const GroupView& group, bool auto_deactivate);
    void unregisterGroup(GroupId group_id);

    AgentApplyResult handlePeerJoined(const PeerJoinedPush& push);
    AgentApplyResult handleRankStateUpdate(const RankStateUpdatePush& push);
    AgentApplyResult handleViewUpdate(const ViewUpdatePush& push);
    AgentApplyResult handleLinkStateChanged(GlobalRank peer, bool connected);

    HeartbeatRequest buildHeartbeat() const;

    AgentApplyResult applyRegisterResponse(const RegisterResponse& resp);
    AgentApplyResult prepareCleanSlateRegister();
    AgentApplyResult markOffline();

    void setAgentSessionEpoch(uint64_t epoch) {
        agent_session_epoch_.store(epoch, std::memory_order_release);
    }

    AgentApplyResult processTransferObservation(
        const TransferObservationEvent& event);

    GroupView getGroupView(GroupId group_id) const;

    enum class CoordinatorConnection { Connected, Registering, Disconnected };
    CoordinatorConnection getCoordinatorConnection() const {
        return coordinator_connection_;
    }
    void setCoordinatorConnected() {
        coordinator_connection_ = CoordinatorConnection::Connected;
    }
    void setCoordinatorRegistering() {
        coordinator_connection_ = CoordinatorConnection::Registering;
    }
    void setCoordinatorDisconnected() {
        coordinator_connection_ = CoordinatorConnection::Disconnected;
    }

    uint64_t getAgentSessionEpoch() const {
        return agent_session_epoch_.load(std::memory_order_acquire);
    }

   private:
    GlobalRank rank_;
    int max_world_size_;

    RankState rank_state_ = RankState::OFFLINE;
    std::atomic<uint64_t> agent_session_epoch_{0};

    struct GroupEntry {
        GroupView view;
        bool auto_deactivate = true;
    };
    std::unordered_map<GroupId, GroupEntry> groups_;

    // Per-GlobalRank state caches.
    std::vector<RankState> global_rank_states_{
        static_cast<size_t>(kMaxNumRanks)};
    std::vector<uint8_t> link_connected_{static_cast<size_t>(kMaxNumRanks)};
    std::vector<uint8_t> last_reported_peer_status_{
        static_cast<size_t>(kMaxNumRanks)};
    std::vector<uint8_t> rank_state_snapshot_{
        static_cast<size_t>(kMaxNumRanks)};
    std::vector<std::optional<RankConnectionMetadata>> rank_connections_{
        static_cast<size_t>(kMaxNumRanks)};

    CoordinatorConnection coordinator_connection_ =
        CoordinatorConnection::Disconnected;

    bool rankInRange(GlobalRank rank) const {
        return 0 <= rank && rank < max_world_size_;
    }

    void syncRankStateSnapshot(AgentApplyResult& effects);
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_AGENT_H
