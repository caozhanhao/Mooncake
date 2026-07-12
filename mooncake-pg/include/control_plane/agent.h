#ifndef MOONCAKE_PG_AGENT_H
#define MOONCAKE_PG_AGENT_H

#include <array>
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
    AgentApplyResult handleLinkStateChange(GlobalRank peer, bool connected);

    HeartbeatRequest buildHeartbeat() const;

    AgentApplyResult applyRegisterAgentResponse(
        const RegisterAgentResponse& resp);
    AgentApplyResult prepareCleanSlateRegister();

    void setAgentSessionEpoch(uint64_t epoch) {
        agent_session_epoch_.store(epoch, std::memory_order_release);
    }

    AgentApplyResult processTransferObservation(
        const TransferObservationEvent& event);

    // Mark the observation data as already reported (used when piggybacking
    // on a syncAfterFailure RPC instead of sending a separate async report).
    void markObservationReported(const TransferObservationEvent& event);

    GroupView getGroupView(GroupId group_id) const;

    enum class CoordinatorConnection {
        Connected,
        AgentRegistering,
        Disconnected
    };
    CoordinatorConnection getCoordinatorConnection() const {
        return coordinator_connection_;
    }
    void setCoordinatorConnection(CoordinatorConnection state) {
        coordinator_connection_ = state;
    }

    uint64_t getAgentSessionEpoch() const {
        return agent_session_epoch_.load(std::memory_order_acquire);
    }

   private:
    GlobalRank rank_;
    int max_world_size_;

    RankState rank_state_ = RankState::OFFLINE;
    std::atomic<uint64_t> agent_session_epoch_{0};

    std::unordered_map<GroupId, GroupView> groups_;

    // Per-GlobalRank state caches.
    std::array<RankState, kMaxNumRanks> global_rank_states_{};
    std::array<bool, kMaxNumRanks> link_connected_{};
    std::array<bool, kMaxNumRanks> last_reported_peer_status_{};
    std::array<uint8_t, kMaxNumRanks> rank_state_snapshot_{};
    std::array<std::optional<RankConnectionMetadata>, kMaxNumRanks>
        rank_connections_{};

    CoordinatorConnection coordinator_connection_ =
        CoordinatorConnection::Disconnected;

    bool rankInRange(GlobalRank rank) const {
        return 0 <= rank && rank < max_world_size_;
    }

    void syncRankStateSnapshot(AgentApplyResult& effects);
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_AGENT_H
