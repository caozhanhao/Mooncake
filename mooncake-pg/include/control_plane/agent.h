#ifndef MOONCAKE_PG_AGENT_H
#define MOONCAKE_PG_AGENT_H

#include <array>
#include <cstdint>
#include <optional>
#include <unordered_map>

#include "rpc.h"

namespace mooncake {

// AgentStateMachine -- Pure state machine for the control-plane client.
class AgentStateMachine {
   public:
    AgentStateMachine(GlobalRank rank, int max_world_size);

    // ---- Group lifecycle (called from executor thread) ----

    void registerGroup(const GroupDeclaration& declaration);
    void unregisterGroup(GroupId group_id);

    // ---- Event handlers (return effects for Host execution) ----

    AgentApplyResult handlePeerJoined(const PeerJoinedPush& push);
    AgentApplyResult handleRankStateUpdate(const RankStateUpdatePush& push);
    AgentApplyResult handleViewUpdate(const ViewUpdatePush& push);
    AgentApplyResult handleLinkStateChanged(GlobalRank peer, bool connected);

    // ---- Heartbeat construction ----

    HeartbeatRequest buildHeartbeat() const;

    // ---- Registration lifecycle ----

    AgentApplyResult applyRegisterResponse(const RegisterResponse& resp);
    AgentApplyResult prepareCleanSlateRegister();
    AgentApplyResult markOffline();

    // Set the agent session epoch (called by Host after incrementing).
    void setAgentSessionEpoch(uint64_t epoch) { agent_session_epoch_ = epoch; }

    // ---- Transfer observation processing ----

    AgentApplyResult processTransferObservation(
        const TransferObservationEvent& event);

    // ---- Queries ----

    GroupView getGroupView(GroupId group_id) const;
    const GroupDescriptor* getGroupDescriptor(GroupId group_id) const;

    enum class CoordinatorConnection { Connected, Disconnected };
    CoordinatorConnection getCoordinatorConnection() const {
        return coordinator_connection_;
    }
    void setCoordinatorConnected() {
        coordinator_connection_ = CoordinatorConnection::Connected;
    }

    // ---- State access for Host ----

    RankState getSelfRankState() const { return rank_state_; }
    uint64_t getAgentSessionEpoch() const { return agent_session_epoch_; }

   private:
    GlobalRank rank_;
    int max_world_size_;

    RankState rank_state_ = RankState::OFFLINE;
    uint64_t agent_session_epoch_ = 0;

    struct GroupEntry {
        GroupDescriptor descriptor;
        GroupView view;
        bool auto_deactivate = true;
    };
    std::unordered_map<GroupId, GroupEntry> groups_;

    std::array<RankState, kMaxNumRanks> global_rank_states_{};
    std::array<bool, kMaxNumRanks> link_connected_{};
    std::array<uint8_t, kMaxNumRanks> link_status_cache_{};
    std::array<uint8_t, kMaxNumRanks> rank_state_snapshot_{};

    std::array<std::optional<RankConnectionMetadata>, kMaxNumRanks>
        rank_connections_;

    CoordinatorConnection coordinator_connection_ =
        CoordinatorConnection::Disconnected;

    // ---- Internal helpers ----

    bool rankInRange(GlobalRank rank) const {
        return rank >= 0 && rank < max_world_size_;
    }

    void syncRankStateSnapshot(AgentApplyResult& effects);
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_AGENT_H
