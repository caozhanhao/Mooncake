#ifndef MOONCAKE_PG_AGENT_H
#define MOONCAKE_PG_AGENT_H

#include <atomic>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>

#include "error_types.h"
#include "rpc.h"

namespace mooncake {

// AgentStateMachine - Pure state machine for the control-plane client.
class AgentStateMachine {
   public:
    AgentStateMachine(GlobalRank rank, int max_world_size);

    AgentApplyResult registerGroup(const GroupView& group);
    void unregisterGroup(GroupId group_id);

    AgentApplyResult handlePeerJoined(const PeerJoinedPush& push);
    AgentApplyResult handleRankStateUpdate(const RankStatePush& push);
    PGResult<AgentApplyResult> applyGroupView(const GroupView& view);
    PGResult<AgentApplyResult> handleViewUpdate(const ViewUpdatePush& push);
    PGResult<AgentApplyResult> handleTransferEndpointUpdate(
        uint64_t request_id, const TransferEndpointUpdatePush& push) const;

    PGResult<AgentApplyResult> applyRegistrationSnapshot(
        const ConfirmAgentRegistrationResponse& response);
    AgentApplyResult reset();

    AgentApplyResult pushLinkEvent(const LinkEvent& event);
    std::optional<LinkEventReport> getLinkEventReport() const;
    void handleLinkEventReportAck(const LinkEventReportAck& ack);

    GroupView getGroupView(GroupId group_id) const;

    enum class RegistrationPhase {
        Unregistered,
        Registering,
        Confirming,
        Registered
    };
    RegistrationPhase registrationPhase() const {
        return registration_phase_.load(std::memory_order_acquire);
    }
    void setRegistrationPhase(RegistrationPhase phase) {
        registration_phase_.store(phase, std::memory_order_release);
    }

    RankIdentity identity() const {
        return {rank_, self_rank_epoch_.load(std::memory_order_acquire)};
    }
    void setIdentity(RankIdentity identity);
    bool accepts(RankIdentity identity) const;

   private:
    GlobalRank rank_;
    int max_world_size_;

    std::atomic<uint64_t> self_rank_epoch_{0};
    std::atomic<RegistrationPhase> registration_phase_{
        RegistrationPhase::Unregistered};

    std::unordered_map<GroupId, GroupView> groups_;

    std::vector<RankState> global_rank_states_;
    std::vector<uint64_t> global_rank_epochs_;
    std::vector<uint64_t> global_rank_state_versions_;
    std::vector<std::optional<RankConnectionMetadata>> rank_connections_;

    std::vector<LinkEvent::EventType> observed_link_state_;
    std::vector<uint64_t> observed_target_rank_epochs_;
    uint64_t link_state_version_ = 0;
    uint64_t acked_link_state_version_ = 0;

    bool rankInRange(GlobalRank rank) const {
        return 0 <= rank && rank < max_world_size_;
    }

    void appendApplyGroupStateEffect(const GroupView& view,
                                     AgentApplyResult& effects) const;
    void appendApplyRankStateEffects(GlobalRank rank,
                                     AgentApplyResult& effects) const;
    void resetRankForNewEpoch(GlobalRank rank, uint64_t rank_epoch,
                              AgentApplyResult& effects);
    bool recordLinkEvent(GlobalRank peer, uint64_t target_rank_epoch,
                         LinkEvent::EventType type);
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_AGENT_H
