#include "control_plane/agent.h"

#include <algorithm>
#include <glog/logging.h>

namespace mooncake {

AgentStateMachine::AgentStateMachine(GlobalRank rank, int max_world_size)
    : rank_(rank), max_world_size_(max_world_size) {
    CHECK_GT(max_world_size_, 0);
    CHECK_LE(max_world_size_, kMaxNumRanks)
        << "max_world_size " << max_world_size_ << " exceeds kMaxNumRanks ("
        << kMaxNumRanks << ")";
    global_rank_states_.assign(max_world_size_, RankState::Offline);
    link_connected_.assign(max_world_size_, false);
    last_reported_peer_status_.assign(max_world_size_, false);
    rank_state_snapshot_.assign(max_world_size_,
                                static_cast<uint8_t>(RankState::Offline));
    rank_connections_.resize(max_world_size_);
}

void AgentStateMachine::registerGroup(const GroupView& group,
                                      bool auto_deactivate) {
    GroupView view = group;
    view.auto_deactivate = auto_deactivate;
    groups_[group.group_id] = std::move(view);
}

void AgentStateMachine::unregisterGroup(GroupId group_id) {
    groups_.erase(group_id);
}

GroupView AgentStateMachine::getGroupView(GroupId group_id) const {
    auto it = groups_.find(group_id);
    if (it != groups_.end()) {
        return it->second;
    }
    return GroupView{};
}

AgentApplyResult AgentStateMachine::handlePeerJoined(
    const PeerJoinedPush& push) {
    AgentApplyResult effects;
    if (!rankInRange(push.rank)) {
        LOG(WARNING) << "AgentStateMachine: handlePeerJoined out-of-range rank "
                     << push.rank;
        return effects;
    }
    if (push.rank == rank_) return effects;

    // If the TE server name changed (replacement process), tear down the
    // old link before probing the new one.  Without this the poller skips
    // the peer because the old Connected state lingers until the transport
    // engine detects the peer failure, which can take seconds.
    auto old = rank_connections_[push.rank];
    if (old.has_value() && old->te_server_name != push.te_server_name) {
        effects.push_back(DisconnectLink{push.rank});
    }

    rank_connections_[push.rank] = RankConnectionMetadata{
        .rank = push.rank,
        .te_server_name = push.te_server_name,
        .warmup_recv_addr = push.warmup_recv_addr,
    };
    // Seed so the first observation of any kind triggers a report.
    last_reported_peer_status_[push.rank] = true;

    effects.push_back(
        EnablePeerProbe{push.rank, push.te_server_name, push.warmup_recv_addr});
    return effects;
}

AgentApplyResult AgentStateMachine::handleRankStateUpdate(
    const RankStateUpdatePush& push) {
    AgentApplyResult effects;
    if (!rankInRange(push.rank)) {
        LOG(WARNING) << "AgentStateMachine: handleRankStateUpdate out-of-range "
                     << push.rank;
        return effects;
    }

    global_rank_states_[push.rank] = static_cast<RankState>(push.new_state);

    // Remote Offline: the control plane is dead (process may have exited).
    // Tear down TE link AND stop candidate probe.
    if (push.rank != rank_ &&
        push.new_state == static_cast<uint8_t>(RankState::Offline)) {
        effects.push_back(DisconnectLink{push.rank});
        effects.push_back(StopReconnect{push.rank});
    }

    syncRankStateSnapshot(effects);
    return effects;
}

AgentApplyResult AgentStateMachine::handleViewUpdate(
    const ViewUpdatePush& push) {
    AgentApplyResult effects;
    auto it = groups_.find(push.group_id);
    if (it == groups_.end()) {
        LOG(WARNING) << "[AGENT] handleViewUpdate group=" << push.group_id
                     << " NOT FOUND in groups_ (epoch=" << push.view.epoch
                     << ")";
        return effects;
    }

    // Detect endpoint updates.
    const auto& old_view = it->second;
    for (size_t r = 0; r < push.view.members.size(); ++r) {
        if (r == static_cast<size_t>(rank_)) continue;
        if (!push.view.members[r].isMember()) continue;
        uint64_t old_epoch = 0;
        uint64_t new_epoch = 0;
        if (old_view.members[r].hasEndpoint()) {
            old_epoch = old_view.members[r].endpoint->endpoint_epoch;
        }
        if (push.view.members[r].hasEndpoint()) {
            new_epoch = push.view.members[r].endpoint->endpoint_epoch;
        }
        if (old_epoch != 0 && new_epoch != 0 && new_epoch != old_epoch) {
            effects.push_back(RefreshPeerLink{r});
        }
    }

    // Detect rank activation transitions.
    if (!old_view.members.empty()) {
        std::vector<GlobalRank> newly_activated;
        for (size_t igr = 0; igr < push.view.rank_order.size(); ++igr) {
            GlobalRank gr = push.view.rank_order[igr];
            if (!old_view.members[gr].isActive() &&
                push.view.members[gr].isActive()) {
                newly_activated.push_back(gr);
            }
        }
        if (!newly_activated.empty()) {
            effects.push_back(NotifyRanksActivated{push.group_id,
                                                   std::move(newly_activated)});
        }
    }

    // Detect Ready transition.
    if (old_view.status != GroupStatus::Ready &&
        push.view.status == GroupStatus::Ready) {
        effects.push_back(NotifyGroupReady{push.group_id});
    }

    it->second = push.view;

    effects.push_back(ApplyViewToBackend{push.group_id, push.view});
    return effects;
}

AgentApplyResult AgentStateMachine::handleLinkStateChange(GlobalRank peer,
                                                          bool connected) {
    AgentApplyResult effects;
    if (!rankInRange(peer)) {
        LOG(WARNING) << "AgentStateMachine: handleLinkStateChange out-of-range "
                     << peer;
        return effects;
    }
    link_connected_[peer] = connected;

    if (!connected) {
        effects.push_back(NotifyTEUnreachable{peer});
    } else {
        // Link came up: re-apply current Ready views so backends refresh their
        // cached segment IDs for this peer.
        for (const auto& [group_id, view] : groups_) {
            if (view.status == GroupStatus::Ready) {
                effects.push_back(ApplyViewToBackend{group_id, view});
            }
        }
    }
    return effects;
}

HeartbeatRequest AgentStateMachine::buildHeartbeat() const {
    HeartbeatRequest req;
    req.rank = rank_;
    return req;
}

AgentApplyResult AgentStateMachine::applyRegisterAgentResponse(
    const RegisterAgentResponse& resp) {
    AgentApplyResult effects;

    if (!resp.success) {
        coordinator_connection_ = CoordinatorConnection::Disconnected;
        return effects;
    }

    // Populate global rank states.
    if (static_cast<int>(resp.all_rank_states.size()) != max_world_size_) {
        LOG(ERROR) << "AgentStateMachine: all_rank_states size mismatch (got "
                   << resp.all_rank_states.size() << ", expected "
                   << max_world_size_ << "); rejecting.";
        return effects;
    }
    for (int32_t r = 0; r < max_world_size_; ++r) {
        global_rank_states_[r] = resp.all_rank_states[r];
    }

    // Populate groups (view includes rank_order and member state).
    for (const auto& gv : resp.groups) {
        groups_[gv.group_id] = gv;
        effects.push_back(ApplyViewToBackend{gv.group_id, gv});
    }

    // Populate connection metadata and trigger peer probes.
    // Seed so the first observation of any kind triggers a report.
    for (const auto& conn : resp.rank_connections) {
        if (conn.rank == rank_) continue;
        rank_connections_[conn.rank] = conn;
        last_reported_peer_status_[conn.rank] = true;
        effects.push_back(EnablePeerProbe{conn.rank, conn.te_server_name,
                                          conn.warmup_recv_addr});
    }

    coordinator_connection_ = CoordinatorConnection::Connected;
    syncRankStateSnapshot(effects);
    return effects;
}

AgentApplyResult AgentStateMachine::prepareCleanSlateRegister() {
    AgentApplyResult effects;

    rank_state_ = RankState::Offline;
    groups_.clear();
    std::fill(global_rank_states_.begin(), global_rank_states_.end(),
              RankState::Offline);
    std::fill(link_connected_.begin(), link_connected_.end(), false);
    std::fill(last_reported_peer_status_.begin(),
              last_reported_peer_status_.end(), false);
    std::fill(rank_state_snapshot_.begin(), rank_state_snapshot_.end(),
              static_cast<uint8_t>(RankState::Offline));
    for (auto& conn : rank_connections_) conn.reset();

    effects.push_back(DisconnectAllLinks{});
    effects.push_back(ClearAllPeerMetadata{});
    PublishRankStateSnapshot snapshot;
    snapshot.states.assign(rank_state_snapshot_.begin(),
                           rank_state_snapshot_.begin() + max_world_size_);
    effects.push_back(std::move(snapshot));

    return effects;
}

std::optional<GroupObservation> AgentStateMachine::processTransferObservation(
    const TransferObservationEvent& event) {
    if (rank_state_ == RankState::Offline) return std::nullopt;

    if (event.attempted_ranks.size() != static_cast<size_t>(max_world_size_) ||
        event.succeeded_ranks.size() != static_cast<size_t>(max_world_size_) ||
        event.failed_ranks_hint.size() !=
            static_cast<size_t>(max_world_size_)) {
        LOG(WARNING)
            << "AgentStateMachine: invalid TransferObservationEvent vectors. "
            << "Expected max_world_size=" << max_world_size_ << "; dropping.";
        return std::nullopt;
    }

    GroupObservation obs;
    obs.group_id = event.group_id;
    obs.group_epoch = getGroupView(event.group_id).epoch;
    obs.attempted_ranks.assign(max_world_size_, 0);
    obs.failed_ranks_hint.assign(max_world_size_, 0);
    obs.succeeded_ranks.assign(max_world_size_, 0);

    bool has_changed = false;

    for (int peer = 0; peer < max_world_size_; ++peer) {
        if (!event.attempted_ranks[peer]) continue;
        bool current = event.succeeded_ranks[peer];
        if (current != last_reported_peer_status_[peer]) {
            last_reported_peer_status_[peer] = current;
            obs.attempted_ranks[peer] = 1;
            obs.succeeded_ranks[peer] = current ? 1 : 0;
            obs.failed_ranks_hint[peer] = current ? 0 : 1;
            has_changed = true;
        }
    }

    if (has_changed) return obs;
    return std::nullopt;
}

void AgentStateMachine::mergeObservationEvent(
    TransferObservationEvent& acc, const TransferObservationEvent& next) {
    for (int peer = 0; peer < max_world_size_; ++peer) {
        if (!next.attempted_ranks[peer]) continue;
        acc.attempted_ranks[peer] = 1;
        acc.succeeded_ranks[peer] = next.succeeded_ranks[peer];
        acc.failed_ranks_hint[peer] = next.failed_ranks_hint[peer];
    }
}

void AgentStateMachine::syncRankStateSnapshot(AgentApplyResult& effects) {
    bool snapshot_changed = false;
    for (int r = 0; r < max_world_size_; ++r) {
        uint8_t new_val = static_cast<uint8_t>(global_rank_states_[r]);
        if (rank_state_snapshot_[r] != new_val) {
            rank_state_snapshot_[r] = new_val;
            snapshot_changed = true;
        }
    }

    // Update self rank_state_ to track the Coordinator-authoritative state.
    rank_state_ = global_rank_states_[rank_];

    if (snapshot_changed) {
        PublishRankStateSnapshot snapshot;
        snapshot.states.assign(rank_state_snapshot_.begin(),
                               rank_state_snapshot_.begin() + max_world_size_);
        effects.push_back(std::move(snapshot));
    }
}

}  // namespace mooncake
