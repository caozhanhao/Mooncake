#include "control_plane/agent.h"

#include <algorithm>
#include <glog/logging.h>

namespace mooncake {

// Constructor

AgentStateMachine::AgentStateMachine(GlobalRank rank, int max_world_size)
    : rank_(rank), max_world_size_(max_world_size) {
    CHECK_GT(max_world_size_, 0);
    CHECK_LE(max_world_size_, kMaxNumRanks)
        << "max_world_size " << max_world_size_ << " exceeds kMaxNumRanks ("
        << kMaxNumRanks << ")";
    global_rank_states_.fill(RankState::OFFLINE);
    link_connected_.fill(false);
    last_reported_peer_status_.fill(false);
    rank_state_snapshot_.fill(static_cast<uint8_t>(RankState::OFFLINE));
}

// Group lifecycle

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
    // the peer because the old CONNECTED state lingers until the transport
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

// handleRankStateUpdate  - Coordinator-authoritative state change

AgentApplyResult AgentStateMachine::handleRankStateUpdate(
    const RankStateUpdatePush& push) {
    AgentApplyResult effects;
    if (!rankInRange(push.rank)) {
        LOG(WARNING) << "AgentStateMachine: handleRankStateUpdate out-of-range "
                     << push.rank;
        return effects;
    }

    global_rank_states_[push.rank] = static_cast<RankState>(push.new_state);

    // Remote OFFLINE: the control plane is dead (process may have exited).
    // Tear down TE link AND stop candidate probe.
    if (push.rank != rank_ &&
        push.new_state == static_cast<uint8_t>(RankState::OFFLINE)) {
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

    LOG(INFO) << "[AGENT] handleViewUpdate rank=" << rank_
              << " group=" << push.group_id << " epoch=" << push.view.epoch
              << " #effects=1";

    // Detect coordinator-assigned endpoint_epoch changes.  A change means the
    // remote endpoint has been republished (e.g. replacement process) and the
    // LinkManager must drop the old link and re-probe with the new server info.
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
            const auto& conn = rank_connections_[r];
            if (conn.has_value()) {
                effects.push_back(DisconnectLink{r});
                effects.push_back(EnablePeerProbe{r, conn->te_server_name,
                                                  conn->warmup_recv_addr});
                LOG(INFO) << "[AGENT] endpoint_epoch changed rank=" << rank_
                          << " peer=" << r << " old_epoch=" << old_epoch
                          << " new_epoch=" << new_epoch;
            }
        }
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
        // cached segment IDs for this peer.  This is required for link recovery
        // (the view stays Ready while the TE link is rebuilt) and covers any
        // edge case where a Ready view was applied before the local TE link
        // finished establishing.
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
        LOG(WARNING) << "AgentStateMachine: all_rank_states size mismatch (got "
                     << resp.all_rank_states.size() << ", expected "
                     << max_world_size_ << "); truncating.";
    }
    int available_states = static_cast<int>(resp.all_rank_states.size());
    for (int32_t r = 0; r < std::min(max_world_size_, available_states); ++r) {
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

    rank_state_ = RankState::OFFLINE;
    global_rank_states_.fill(RankState::OFFLINE);
    link_connected_.fill(false);
    last_reported_peer_status_.fill(false);
    rank_state_snapshot_.fill(static_cast<uint8_t>(RankState::OFFLINE));
    for (auto& conn : rank_connections_) conn.reset();

    effects.push_back(DisconnectAllLinks{});
    effects.push_back(ClearAllPeerMetadata{});
    PublishRankStateSnapshot snapshot;
    snapshot.states.assign(rank_state_snapshot_.begin(),
                           rank_state_snapshot_.begin() + max_world_size_);
    effects.push_back(std::move(snapshot));

    return effects;
}

// processTransferObservation  - report when observation differs from last
//
// Input bit-vectors are indexed by GlobalRank (producers translate through
// rank_order before reporting).  We compare each attempted peer against
// last_reported_peer_status_ and emit a TransferObservationReport when
// anything changed.
AgentApplyResult AgentStateMachine::processTransferObservation(
    const TransferObservationEvent& event) {
    AgentApplyResult effects;

    if (rank_state_ == RankState::OFFLINE) return effects;

    TransferObservationReport req;
    req.group_id = event.group_id;
    req.reporter_rank = rank_;

    bool has_changed = false;

    // Bit-vectors are GlobalRank-indexed.  Iterate over the input size and
    // ignore entries beyond our current max_world_size_.
    for (size_t peer = 0; peer < event.attempted_ranks.size(); ++peer) {
        if (peer >= max_world_size_) continue;
        if (!event.attempted_ranks[peer]) continue;

        bool succeeded = peer < event.succeeded_ranks.size()
                             ? event.succeeded_ranks[peer]
                             : 0;
        bool failed = peer < event.failed_ranks_hint.size()
                          ? event.failed_ranks_hint[peer]
                          : 0;

        // Determine current observation: failed takes precedence.
        bool current = succeeded && !failed;

        if (current != last_reported_peer_status_[peer]) {
            last_reported_peer_status_[peer] = current;
            // Lazy-init the vectors on first change.
            if (!has_changed) {
                req.attempted_ranks.assign(max_world_size_, 0);
                req.failed_ranks_hint.assign(max_world_size_, 0);
                req.succeeded_ranks.assign(max_world_size_, 0);
            }
            req.attempted_ranks[peer] = 1;
            req.succeeded_ranks[peer] = current ? 1 : 0;
            req.failed_ranks_hint[peer] = current ? 0 : 1;
            has_changed = true;
        }
    }

    if (has_changed) {
        effects.push_back(SendTransferObservation{std::move(req)});
        LOG(INFO) << "[AGENT] processTransferObservation has_changed rank="
                  << rank_ << " attempted=";
        for (size_t peer = 0; peer < req.attempted_ranks.size(); ++peer) {
            if (req.attempted_ranks[peer]) {
                LOG(INFO) << "[AGENT]   peer=" << peer
                          << " succeeded=" << (int)req.succeeded_ranks[peer]
                          << " failed=" << (int)req.failed_ranks_hint[peer];
            }
        }
    } else {
        LOG(INFO) << "[AGENT] processTransferObservation no_change rank="
                  << rank_;
    }

    return effects;
}

void AgentStateMachine::markObservationReported(
    const TransferObservationEvent& event) {
    for (size_t peer = 0; peer < event.attempted_ranks.size(); ++peer) {
        if (peer >= max_world_size_) continue;
        if (!event.attempted_ranks[peer]) continue;

        bool succeeded = peer < event.succeeded_ranks.size()
                             ? event.succeeded_ranks[peer]
                             : 0;
        bool failed = peer < event.failed_ranks_hint.size()
                          ? event.failed_ranks_hint[peer]
                          : 0;
        bool current = succeeded && !failed;
        last_reported_peer_status_[peer] = current;
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
