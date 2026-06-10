#include "control_plane/agent.h"

#include <glog/logging.h>

namespace mooncake {

// =========================================================================
// Constructor
// =========================================================================

AgentStateMachine::AgentStateMachine(GlobalRank rank, int max_world_size)
    : rank_(rank), max_world_size_(max_world_size) {
    global_rank_states_.fill(RankState::OFFLINE);
    link_connected_.fill(false);
    link_status_cache_.fill(0);
    rank_state_snapshot_.fill(static_cast<uint8_t>(RankState::OFFLINE));
}

// =========================================================================
// Group lifecycle
// =========================================================================

void AgentStateMachine::registerGroup(const GroupDeclaration& declaration) {
    GroupId group_id = declaration.descriptor.group_id;
    GroupEntry entry;
    entry.descriptor = declaration.descriptor;
    entry.auto_deactivate = declaration.auto_deactivate;
    // View is populated later via handleViewUpdate.
    groups_[group_id] = std::move(entry);
}

void AgentStateMachine::unregisterGroup(GroupId group_id) {
    groups_.erase(group_id);
}

GroupView AgentStateMachine::getGroupView(GroupId group_id) const {
    auto it = groups_.find(group_id);
    if (it != groups_.end()) {
        return it->second.view;
    }
    return GroupView{};
}

const GroupDescriptor* AgentStateMachine::getGroupDescriptor(
    GroupId group_id) const {
    auto it = groups_.find(group_id);
    if (it != groups_.end()) {
        return &it->second.descriptor;
    }
    return nullptr;
}

// =========================================================================
// handlePeerJoined — new rank registered
// =========================================================================

AgentApplyResult AgentStateMachine::handlePeerJoined(
    const PeerJoinedPush& push) {
    AgentApplyResult effects;
    if (!rankInRange(push.rank)) {
        LOG(WARNING) << "AgentStateMachine: handlePeerJoined out-of-range rank "
                     << push.rank;
        return effects;
    }
    if (push.rank == rank_) return effects;

    rank_connections_[push.rank] = RankConnectionMetadata{
        .rank = push.rank,
        .te_server_name = push.te_server_name,
        .warmup_recv_addr = push.warmup_recv_addr,
    };

    effects.push_back(
        EnablePeerProbe{push.rank, push.te_server_name, push.warmup_recv_addr});
    return effects;
}

// =========================================================================
// handleRankStateUpdate — Coordinator-authoritative state change
// =========================================================================

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

// =========================================================================
// handleViewUpdate — new GroupView received
// =========================================================================

AgentApplyResult AgentStateMachine::handleViewUpdate(
    const ViewUpdatePush& push) {
    AgentApplyResult effects;
    auto it = groups_.find(push.group_id);
    if (it == groups_.end()) return effects;

    // Update descriptor (rank_order may have grown via activate) and view.
    it->second.descriptor = push.descriptor;
    it->second.view = push.view;

    effects.push_back(
        ApplyViewToBackend{push.group_id, push.descriptor, push.view});
    return effects;
}

// =========================================================================
// handleLinkStateChanged — TELinkManager link up/down event
// =========================================================================

AgentApplyResult AgentStateMachine::handleLinkStateChanged(GlobalRank peer,
                                                           bool connected) {
    AgentApplyResult effects;
    if (!rankInRange(peer)) {
        LOG(WARNING)
            << "AgentStateMachine: handleLinkStateChanged out-of-range "
            << peer;
        return effects;
    }
    link_connected_[peer] = connected;
    // NOTE: Do NOT touch link_status_cache_ here — it is reserved for
    // transfer-observation debounce.  Physical link state is reported
    // via link_connected_ in buildHeartbeat.

    syncRankStateSnapshot(effects);
    return effects;
}

// =========================================================================
// buildHeartbeat
// =========================================================================

HeartbeatRequest AgentStateMachine::buildHeartbeat() const {
    HeartbeatRequest req;
    req.rank = rank_;
    // Report physical link state, not the transfer-observation debounce cache.
    req.link_status.resize(max_world_size_, 0);
    for (int i = 0; i < max_world_size_; ++i)
        req.link_status[i] = link_connected_[i] ? 1 : 0;
    req.link_status[rank_] = 1;
    return req;
}

// =========================================================================
// applyRegisterResponse — process full state from Coordinator
// =========================================================================

AgentApplyResult AgentStateMachine::applyRegisterResponse(
    const RegisterResponse& resp) {
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
    for (int i = 0; i < max_world_size_ &&
                    i < static_cast<int>(resp.all_rank_states.size());
         ++i) {
        global_rank_states_[i] =
            static_cast<RankState>(resp.all_rank_states[i]);
    }

    // Populate group descriptors.
    for (const auto& desc : resp.group_descriptors) {
        groups_[desc.group_id].descriptor = desc;
    }

    // Populate group views.
    for (const auto& view : resp.current_views) {
        auto& group = groups_[view.group_id];
        group.view = view;
        effects.push_back(
            ApplyViewToBackend{view.group_id, group.descriptor, view});
    }

    // Populate connection metadata and trigger peer probes.
    for (const auto& conn : resp.rank_connections) {
        if (conn.rank == rank_) continue;
        rank_connections_[conn.rank] = conn;
        effects.push_back(EnablePeerProbe{conn.rank, conn.te_server_name,
                                          conn.warmup_recv_addr});
    }

    coordinator_connection_ = CoordinatorConnection::Connected;
    syncRankStateSnapshot(effects);
    return effects;
}

// =========================================================================
// prepareCleanSlateRegister — full reset before re-registration
// =========================================================================

AgentApplyResult AgentStateMachine::prepareCleanSlateRegister() {
    AgentApplyResult effects;

    rank_state_ = RankState::OFFLINE;
    global_rank_states_.fill(RankState::OFFLINE);
    link_connected_.fill(false);
    link_status_cache_.fill(0);
    rank_state_snapshot_.fill(static_cast<uint8_t>(RankState::OFFLINE));
    for (auto& conn : rank_connections_) conn.reset();

    effects.push_back(DisconnectAllLinks{});
    effects.push_back(ClearAllPeerMetadata{});
    effects.push_back(PublishRankStateSnapshot{
        std::vector<uint8_t>(rank_state_snapshot_.begin(),
                             rank_state_snapshot_.begin() + max_world_size_)});

    return effects;
}

// =========================================================================
// markOffline — Coordinator connection lost
// =========================================================================

AgentApplyResult AgentStateMachine::markOffline() {
    AgentApplyResult effects;

    rank_state_ = RankState::OFFLINE;
    coordinator_connection_ = CoordinatorConnection::Disconnected;
    global_rank_states_.fill(RankState::OFFLINE);
    link_connected_.fill(false);
    link_status_cache_.fill(0);
    rank_state_snapshot_.fill(static_cast<uint8_t>(RankState::OFFLINE));
    for (auto& conn : rank_connections_) conn.reset();

    effects.push_back(DisconnectAllLinks{});
    effects.push_back(ClearAllPeerMetadata{});
    effects.push_back(PublishRankStateSnapshot{
        std::vector<uint8_t>(rank_state_snapshot_.begin(),
                             rank_state_snapshot_.begin() + max_world_size_)});

    for (auto& [group_id, entry] : groups_) {
        effects.push_back(MarkBackendViewStale{group_id});
    }

    return effects;
}

// =========================================================================
// processTransferObservation — Debouncing logic
// =========================================================================

AgentApplyResult AgentStateMachine::processTransferObservation(
    const TransferObservationEvent& event) {
    AgentApplyResult effects;

    if (rank_state_ == RankState::OFFLINE) return effects;

    // Track which peers just transitioned from 1→0 (first failure).
    // These trigger an immediate RPC to the Coordinator.
    std::vector<GlobalRank> new_failures;

    for (int j = 0; j < max_world_size_; ++j) {
        if (static_cast<size_t>(j) >= event.attempted_ranks.size()) continue;
        if (!event.attempted_ranks[j]) continue;

        if (static_cast<size_t>(j) < event.succeeded_ranks.size() &&
            event.succeeded_ranks[j]) {
            // Success: update local cache, piggyback on next heartbeat.
            link_status_cache_[j] = 1;
        }

        if (static_cast<size_t>(j) < event.failed_ranks.size() &&
            event.failed_ranks[j]) {
            // First failure (1→0 transition) triggers immediate RPC.
            if (link_status_cache_[j] == 1) {
                new_failures.push_back(j);
            }
            link_status_cache_[j] = 0;
        }
    }

    if (!new_failures.empty()) {
        TransferObservationReport req;
        req.group_id = event.group_id;
        req.reporter_rank = rank_;
        req.attempted_ranks.assign(max_world_size_, 0);
        req.failed_ranks.assign(max_world_size_, 0);
        req.succeeded_ranks.assign(max_world_size_, 0);

        // Only include newly-failed peers (1→0 transition).
        for (GlobalRank j : new_failures) {
            req.attempted_ranks[j] = 1;
            req.failed_ranks[j] = 1;
        }

        effects.push_back(SendTransferObservation{std::move(req)});
    }

    return effects;
}

// =========================================================================
// syncRankStateSnapshot — sync state snapshot for TELinkManager
// =========================================================================

void AgentStateMachine::syncRankStateSnapshot(AgentApplyResult& effects) {
    // rank_state_ strictly tracks the Coordinator-authoritative broadcast.
    // No local HEALTHY downgrade is performed here.

    bool snapshot_changed = false;
    for (int j = 0; j < max_world_size_; ++j) {
        uint8_t new_val = static_cast<uint8_t>(global_rank_states_[j]);
        if (rank_state_snapshot_[j] != new_val) {
            rank_state_snapshot_[j] = new_val;
            snapshot_changed = true;
        }
    }

    // Update self rank_state_ to track the Coordinator-authoritative state.
    rank_state_ = global_rank_states_[rank_];

    if (snapshot_changed) {
        effects.push_back(PublishRankStateSnapshot{std::vector<uint8_t>(
            rank_state_snapshot_.begin(),
            rank_state_snapshot_.begin() + max_world_size_)});
    }
}

}  // namespace mooncake
