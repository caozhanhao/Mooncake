#include "control_plane/coordinator.h"

#include <algorithm>
#include <bit>
#include <numeric>
#include <set>

#include <glog/logging.h>

namespace mooncake {

// =========================================================================
// Constructor
// =========================================================================

CentralizedCoordinatorStateMachine::CentralizedCoordinatorStateMachine(
    int max_world_size)
    : max_world_size_(max_world_size) {
    CHECK_GT(max_world_size_, 0);
    CHECK_LE(max_world_size_, kMaxNumRanks)
        << "max_world_size " << max_world_size_ << " exceeds kMaxNumRanks ("
        << kMaxNumRanks << ")";
    for (int i = 0; i < kMaxNumRanks; ++i) {
        ranks_[i].link_status.assign(max_world_size_, 0);
        ranks_[i].link_status[i] = 1;
    }
}

// =========================================================================
// §7.5  registerAgent
// =========================================================================

CoordinatorApplyResult<RegisterResponse>
CentralizedCoordinatorStateMachine::handleRegister(const RegisterRequest& req) {
    CoordinatorApplyResult<RegisterResponse> result;
    if (!rankInValidRange(req.rank)) {
        result.response.success = false;
        result.response.error_msg = "rank out of valid range";
        return result;
    }
    auto& info = ranks_[req.rank];

    // --- Identity check ---
    // If the rank already has an active registration AND the request comes
    // from a different process → reject.  Replacement must wait for the old
    // process to be heartbeat-timed-out to OFFLINE.
    bool same_peer = (info.agent_addr == req.agent_addr &&
                      info.te_server_name == req.te_server_name);

    if (info.state != RankState::OFFLINE && !same_peer) {
        result.response.success = false;
        result.response.error_msg =
            "rank already registered, and is not offline.";
        return result;
    }

    // A new session makes old endpoints invalid.
    // We do NOT reset endpoint_epoch (it stays monotonically increasing).
    // Validity is checked via (agent_session_epoch match, endpoint_epoch !=
    // kInvalidEpoch) in isRankActivatable().

    info.agent_addr = req.agent_addr;
    info.te_server_name = req.te_server_name;
    info.agent_session_epoch = req.agent_session_epoch;
    info.warmup_recv_addr = req.warmup_recv_addr;
    info.last_heartbeat = std::chrono::steady_clock::now();
    info.link_status.assign(max_world_size_, 0);
    info.link_status[req.rank] = 1;

    result.effects.push_back(makePeerJoinedEffect(req.rank));
    transitionToSynced(req.rank, result.effects);
    result.response = buildRegisterResponse(req.rank);
    result.response.success = true;
    return result;
}

// =========================================================================
// §7.5  heartbeat
// =========================================================================

CoordinatorApplyResult<HeartbeatResponse>
CentralizedCoordinatorStateMachine::handleHeartbeat(
    const HeartbeatRequest& req) {
    CoordinatorApplyResult<HeartbeatResponse> result;
    if (!rankInValidRange(req.rank)) {
        result.response.acknowledge = false;
        result.response.require_reregister = true;
        return result;
    }
    auto& info = ranks_[req.rank];

    if (info.state == RankState::OFFLINE ||
        info.agent_session_epoch != req.agent_session_epoch) {
        result.response.acknowledge = false;
        result.response.require_reregister = true;
        return result;
    }

    info.last_heartbeat = std::chrono::steady_clock::now();
    info.link_status = req.link_status;
    if (info.link_status.size() < static_cast<size_t>(max_world_size_)) {
        info.link_status.resize(max_world_size_, 0);
    }
    info.link_status[req.rank] = 1;

    result.response.acknowledge = true;
    result.response.require_reregister = false;
    updateRankHealth(result.effects);
    return result;
}

// =========================================================================
// §7.5  declareGroup
// =========================================================================

CoordinatorApplyResult<DeclareGroupResponse>
CentralizedCoordinatorStateMachine::handleDeclareGroup(
    const DeclareGroupRequest& req) {
    CoordinatorApplyResult<DeclareGroupResponse> result;
    if (!rankInValidRange(req.rank)) {
        result.response.success = false;
        result.response.reject_reason = "rank out of valid range";
        return result;
    }
    auto& info = ranks_[req.rank];

    if (info.state == RankState::OFFLINE ||
        info.agent_session_epoch != req.agent_session_epoch) {
        result.response.success = false;
        result.response.reject_reason = "stale agent session epoch";
        return result;
    }

    bool ok = declareGroup(req.group, result.response);
    if (!ok) return result;

    result.response.success = true;
    return result;
}

// =========================================================================
// §7.7  handleViewUpdate (activate / deactivate)
// =========================================================================

CoordinatorApplyResult<void>
CentralizedCoordinatorStateMachine::handleViewUpdate(
    uint64_t propose_id, const ProposeViewUpdateRequest& req) {
    CoordinatorApplyResult<void> result;
    auto it = group_views_.find(req.group_id);
    if (it == group_views_.end()) {
        result.effects.push_back(ReplyViewUpdateEffect{
            propose_id,
            {ViewUpdateStatus::Rejected, 0, {}, "group not found"}});
        return result;
    }

    GroupView& view = it->second;
    GroupView old_view = view;
    bool changed = false;

    // Reject stale or offline proposer.
    if (ranks_[req.source_rank].state == RankState::OFFLINE ||
        ranks_[req.source_rank].agent_session_epoch !=
            req.agent_session_epoch) {
        result.effects.push_back(ReplyViewUpdateEffect{
            propose_id,
            {ViewUpdateStatus::Rejected,
             view.epoch,
             {},
             "source rank is OFFLINE or stale session epoch"}});
        return result;
    }

    // Validate all targets are within valid GlobalRank range.
    for (GlobalRank rank : req.requested_ranks) {
        if (!rankInValidRange(rank)) {
            result.effects.push_back(
                ReplyViewUpdateEffect{propose_id,
                                      {ViewUpdateStatus::Rejected,
                                       view.epoch,
                                       {},
                                       "target rank is out of valid range"}});
            return result;
        }
    }

    if (req.is_activate) {
        // group_descriptors_ is populated synchronously with group_views_ in
        // declareGroup; both should always exist together.
        auto desc_it = group_descriptors_.find(req.group_id);
        if (desc_it == group_descriptors_.end()) {
            result.effects.push_back(ReplyViewUpdateEffect{
                propose_id,
                {ViewUpdateStatus::Rejected,
                 view.epoch,
                 {},
                 "group descriptor not found (internal error)"}});
            return result;
        }
        auto& rank_order = desc_it->second.rank_order;

        // Append new ranks to rank_order.
        for (GlobalRank rank : req.requested_ranks) {
            if (std::find(rank_order.begin(), rank_order.end(), rank) ==
                rank_order.end()) {
                rank_order.push_back(rank);
            }
        }

        // Check activatable before applying changes.
        for (GlobalRank rank : req.requested_ranks) {
            if (!view.members[rank].active) changed = true;
        }

        if (changed &&
            !isActivatableSet(req.group_id, req.requested_ranks, view)) {
            result.effects.push_back(
                ReplyViewUpdateEffect{propose_id,
                                      {ViewUpdateStatus::Rejected,
                                       view.epoch,
                                       {},
                                       "new active set is not activatable"}});
            return result;
        }

        // Apply.
        for (GlobalRank rank : req.requested_ranks) {
            view.members[rank].active = true;
        }
    } else {
        // deactivate
        for (GlobalRank rank : req.requested_ranks) {
            if (view.members[rank].active) {
                view.members[rank].active = false;
                changed = true;
            }
        }
        // rank_order is NOT changed on deactivate ($2.3).
    }

    if (!changed) {
        result.effects.push_back(ReplyViewUpdateEffect{
            propose_id, {ViewUpdateStatus::Applied, view.epoch, {}, ""}});
        return result;
    }

    view.epoch++;

    auto required_acks =
        computeRequiredViewAcks(old_view, view, req.source_rank);

    if (required_acks.empty()) {
        // Best-effort: broadcast + reply immediately.
        result.effects.push_back(
            ViewUpdateEffect{group_descriptors_[req.group_id], view, {}});
        result.effects.push_back(ReplyViewUpdateEffect{
            propose_id, {ViewUpdateStatus::Applied, view.epoch, {}, ""}});
    } else {
        // Strict Barrier: store pending, let Host broadcast, wait for ACKs.
        pending_proposes_[propose_id] = PendingPropose{
            propose_id,
            req.group_id,
            {ViewUpdateStatus::Applied, view.epoch, {}, ""},
            std::unordered_set<GlobalRank>(required_acks.begin(),
                                           required_acks.end()),
            std::chrono::steady_clock::now() + kProposeTimeout,
        };
        result.effects.push_back(ViewUpdateEffect{
            group_descriptors_[req.group_id], view, required_acks});
    }

    return result;
}

// =========================================================================
// §7.7b  handleViewUpdateAck
// =========================================================================

CoordinatorApplyResult<void>
CentralizedCoordinatorStateMachine::handleViewUpdateAck(uint64_t propose_id,
                                                        GlobalRank rank,
                                                        uint64_t epoch,
                                                        bool applied) {
    CoordinatorApplyResult<void> result;
    auto it = pending_proposes_.find(propose_id);
    if (it == pending_proposes_.end()) return result;  // already resolved

    if (!applied || epoch != it->second.eventual_response.new_epoch)
        return result;

    it->second.waiting_acks.erase(rank);
    if (it->second.waiting_acks.empty()) {
        result.effects.push_back(
            ReplyViewUpdateEffect{propose_id, it->second.eventual_response});
        pending_proposes_.erase(it);
    }
    return result;
}

// =========================================================================
// §7.7c  checkTimeouts
// =========================================================================

CoordinatorApplyResult<void>
CentralizedCoordinatorStateMachine::checkTimeouts() {
    CoordinatorApplyResult<void> result;
    auto now = std::chrono::steady_clock::now();

    // --- Heartbeat timeout ---
    for (int rank = 0; rank < max_world_size_; ++rank) {
        auto& info = ranks_[rank];
        if (info.state == RankState::OFFLINE) continue;
        if (now - info.last_heartbeat > kHeartbeatTimeout) {
            transitionToOffline(rank, result.effects);
        }
    }

    // --- Propose ACK timeout (Strict Barrier & Prune) ---
    // NOTE: transitionToOffline below does NOT modify pending_proposes_,
    // so the iterator remains valid across the call.
    for (auto it = pending_proposes_.begin(); it != pending_proposes_.end();) {
        if (now > it->second.deadline) {
            auto& pending = it->second;
            pending.eventual_response.status =
                ViewUpdateStatus::AppliedWithDroppedRanks;
            for (GlobalRank rank : pending.waiting_acks) {
                transitionToOffline(rank, result.effects);
                pending.eventual_response.dropped_ranks.push_back(rank);
            }
            result.effects.push_back(ReplyViewUpdateEffect{
                pending.propose_id, pending.eventual_response});
            it = pending_proposes_.erase(it);
        } else {
            ++it;
        }
    }

    return result;
}

// =========================================================================
// §7.8  publishEndpoint
// =========================================================================

CoordinatorApplyResult<PublishEndpointResponse>
CentralizedCoordinatorStateMachine::handlePublishEndpoint(
    const PublishEndpointRequest& req) {
    CoordinatorApplyResult<PublishEndpointResponse> result;
    if (!rankInValidRange(req.rank)) {
        result.response.success = false;
        result.response.reject_reason = "rank out of valid range";
        return result;
    }
    auto& info = ranks_[req.rank];

    if (info.state == RankState::OFFLINE ||
        info.agent_session_epoch != req.agent_session_epoch) {
        result.response.success = false;
        result.response.reject_reason = "stale agent session epoch";
        return result;
    }

    for (const auto& ep : req.endpoints) {
        auto it = group_views_.find(ep.group_id);
        if (it == group_views_.end()) {
            result.response.success = false;
            result.response.reject_reason = "group not found";
            return result;
        }

        auto& member = it->second.members[req.rank];
        if (member.active) {
            // Rank is already active → accept new endpoint, push best-effort
            // view update.  (e.g., replacement rank refreshed its endpoint.)
            member.agent_session_epoch = req.agent_session_epoch;
            member.endpoint_epoch = ep.endpoint_epoch;
            member.endpoint_info = ep.endpoint_info;
            it->second.epoch++;
            result.effects.push_back(ViewUpdateEffect{
                group_descriptors_[ep.group_id], it->second, {}});
        } else {
            // Inactive → store locally only.  No broadcast.
            member.agent_session_epoch = req.agent_session_epoch;
            member.endpoint_epoch = ep.endpoint_epoch;
            member.endpoint_info = ep.endpoint_info;
        }
    }

    result.response.success = true;
    return result;
}

// =========================================================================
// §7.9  transferObservation
// =========================================================================

CoordinatorApplyResult<void>
CentralizedCoordinatorStateMachine::handleTransferObservation(
    const TransferObservationReport& req) {
    CoordinatorApplyResult<void> result;
    if (!rankInValidRange(req.reporter_rank)) return result;
    auto& reporter = ranks_[req.reporter_rank];

    if (reporter.agent_session_epoch != req.agent_session_epoch) {
        return result;  // stale reporter
    }

    if (reporter.link_status.size() != static_cast<size_t>(max_world_size_)) {
        reporter.link_status.resize(max_world_size_, 0);
    }

    auto bitAt = [](const std::vector<uint8_t>& bits, int idx) -> bool {
        return idx >= 0 && static_cast<size_t>(idx) < bits.size() &&
               bits[idx] != 0;
    };

    for (int j = 0; j < max_world_size_; ++j) {
        if (!bitAt(req.attempted_ranks, j)) continue;
        if (bitAt(req.succeeded_ranks, j)) reporter.link_status[j] = 1;
        if (bitAt(req.failed_ranks, j)) reporter.link_status[j] = 0;
    }

    // Recompute authoritative health (updateRankHealth now also handles
    // auto_deactivate for all groups, not just the reporting group).
    updateRankHealth(result.effects);

    return result;
}

// =========================================================================
// Private: state transitions
// =========================================================================

void CentralizedCoordinatorStateMachine::transitionToOffline(
    GlobalRank rank, std::vector<CoordinatorEffect>& effects) {
    ranks_[rank].state = RankState::OFFLINE;
    ranks_[rank].link_status.assign(max_world_size_, 0);

    // Clear this rank's connectivity from all peers.
    for (auto& peer : ranks_) {
        if (peer.link_status.size() > static_cast<size_t>(rank))
            peer.link_status[rank] = 0;
    }

    // Per-group endpoint validity is determined by agent_session_epoch in
    // isRankActivatable().  Old endpoints become invalid automatically when
    // the Agent re-registers with a new session epoch — no explicit clearing
    // needed.

    effects.push_back(makeRankStateEffect(rank));

    // DO NOT modify active_ranks here.  Membership changes come from:
    //   - auto_deactivate=true → handleTransferObservation
    //   - Explicit deactivate_rank / disconnect
    //   - Strict Barrier & Prune → AppliedWithDroppedRanks

    updateRankHealth(effects);
}

void CentralizedCoordinatorStateMachine::transitionToSynced(
    GlobalRank rank, std::vector<CoordinatorEffect>& effects) {
    ranks_[rank].state = RankState::SYNCED;
    effects.push_back(makeRankStateEffect(rank));
    // Do NOT recompute healthy set here — a newly-SYNCED rank has
    // link_status_ all zeros, so it cannot affect the healthy set.
}

// =========================================================================
// Private: authoritative HEALTHY computation
// =========================================================================

bool CentralizedCoordinatorStateMachine::isMutuallyConnected(
    GlobalRank a, GlobalRank b) const {
    if (a == b) return true;
    if (ranks_[a].state == RankState::OFFLINE ||
        ranks_[b].state == RankState::OFFLINE)
        return false;
    return ranks_[a].link_status.size() > static_cast<size_t>(b) &&
           ranks_[b].link_status.size() > static_cast<size_t>(a) &&
           ranks_[a].link_status[b] != 0 && ranks_[b].link_status[a] != 0;
}

std::vector<GlobalRank> CentralizedCoordinatorStateMachine::findHealthySet()
    const {
    std::vector<GlobalRank> candidates;
    for (int i = 0; i < max_world_size_; ++i) {
        if (ranks_[i].state != RankState::OFFLINE) {
            candidates.push_back(i);
        }
    }

    int n = static_cast<int>(candidates.size());
    if (n == 0) return {};

    std::vector<std::vector<uint8_t>> adj(n, std::vector<uint8_t>(n, 0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (isMutuallyConnected(candidates[i], candidates[j])) {
                adj[i][j] = 1;
            }
        }
    }

    std::vector<GlobalRank> best;
    std::vector<GlobalRank> current;

    auto isBetter = [&](const std::vector<GlobalRank>& a,
                        const std::vector<GlobalRank>& b) -> bool {
        if (a.size() != b.size()) return a.size() > b.size();
        bool a_has_0 = std::find(a.begin(), a.end(), 0) != a.end();
        bool b_has_0 = std::find(b.begin(), b.end(), 0) != b.end();
        if (a_has_0 != b_has_0) return a_has_0;
        return std::lexicographical_compare(a.begin(), a.end(), b.begin(),
                                            b.end());
    };

    static constexpr int kMaxSearchSteps = 100000;
    int search_steps = 0;

    std::function<void(const std::vector<int>&)> search =
        [&](const std::vector<int>& allowed) {
            if (++search_steps > kMaxSearchSteps) return;

            if (current.size() + allowed.size() <= best.size()) return;

            for (size_t i = 0; i < allowed.size(); ++i) {
                int v = allowed[i];

                current.push_back(candidates[v]);

                std::vector<int> next_allowed;
                next_allowed.reserve(allowed.size() - i - 1);

                for (size_t j = i + 1; j < allowed.size(); ++j) {
                    int u = allowed[j];
                    if (adj[v][u]) {
                        next_allowed.push_back(u);
                    }
                }

                search(next_allowed);

                if (isBetter(current, best)) {
                    best = current;
                }
                current.pop_back();
            }
        };

    std::vector<int> initial_allowed(n);
    std::iota(initial_allowed.begin(), initial_allowed.end(), 0);

    search(initial_allowed);

    if (best.empty() && !candidates.empty()) {
        best.push_back(candidates[0]);
    }

    return best;
}

void CentralizedCoordinatorStateMachine::updateRankHealth(
    std::vector<CoordinatorEffect>& effects) {
    auto healthy_set = findHealthySet();

    // 1. Update per-rank HEALTHY / SYNCED state.
    for (int i = 0; i < max_world_size_; ++i) {
        if (ranks_[i].state == RankState::OFFLINE) continue;

        bool in_healthy = std::find(healthy_set.begin(), healthy_set.end(),
                                    i) != healthy_set.end();

        if (in_healthy && ranks_[i].state != RankState::HEALTHY) {
            ranks_[i].state = RankState::HEALTHY;
            effects.push_back(makeRankStateEffect(i));
        } else if (!in_healthy && ranks_[i].state == RankState::HEALTHY) {
            ranks_[i].state = RankState::SYNCED;
            effects.push_back(makeRankStateEffect(i));
        }
    }

    // 2. For auto_deactivate groups, remove unhealthy ranks from the active
    //    set.  This subsumes the per-group deactivation that was previously in
    //    handleTransferObservation, and also fires on heartbeat-timeout →
    //    transitionToOffline → updateRankHealth, fixing the bug where LinkDown
    //    could suppress transfer-observation reports and leave a dead rank
    //    stuck as active=true forever.
    for (auto& [group_id, view] : group_views_) {
        if (!group_auto_deactivate_[group_id]) continue;
        bool changed = false;
        for (int i = 0; i < max_world_size_; ++i) {
            if (!view.members[i].active) continue;
            bool in_healthy = std::find(healthy_set.begin(), healthy_set.end(),
                                        i) != healthy_set.end();
            if (!in_healthy) {
                view.members[i].active = false;
                changed = true;
            }
        }
        if (changed) {
            view.epoch++;
            effects.push_back(
                ViewUpdateEffect{group_descriptors_[group_id], view, {}});
        }
    }
}

// =========================================================================
// Private: activatable predicates
// =========================================================================

bool CentralizedCoordinatorStateMachine::rankInValidRange(
    GlobalRank rank) const {
    return rank >= 0 && rank < max_world_size_;
}

bool CentralizedCoordinatorStateMachine::isRankActivatable(
    GroupId group_id, GlobalRank rank) const {
    if (!rankInValidRange(rank)) return false;
    if (ranks_[rank].state != RankState::HEALTHY) return false;

    auto it = group_views_.find(group_id);
    if (it == group_views_.end()) return false;

    const auto& member = it->second.members[rank];
    // endpoint_epoch is monotonically increasing and never reset.
    // Validity requires the Agent session epoch to match the current session.
    return member.endpoint_epoch != kInvalidEpoch &&
           member.agent_session_epoch == ranks_[rank].agent_session_epoch;
}

bool CentralizedCoordinatorStateMachine::isActivatableSet(
    GroupId group_id, const std::vector<GlobalRank>& new_ranks,
    const GroupView& old_view) const {
    // Check all currently-active ranks + new ranks.
    // Degraded groups (with non-activatable active members) will fail.
    for (int i = 0; i < max_world_size_; ++i) {
        if (!old_view.members[i].active) continue;
        if (!isRankActivatable(group_id, i)) return false;
    }
    for (GlobalRank r : new_ranks) {
        if (old_view.members[r].active) continue;  // already checked above
        if (!isRankActivatable(group_id, r)) return false;
    }
    return true;
}

// =========================================================================
// Private: declareGroup
// =========================================================================

bool CentralizedCoordinatorStateMachine::declareGroup(
    const GroupDeclaration& declaration, DeclareGroupResponse& response) {
    GroupId group_id = declaration.descriptor.group_id;

    // Validate rank_order elements.
    for (GlobalRank r : declaration.descriptor.rank_order) {
        if (!rankInValidRange(r)) {
            response.success = false;
            response.reject_reason = "rank_order contains invalid GlobalRank";
            return false;
        }
    }

    // Validate no duplicates in rank_order.
    {
        std::set<GlobalRank> seen(declaration.descriptor.rank_order.begin(),
                                  declaration.descriptor.rank_order.end());
        if (seen.size() != declaration.descriptor.rank_order.size()) {
            response.success = false;
            response.reject_reason = "rank_order contains duplicate ranks";
            return false;
        }
    }

    // Validate initial_view.members size.
    const auto& initial = declaration.initial_view;
    if (initial.members.size() != static_cast<size_t>(max_world_size_)) {
        response.success = false;
        response.reject_reason = "initial_view.members size != max_world_size";
        return false;
    }

    // Active ranks must appear in rank_order.
    for (int i = 0; i < max_world_size_; ++i) {
        if (initial.members[i].active) {
            if (std::find(declaration.descriptor.rank_order.begin(),
                          declaration.descriptor.rank_order.end(),
                          i) == declaration.descriptor.rank_order.end()) {
                response.success = false;
                response.reject_reason =
                    "initial active rank not in rank_order";
                return false;
            }
        }
    }

    auto it = group_views_.find(group_id);
    if (it == group_views_.end()) {
        // First declaration → create group.
        group_descriptors_[group_id] = declaration.descriptor;
        group_views_[group_id] = declaration.initial_view;
        group_auto_deactivate_[group_id] = declaration.auto_deactivate;
        response.current_view = declaration.initial_view;
        response.success = true;
        return true;
    }

    // Group already exists → validate rank_order prefix.
    const auto& existing_order = group_descriptors_[group_id].rank_order;
    const auto& new_order = declaration.descriptor.rank_order;

    if (new_order.size() > existing_order.size()) {
        response.success = false;
        response.reject_reason =
            "new rank_order is longer than existing (must be a prefix)";
        return false;
    }

    for (size_t i = 0; i < new_order.size(); ++i) {
        if (new_order[i] != existing_order[i]) {
            response.success = false;
            response.reject_reason =
                "new rank_order is not a prefix of existing rank_order";
            return false;
        }
    }

    // Return current authoritative view.
    response.current_view = it->second;
    response.success = true;
    return true;
}

// =========================================================================
// Private: helpers
// =========================================================================

std::vector<GlobalRank>
CentralizedCoordinatorStateMachine::computeRequiredViewAcks(
    const GroupView& old_view, const GroupView& new_view,
    GlobalRank proposer) const {
    // All online ranks that are active in EITHER old or new view must ACK.
    // This ensures newly-activated ranks have received the view, and
    // deactivated ranks know they're removed.
    std::set<GlobalRank> required;
    for (int i = 0; i < max_world_size_; ++i) {
        if (i == proposer) continue;
        if (ranks_[i].state == RankState::OFFLINE) continue;
        if (old_view.members[i].active || new_view.members[i].active) {
            required.insert(i);
        }
    }

    return std::vector<GlobalRank>(required.begin(), required.end());
}

RegisterResponse CentralizedCoordinatorStateMachine::buildRegisterResponse(
    GlobalRank for_rank) const {
    RegisterResponse resp;
    resp.success = true;

    // All rank states.
    resp.all_rank_states.resize(max_world_size_);
    for (int i = 0; i < max_world_size_; ++i) {
        resp.all_rank_states[i] = static_cast<uint8_t>(ranks_[i].state);
    }

    // All group descriptors.
    for (const auto& [gid, desc] : group_descriptors_) {
        resp.group_descriptors.push_back(desc);
    }

    // All current views.
    for (const auto& [gid, view] : group_views_) {
        resp.current_views.push_back(view);
    }

    // All rank connection metadata (for TELinkManager).
    for (int i = 0; i < max_world_size_; ++i) {
        if (i == for_rank) continue;
        if (ranks_[i].state == RankState::OFFLINE) continue;

        RankConnectionMetadata conn;
        conn.rank = i;
        conn.agent_session_epoch = ranks_[i].agent_session_epoch;
        conn.agent_addr = ranks_[i].agent_addr;
        conn.te_server_name = ranks_[i].te_server_name;
        conn.warmup_recv_addr = ranks_[i].warmup_recv_addr;
        resp.rank_connections.push_back(conn);
    }

    return resp;
}

// =========================================================================
// Effect factories
// =========================================================================

CoordinatorEffect CentralizedCoordinatorStateMachine::makeRankStateEffect(
    GlobalRank rank) {
    return PushEffect<RankStateUpdatePush>{
        EffectTarget{EffectTarget::Kind::BroadcastOnline},
        RankStateUpdatePush{rank, static_cast<uint8_t>(ranks_[rank].state)}};
}

CoordinatorEffect CentralizedCoordinatorStateMachine::makePeerJoinedEffect(
    GlobalRank rank) {
    return PushEffect<PeerJoinedPush>{
        EffectTarget{EffectTarget::Kind::BroadcastOnline},
        PeerJoinedPush{rank, ranks_[rank].te_server_name,
                       ranks_[rank].warmup_recv_addr}};
}

}  // namespace mooncake
