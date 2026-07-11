#include "control_plane/coordinator.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <set>

#include <glog/logging.h>

namespace mooncake {

// Constructor

CentralizedCoordinatorStateMachine::CentralizedCoordinatorStateMachine(
    int max_world_size, std::chrono::microseconds fault_reconciliation_window)
    : max_world_size_(max_world_size),
      fault_reconciliation_window_(fault_reconciliation_window) {
    CHECK_GT(max_world_size_, 0);
    CHECK_LE(max_world_size_, kMaxNumRanks)
        << "max_world_size " << max_world_size_ << " exceeds kMaxNumRanks ("
        << kMaxNumRanks << ")";
    ranks_.assign(kMaxNumRanks, RankInfo{});
    for (GlobalRank r{0}; r < GlobalRank{kMaxNumRanks}; ++r) {
        ranks_[r].link_status.assign(max_world_size_, 0);
        if (r < max_world_size_) {
            ranks_[r].link_status[r] = 1;
        }
    }
}

// registerAgent

CoordinatorApplyResult<RegisterResponse>
CentralizedCoordinatorStateMachine::handleRegister(const RegisterRequest& req) {
    CoordinatorApplyResult<RegisterResponse> result;
    if (!rankInValidRange(req.rank)) {
        result.response.success = false;
        result.response.error_msg = "rank out of valid range";
        return result;
    }
    auto& info = ranks_[req.rank];

    // Identity check
    // If the rank is currently HEALTHY AND the request comes from a different
    // process -> reject.  A failed / auto-deactivated rank (SYNCED or OFFLINE)
    // may be replaced immediately; waiting for heartbeat timeout is not
    // required because the Coordinator has already removed it from the healthy
    // set.
    bool same_peer = (info.agent_addr == req.agent_addr &&
                      info.te_server_name == req.te_server_name);

    if (info.state == RankState::HEALTHY && !same_peer) {
        result.response.success = false;
        result.response.error_msg =
            "rank already registered and is HEALTHY; replacement must wait "
            "for the old process to leave the healthy set.";
        return result;
    }

    LOG(INFO) << "[COORD] registerAgent rank=" << req.rank
              << " addr=" << req.agent_addr
              << " session=" << req.agent_session_epoch;

    // A new session makes old endpoints invalid.  Clear any endpoint that
    // belongs to a previous session so that GroupMember::endpoint being set is
    // authoritative for "endpoint present and fresh for the current session".
    bool session_changed =
        (info.state != RankState::OFFLINE &&
         info.agent_session_epoch != req.agent_session_epoch);

    info.agent_addr = req.agent_addr;
    info.te_server_name = req.te_server_name;
    info.agent_session_epoch = req.agent_session_epoch;
    info.warmup_recv_addr = req.warmup_recv_addr;
    info.last_heartbeat = std::chrono::steady_clock::now();
    info.link_status.assign(max_world_size_, 0);
    info.link_status[req.rank] = 1;

    if (session_changed) {
        for (auto& [group_id, view] : group_views_) {
            auto& member = view.members[req.rank];
            if (member.agent_session_epoch.has_value() &&
                *member.agent_session_epoch != req.agent_session_epoch) {
                member.agent_session_epoch = std::nullopt;
                member.endpoint = std::nullopt;
                LOG(INFO) << "[COORD] registerAgent cleared stale endpoint"
                          << " group=" << group_id << " rank=" << req.rank;
            }
        }
    }

    result.effects.push_back(makePeerJoinedEffect(req.rank));
    transitionToSynced(req.rank, result.effects);
    result.response = buildRegisterResponse(req.rank);
    result.response.success = true;
    return result;
}

// heartbeat

CoordinatorApplyResult<HeartbeatResponse>
CentralizedCoordinatorStateMachine::handleHeartbeat(
    const HeartbeatRequest& req) {
    CoordinatorApplyResult<HeartbeatResponse> result;
    if (!rankInValidRange(req.rank)) {
        result.response.acknowledge = false;
        result.response.require_reregister = true;
        LOG(INFO) << "[COORD] heartbeat rank=" << req.rank
                  << " rejected: rank out of range";
        return result;
    }
    auto& info = ranks_[req.rank];

    if (info.state == RankState::OFFLINE ||
        info.agent_session_epoch != req.agent_session_epoch) {
        result.response.acknowledge = false;
        result.response.require_reregister = true;
        LOG(INFO) << "[COORD] heartbeat rank=" << req.rank
                  << " rejected: stale session (state="
                  << static_cast<int>(info.state)
                  << " session=" << info.agent_session_epoch
                  << " req_session=" << req.agent_session_epoch << ")";
        return result;
    }

    info.last_heartbeat = std::chrono::steady_clock::now();

    result.response.acknowledge = true;
    result.response.require_reregister = false;
    return result;
}

// registerGroup

CoordinatorApplyResult<RegisterGroupResponse>
CentralizedCoordinatorStateMachine::handleRegisterGroup(
    const RegisterGroupRequest& req) {
    CoordinatorApplyResult<RegisterGroupResponse> result;
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
        LOG(INFO) << "[COORD] registerGroup rank=" << req.rank
                  << " group=" << req.group.group_id
                  << " rejected: stale session (state="
                  << static_cast<int>(info.state)
                  << " session=" << info.agent_session_epoch
                  << " req_session=" << req.agent_session_epoch << ")";
        return result;
    }

    bool ok = registerGroup(req.rank, req.group, req.auto_deactivate,
                            result.response, result.effects);
    if (!ok) return result;

    result.response.success = true;
    return result;
}

// handleProposeViewUpdate (activate / deactivate)

CoordinatorApplyResult<void>
CentralizedCoordinatorStateMachine::handleProposeViewUpdate(
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
        // group_views_ is populated synchronously with group_views_ in
        // registerGroup; both should always exist together.
        auto desc_it = group_views_.find(req.group_id);
        if (desc_it == group_views_.end()) {
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
            if (!view.members[rank].isActive()) changed = true;
        }

        LOG(INFO) << "[COORD] handleProposeViewUpdate activate group="
                  << req.group_id << " source=" << req.source_rank
                  << " requested_ranks=["
                  << [&req]() {
                         std::string s;
                         for (auto r : req.requested_ranks)
                             s += std::to_string(r) + " ";
                         return s;
                     }()
                  << "] changed=" << changed << " epoch=" << view.epoch;

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
            view.members[rank].status = GroupMemberStatus::kActive;
        }
    } else {
        // deactivate
        for (GlobalRank rank : req.requested_ranks) {
            if (view.members[rank].isActive()) {
                view.members[rank].status = GroupMemberStatus::kInactive;
                view.members[rank].agent_session_epoch = std::nullopt;
                view.members[rank].endpoint = std::nullopt;
                changed = true;
            }
        }
        // rank_order is NOT changed on deactivate.
    }

    if (!changed) {
        result.effects.push_back(ReplyViewUpdateEffect{
            propose_id, {ViewUpdateStatus::Applied, view.epoch, {}, ""}});
        return result;
    }

    view.epoch++;

    // Resolve any pending syncAfterFailure callers whose view is now stale.
    flushPendingSyncs(req.group_id, result.effects);

    auto required_acks =
        computeRequiredViewAcks(old_view, view, req.source_rank);

    if (required_acks.empty()) {
        // Best-effort: broadcast + reply immediately.
        result.effects.push_back(
            ViewUpdateEffect{view, {}, ProposalAckRoute{propose_id}});
        result.effects.push_back(ReplyViewUpdateEffect{
            propose_id, {ViewUpdateStatus::Applied, view.epoch, {}, ""}});
    } else {
        // Strict Barrier: store pending, let Host broadcast, wait for ACKs.
        pending_proposal_acks_[propose_id] = PendingProposal{
            propose_id,
            req.group_id,
            {ViewUpdateStatus::Applied, view.epoch, {}, ""},
            std::unordered_set<GlobalRank>(required_acks.begin(),
                                           required_acks.end()),
            std::chrono::steady_clock::now() + kProposeTimeout,
        };
        result.effects.push_back(ViewUpdateEffect{
            view, required_acks, ProposalAckRoute{propose_id}});
    }

    return result;
}

// checkTimeouts  - heartbeat timeout + proposal ACK timeout + fault
// reconciliation window

CoordinatorApplyResult<void>
CentralizedCoordinatorStateMachine::checkTimeouts() {
    CoordinatorApplyResult<void> result;
    auto now = std::chrono::steady_clock::now();

    // Heartbeat timeout
    for (GlobalRank rank{0}; rank < max_world_size_; ++rank) {
        auto& info = ranks_[rank];
        if (info.state == RankState::OFFLINE) continue;
        if (now - info.last_heartbeat > kHeartbeatTimeout) {
            transitionToOffline(rank, result.effects);
        }
    }

    // Propose ACK timeout (Strict Barrier & Prune)
    for (auto it = pending_proposal_acks_.begin();
         it != pending_proposal_acks_.end();) {
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
            it = pending_proposal_acks_.erase(it);
        } else {
            ++it;
        }
    }

    // Fault reconciliation window.  link_status was updated immediately when
    // each report arrived; now that the window has closed we recompute health,
    // prune unhealthy members from all auto_deactivate groups, and seal the
    // epoch for groups that participated in the window so late reports cannot
    // reopen a decision for the sealed view.
    if (reconciliation_ctx_.active && now >= reconciliation_ctx_.deadline) {
        reconciliation_ctx_.active = false;
        auto groups_to_seal = std::move(reconciliation_ctx_.groups_in_window);
        reconciliation_ctx_.groups_in_window.clear();

        updateRankStates(result.effects);
        applyAutoDeactivate(result.effects);

        for (const auto& [group_id, entry_epoch] : groups_to_seal) {
            auto it = group_views_.find(group_id);
            if (it == group_views_.end()) {
                flushPendingSyncs(group_id, result.effects);
                continue;
            }
            GroupView& view = it->second;

            // If the group's epoch has already moved on (e.g. a concurrent
            // manual activate/deactivate proposal, or applyAutoDeactivate
            // already sealed it), flush any pending syncs immediately.
            if (view.epoch != entry_epoch) {
                flushPendingSyncs(group_id, result.effects);
                continue;
            }

            // Epoch unchanged — seal the group now.
            view.epoch++;

            // If there are pending syncAfterFailure callers for this group,
            // require their ACK so the sync RPC is resolved only after the
            // Agent has applied the ViewUpdate locally.
            auto sync_it = pending_syncs_.find(group_id);
            if (sync_it != pending_syncs_.end()) {
                std::vector<GlobalRank> ack_ranks;
                for (const auto& [rank, ids] : sync_it->second) {
                    ack_ranks.push_back(rank);
                }
                result.effects.push_back(ViewUpdateEffect{
                    view, ack_ranks, GeneralAckRoute{}});
            } else {
                result.effects.push_back(
                    ViewUpdateEffect{view, {}, GeneralAckRoute{}});
            }

            LOG(INFO) << "[COORD] sealed group=" << group_id
                      << " epoch=" << view.epoch;
        }

        checkGroupTransitions(result.effects);
    }

    return result;
}

// publishEndpoint - endpoint registration

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

        auto& view = it->second;
        auto& member = view.members[req.rank];
        bool had_endpoint = member.hasEndpoint();
        member.agent_session_epoch = req.agent_session_epoch;
        member.endpoint = ep.endpoint_info;
        member.endpoint->endpoint_epoch = ++endpoint_epochs_[req.rank];

        LOG(INFO) << "[COORD] handlePublishEndpoint rank=" << req.rank
                  << " group=" << ep.group_id
                  << " had_endpoint=" << had_endpoint
                  << " endpoint_epoch=" << member.endpoint->endpoint_epoch
                  << " agent_session_epoch=" << req.agent_session_epoch
                  << " is_member=" << member.isMember()
                  << " group_status=" << static_cast<int>(view.status);

        if (member.isMember() && view.status == GroupStatus::Ready) {
            // Group already activated -> push best-effort view update so that
            // all members (including existing survivors) see the new endpoint
            // for a replacement/extension rank before it is activated.
            view.epoch++;
            result.effects.push_back(
                ViewUpdateEffect{view, {}, GeneralAckRoute{}});
            flushPendingSyncs(ep.group_id, result.effects);
            LOG(INFO) << "[COORD] handlePublishEndpoint pushed view update"
                      << " group=" << ep.group_id
                      << " epoch=" << view.epoch;
        }
    }

    result.response.success = true;
    checkGroupTransitions(result.effects);
    return result;
}

// transferObservation  - update link_status from data-plane evidence
//
// The link_status is updated immediately so subsequent reports and link-state
// events see the latest connectivity snapshot.  However, the coordinator does
// NOT recompute the authoritative healthy set here.  Instead it opens (or
// extends) a fault_reconciliation_window; the membership decision is deferred
// until the window closes.  This gives multiple survivors time to report the
// same failure and prevents a single fast reporter from causing a premature
// auto-deactivation that hides the failure from slower survivors.

CoordinatorApplyResult<void>
CentralizedCoordinatorStateMachine::handleTransferObservation(
    const TransferObservationReport& req) {
    CoordinatorApplyResult<void> result;
    if (!rankInValidRange(req.reporter_rank)) return result;
    auto& reporter = ranks_[req.reporter_rank];

    if (reporter.agent_session_epoch != req.agent_session_epoch) {
        return result;  // stale reporter
    }

    // Determine authoritative epoch for the reported group.  Reports for
    // unknown groups or from an already-sealed view are dropped.
    auto view_it = group_views_.find(req.group_id);
    if (view_it == group_views_.end()) return result;
    uint64_t current_epoch = view_it->second.epoch;

    if (req.epoch < current_epoch) {
        LOG(INFO) << "[COORD] handleTransferObservation dropped stale report"
                  << " reporter=" << req.reporter_rank
                  << " group=" << req.group_id << " report_epoch=" << req.epoch
                  << " current_epoch=" << current_epoch;
        return result;
    }

    // Open or extend the fault reconciliation window.  The actual membership
    // decision is deferred until checkTimeouts() closes the window.  A single
    // global deadline batches all groups that observe the same physical fault,
    // but each group records its own entry epoch so that multi-group scenarios
    // do not suffer from epoch skew.
    if (!reconciliation_ctx_.active) {
        reconciliation_ctx_.active = true;
        reconciliation_ctx_.deadline =
            std::chrono::steady_clock::now() + fault_reconciliation_window_;
        reconciliation_ctx_.groups_in_window.clear();
    }

    // Record the group and the epoch at which it entered the window.  A group's
    // authoritative epoch must remain constant for the lifetime of the window.
    // If a concurrent manual proposal advances the epoch while the window is
    // still open, we adopt the newer epoch so the window can still seal the
    // current authoritative view.  Reports older than the recorded entry epoch
    // are dropped as they belong to a view that has already been superseded.
    auto it = reconciliation_ctx_.groups_in_window.find(req.group_id);
    if (it == reconciliation_ctx_.groups_in_window.end()) {
        reconciliation_ctx_.groups_in_window[req.group_id] = req.epoch;
    } else if (req.epoch > it->second) {
        LOG(INFO) << "[COORD] handleTransferObservation adopted newer epoch"
                  << " group=" << req.group_id
                  << " old_window_epoch=" << it->second
                  << " new_window_epoch=" << req.epoch;
        it->second = req.epoch;
    } else if (req.epoch < it->second) {
        LOG(INFO) << "[COORD] handleTransferObservation dropped obsolete report"
                  << " reporter=" << req.reporter_rank
                  << " group=" << req.group_id << " report_epoch=" << req.epoch
                  << " window_epoch=" << it->second;
        return result;
    }

    if (reporter.link_status.size() != static_cast<size_t>(max_world_size_)) {
        reporter.link_status.resize(max_world_size_, 0);
    }

    for (int32_t peer = 0; peer < max_world_size_; ++peer) {
        if (!req.attempted_ranks[peer]) continue;
        if (req.succeeded_ranks[peer]) reporter.link_status[peer] = 1;
        if (req.failed_ranks_hint[peer]) {
            reporter.link_status[peer] = 0;
        }
    }

    return result;
}

// handleLinkStateChange - per-peer link state change from LinkManager events

CoordinatorApplyResult<void>
CentralizedCoordinatorStateMachine::handleLinkStateChange(
    const LinkStateChangeReport& req) {
    CoordinatorApplyResult<void> result;
    if (!rankInValidRange(req.reporter_rank)) return result;
    if (!rankInValidRange(req.peer)) return result;
    auto& reporter = ranks_[req.reporter_rank];

    if (reporter.agent_session_epoch != req.agent_session_epoch)
        return result;  // stale reporter

    if (reporter.link_status.size() != static_cast<size_t>(max_world_size_)) {
        reporter.link_status.resize(max_world_size_, 0);
    }

    reporter.link_status[req.peer] = req.is_up ? 1 : 0;
    reporter.link_status[req.reporter_rank] = 1;  // self is always connected

    LOG(INFO) << "[COORD] handleLinkStateChange reporter=" << req.reporter_rank
              << " peer=" << req.peer << " is_up=" << req.is_up;

    updateRankStates(result.effects);
    applyAutoDeactivate(result.effects);
    checkGroupTransitions(result.effects);
    return result;
}

// handleUnregisterGroup - explicit, fire-and-forget group departure.

CoordinatorApplyResult<void>
CentralizedCoordinatorStateMachine::handleUnregisterGroup(
    const UnregisterGroupRequest& req) {
    CoordinatorApplyResult<void> result;
    if (!rankInValidRange(req.rank)) {
        LOG(INFO) << "[COORD] unregisterGroup rank=" << req.rank
                  << " group=" << req.group_id
                  << " rejected: rank out of range";
        return result;
    }

    auto it = group_views_.find(req.group_id);
    if (it == group_views_.end()) {
        LOG(INFO) << "[COORD] unregisterGroup rank=" << req.rank
                  << " group=" << req.group_id << " rejected: group not found";
        return result;
    }

    // Reject stale session epochs (e.g. the Agent re-registered after a
    // disconnect and this is a delayed unregister from the old session).
    if (ranks_[req.rank].agent_session_epoch != req.agent_session_epoch) {
        LOG(INFO) << "[COORD] unregisterGroup rank=" << req.rank
                  << " group=" << req.group_id << " rejected: stale session";
        return result;
    }

    auto& view = it->second;
    auto& member = view.members[req.rank];
    if (member.hasLeft()) {
        LOG(INFO) << "[COORD] unregisterGroup rank=" << req.rank
                  << " group=" << req.group_id << " ignored: already left";
        return result;
    }

    LOG(INFO) << "[COORD] unregisterGroup rank=" << req.rank
              << " group=" << req.group_id
              << " old_status=" << static_cast<int>(member.status);

    member.status = GroupMemberStatus::kLeft;
    member.agent_session_epoch = std::nullopt;
    member.endpoint = std::nullopt;
    view.epoch++;

    flushPendingSyncs(req.group_id, result.effects);

    if (canEraseGroup(view)) {
        eraseGroup(req.group_id, result.effects);
    }
    return result;
}

// Private: state transitions

void CentralizedCoordinatorStateMachine::transitionToOffline(
    GlobalRank rank, std::vector<CoordinatorEffect>& effects) {
    LOG(INFO) << "[COORD] transitionToOffline rank=" << rank
              << " state=" << static_cast<int>(ranks_[rank].state);
    ranks_[rank].state = RankState::OFFLINE;
    ranks_[rank].link_status.assign(max_world_size_, 0);

    // Clear this rank's connectivity from all peers.
    for (auto& peer : ranks_) {
        if (static_cast<size_t>(rank) < peer.link_status.size())
            peer.link_status[rank] = 0;
    }

    // Mark the failed rank as inactive in every group it still belongs to.
    std::vector<GroupId> groups_to_erase;
    for (auto& [group_id, view] : group_views_) {
        if (!rankInValidRange(rank)) continue;
        auto& member = view.members[rank];
        if (member.status == GroupMemberStatus::kActive) {
            member.status = GroupMemberStatus::kInactive;
            member.agent_session_epoch = std::nullopt;
            member.endpoint = std::nullopt;
            view.epoch++;
            effects.push_back(ViewUpdateEffect{view, {}, GeneralAckRoute{}});
            flushPendingSyncs(group_id, effects);
        }
        if (canEraseGroup(view)) {
            groups_to_erase.push_back(group_id);
        }
    }
    for (GroupId group_id : groups_to_erase) {
        eraseGroup(group_id, effects);
    }

    effects.push_back(makeRankStateEffect(rank));
    updateRankStates(effects);
    applyAutoDeactivate(effects);
    checkGroupTransitions(effects);
}

void CentralizedCoordinatorStateMachine::transitionToSynced(
    GlobalRank rank, std::vector<CoordinatorEffect>& effects) {
    ranks_[rank].state = RankState::SYNCED;
    effects.push_back(makeRankStateEffect(rank));
}

// Private: authoritative HEALTHY computation

bool CentralizedCoordinatorStateMachine::isMutuallyConnected(
    GlobalRank a, GlobalRank b) const {
    if (a == b) return true;
    if (ranks_[a].state == RankState::OFFLINE ||
        ranks_[b].state == RankState::OFFLINE)
        return false;
    return static_cast<size_t>(b) < ranks_[a].link_status.size() &&
           static_cast<size_t>(a) < ranks_[b].link_status.size() &&
           ranks_[a].link_status[b] != 0 && ranks_[b].link_status[a] != 0;
}

std::vector<GlobalRank> CentralizedCoordinatorStateMachine::extendHealthySet()
    const {
    // Step 1: Collect current HEALTHY ranks.
    std::vector<GlobalRank> result;
    for (GlobalRank i{0}; i < max_world_size_; ++i) {
        if (ranks_[i].state == RankState::HEALTHY) {
            result.push_back(i);
        }
    }

    // Step 2: Evict the least-connected rank until the set is a clique.
    // (Focuses strictly on connection density; naturally terminates on singletons).
    while (true) {
        GlobalRank worst = kInvalidGlobalRank;
        int worst_degree = std::numeric_limits<int>::max();

        for (GlobalRank r : result) {
            int degree = 0;
            for (GlobalRank other : result) {
                if (r == other) continue;
                if (isMutuallyConnected(r, other)) ++degree;
            }
            if (degree < worst_degree ||
                (degree == worst_degree &&
                 (worst == kInvalidGlobalRank || r > worst))) {
                worst_degree = degree;
                worst = r;
            }
        }

        int expected = static_cast<int>(result.size()) - 1;
        if (worst_degree >= expected) break;

        result.erase(std::remove(result.begin(), result.end(), worst),
                     result.end());
    }

    // Step 2.5: Enforce business rule - evict isolated singletons.
    // (Clears the path so Step 3 can bootstrap other healthy candidates).
    if (result.size() == 1) {
        GlobalRank singleton = result[0];
        bool has_connections = false;
        for (GlobalRank other{0}; other < max_world_size_; ++other) {
            if (other == singleton) continue;
            if (ranks_[other].state == RankState::OFFLINE) continue;
            if (isMutuallyConnected(singleton, other)) {
                has_connections = true;
                break;
            }
        }
        if (!has_connections) {
            result.clear();
        }
    }

    // Step 3: Extend with new mutually-connected candidates.
    for (GlobalRank i{0}; i < max_world_size_; ++i) {
        if (ranks_[i].state == RankState::OFFLINE) continue;
        if (std::find(result.begin(), result.end(), i) != result.end())
            continue;
        bool connected_to_all = true;
        for (GlobalRank existing : result) {
            if (!isMutuallyConnected(i, existing)) {
                connected_to_all = false;
                break;
            }
        }
        if (connected_to_all) {
            result.push_back(i);
        }
    }

    return result;
}

void CentralizedCoordinatorStateMachine::updateRankStates(
    std::vector<CoordinatorEffect>& effects) {
    auto healthy_set = extendHealthySet();

    std::string healthy_str;
    for (GlobalRank r : healthy_set) {
        healthy_str += std::to_string(r) + " ";
    }
    LOG(INFO) << "[COORD] updateRankStates healthy_set=[" << healthy_str << "]";

    // Update per-rank HEALTHY / SYNCED state.
    for (GlobalRank i{0}; i < max_world_size_; ++i) {
        if (ranks_[i].state == RankState::OFFLINE) continue;

        bool in_healthy = std::find(healthy_set.begin(), healthy_set.end(),
                                    i) != healthy_set.end();

        if (in_healthy && ranks_[i].state != RankState::HEALTHY) {
            LOG(INFO) << "[COORD] rank=" << i << " transitioning to HEALTHY";
            ranks_[i].state = RankState::HEALTHY;
            effects.push_back(makeRankStateEffect(i));
        } else if (!in_healthy && ranks_[i].state == RankState::HEALTHY) {
            LOG(INFO) << "[COORD] rank=" << i << " transitioning to SYNCED";
            ranks_[i].state = RankState::SYNCED;
            effects.push_back(makeRankStateEffect(i));
        }
    }
}

void CentralizedCoordinatorStateMachine::applyAutoDeactivate(
    std::vector<CoordinatorEffect>& effects) {
    auto healthy_set = extendHealthySet();

    // For auto_deactivate groups, remove unhealthy ranks from the active set.
    // However, during bootstrap / BootstrapSyncing we do NOT do this: we wait
    // for full mutual connectivity and let waitUntilGroupReady() time out if a
    // peer is truly dead.
    for (auto& [group_id, view] : group_views_) {
        if (!group_auto_deactivate_[group_id]) continue;
        if (view.status != GroupStatus::Ready) continue;
        std::vector<GlobalRank> deactivated_ranks;
        for (GlobalRank i{0}; i < max_world_size_; ++i) {
            if (!view.members[i].isActive()) continue;
            bool in_healthy = std::find(healthy_set.begin(), healthy_set.end(),
                                        i) != healthy_set.end();
            if (!in_healthy) {
                view.members[i].status = GroupMemberStatus::kInactive;
                view.members[i].agent_session_epoch = std::nullopt;
                view.members[i].endpoint = std::nullopt;
                deactivated_ranks.push_back(i);
                LOG(INFO) << "[COORD] auto_deactivate group=" << group_id
                          << " rank=" << i;
            }
        }
        if (!deactivated_ranks.empty()) {
            view.epoch++;
            effects.push_back(ViewUpdateEffect{view, {}, GeneralAckRoute{}});
            flushPendingSyncs(group_id, effects);
            LOG(INFO) << "[COORD] auto_deactivate view update group="
                      << group_id << " epoch=" << view.epoch;
        }
    }
}

// Private: activatable predicates

bool CentralizedCoordinatorStateMachine::isActivatableSet(
    GroupId group_id, const std::vector<GlobalRank>& new_ranks,
    const GroupView& old_view) const {
    // Build the future active set: old active ∪ new ranks.
    std::vector<GlobalRank> future_active;
    for (GlobalRank i{0}; i < max_world_size_; ++i) {
        if (old_view.members[i].isActive()) {
            future_active.push_back(i);
        }
    }
    for (GlobalRank r : new_ranks) {
        if (!old_view.members[r].isActive()) {
            future_active.push_back(r);
        }
    }

    std::string future_active_str;
    for (GlobalRank r : future_active) {
        future_active_str += std::to_string(r) + " ";
    }
    LOG(INFO) << "[COORD] isActivatableSet group=" << group_id
              << " new_ranks=[" << [&new_ranks]() {
                     std::string s;
                     for (auto r : new_ranks) s += std::to_string(r) + " ";
                     return s;
                 }()
              << "] future_active=[" << future_active_str << "]";

    // Every rank in the future set must be activatable with respect to the
    // full future set.  This guarantees all-to-all mutual connectivity:
    // old <-> old, old <-> new, and new <-> new.
    for (GlobalRank r : future_active) {
        if (!isRankActivatable(group_id, r, future_active)) {
            LOG(INFO) << "[COORD] isActivatableSet group=" << group_id
                      << " rejected: rank=" << r
                      << " not activatable";
            return false;
        }
    }
    LOG(INFO) << "[COORD] isActivatableSet group=" << group_id
              << " accepted";
    return true;
}

bool CentralizedCoordinatorStateMachine::isRankActivatable(
    GroupId group_id, GlobalRank rank,
    const std::vector<GlobalRank>& peer_ranks) const {
    if (!rankInValidRange(rank)) {
        LOG(INFO) << "[COORD] isRankActivatable group=" << group_id
                  << " rank=" << rank << " false: out of range";
        return false;
    }
    if (ranks_[rank].state != RankState::HEALTHY) {
        LOG(INFO) << "[COORD] isRankActivatable group=" << group_id
                  << " rank=" << rank
                  << " false: rank_state=" << static_cast<int>(ranks_[rank].state)
                  << " (expected HEALTHY="
                  << static_cast<int>(RankState::HEALTHY) << ")";
        return false;
    }

    for (GlobalRank other : peer_ranks) {
        if (other == rank) continue;
        if (!isMutuallyConnected(rank, other)) {
            LOG(INFO) << "[COORD] isRankActivatable group=" << group_id
                      << " rank=" << rank
                      << " false: not mutually connected to peer="
                      << other;
            return false;
        }
    }

    auto group = group_views_.find(group_id);
    if (group == group_views_.end()) {
        LOG(INFO) << "[COORD] isRankActivatable group=" << group_id
                  << " rank=" << rank
                  << " false: group not found";
        return false;
    }

    const auto& member = group->second.members[rank];
    bool member_ok = member.isMember();
    bool endpoint_ok = member.hasEndpoint();
    bool session_ok = member.agent_session_epoch.has_value() &&
                      *member.agent_session_epoch ==
                          ranks_[rank].agent_session_epoch;
    bool result = member_ok && endpoint_ok && session_ok;
    if (!result) {
        LOG(INFO) << "[COORD] isRankActivatable group=" << group_id
                  << " rank=" << rank
                  << " false: member_ok=" << member_ok
                  << " endpoint_ok=" << endpoint_ok
                  << " session_ok=" << session_ok
                  << " member.status="
                  << static_cast<int>(member.status)
                  << " member.has_endpoint=" << member.hasEndpoint()
                  << " member.agent_session_epoch="
                  << (member.agent_session_epoch.has_value()
                         ? std::to_string(*member.agent_session_epoch)
                         : "nullopt")
                  << " rank.agent_session_epoch="
                  << ranks_[rank].agent_session_epoch;
    }
    return result;
}

// checkGroupTransitions - bootstrap state machine driver.
//
// Group lifecycle:
//   registerGroup() creates a group in Bootstrapping status.  Once all active
//   ranks have valid endpoints, are HEALTHY, and are mutually TE-connected,
//   this function transitions to BootstrapSyncing and broadcasts a ViewUpdate
//   to collect ACKs from all active ranks (a 2PC barrier).  Once all ACKs
//   arrive, the group transitions to Ready. waitUntilGroupReady() unblocks only
//   when status == Ready.
void CentralizedCoordinatorStateMachine::checkGroupTransitions(
    std::vector<CoordinatorEffect>& effects) {
    for (auto& [group_id, view] : group_views_) {
        if (view.status == GroupStatus::Bootstrapping) {
            // Collect all active ranks.
            std::vector<GlobalRank> peer_ranks;
            bool has_any_active = false;
            for (GlobalRank i{0}; i < max_world_size_; ++i) {
                if (!view.members[i].isActive()) continue;
                has_any_active = true;
                peer_ranks.push_back(i);
            }

            bool all_ready = true;
            for (GlobalRank r : peer_ranks) {
                if (!isRankActivatable(group_id, r, peer_ranks)) {
                    all_ready = false;
                    break;
                }
            }

            if (has_any_active && all_ready) {
                // All active ranks have endpoints and are HEALTHY.
                // Transition to BootstrapSyncing and initiate 2PC.
                view.status = GroupStatus::BootstrapSyncing;
                view.epoch++;

                auto acks_needed =
                    computeRequiredViewAcks(view, view, kInvalidGlobalRank);
                pending_bootstrap_acks_[group_id] =
                    std::unordered_set<GlobalRank>(acks_needed.begin(),
                                                   acks_needed.end());

                effects.push_back(
                    ViewUpdateEffect{view, acks_needed, GeneralAckRoute{}});
                LOG(INFO) << "[COORD] group=" << group_id
                          << " transitioned to BootstrapSyncing epoch="
                          << view.epoch;
            }
        } else if (view.status == GroupStatus::BootstrapSyncing) {
            // If a peer dies during this phase, its ACK never arrives.
            // The group stays in BootstrapSyncing; waitUntilGroupReady()
            // will time out on the dead peer's Agent.
            auto it = pending_bootstrap_acks_.find(group_id);
            if (it == pending_bootstrap_acks_.end()) continue;
            auto& pending = it->second;

            if (pending.empty()) {
                // All ranks have ACKed.  Transition to Ready.
                view.status = GroupStatus::Ready;
                view.epoch++;
                pending_bootstrap_acks_.erase(it);
                effects.push_back(
                    ViewUpdateEffect{view, {}, GeneralAckRoute{}});
                flushPendingSyncs(group_id, effects);
                LOG(INFO) << "[COORD] group=" << group_id
                          << " transitioned to Ready epoch=" << view.epoch;
            }
        }
    }
}

// Private: registerGroup

bool CentralizedCoordinatorStateMachine::registerGroup(
    GlobalRank joining_rank, const GroupView& group, bool auto_deactivate,
    RegisterGroupResponse& response, std::vector<CoordinatorEffect>& effects) {
    GroupId group_id = group.group_id;

    // Validate rank_order elements.
    for (GlobalRank r : group.rank_order) {
        if (!rankInValidRange(r)) {
            response.success = false;
            response.reject_reason = "rank_order contains invalid GlobalRank";
            return false;
        }
    }

    // Validate no duplicates in rank_order.
    {
        std::set<GlobalRank> seen(group.rank_order.begin(),
                                  group.rank_order.end());
        if (seen.size() != group.rank_order.size()) {
            response.success = false;
            response.reject_reason = "rank_order contains duplicate ranks";
            return false;
        }
    }

    auto it = group_views_.find(group_id);
    if (it == group_views_.end()) {
        // First declaration -> create group.
        // Founding members are all entries in rank_order.
        GroupView view;
        view.group_id = group_id;
        view.rank_order = group.rank_order;
        view.members.resize(max_world_size_);
        for (GlobalRank r : group.rank_order) {
            view.members[r].status = GroupMemberStatus::kActive;
        }
        view.status = GroupStatus::Bootstrapping;
        group_views_[group_id] = std::move(view);
        group_auto_deactivate_[group_id] = auto_deactivate;
        response.success = true;
        return true;
    }

    // Group already exists -> validate that the new rank_order is compatible
    // with the existing one.  The first backend to declare the group sets the
    // initial rank_order; later backends must agree on all overlapping
    // positions.  rank_order extension and member activation are handled
    // exclusively through proposeViewUpdate (activate_rank / recover_ranks).
    const auto& existing_order = group_views_[group_id].rank_order;
    const auto& new_order = group.rank_order;

    auto common_len = std::min(existing_order.size(), new_order.size());
    for (InGroupRank i{0}; i < static_cast<int32_t>(common_len); ++i) {
        if (new_order[i] != existing_order[i]) {
            response.success = false;
            response.reject_reason =
                "rank_order mismatch at position " + std::to_string(i);
            return false;
        }
    }

    // Note: if new_order is longer than existing_order, the extra ranks are
    // not activated here.  They must be activated via a subsequent
    // proposeViewUpdate (activate_rank / recover_ranks) from an existing
    // active member.  Until then, the extension backend operates in local-only
    // mode (its own rank is masked out of collectives by activeRanks).
    //
    // However, extend the existing rank_order with the new ranks now so that
    // every member's ViewUpdate already carries the correct local→global
    // mapping.  getPeerState() relies on rank_order to resolve in-group ranks,
    // and without this extension an existing member would read garbage/identity
    // for the joiner's slot and never see member=1.
    if (new_order.size() > existing_order.size()) {
        auto& ext_order = group_views_[group_id].rank_order;
        for (size_t i = existing_order.size(); i < new_order.size(); ++i) {
            ext_order.push_back(new_order[i]);
        }
        LOG(INFO) << "[COORD] registerGroup extended rank_order for group="
                  << group_id << " old_size=" << existing_order.size()
                  << " new_size=" << ext_order.size();
    }

    // The joining rank needs to be able to receive view updates from the
    // Coordinator, even if it is not yet active.  Promote kNone to kInactive
    // so that pushViewUpdate() will deliver the authoritative view to it.
    auto& view = group_views_[group_id];
    if (rankInValidRange(joining_rank) &&
        view.members[joining_rank].status == GroupMemberStatus::kNone) {
        view.members[joining_rank].status = GroupMemberStatus::kInactive;
        LOG(INFO) << "[COORD] registerGroup promoted joining_rank="
                  << joining_rank << " to Inactive in group=" << group_id
                  << " view_epoch=" << view.epoch;
    }

    // A Ready group that receives a registerGroup should push the authoritative
    // Ready view to all members (including the newly-joined inactive rank) so
    // that joining ranks can observe Ready and unblock waitUntilGroupReady().
    // Because the Coordinator executor is serialized, multiple simultaneous
    // joiners are naturally ordered into sequential pushes.
    if (view.status == GroupStatus::Ready) {
        effects.push_back(ViewUpdateEffect{view, {}, GeneralAckRoute{}});
        LOG(INFO) << "[COORD] registerGroup pushed Ready view group="
                  << group_id << " epoch=" << view.epoch;
    }

    response.success = true;
    return true;
}

// Private: helpers

bool CentralizedCoordinatorStateMachine::canEraseGroup(
    const GroupView& view) const {
    return std::all_of(view.members.begin(), view.members.end(),
                       [](const GroupMember& m) {
                           return m.status == GroupMemberStatus::kNone ||
                                  m.status == GroupMemberStatus::kLeft;
                       });
}

void CentralizedCoordinatorStateMachine::eraseGroup(
    GroupId group_id, std::vector<CoordinatorEffect>& effects) {
    // Erase pending bootstrap state if present.
    pending_bootstrap_acks_.erase(group_id);

    // Erase any pending proposals for this group so replies are not sent
    // after the group is gone.
    for (auto it = pending_proposal_acks_.begin();
         it != pending_proposal_acks_.end();) {
        if (it->second.group_id == group_id) {
            effects.push_back(ReplyViewUpdateEffect{
                it->first,
                {ViewUpdateStatus::Rejected, 0, {}, "group was destroyed"}});
            it = pending_proposal_acks_.erase(it);
        } else {
            ++it;
        }
    }

    group_views_.erase(group_id);
    group_auto_deactivate_.erase(group_id);
}

std::vector<GlobalRank>
CentralizedCoordinatorStateMachine::computeRequiredViewAcks(
    const GroupView& old_view, const GroupView& new_view,
    GlobalRank proposer) const {
    // All online ranks that are active in EITHER old or new view must ACK.
    // This ensures newly-activated ranks have received the view, and
    // deactivated ranks know they're removed.
    std::set<GlobalRank> required;
    for (GlobalRank i{0}; i < max_world_size_; ++i) {
        if (i == proposer) continue;
        if (ranks_[i].state == RankState::OFFLINE) continue;
        if (old_view.members[i].isActive() || new_view.members[i].isActive()) {
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
    for (int32_t i = 0; i < max_world_size_; ++i) {
        resp.all_rank_states[i] = ranks_[i].state;
    }

    // All groups (view includes rank_order and member state).
    for (const auto& [gid, view] : group_views_) {
        resp.groups.push_back(view);
    }

    // All rank connection metadata (for LinkManager).
    for (int32_t i = 0; i < max_world_size_; ++i) {
        if (i == for_rank) continue;
        if (ranks_[i].state == RankState::OFFLINE) continue;

        RankConnectionMetadata conn;
        conn.rank = i;
        conn.agent_addr = ranks_[i].agent_addr;
        conn.te_server_name = ranks_[i].te_server_name;
        conn.warmup_recv_addr = ranks_[i].warmup_recv_addr;
        resp.rank_connections.push_back(conn);
    }

    return resp;
}

// Effect factories

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

// handleSyncAfterFailure - sync-after-failure RPC handler.
//
// Flow:
//   1. Validate (rank, session, group).
//   2. Apply piggybacked observation if present (update link_status, open window).
//   3. Epoch guard: if caller is behind, piggyback on an existing pending ACK
//      (when available) or return immediately.
//   4. If a reconciliation window is open for this group, defer until it closes.
//   5. Otherwise return kNoChange immediately.
CoordinatorApplyResult<void>
CentralizedCoordinatorStateMachine::handleSyncAfterFailure(
    uint64_t sync_id, const SyncAfterFailureRequest& req) {
    CoordinatorApplyResult<void> result;

    // 1. Validate.
    if (!rankInValidRange(req.reporter_rank)) {
        result.effects.push_back(ReplySyncEffect{
            sync_id,
            {SyncAfterFailureStatus::kRejected, 0, "rank out of range"}});
        return result;
    }
    auto& reporter = ranks_[req.reporter_rank];
    if (reporter.state == RankState::OFFLINE ||
        reporter.agent_session_epoch != req.agent_session_epoch) {
        result.effects.push_back(ReplySyncEffect{
            sync_id,
            {SyncAfterFailureStatus::kRejected, 0, "stale session epoch"}});
        return result;
    }

    auto view_it = group_views_.find(req.group_id);
    if (view_it == group_views_.end()) {
        result.effects.push_back(ReplySyncEffect{
            sync_id,
            {SyncAfterFailureStatus::kRejected, 0, "group not found"}});
        return result;
    }

    // 2. Apply piggybacked observation inline.
    if (req.has_observation) {
        if (reporter.link_status.size() !=
            static_cast<size_t>(max_world_size_)) {
            reporter.link_status.resize(max_world_size_, 0);
        }
        for (int32_t peer = 0; peer < max_world_size_; ++peer) {
            if (!req.attempted_ranks[peer]) continue;
            if (req.succeeded_ranks[peer]) reporter.link_status[peer] = 1;
            if (req.failed_ranks_hint[peer]) {
                reporter.link_status[peer] = 0;
                if (rankInValidRange(peer) &&
                    ranks_[peer].state != RankState::OFFLINE) {
                    ranks_[peer].link_status.assign(max_world_size_, 0);
                    ranks_[peer].link_status[peer] = 1;
                }
            }
        }

        if (!reconciliation_ctx_.active) {
            reconciliation_ctx_.active = true;
            reconciliation_ctx_.deadline =
                std::chrono::steady_clock::now() +
                fault_reconciliation_window_;
            reconciliation_ctx_.groups_in_window.clear();
        }
        auto win_it =
            reconciliation_ctx_.groups_in_window.find(req.group_id);
        if (win_it == reconciliation_ctx_.groups_in_window.end()) {
            reconciliation_ctx_.groups_in_window[req.group_id] =
                req.observation_epoch;
        } else if (req.observation_epoch > win_it->second) {
            win_it->second = req.observation_epoch;
        }
    }

    // Re-fetch after possible state changes.
    view_it = group_views_.find(req.group_id);
    if (view_it == group_views_.end()) {
        result.effects.push_back(ReplySyncEffect{
            sync_id,
            {SyncAfterFailureStatus::kRejected, 0, "group not found"}});
        return result;
    }
    uint64_t current_epoch = view_it->second.epoch;

    // 3. Epoch guard.
    if (req.current_epoch > current_epoch) {
        result.effects.push_back(ReplySyncEffect{
            sync_id,
            {SyncAfterFailureStatus::kRejected, 0,
             "epoch ahead of coordinator"}});
        return result;
    }

    if (req.current_epoch < current_epoch) {
        // Epoch already advanced.  Check whether the caller is already
        // waiting on a pending ACK (piggyback).
        auto group_it = pending_syncs_.find(req.group_id);
        if (group_it != pending_syncs_.end()) {
            auto rank_it = group_it->second.find(req.reporter_rank);
            if (rank_it != group_it->second.end()) {
                rank_it->second.push_back(sync_id);
                return result;  // defer — no ReplySyncEffect
            }
        }
        // No pending ACK for this caller → decision already delivered.
        result.effects.push_back(ReplySyncEffect{
            sync_id, buildSyncAfterFailureResponse(req.group_id)});
        return result;
    }

    // 4. req.current_epoch == current_epoch.
    if (reconciliation_ctx_.active &&
        reconciliation_ctx_.groups_in_window.count(req.group_id)) {
        // Window open → defer until window closes.
        pending_syncs_[req.group_id][req.reporter_rank].push_back(sync_id);
        return result;
    }

    // 5. No window, epoch matches → no pending decision.
    result.effects.push_back(ReplySyncEffect{
        sync_id, {SyncAfterFailureStatus::kNoChange, current_epoch, ""}});
    return result;
}

// handleViewUpdateAck - unified ACK handler for all ViewUpdate pushes.
// Checks proposal 2PC, bootstrap 2PC, and pending sync callers.
CoordinatorApplyResult<void>
CentralizedCoordinatorStateMachine::handleViewUpdateAck(
    GroupId group_id, GlobalRank rank, uint64_t epoch, bool applied,
    const ViewUpdateAckRoute& route) {
    CoordinatorApplyResult<void> result;

    if (!applied) return result;

    auto view_it = group_views_.find(group_id);
    if (view_it == group_views_.end()) return result;
    if (epoch != view_it->second.epoch) return result;  // stale ACK

    // 1. Proposal 2PC — carry the propose_id to resolve the right proposal.
    if (auto* p = std::get_if<ProposalAckRoute>(&route)) {
        auto it = pending_proposal_acks_.find(p->propose_id);
        if (it != pending_proposal_acks_.end()) {
            it->second.waiting_acks.erase(rank);
            if (it->second.waiting_acks.empty()) {
                result.effects.push_back(ReplyViewUpdateEffect{
                    p->propose_id, it->second.eventual_response});
                pending_proposal_acks_.erase(it);
            }
        }
    }

    // 2. Bootstrap 2PC — only during BootstrapSyncing.
    auto ack_it = pending_bootstrap_acks_.find(group_id);
    if (ack_it != pending_bootstrap_acks_.end()) {
        ack_it->second.erase(rank);
        checkGroupTransitions(result.effects);
    }

    // 3. Sync-after-failure callers waiting on this ACK.
    flushPendingSyncs(group_id, rank, result.effects);

    return result;
}

// flushPendingSyncs - resolve all pending syncs for a group.
void CentralizedCoordinatorStateMachine::flushPendingSyncs(
    GroupId group_id, std::vector<CoordinatorEffect>& effects) {
    auto it = pending_syncs_.find(group_id);
    if (it == pending_syncs_.end()) return;

    auto resp = buildSyncAfterFailureResponse(group_id);
    for (auto& [rank, sync_ids] : it->second) {
        for (uint64_t sync_id : sync_ids) {
            effects.push_back(ReplySyncEffect{sync_id, resp});
        }
    }
    pending_syncs_.erase(it);
}

// flushPendingSyncs - resolve pending syncs for a specific rank.
void CentralizedCoordinatorStateMachine::flushPendingSyncs(
    GroupId group_id, GlobalRank rank,
    std::vector<CoordinatorEffect>& effects) {
    auto it = pending_syncs_.find(group_id);
    if (it == pending_syncs_.end()) return;

    auto rank_it = it->second.find(rank);
    if (rank_it == it->second.end()) return;

    auto resp = buildSyncAfterFailureResponse(group_id);
    for (uint64_t sync_id : rank_it->second) {
        effects.push_back(ReplySyncEffect{sync_id, resp});
    }
    it->second.erase(rank_it);
    if (it->second.empty()) {
        pending_syncs_.erase(it);
    }
}

SyncAfterFailureResponse
CentralizedCoordinatorStateMachine::buildSyncAfterFailureResponse(
    GroupId group_id) const {
    SyncAfterFailureResponse resp;
    auto it = group_views_.find(group_id);
    if (it != group_views_.end()) {
        resp.status = SyncAfterFailureStatus::kDecisionApplied;
        resp.new_epoch = it->second.epoch;
    }
    return resp;
}

}  // namespace mooncake
