#include "control_plane/coordinator.h"

#include <algorithm>
#include <numeric>
#include <set>
#include <sstream>

#include <glog/logging.h>

namespace mooncake {

// Constructor

CentralizedCoordinatorStateMachine::CentralizedCoordinatorStateMachine(
    int max_world_size)
    : max_world_size_(max_world_size) {
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
    // If the rank already has an active registration AND the request comes
    // from a different process -> reject.  Replacement must wait for the old
    // process to be heartbeat-timed-out to OFFLINE.
    bool same_peer = (info.agent_addr == req.agent_addr &&
                      info.te_server_name == req.te_server_name);

    if (info.state != RankState::OFFLINE && !same_peer) {
        result.response.success = false;
        result.response.error_msg =
            "rank already registered, and is not offline.";
        LOG(INFO) << "[COORD] registerAgent rank=" << req.rank
                  << " rejected: already registered";
        return result;
    }

    LOG(INFO) << "[COORD] registerAgent rank=" << req.rank
              << " addr=" << req.agent_addr
              << " session=" << req.agent_session_epoch;

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

// joinGroup

CoordinatorApplyResult<JoinGroupResponse>
CentralizedCoordinatorStateMachine::handleJoinGroup(
    const JoinGroupRequest& req) {
    CoordinatorApplyResult<JoinGroupResponse> result;
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
        LOG(INFO) << "[COORD] joinGroup rank=" << req.rank
                  << " group=" << req.group.group_id
                  << " rejected: stale session (state="
                  << static_cast<int>(info.state)
                  << " session=" << info.agent_session_epoch
                  << " req_session=" << req.agent_session_epoch << ")";
        return result;
    }

    bool ok = joinGroup(req.group, req.auto_deactivate, result.response);
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
        // joinGroup; both should always exist together.
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
            if (!view.member(rank).isActive()) changed = true;
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
            view.member(rank).status = GroupMemberStatus::kActive;
        }
    } else {
        // deactivate
        for (GlobalRank rank : req.requested_ranks) {
            if (view.member(rank).isActive()) {
                view.member(rank).status = GroupMemberStatus::kInactive;
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

// handleProposalAck  - 2PC second phase: collect ACKs from agents

CoordinatorApplyResult<void>
CentralizedCoordinatorStateMachine::handleProposalAck(uint64_t propose_id,
                                                      GlobalRank rank,
                                                      uint64_t epoch,
                                                      bool applied) {
    CoordinatorApplyResult<void> result;
    auto it = pending_proposal_acks_.find(propose_id);
    if (it == pending_proposal_acks_.end()) return result;  // already resolved

    if (!applied || epoch != it->second.eventual_response.new_epoch)
        return result;

    it->second.waiting_acks.erase(rank);
    if (it->second.waiting_acks.empty()) {
        result.effects.push_back(
            ReplyViewUpdateEffect{propose_id, it->second.eventual_response});
        pending_proposal_acks_.erase(it);
    }
    return result;
}

// handleBootstrapAck  - collect ACKs during BootstrapSyncing phase

CoordinatorApplyResult<void>
CentralizedCoordinatorStateMachine::handleBootstrapAck(GroupId group_id,
                                                       GlobalRank rank,
                                                       uint64_t epoch) {
    CoordinatorApplyResult<void> result;
    auto it = group_views_.find(group_id);
    if (it == group_views_.end()) return result;
    if (it->second.status != GroupStatus::BootstrapSyncing) return result;
    if (epoch != it->second.epoch) return result;  // stale ACK

    auto ack_it = pending_bootstrap_acks_.find(group_id);
    if (ack_it == pending_bootstrap_acks_.end()) return result;
    ack_it->second.erase(rank);
    checkGroupTransitions(result.effects);
    return result;
}

// checkTimeouts  - heartbeat timeout + proposal ACK timeout

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
        auto& member = view.member(req.rank);
        member.agent_session_epoch = req.agent_session_epoch;
        member.endpoint_epoch = ep.endpoint_epoch;
        member.endpoint_info = ep.endpoint_info;

        if (member.isActive()) {
            if (view.status == GroupStatus::Ready) {
                // Group already activated -> push best-effort view update.
                view.epoch++;
                result.effects.push_back(
                    ViewUpdateEffect{view, {}, BootstrapAckRoute{}});
            }
        }
    }

    result.response.success = true;
    checkGroupTransitions(result.effects);
    return result;
}

// transferObservation  - update link_status from data-plane evidence

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

    for (GlobalRank peer : globalRankRange(max_world_size_)) {
        if (!req.attempted_ranks[peer]) continue;
        if (req.succeeded_ranks[peer]) reporter.link_status[peer] = 1;
        if (req.failed_ranks[peer]) {
            reporter.link_status[peer] = 0;
            // When a transfer observation reports peer P as failed, P is
            // likely dead.  Clear P's entire link_status to prevent stale
            // data from P (collected before it died) from keeping P in the
            // max-clique healthy set.  P keeps self=1 so it can recover if
            // it comes back.
            if (rankInValidRange(peer) &&
                ranks_[peer].state != RankState::OFFLINE) {
                ranks_[peer].link_status.assign(max_world_size_, 0);
                ranks_[peer].link_status[peer] = 1;
                LOG(INFO) << "[COORD] handleTransferObservation cleared "
                          << "link_status for failed peer=" << peer;
            }
        }
    }

    LOG(INFO) << "[COORD] handleTransferObservation reporter="
              << req.reporter_rank << " link_status=";
    for (GlobalRank peer{0}; peer < max_world_size_; ++peer) {
        LOG(INFO) << "[COORD]   peer=" << peer
                  << " status=" << (int)reporter.link_status[peer];
    }

    // Recompute authoritative health
    updateRankHealth(result.effects);

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

    updateRankHealth(result.effects);
    return result;
}

// handleLeaveGroup - explicit, fire-and-forget group departure.

CoordinatorApplyResult<void>
CentralizedCoordinatorStateMachine::handleLeaveGroup(
    const LeaveGroupRequest& req) {
    CoordinatorApplyResult<void> result;
    if (!rankInValidRange(req.rank)) {
        LOG(INFO) << "[COORD] leaveGroup rank=" << req.rank
                  << " group=" << req.group_id
                  << " rejected: rank out of range";
        return result;
    }

    auto it = group_views_.find(req.group_id);
    if (it == group_views_.end()) {
        LOG(INFO) << "[COORD] leaveGroup rank=" << req.rank
                  << " group=" << req.group_id << " rejected: group not found";
        return result;
    }

    // Reject stale session epochs (e.g. the Agent re-registered after a
    // disconnect and this is a delayed leave from the old session).
    if (ranks_[req.rank].agent_session_epoch != req.agent_session_epoch) {
        LOG(INFO) << "[COORD] leaveGroup rank=" << req.rank
                  << " group=" << req.group_id << " rejected: stale session";
        return result;
    }

    auto& view = it->second;
    auto& member = view.member(req.rank);
    if (member.hasLeft()) {
        LOG(INFO) << "[COORD] leaveGroup rank=" << req.rank
                  << " group=" << req.group_id << " ignored: already left";
        return result;
    }

    LOG(INFO) << "[COORD] leaveGroup rank=" << req.rank
              << " group=" << req.group_id
              << " old_status=" << static_cast<int>(member.status);

    member.status = GroupMemberStatus::kLeft;
    member.endpoint_epoch = kInvalidEpoch;
    view.epoch++;

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
        if (peer.link_status.hasIndex(rank)) peer.link_status[rank] = 0;
    }

    // Mark the failed rank as inactive in every group it still belongs to.
    std::vector<GroupId> groups_to_erase;
    for (auto& [group_id, view] : group_views_) {
        if (!rankInValidRange(rank)) continue;
        auto& member = view.member(rank);
        if (member.status == GroupMemberStatus::kActive) {
            member.status = GroupMemberStatus::kInactive;
            view.epoch++;
            effects.push_back(ViewUpdateEffect{view, {}, BootstrapAckRoute{}});
        }
        if (canEraseGroup(view)) {
            groups_to_erase.push_back(group_id);
        }
    }
    for (GroupId group_id : groups_to_erase) {
        eraseGroup(group_id, effects);
    }

    effects.push_back(makeRankStateEffect(rank));
    updateRankHealth(effects);
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
    return ranks_[a].link_status.hasIndex(b) &&
           ranks_[b].link_status.hasIndex(a) && ranks_[a].link_status[b] != 0 &&
           ranks_[b].link_status[a] != 0;
}

// Find the largest all-to-all mutually connected set of ranks.
// This is a maximum clique problem on the mutual-connectivity graph.
// Uses exhaustive branch-and-bound search with a step limit to prevent
// hanging on pathological topologies.
//
// Tie-breaking: prefer cliques containing Rank 0 (it hosts the
// Coordinator and Store), then lexicographically smallest.
std::vector<GlobalRank> CentralizedCoordinatorStateMachine::findHealthySet()
    const {
    std::vector<GlobalRank> candidates;
    for (GlobalRank i{0}; i < max_world_size_; ++i) {
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

    // Debug: show all rank link_status and healthy_set
    {
        std::ostringstream oss;
        oss << "[COORD] updateRankHealth healthy_set={";
        for (size_t i = 0; i < healthy_set.size(); ++i) {
            if (i > 0) oss << ",";
            oss << healthy_set[i];
        }
        oss << "} link_status_per_rank=";
        for (GlobalRank r{0}; r < 4 && r < max_world_size_; ++r) {
            oss << " rank" << r << "=[";
            for (GlobalRank p{0}; p < 4 && p < max_world_size_; ++p) {
                if (p > 0) oss << ",";
                oss << (int)ranks_[r].link_status[p];
            }
            oss << "]";
        }
        LOG(INFO) << oss.str();
    }

    // 1. Update per-rank HEALTHY / SYNCED state.
    for (GlobalRank i{0}; i < max_world_size_; ++i) {
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
    //    set. However, during bootstrap / BootstrapSyncing we do NOT do this:
    //    we wait for full mutual connectivity and let waitUntilGroupReady()
    //    time out if a peer is truly dead.
    for (auto& [group_id, view] : group_views_) {
        if (!group_auto_deactivate_[group_id]) continue;
        if (view.status != GroupStatus::Ready) continue;
        bool changed = false;
        for (GlobalRank i{0}; i < max_world_size_; ++i) {
            if (!view.member(i).isActive()) continue;
            bool in_healthy = std::find(healthy_set.begin(), healthy_set.end(),
                                        i) != healthy_set.end();
            if (!in_healthy) {
                view.member(i).status = GroupMemberStatus::kInactive;
                changed = true;
                LOG(INFO) << "[COORD] auto_deactivate group=" << group_id
                          << " rank=" << i;
            }
        }
        if (changed) {
            view.epoch++;
            effects.push_back(ViewUpdateEffect{view, {}, BootstrapAckRoute{}});
            LOG(INFO) << "[COORD] auto_deactivate view update group="
                      << group_id << " epoch=" << view.epoch;
        }
    }

    checkGroupTransitions(effects);
}

// Private: activatable predicates

bool CentralizedCoordinatorStateMachine::rankInValidRange(
    GlobalRank rank) const {
    return rank >= 0 && rank < max_world_size_;
}

bool CentralizedCoordinatorStateMachine::isActivatableSet(
    GroupId group_id, const std::vector<GlobalRank>& new_ranks,
    const GroupView& old_view) const {
    // Build the future active set: old active ∪ new ranks.
    std::vector<GlobalRank> future_active;
    for (GlobalRank i{0}; i < max_world_size_; ++i) {
        if (old_view.member(i).isActive()) {
            future_active.push_back(i);
        }
    }
    for (GlobalRank r : new_ranks) {
        if (!old_view.member(r).isActive()) {
            future_active.push_back(r);
        }
    }

    // Every rank in the future set must be activatable with respect to the
    // full future set.  This guarantees all-to-all mutual connectivity:
    // old <-> old, old <-> new, and new <-> new.
    for (GlobalRank r : future_active) {
        if (!isRankActivatable(group_id, r, future_active)) return false;
    }
    return true;
}

bool CentralizedCoordinatorStateMachine::isRankActivatable(
    GroupId group_id, GlobalRank rank,
    const std::vector<GlobalRank>& peer_ranks) const {
    if (!rankInValidRange(rank)) return false;
    if (ranks_[rank].state != RankState::HEALTHY) return false;

    for (GlobalRank other : peer_ranks) {
        if (other == rank) continue;
        if (!isMutuallyConnected(rank, other)) return false;
    }

    auto group = group_views_.find(group_id);
    if (group == group_views_.end()) return false;

    const auto& member = group->second.member(rank);
    return (member.status == GroupMemberStatus::kActive ||
            member.status == GroupMemberStatus::kInactive) &&
           member.endpoint_epoch != kInvalidEpoch &&
           member.agent_session_epoch == ranks_[rank].agent_session_epoch;
}

// checkGroupTransitions - bootstrap state machine driver.
//
// Group lifecycle:
//   joinGroup() creates a group in Bootstrapping status.
//   Once all active ranks have valid endpoints, are HEALTHY, and are mutually
//   TE-connected, this function transitions to BootstrapSyncing and broadcasts
//   a ViewUpdate to collect ACKs from all active ranks (a 2PC barrier).
//   Once all ACKs arrive, the group transitions to Ready.
//   waitUntilGroupReady() unblocks only when status == Ready.
void CentralizedCoordinatorStateMachine::checkGroupTransitions(
    std::vector<CoordinatorEffect>& effects) {
    for (auto& [group_id, view] : group_views_) {
        if (view.status == GroupStatus::Bootstrapping) {
            // Collect all active ranks.
            std::vector<GlobalRank> peer_ranks;
            bool has_any_active = false;
            for (GlobalRank i{0}; i < max_world_size_; ++i) {
                if (!view.member(i).isActive()) continue;
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
                view.epoch = (view.epoch == kInvalidEpoch) ? 1 : view.epoch + 1;

                auto acks_needed =
                    computeRequiredViewAcks(view, view, kInvalidGlobalRank);
                pending_bootstrap_acks_[group_id] =
                    std::unordered_set<GlobalRank>(acks_needed.begin(),
                                                   acks_needed.end());

                effects.push_back(
                    ViewUpdateEffect{view, acks_needed, BootstrapAckRoute{}});
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
                    ViewUpdateEffect{view, {}, BootstrapAckRoute{}});
                LOG(INFO) << "[COORD] group=" << group_id
                          << " transitioned to Ready epoch=" << view.epoch;
            }
        }
    }
}

// Private: joinGroup

bool CentralizedCoordinatorStateMachine::joinGroup(
    const GroupView& group, bool auto_deactivate, JoinGroupResponse& response) {
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
            view.member(r).status = GroupMemberStatus::kActive;
        }
        view.status = GroupStatus::Bootstrapping;
        group_views_[group_id] = std::move(view);
        group_auto_deactivate_[group_id] = auto_deactivate;
        response.success = true;
        return true;
    }

    // Group already exists -> validate rank_order prefix.
    const auto& existing_order = group_views_[group_id].rank_order;
    const auto& new_order = group.rank_order;

    if (new_order.size() > existing_order.size()) {
        response.success = false;
        response.reject_reason =
            "new rank_order is longer than existing (must be a prefix)";
        return false;
    }

    for (InGroupRank i : new_order.indices()) {
        if (new_order[i] != existing_order[i]) {
            response.success = false;
            response.reject_reason =
                "new rank_order is not a prefix of existing rank_order";
            return false;
        }
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
        if (old_view.member(i).isActive() || new_view.member(i).isActive()) {
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
    for (GlobalRank i : globalRankRange(max_world_size_)) {
        resp.all_rank_states[i] = ranks_[i].state;
    }

    // All groups (view includes rank_order and member state).
    for (const auto& [gid, view] : group_views_) {
        resp.groups.push_back(view);
    }

    // All rank connection metadata (for LinkManager).
    for (GlobalRank i : globalRankRange(max_world_size_)) {
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

}  // namespace mooncake
