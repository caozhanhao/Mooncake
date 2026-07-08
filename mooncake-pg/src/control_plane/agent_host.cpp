#include "control_plane/agent_host.h"

#include <exception>
#include <stdexcept>

#include <glog/logging.h>

#include "mooncake_backend.h"
#include "control_plane/link_manager.h"
#include "control_plane/rpc_runtime.h"

namespace mooncake {

// AgentRpcServiceImpl

void AgentRpcServiceImpl::onPeerJoined(PeerJoinedPush push) {
    host_.postPeerJoined(std::move(push));
}

void AgentRpcServiceImpl::onRankStateUpdate(RankStateUpdatePush push) {
    host_.postRankStateUpdate(std::move(push));
}

void AgentRpcServiceImpl::onViewUpdate(coro_rpc::context<ViewUpdateAck> ctx,
                                       ViewUpdatePush push) {
    host_.postViewUpdate(std::move(ctx), std::move(push));
}

// AgentHost

AgentHost::AgentHost(c10::intrusive_ptr<c10d::Store> store,
                     const std::string& host_ip, GlobalRank rank,
                     int max_world_size, LinkManager& link_manager)
    : agent_(rank, max_world_size),
      executor_("AgentHost"),
      link_manager_(link_manager),
      store_(std::move(store)),
      host_ip_(host_ip),
      rank_(rank),
      max_world_size_(max_world_size),
      rpc_client_(std::make_unique<RpcClient>()) {}

AgentHost::~AgentHost() { shutdown(); }

void AgentHost::start() {
    // Register LinkManager event callback.
    link_manager_.setEventCallback(
        [this](TELinkEvent event) { postTELinkEvent(std::move(event)); });

    // Start Agent RPC server.
    rpc_server_ = std::make_unique<RpcServer>(/*port=*/0, /*thread_num=*/2);
    rpc_impl_ = std::make_unique<AgentRpcServiceImpl>(*this);
    rpc_server_->registerHandler<&AgentRpcService::onPeerJoined,
                                 &AgentRpcService::onRankStateUpdate,
                                 &AgentRpcService::onViewUpdate>(
        rpc_impl_.get());
    bool server_started = rpc_server_->start();
    if (!server_started) {
        LOG(ERROR) << "AgentHost: failed to start RPC server rank=" << rank_;
    }

    // Read Coordinator address from Store with backoff poll.
    // We cannot guarantee the Coordinator is initialised before this
    // Agent  - non-rank-0 processes may reach this point before rank 0
    // has written the key.  wait() blocks the predicate thread until the
    // key is set (or the Store connection fails), so we run it inside the
    // BackoffWaiter predicate which catches exceptions and retries.
    BackoffWaiter waiter(BackoffWaiterConfig::constantSleep(
        AgentHost::kCoordinatorAddrPollInterval));

    bool found = waiter.wait_for(AgentHost::kCoordinatorAddrTimeout, [this]() {
        try {
            store_->wait({"coordinator_addr"});
            coordinator_addr_ = store_->get_to_str("coordinator_addr");
            return !coordinator_addr_.empty();
        } catch (const std::exception& e) {
            LOG(WARNING) << "AgentHost: store access failed rank=" << rank_
                         << ": " << e.what();
        }
        return false;
    });

    if (!found) {
        LOG(FATAL) << "AgentHost: timed out after "
                   << std::chrono::duration_cast<std::chrono::seconds>(
                          AgentHost::kCoordinatorAddrTimeout)
                          .count()
                   << "s waiting for coordinator_addr in Store";
    }

    // Set up periodic tick.
    executor_.setTickCallback([this]() { tick(); });
    executor_.start();

    // Initial registration.
    executor_.post([this]() { startRegisterRpc(); });
}

void AgentHost::shutdown() {
    // Stop RPC server first — no new pushes accepted, in-flight handlers
    // are guaranteed to have finished posting to the executor.
    if (rpc_server_) rpc_server_->shutdown();
    // Drain the executor next so that queued tasks (including leaveGroup /
    // leaveGroup sends) finish before we tear down the RpcClient.
    executor_.shutdown();
    // Mark RpcClient as shutting down last.  In-flight async coroutines on the
    // global I/O executor may still fire after this point, but their connection
    // errors are suppressed by the VLOG in rpc_runtime.h.
    if (rpc_client_) rpc_client_->shutdown();
    // Drop the shared-state reference.  At this point the executor is stopped
    // and in-flight coroutines have been told to drop, so no code path will
    // dereference rpc_client_.
    if (rpc_client_) rpc_client_.reset();
    link_manager_.setEventCallback(nullptr);
}

// Agent interface: Bootstrap

bool AgentHost::waitUntilRegistered(std::chrono::milliseconds timeout) {
    auto promise = std::make_shared<std::promise<void>>();
    auto future = promise->get_future();

    executor_.post([this, promise]() {
        if (registration_done_) {
            promise->set_value();
        } else {
            registration_promises_.push_back(promise);
        }
    });

    if (future.wait_for(timeout) != std::future_status::ready) {
        // Timeout: remove the dangling promise on the executor thread.
        executor_.post(
            [this, promise]() { std::erase(registration_promises_, promise); });
        return false;
    }
    return true;
}

// Block until the group reaches Ready status (all active ranks ACKed the
// bootstrap ViewUpdate).  Throws on timeout.
//
// Note: if a peer dies during the BootstrapSyncing phase, the Coordinator
// will not transition the group to Ready, and this call will hang until
// the timeout expires.  The caller should handle this as a bootstrap failure.
GroupView AgentHost::waitUntilGroupReady(GroupId group_id,
                                         std::chrono::milliseconds timeout) {
    auto promise = std::make_shared<std::promise<GroupView>>();
    auto future = promise->get_future();

    executor_.post([this, group_id, promise]() {
        auto view = agent_.getGroupView(group_id);
        if (view.status == GroupStatus::Ready) {
            promise->set_value(view);
        } else {
            group_ready_promises_[group_id].push_back(promise);
        }
    });

    if (future.wait_for(timeout) != std::future_status::ready) {
        // Clean up the dangling promise before throwing.
        executor_.post([this, group_id, promise]() {
            auto it = group_ready_promises_.find(group_id);
            if (it != group_ready_promises_.end()) {
                auto& vec = it->second;
                vec.erase(std::remove(vec.begin(), vec.end(), promise),
                          vec.end());
                if (vec.empty()) group_ready_promises_.erase(it);
            }
        });
        throw std::runtime_error("waitUntilGroupReady timed out for group " +
                                 std::to_string(group_id));
    }
    return future.get();
}

void AgentHost::waitUntilRankActive(GroupId group_id, GlobalRank rank,
                                    std::chrono::milliseconds timeout) {
    auto promise = std::make_shared<std::promise<void>>();
    auto future = promise->get_future();

    executor_.post([this, group_id, rank, promise]() {
        auto view = agent_.getGroupView(group_id);
        if (view.member(rank).isActive()) {
            promise->set_value();
        } else {
            rank_active_promises_[group_id][rank].push_back(promise);
        }
    });

    if (future.wait_for(timeout) != std::future_status::ready) {
        executor_.post([this, group_id, rank, promise]() {
            auto it = rank_active_promises_.find(group_id);
            if (it != rank_active_promises_.end()) {
                auto rit = it->second.find(rank);
                if (rit != it->second.end()) {
                    auto& vec = rit->second;
                    vec.erase(std::remove(vec.begin(), vec.end(), promise),
                              vec.end());
                    if (vec.empty()) it->second.erase(rit);
                }
                if (it->second.empty()) rank_active_promises_.erase(it);
            }
        });
        throw std::runtime_error("waitUntilRankActive timed out for rank " +
                                 std::to_string(rank.value) + " in group " +
                                 std::to_string(group_id));
    }
}

// Agent interface: Group management

void AgentHost::doJoinGroup(GroupView group, bool auto_deactivate,
                            c10::intrusive_ptr<MooncakeBackend> backend) {
    auto group_id = group.group_id;
    backends_.insert_or_assign(group_id, std::move(backend));
    agent_.joinGroup(group, auto_deactivate);

    JoinGroupRequest req;
    req.rank = rank_;
    req.agent_session_epoch = agent_.getAgentSessionEpoch();
    req.group = std::move(group);
    req.auto_deactivate = auto_deactivate;

    auto resp = rpc_client_->call<&CoordinatorRpcService::joinGroup>(
        coordinator_addr_, std::move(req));

    LOG(INFO) << "[AGENT] joinGroup rank=" << rank_ << " group=" << group_id
              << " success=" << resp.success;

    if (!resp.success) {
        LOG(ERROR) << "AgentHost: joinGroup failed for group " << group_id
                   << ": " << resp.reject_reason;
        auto it = group_ready_promises_.find(group_id);
        if (it != group_ready_promises_.end()) {
            for (auto& p : it->second) {
                p->set_exception(std::make_exception_ptr(std::runtime_error(
                    "joinGroup rejected: " + resp.reject_reason)));
            }
            group_ready_promises_.erase(it);
        }
    }
}

void AgentHost::joinGroup(const GroupView& group, bool auto_deactivate,
                          c10::intrusive_ptr<MooncakeBackend> backend) {
    executor_.postAndWait([this, group = group, auto_deactivate,
                           backend = std::move(backend)]() mutable {
        doJoinGroup(std::move(group), auto_deactivate, std::move(backend));
    });
}

void AgentHost::leaveGroup(GroupId group_id) {
    executor_.post([this, group_id]() {
        LOG(INFO) << "[AGENT] leaveGroup rank=" << rank_
                  << " group=" << group_id;
        agent_.leaveGroup(group_id);
        backends_.erase(group_id);

        LeaveGroupRequest req;
        req.group_id = group_id;
        req.rank = rank_;
        req.agent_session_epoch = agent_.getAgentSessionEpoch();
        rpc_client_->send<&CoordinatorRpcService::leaveGroup>(coordinator_addr_,
                                                              req);
    });
}

GroupView AgentHost::getGroupView(GroupId group_id) {
    auto promise = std::make_shared<std::promise<GroupView>>();
    auto future = promise->get_future();
    executor_.post([this, group_id, promise]() {
        promise->set_value(agent_.getGroupView(group_id));
    });
    return future.get();
}

void AgentHost::doPublishLocalEndpoint(GroupEndpointPublication endpoint) {
    PublishEndpointRequest req;
    req.rank = rank_;
    req.agent_session_epoch = agent_.getAgentSessionEpoch();
    req.endpoints.push_back(std::move(endpoint));
    rpc_client_->call<&CoordinatorRpcService::publishEndpoint>(
        coordinator_addr_, std::move(req));
}

void AgentHost::publishLocalEndpoint(GroupEndpointPublication endpoint) {
    executor_.postAndWait([this, endpoint = std::move(endpoint)]() mutable {
        doPublishLocalEndpoint(std::move(endpoint));
    });
}

// Agent interface: Membership proposals (synchronous)

ProposeViewUpdateResponse AgentHost::proposeActivate(
    GroupId group_id, const std::vector<GlobalRank>& ranks) {
    ProposeViewUpdateRequest req;
    req.group_id = group_id;
    req.source_rank = rank_;
    req.agent_session_epoch = agent_.getAgentSessionEpoch();
    req.requested_ranks = ranks;
    req.is_activate = true;
    return rpc_client_->call<&CoordinatorRpcService::proposeViewUpdate>(
        coordinator_addr_, req);
}

ProposeViewUpdateResponse AgentHost::proposeDeactivate(
    GroupId group_id, const std::vector<GlobalRank>& ranks) {
    ProposeViewUpdateRequest req;
    req.group_id = group_id;
    req.source_rank = rank_;
    req.agent_session_epoch = agent_.getAgentSessionEpoch();
    req.requested_ranks = ranks;
    req.is_activate = false;
    return rpc_client_->call<&CoordinatorRpcService::proposeViewUpdate>(
        coordinator_addr_, req);
}

// Agent interface: Transfer observation (thread-safe)

void AgentHost::pushTransferObservation(
    GroupId group_id, IndexedVector<uint8_t, GlobalRankTag> attempted_ranks,
    IndexedVector<uint8_t, GlobalRankTag> failed_ranks,
    IndexedVector<uint8_t, GlobalRankTag> succeeded_ranks) {
    observation_queue_.enqueue(TransferObservationEvent{
        group_id, std::move(attempted_ranks), std::move(failed_ranks),
        std::move(succeeded_ranks)});
}

// RPC push callbacks

void AgentHost::postPeerJoined(PeerJoinedPush push) {
    executor_.post([this, push = std::move(push)]() {
        runEffects(agent_.handlePeerJoined(push));
    });
}

void AgentHost::postRankStateUpdate(RankStateUpdatePush push) {
    executor_.post([this, push = std::move(push)]() {
        runEffects(agent_.handleRankStateUpdate(push));
    });
}

void AgentHost::postViewUpdate(coro_rpc::context<ViewUpdateAck> ctx,
                               ViewUpdatePush push) {
    auto group_id = push.group_id;
    auto epoch = push.view.epoch;
    LOG(INFO) << "[AGENT] postViewUpdate rank=" << rank_
              << " group=" << group_id << " epoch=" << epoch
              << " status=" << static_cast<int>(push.view.status);

    executor_.post([this, ctx = std::move(ctx), push = std::move(push),
                    group_id, epoch]() mutable {
        // Snapshot ranks that transition from inactive -> active in this
        // update. Must be done on the executor thread because it reads
        // agent_.groups_, which is only safe to access from the executor.
        auto old_view = agent_.getGroupView(group_id);
        std::vector<GlobalRank> newly_activated;
        for (InGroupRank i : push.view.rank_order.indices()) {
            GlobalRank gr = push.view.globalRank(i);
            if (!old_view.member(gr).isActive() &&
                push.view.member(gr).isActive()) {
                newly_activated.push_back(gr);
            }
        }

        runEffects(agent_.handleViewUpdate(push));

        // Wake up waitUntilGroupReady callers when group reaches Ready status.
        if (push.view.status == GroupStatus::Ready) {
            auto it = group_ready_promises_.find(group_id);
            if (it != group_ready_promises_.end()) {
                for (auto& p : it->second) {
                    p->set_value(push.view);
                }
                group_ready_promises_.erase(it);
            }
        }

        // Wake up waitUntilRankActive callers for newly activated ranks.
        for (GlobalRank gr : newly_activated) {
            auto it = rank_active_promises_.find(group_id);
            if (it != rank_active_promises_.end()) {
                auto rit = it->second.find(gr);
                if (rit != it->second.end()) {
                    for (auto& p : rit->second) {
                        p->set_value();
                    }
                    it->second.erase(rit);
                }
                if (it->second.empty()) rank_active_promises_.erase(it);
            }
        }

        ctx.response_msg(ViewUpdateAck{.rank = rank_,
                                       .group_id = group_id,
                                       .epoch = epoch,
                                       .applied = true,
                                       .error_msg = ""});
    });
}

// LinkManager event

void AgentHost::postTELinkEvent(TELinkEvent event) {
    executor_.post([this, event = std::move(event)]() {
        if (event.kind == TELinkEvent::Kind::LinkUp) {
            if (event.target_id.has_value()) {
                // LinkManager already updated the read model (publishLinkUp)
                // before emitting this event  - just update the state machine.
                runEffects(agent_.handleLinkStateChanged(event.peer, true));
            } else {
                LOG(WARNING) << "AgentHost: LinkUp event for peer "
                             << event.peer << " without target_id; ignoring.";
                return;
            }
        } else {
            // LinkDown: LinkManager already called publishLinkDown in
            // tearDownPeerLink.  P2P reset is modeled as a
            // ResetPeerP2PState effect from handleLinkStateChanged.
            runEffects(agent_.handleLinkStateChanged(event.peer, false));
        }

        // Report the link state change event to the Coordinator so it can
        // update the authoritative link_status.  This is event-driven (only
        // fires on actual LinkUp/LinkDown transitions), avoiding the
        // periodic-overwrite problem that heartbeat-based link_status had.
        if (rpc_client_ && !coordinator_addr_.empty()) {
            rpc_client_->send<&CoordinatorRpcService::reportLinkStateChange>(
                coordinator_addr_,
                LinkStateChangeReport{
                    .reporter_rank = rank_,
                    .peer = event.peer,
                    .is_up = (event.kind == TELinkEvent::Kind::LinkUp),
                    .agent_session_epoch = agent_.getAgentSessionEpoch(),
                });
        }
    });
}

// Internal: startRegisterRpc

void AgentHost::startRegisterRpc() {
    // Avoid duplicate registration RPCs.  This also covers the case where a
    // heartbeat response callback asks for re-registration while another
    // registration is already in flight.
    if (agent_.getCoordinatorConnection() ==
        AgentStateMachine::CoordinatorConnection::Registering) {
        return;
    }
    agent_.setCoordinatorRegistering();

    RegisterRequest req;
    req.rank = rank_;
    req.agent_addr = rpc_server_->getListenAddr(host_ip_);
    req.te_server_name = link_manager_.localServerName();
    req.warmup_recv_addr = link_manager_.getWarmupRecvAddr();
    req.agent_session_epoch = ++agent_session_epoch_;
    agent_.setAgentSessionEpoch(agent_session_epoch_);

    rpc_client_->callAsync<&CoordinatorRpcService::registerAgent>(
        coordinator_addr_, std::move(req), [this](RegisterResponse resp) {
            executor_.post([this, resp = std::move(resp)]() mutable {
                auto effects = agent_.applyRegisterResponse(resp);
                runEffects(effects);

                if (resp.success) {
                    if (!registration_done_) {
                        registration_done_ = true;
                        for (auto& p : registration_promises_) {
                            p->set_value();
                        }
                        registration_promises_.clear();
                    }

                    // Re-publish all local backends' endpoints after (re-)reg.
                    // (Old session endpoints were cleared by Coordinator.)
                    forEachBackend([&](auto backend) {
                        doPublishLocalEndpoint(
                            backend->buildEndpointMetadata());
                    });
                } else {
                    LOG(ERROR)
                        << "AgentHost: registerAgent failed: " << resp.error_msg
                        << " (will retry after heartbeat interval; if this "
                           "persists, the Coordinator may be rejecting a "
                           "replacement rank before the old one times out)";
                }
            });
        });
}

// Internal: tick

void AgentHost::tick() {
    if (!rpc_client_) return;

    // Consume transfer observation queue.
    TransferObservationEvent event;
    while (observation_queue_.try_dequeue(event)) {
        auto effects = agent_.processTransferObservation(event);
        for (auto& effect : effects) {
            if (auto* e = std::get_if<SendTransferObservation>(&effect)) {
                e->request.agent_session_epoch = agent_.getAgentSessionEpoch();
                e->request.epoch =
                    agent_.getGroupView(e->request.group_id).epoch;
            }
        }
        runEffects(effects);
    }

    // Check if reconnect is needed.
    if (agent_.getCoordinatorConnection() ==
        AgentStateMachine::CoordinatorConnection::Disconnected) {
        if (rpc_client_->tryReconnect(coordinator_addr_)) {
            startRegisterRpc();
        }
        return;
    }

    // Do not send heartbeats while a registration is in flight; wait for the
    // registerAgent response first.
    if (agent_.getCoordinatorConnection() ==
        AgentStateMachine::CoordinatorConnection::Registering) {
        return;
    }

    // Send heartbeat.
    auto req = agent_.buildHeartbeat();
    req.agent_session_epoch = agent_.getAgentSessionEpoch();

    rpc_client_->callAsync<&CoordinatorRpcService::heartbeat>(
        coordinator_addr_, std::move(req), [this](HeartbeatResponse resp) {
            executor_.post([this, resp]() {
                if (resp.require_reregister) {
                    runEffects(agent_.prepareCleanSlateRegister());
                    startRegisterRpc();
                }
            });
        });
}

// Internal: runEffects

void AgentHost::runEffects(const AgentApplyResult& effects) {
    for (const auto& effect : effects) {
        std::visit(
            overloaded{
                [this](const SendTransferObservation& e) {
                    rpc_client_->send<
                        &CoordinatorRpcService::reportTransferObservation>(
                        coordinator_addr_, e.request);
                },
                [this](const EnablePeerProbe& e) {
                    link_manager_.enablePeerProbe(e.rank, e.te_server_name,
                                                  e.warmup_recv_addr);
                },
                [this](const DisconnectLink& e) {
                    link_manager_.disconnect(e.peer);
                },
                [this](const StopReconnect& e) {
                    link_manager_.stopReconnect(e.peer);
                },
                [this](const ClearPeerMetadata& e) {
                    link_manager_.publishLinkDown(e.peer);
                },
                [this](const DisconnectAllLinks&) {
                    for (GlobalRank i{0}; i < max_world_size_; ++i) {
                        if (i != rank_) {
                            link_manager_.disconnect(i);
                        }
                    }
                },
                [this](const ClearAllPeerMetadata&) {
                    for (GlobalRank i{0}; i < max_world_size_; ++i) {
                        if (i != rank_) {
                            link_manager_.publishLinkDown(i);
                        }
                    }
                },
                [this](const PublishRankStateSnapshot& e) {
                    link_manager_.setRankStates(e.states);
                },
                [this](const ApplyViewToBackend& e) {
                    withBackend(e.group_id, [&](auto backend) {
                        backend->applyViewChange(e.view);
                    });
                },
                [this](const MarkBackendViewStale& e) {
                    withBackend(e.group_id, [&](auto backend) {
                        backend->markViewStale();
                    });
                },
                [this](const NotifyTEUnreachable& e) {
                    forEachBackend([&](auto backend) {
                        backend->onPeerLinkReset(e.peer);
                    });
                },
            },
            effect);
    }
}

}  // namespace mooncake
