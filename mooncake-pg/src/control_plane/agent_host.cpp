#include "control_plane/agent_host.h"

#include <stdexcept>

#include <glog/logging.h>

#include "mooncake_backend.h"
#include "control_plane/te_link_manager.h"
#include "control_plane/rpc_runtime.h"

namespace mooncake {

// =========================================================================
// AgentRpcServiceImpl
// =========================================================================

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

// =========================================================================
// AgentHost
// =========================================================================

AgentHost::AgentHost(c10::intrusive_ptr<c10d::Store> store,
                     const std::string& host_ip, GlobalRank rank,
                     int max_world_size, TELinkManager& te_link_manager)
    : agent_(rank, max_world_size),
      executor_("AgentHost"),
      te_link_manager_(te_link_manager),
      store_(std::move(store)),
      host_ip_(host_ip),
      rank_(rank),
      max_world_size_(max_world_size),
      rpc_client_(std::make_unique<RpcClient>()) {}

AgentHost::~AgentHost() { shutdown(); }

void AgentHost::start() {
    // Register TELinkManager event callback.
    te_link_manager_.setEventCallback(
        [this](TELinkEvent event) { postTELinkEvent(std::move(event)); });

    // Start Agent RPC server.
    rpc_server_ = std::make_unique<RpcServer>(/*port=*/0, /*thread_num=*/2);
    auto rpc_impl = std::make_unique<AgentRpcServiceImpl>(*this);
    rpc_server_->registerHandler<AgentRpcService>(std::move(rpc_impl));
    if (!rpc_server_->start()) {
        LOG(ERROR) << "AgentHost: failed to start RPC server";
    }

    // Read Coordinator address from Store with backoff poll.
    // We cannot guarantee the Coordinator is initialised before this
    // Agent — non-rank-0 processes may reach this point before rank 0
    // has written the key.  check() is non-blocking; the predicate
    // catches exceptions so a transient Store failure retries instead
    // of crashing the process.
    BackoffWaiter waiter(BackoffWaiterConfig::constantSleep(
        AgentHost::kCoordinatorAddrPollInterval));

    bool found = waiter.wait_for(AgentHost::kCoordinatorAddrTimeout, [this]() {
        try {
            if (store_->check({"coordinator_addr"})) {
                coordinator_addr_ = store_->get_to_str("coordinator_addr");
                return !coordinator_addr_.empty();
            }
        } catch (const std::exception& e) {
            LOG(WARNING) << "AgentHost: store access failed: " << e.what();
        }
        return false;
    });

    if (!found) {
        LOG(FATAL) << "AgentHost: timed out after "
                   << std::chrono::duration_cast<std::chrono::seconds>(
                          AgentHost::kCoordinatorAddrTimeout)
                          .count()
                   << "s waiting for coordinator_addr in Store";
        std::abort();
    }

    // Set up periodic tick.
    executor_.setTickCallback([this]() { tick(); });

    executor_.start();

    // Initial registration.
    executor_.post([this]() { startRegisterRpc(); });
}

void AgentHost::shutdown() {
    // Release our reference to the RPC client pool.  In-flight coroutines
    // on the global I/O executor hold their own shared_ptr copies of the
    // underlying coro_rpc_client objects, so they complete safely.
    if (rpc_client_) rpc_client_.reset();
    // Drain the executor — any callbacks already posted will complete.
    executor_.shutdown();
    te_link_manager_.setEventCallback(nullptr);
    if (rpc_server_) rpc_server_->shutdown();
}

// =========================================================================
// Agent interface: Bootstrap
// =========================================================================

bool AgentHost::waitUntilRegistered(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(registration_mutex_);
    return registration_cv_.wait_for(lock, timeout,
                                     [this] { return registration_done_; });
}

GroupView AgentHost::waitUntilGroupReady(GroupId group_id,
                                         std::chrono::milliseconds timeout) {
    auto promise = std::make_shared<std::promise<GroupView>>();
    auto future = promise->get_future();

    executor_.post([this, group_id, promise]() {
        auto view = agent_.getGroupView(group_id);
        if (view.epoch != kInvalidEpoch) {
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

// =========================================================================
// Agent interface: Group management
// =========================================================================

void AgentHost::registerGroup(GroupDeclaration declaration,
                              MooncakeBackend* backend) {
    auto group_id = declaration.descriptor.group_id;
    executor_.post([this, declaration = std::move(declaration), group_id,
                    backend]() mutable {
        backends_[group_id] = backend;
        agent_.registerGroup(declaration);

        DeclareGroupRequest req;
        req.rank = rank_;
        req.agent_session_epoch = agent_.getAgentSessionEpoch();
        req.group = std::move(declaration);

        rpc_client_->callAsync<&CoordinatorRpcService::declareGroup>(
            coordinator_addr_, std::move(req),
            [this, group_id](DeclareGroupResponse resp) {
                executor_.post([this, group_id, resp = std::move(resp)]() {
                    if (resp.success) {
                        // Use the descriptor (with rank_order) that was stored
                        // in the agent during registerGroup, not an empty one.
                        ViewUpdatePush push{
                            resp.current_view.group_id, {}, resp.current_view};
                        if (auto* desc = agent_.getGroupDescriptor(group_id)) {
                            push.descriptor = *desc;
                        } else {
                            push.descriptor.group_id = group_id;
                        }
                        runEffects(agent_.handleViewUpdate(push));

                        // Wake up waitUntilGroupReady callers.
                        auto it = group_ready_promises_.find(group_id);
                        if (it != group_ready_promises_.end()) {
                            for (auto& p : it->second) {
                                p->set_value(resp.current_view);
                            }
                            group_ready_promises_.erase(it);
                        }
                    } else {
                        LOG(ERROR)
                            << "AgentHost: declareGroup failed for group "
                            << group_id << ": " << resp.reject_reason;
                        // Fulfill pending waitUntilGroupReady promises so
                        // callers don't hang until timeout (30s).
                        auto it = group_ready_promises_.find(group_id);
                        if (it != group_ready_promises_.end()) {
                            for (auto& p : it->second) {
                                try {
                                    throw std::runtime_error(
                                        "declareGroup rejected: " +
                                        resp.reject_reason);
                                } catch (...) {
                                    p->set_exception(std::current_exception());
                                }
                            }
                            group_ready_promises_.erase(it);
                        }
                    }
                });
            });
    });
}

void AgentHost::unregisterGroup(GroupId group_id) {
    executor_.post([this, group_id]() {
        agent_.unregisterGroup(group_id);
        backends_.erase(group_id);
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

void AgentHost::publishLocalEndpoint(GroupEndpointPublication endpoint) {
    executor_.post([this, endpoint = std::move(endpoint)]() mutable {
        PublishEndpointRequest req;
        req.rank = rank_;
        req.agent_session_epoch = agent_.getAgentSessionEpoch();
        req.endpoints.push_back(std::move(endpoint));
        rpc_client_->send<&CoordinatorRpcService::publishEndpoint>(
            coordinator_addr_, std::move(req));
    });
}

// =========================================================================
// Agent interface: Membership proposals (synchronous)
// =========================================================================

ProposeViewUpdateResponse AgentHost::proposeActivate(
    GroupId group_id, const std::vector<GlobalRank>& ranks) {
    ProposeViewUpdateRequest req;
    req.group_id = group_id;
    req.source_rank = rank_;
    req.agent_session_epoch = agent_.getAgentSessionEpoch();
    req.target_ranks = ranks;
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
    req.target_ranks = ranks;
    req.is_activate = false;
    return rpc_client_->call<&CoordinatorRpcService::proposeViewUpdate>(
        coordinator_addr_, req);
}

// =========================================================================
// Agent interface: Transfer observation (thread-safe)
// =========================================================================

void AgentHost::pushTransferObservation(GroupId group_id,
                                        std::vector<uint8_t> attempted_ranks,
                                        std::vector<uint8_t> failed_ranks,
                                        std::vector<uint8_t> succeeded_ranks) {
    observation_queue_.enqueue(TransferObservationEvent{
        group_id, std::move(attempted_ranks), std::move(failed_ranks),
        std::move(succeeded_ranks)});
}

// =========================================================================
// RPC push callbacks
// =========================================================================

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
    executor_.post([this, ctx = std::move(ctx), push = std::move(push),
                    group_id, epoch]() mutable {
        runEffects(agent_.handleViewUpdate(push));
        ctx.response_msg(ViewUpdateAck{.rank = rank_,
                                       .group_id = group_id,
                                       .epoch = epoch,
                                       .applied = true,
                                       .error_msg = ""});
    });
}

// =========================================================================
// TELinkManager event
// =========================================================================

void AgentHost::postTELinkEvent(TELinkEvent event) {
    executor_.post([this, event = std::move(event)]() {
        if (event.kind == TELinkEvent::Kind::LinkUp) {
            if (event.target_id.has_value()) {
                // TELinkManager already updated the read model (publishLinkUp)
                // before emitting this event — just update the state machine.
                runEffects(agent_.handleLinkStateChanged(event.peer, true));
            } else {
                LOG(WARNING) << "AgentHost: LinkUp event for peer "
                             << event.peer << " without target_id; ignoring.";
            }
        } else {
            // LinkDown: TELinkManager already called publishLinkDown in
            // tearDownPeerLink.  We just need the state machine update and
            // P2P reset fanout to all backends.
            forEachBackend([&](MooncakeBackend* backend) {
                backend->onPeerLinkReset(event.peer);
            });
            runEffects(agent_.handleLinkStateChanged(event.peer, false));
        }
    });
}

// =========================================================================
// Internal: startRegisterRpc
// =========================================================================

void AgentHost::startRegisterRpc() {
    RegisterRequest req;
    req.rank = rank_;
    req.agent_addr = rpc_server_->getListenAddr(host_ip_);
    req.te_server_name = te_link_manager_.localServerName();
    req.warmup_recv_addr = te_link_manager_.getWarmupRecvAddr();
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
                        registration_cv_.notify_all();
                    }

                    // Re-publish all local backends' endpoints after (re-)reg.
                    // (Old session endpoints were cleared by Coordinator.)
                    for (auto& [group_id, backend] : backends_) {
                        publishLocalEndpoint(backend->buildEndpointMetadata());
                    }
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

// =========================================================================
// Internal: tick
// =========================================================================

void AgentHost::tick() {
    // Consume transfer observation queue.
    TransferObservationEvent event;
    while (observation_queue_.try_dequeue(event)) {
        auto effects = agent_.processTransferObservation(event);
        for (auto& effect : effects) {
            if (auto* e = std::get_if<SendTransferObservation>(&effect)) {
                e->request.agent_session_epoch = agent_.getAgentSessionEpoch();
            }
        }
        runEffects(effects);
    }

    // Check if reconnect is needed.
    if (agent_.getCoordinatorConnection() ==
        AgentStateMachine::CoordinatorConnection::Disconnected) {
        if (rpc_client_->tryReconnect(coordinator_addr_)) {
            agent_.setCoordinatorConnected();
            startRegisterRpc();
        }
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

    // Check connection to Coordinator.
    if (!rpc_client_->isConnected(coordinator_addr_)) {
        runEffects(agent_.markOffline());
    }
}

// =========================================================================
// Internal: runEffects
// =========================================================================

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
                    te_link_manager_.enablePeerProbe(e.rank, e.te_server_name,
                                                     e.warmup_recv_addr);
                },
                [this](const DisconnectLink& e) {
                    te_link_manager_.disconnect(e.peer);
                },
                [this](const StopReconnect& e) {
                    te_link_manager_.stopReconnect(e.peer);
                },
                [this](const ClearPeerMetadata& e) {
                    te_link_manager_.publishLinkDown(e.peer);
                },
                [this](const DisconnectAllLinks&) {
                    for (int i = 0; i < max_world_size_; ++i) {
                        if (i != rank_) {
                            te_link_manager_.disconnect(i);
                        }
                    }
                },
                [this](const ClearAllPeerMetadata&) {
                    for (int i = 0; i < max_world_size_; ++i) {
                        if (i != rank_) {
                            te_link_manager_.publishLinkDown(i);
                        }
                    }
                },
                [this](const PublishRankStateSnapshot& e) {
                    te_link_manager_.setRankStates(e.states);
                },
                [this](const ApplyViewToBackend& e) {
                    auto it = backends_.find(e.group_id);
                    if (it != backends_.end()) {
                        it->second->applyViewChange(e.descriptor, e.view);
                    }
                },
                [this](const MarkBackendViewStale& e) {
                    auto it = backends_.find(e.group_id);
                    if (it != backends_.end()) {
                        it->second->markViewStale();
                    }
                },
            },
            effect);
    }
}

void AgentHost::forEachBackend(std::function<void(MooncakeBackend*)> func) {
    for (auto& [group_id, backend] : backends_) {
        func(backend);
    }
}

}  // namespace mooncake
