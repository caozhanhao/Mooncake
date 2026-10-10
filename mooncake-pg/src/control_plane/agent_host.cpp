#include "control_plane/agent_host.h"

#include <algorithm>
#include <chrono>
#include <unistd.h>

#include <glog/logging.h>

#include "mooncake_communicator.h"
#include "control_plane/link_manager.h"
#include "control_plane/rpc_runtime.h"
#include "device_comm/device_collective/device_collective_feature.h"
#if MOONCAKE_PG_HAS_COLLECTIVE_V2
#include "device_comm/device_collective/device_collective_workspace.h"
#include "device_comm/device_transfer/transfer_service.h"
#endif
#include "pg_utils.h"

namespace mooncake {

namespace {

// Generate a process-unique key for one logical registration.
uint64_t generateInitialRegistrationId() {
    auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    uint64_t pid = static_cast<uint64_t>(getpid());
    uint64_t base = (pid << 32) ^ static_cast<uint64_t>(now);
    return base == 0 ? 1 : base;
}

// Runs on the installation executor with no access to AgentHost state.
PGResult<DeviceTransferEndpoint> installTransferEndpoints(
    DeviceTransferService* service, const InstallTransferEndpoints& effect) {
#if MOONCAKE_PG_HAS_COLLECTIVE_V2
    PG_VALIDATE_STATE(service, "device endpoint service is missing");
    return service->installEndpoints(effect.snapshot,
                                     effect.reclaim_before_version);
#else
    (void)service;
    (void)effect;
    return makePGError(PGErrorCode::NotSupported,
                       "device endpoints require CUDA");
#endif
}

}  // namespace

void AgentRpcServiceImpl::onPeerJoined(PeerJoinedPush push) {
    host_.postPeerJoined(std::move(push));
}

void AgentRpcServiceImpl::onRankStateUpdate(RankStatePush push) {
    host_.postRankStateUpdate(std::move(push));
}

void AgentRpcServiceImpl::onViewUpdate(coro_rpc::context<ViewUpdateAck> ctx,
                                       ViewUpdatePush push) {
    host_.postViewUpdate(std::move(ctx), std::move(push));
}

void AgentRpcServiceImpl::onTransferEndpointUpdate(
    coro_rpc::context<TransferEndpointUpdateAck> ctx,
    TransferEndpointUpdatePush push) {
    host_.postTransferEndpointUpdate(std::move(ctx), std::move(push));
}

AgentHost::AgentHost(std::string coordinator_addr, const std::string& host_ip,
                     GlobalRank rank, int max_world_size,
                     DeviceTransferService* device_transfer_service,
                     DeviceCollectiveWorkspace* device_collective_workspace,
                     LinkManager& link_manager,
                     int64_t fault_reconciliation_window_us)
    : agent_(rank, max_world_size),
      executor_("AgentHost"),
      device_transfer_service_(device_transfer_service),
      device_collective_workspace_(device_collective_workspace),
      link_manager_(link_manager),
      host_ip_(host_ip),
      rank_(rank),
      max_world_size_(max_world_size),
      coordinator_addr_(std::move(coordinator_addr)),
      fault_reconciliation_window_us_(fault_reconciliation_window_us),
      registration_id_(generateInitialRegistrationId()),
      rpc_client_(std::make_unique<RpcClient>()) {}

AgentHost::~AgentHost() { shutdown(); }

void AgentHost::setFaultReconciliationWindow(int64_t timeout_us) {
    fault_reconciliation_window_us_.store(timeout_us,
                                          std::memory_order_relaxed);
}

PGResult<void> AgentHost::start() {
    PG_VALIDATE_ARG(!coordinator_addr_.empty(),
                    "AgentHost coordinator address must not be empty");

    PG_VALIDATE_STATE(!shutdown_requested_.load(std::memory_order_acquire),
                      "AgentHost cannot start after shutdown");

    link_manager_.setEventCallback([this](TELinkUpEvent event) {
        if (shutdown_requested_.load(std::memory_order_acquire)) return;
        if (event.peer < 0 || event.peer >= max_world_size_) return;
        LinkEvent link_event;
        link_event.observer = event.observer;
        link_event.events.assign(max_world_size_, LinkEvent::EventType::None);
        link_event.target_rank_epochs.assign(max_world_size_, 0);
        link_event.events[event.peer] = LinkEvent::EventType::Success;
        link_event.target_rank_epochs[event.peer] = event.target_rank_epoch;
        pushLinkEvent(link_event);
    });

    rpc_server_ = std::make_unique<RpcServer>(/*port=*/0, /*thread_num=*/2);
    rpc_impl_ = std::make_unique<AgentRpcServiceImpl>(*this);
    rpc_server_->registerHandler<&AgentRpcService::onPeerJoined,
                                 &AgentRpcService::onRankStateUpdate,
                                 &AgentRpcService::onViewUpdate,
                                 &AgentRpcService::onTransferEndpointUpdate>(
        rpc_impl_.get());
    bool server_started = rpc_server_->start();
    if (!server_started) {
        link_manager_.setEventCallback(nullptr);
        rpc_impl_.reset();
        rpc_server_.reset();
        return makePGError(PGErrorCode::SystemError,
                           "AgentHost failed to start RPC server for rank " +
                               std::to_string(rank_));
    }

    transfer_endpoint_installer_.start();
    executor_.setTickCallback([this]() { tick(); });
    executor_.start();

    return executor_.post([this]() { startAgentRegistration(); });
}

void AgentHost::shutdown() {
    if (shutdown_requested_.exchange(true, std::memory_order_acq_rel)) return;

    link_manager_.setEventCallback(nullptr);
    // Installations post their completion back to the control executor.
    transfer_endpoint_installer_.shutdown();
    executor_.shutdown();
    if (rpc_server_) rpc_server_->shutdown();
    link_manager_.stop();
    unregisterAgent();
    if (rpc_client_) {
        rpc_client_->shutdown();
        rpc_client_.reset();
    }
}

void AgentHost::unregisterAgent() {
    if (!rpc_client_ || coordinator_addr_.empty()) return;

    const auto identity = agent_.identity();
    if (!identity.valid()) return;

    UnregisterAgentRequest req;
    req.identity = identity;
    auto result = callCoordinator<&CoordinatorRpcService::unregisterAgent>(std::move(req));
    if (!result.has_value()) {
        LOG(WARNING) << "AgentHost: unregisterAgent RPC failed, rank=" << rank_
                     << ": " << result.error().message;
        return;
    }
    const auto& response = result.value();
    if (!response.success) {
        LOG(WARNING) << "AgentHost: unregisterAgent rejected, rank=" << rank_
                     << ": " << response.reject_reason;
    }
}

PGResult<void> AgentHost::waitUntilRegistered(
    std::chrono::milliseconds timeout) {
    auto promise = std::make_shared<std::promise<void>>();
    auto future = promise->get_future();

    PG_TRY(executor_.post([this, promise]() {
        if (agent_.registrationPhase() ==
            AgentStateMachine::RegistrationPhase::Registered) {
            promise->set_value();
        } else {
            agent_registration_promises_.push_back(promise);
        }
    }));

    if (future.wait_for(timeout) != std::future_status::ready) {
        // Timeout: remove the dangling promise on the executor thread.
        executor_.post([this, promise]() {
            std::erase(agent_registration_promises_, promise);
        });
        return makePGError(PGErrorCode::Timeout,
                           "timed out waiting for agent registration");
    }
    return {};
}

// Block until the group reaches Ready status (all active ranks ACKed the
// bootstrap ViewUpdate). Returns a timeout error if the deadline expires.
//
// Note: if a peer dies during the BootstrapSyncing phase, the Coordinator
// will not transition the group to Ready, and this call will hang until
// the timeout expires.  The caller should handle this as a bootstrap failure.
PGResult<GroupView> AgentHost::waitUntilGroupReady(
    GroupId group_id, std::chrono::milliseconds timeout) {
    auto promise = std::make_shared<std::promise<GroupView>>();
    auto future = promise->get_future();

    PG_TRY(executor_.post([this, group_id, promise]() {
        auto view = agent_.getGroupView(group_id);
        if (view.status == GroupStatus::Ready) {
            promise->set_value(view);
        } else {
            group_ready_promises_[group_id].push_back(promise);
        }
    }));

    if (future.wait_for(timeout) != std::future_status::ready) {
        // Clean up the dangling promise before returning the timeout.
        executor_.post([this, group_id, promise]() {
            auto it = group_ready_promises_.find(group_id);
            if (it != group_ready_promises_.end()) {
                auto& vec = it->second;
                vec.erase(std::remove(vec.begin(), vec.end(), promise),
                          vec.end());
                if (vec.empty()) group_ready_promises_.erase(it);
            }
        });
        return makePGError(
            PGErrorCode::Timeout,
            "waitUntilGroupReady timed out for group " + group_id);
    }
    return future.get();
}

PGResult<void> AgentHost::waitUntilRankActive(
    GroupId group_id, GlobalRank rank, std::chrono::milliseconds timeout) {
    auto promise = std::make_shared<std::promise<void>>();
    auto future = promise->get_future();

    PG_TRY(executor_.post([this, group_id, rank, promise]() {
        auto view = agent_.getGroupView(group_id);
        if (view.members[rank].isActive()) {
            promise->set_value();
        } else {
            rank_active_promises_[group_id][rank].push_back(promise);
        }
    }));

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
        return makePGError(PGErrorCode::Timeout,
                           "waitUntilRankActive timed out for rank " +
                               std::to_string(rank) + " in group " + group_id);
    }
    return {};
}

PGResult<GroupId> AgentHost::registerGroup(
    GroupBootstrapId group_bootstrap_id, int32_t max_group_size,
    std::vector<GlobalRank> rank_order,
    std::optional<GpuCollectiveBackend> preferred_gpu_collective_backend,
    GroupBootstrapIdResolvePolicy resolve_policy, bool auto_deactivate,
    MooncakeCommunicator* communicator) {
    return executor_.postAndWait(
        [this, group_bootstrap_id = std::move(group_bootstrap_id),
         max_group_size, rank_order = std::move(rank_order),
         preferred_gpu_collective_backend, resolve_policy, auto_deactivate,
         communicator, identity = agent_.identity()]() mutable
            -> PGResult<GroupId> {
            RegisterGroupRequest req;
            req.identity = identity;
            req.group_bootstrap_id = std::move(group_bootstrap_id);
            req.max_group_size = max_group_size;
            req.rank_order = std::move(rank_order);
            req.preferred_gpu_collective_backend =
                preferred_gpu_collective_backend;
            req.resolve_policy = resolve_policy;
            req.auto_deactivate = auto_deactivate;

            PG_TRY(auto resp,
                   callCoordinator<&CoordinatorRpcService::registerGroup>(
                       std::move(req)));

            if (!resp.success) {
                // A rejected group must not affect the process-scoped Agent.
                // Return an empty id so this communicator can remain
                // group-scoped and execute local-only collectives.
                LOG(WARNING)
                    << "AgentHost: registerGroup rejected for rank=" << rank_
                    << ": " << resp.reject_reason
                    << "; leaving this group out of Agent state and falling "
                       "back to local-only execution";
                return GroupId{};
            }

            const auto& group_id = resp.view.group_id;
            communicators_.insert_or_assign(group_id, communicator);
            runEffects(agent_.registerGroup(resp.view));
            return group_id;
        });
}

void AgentHost::detachCommunicator(GroupId group_id) {
    executor_.postAndWait(
        [this, group_id]() { communicators_.erase(group_id); });
}

PGResult<void> AgentHost::unregisterGroup(GroupId group_id) {
    const auto identity = agent_.identity();
    return executor_.postAndWait([this, group_id, identity]() -> PGResult<void> {
        agent_.unregisterGroup(group_id);

        UnregisterGroupRequest req;
        req.identity = identity;
        req.group_id = group_id;
        return callAndCheck<&CoordinatorRpcService::unregisterGroup>(
            std::move(req));
    });
}

PGResult<void> AgentHost::confirmReadyForActivation(GroupId group_id) {
    ConfirmReadyForActivationRequest req;
    req.identity = agent_.identity();
    req.group_id = std::move(group_id);
    return callAndCheck<&CoordinatorRpcService::confirmReadyForActivation>(
        std::move(req));
}

PGResult<void> AgentHost::sendPublishEndpointRpc(
    GroupEndpointPublication endpoint, RankIdentity identity) {
    PublishEndpointRequest req;
    req.identity = identity;
    req.endpoints.push_back(std::move(endpoint));
    return callAndCheck<&CoordinatorRpcService::publishEndpoint>(std::move(req));
}

PGResult<void> AgentHost::publishLocalEndpoint(
    GroupEndpointPublication endpoint) {
    return executor_.postAndWait(
        [this, endpoint = std::move(endpoint),
         identity = agent_.identity()]() mutable {
            return sendPublishEndpointRpc(std::move(endpoint), identity);
        });
}

void AgentHost::sendLinkEventReport(LinkEventReport report) {
    callCoordinatorAsync<&CoordinatorRpcService::reportLinkEvent>(
        std::move(report), [this](PGResult<LinkEventReportAck> result) {
            if (result.has_value()) {
                agent_.handleLinkEventReportAck(result.value());
            }
        });
}

PGResult<ProposeViewUpdateResponse> AgentHost::proposeViewUpdateInternal(
    GroupId group_id, const std::vector<InGroupRank>& ranks,
    bool is_activation) {
    ProposeViewUpdateRequest req;
    req.identity = agent_.identity();
    req.group_id = group_id;
    req.requested_ranks = ranks;
    req.is_activation = is_activation;

    const auto coordinator_timeout =
        kProposalAdmissionTimeout + kViewUpdateAckTimeout;
    const auto rpc_timeout =
        std::max(RpcClient::kDefaultRequestTimeout,
                 std::chrono::duration_cast<std::chrono::milliseconds>(
                     2 * coordinator_timeout));
    return callCoordinator<&CoordinatorRpcService::proposeViewUpdate>(
        std::move(req), rpc_timeout);
}

PGResult<ProposeViewUpdateResponse> AgentHost::proposeActivate(
    GroupId group_id, const std::vector<InGroupRank>& ranks) {
    return proposeViewUpdateInternal(group_id, ranks, /*is_activation=*/true);
}

PGResult<ProposeViewUpdateResponse> AgentHost::proposeDeactivate(
    GroupId group_id, const std::vector<InGroupRank>& ranks) {
    return proposeViewUpdateInternal(group_id, ranks, /*is_activation=*/false);
}

void AgentHost::pushLinkEvent(const LinkEvent& event) {
    executor_.post(
        [this, event]() {
            if (agent_.accepts(event.observer)) {
                runEffects(agent_.pushLinkEvent(event));
            }
        });
}

PGResult<SyncAfterFailureResponse> AgentHost::syncAfterFailure(
    GroupId group_id) {
    SyncAfterFailureRequest req;
    req.identity = agent_.identity();
    req.group_id = group_id;

    PG_TRY(executor_.postAndWait([this, &req]() -> PGResult<void> {
        PG_TRY(prepareRpc(req));
        req.link_event_report = agent_.getLinkEventReport();
        req.current_epoch = agent_.getGroupView(req.group_id).epoch;
        return {};
    }));

    // Synchronous RPC should be issued outside the executor.
    // Blocking the serialized executor would stall all local state-machine
    // tasks.
    const auto reconciliation_window = std::chrono::microseconds(
        fault_reconciliation_window_us_.load(std::memory_order_relaxed));
    const auto reconciliation_timeout =
        std::chrono::ceil<std::chrono::milliseconds>(reconciliation_window);
    const auto rpc_timeout =
        std::max(RpcClient::kDefaultRequestTimeout, 2 * reconciliation_timeout);
    auto result = rpc_client_->call<&CoordinatorRpcService::syncAfterFailure>(
        coordinator_addr_, req, rpc_timeout);
    return executor_.postAndWait(
        [this, identity = req.identity, result = std::move(result)]() mutable
            -> PGResult<SyncAfterFailureResponse> {
            PG_TRY(auto response, consumeReply(identity, std::move(result)));
            if (response.link_event_report_ack)
                agent_.handleLinkEventReportAck(*response.link_event_report_ack);
            if (response.status != SyncAfterFailureStatus::Rejected) {
                PG_TRY(auto effects, agent_.applyGroupView(response.view));
                runEffects(effects);
            }
            return response;
        });
}

void AgentHost::postPeerJoined(PeerJoinedPush push) {
    executor_.post([this, push = std::move(push)]() {
        if (!acceptsRpc(push.identity)) return;
        runEffects(agent_.handlePeerJoined(push));
    });
}

void AgentHost::postRankStateUpdate(RankStatePush push) {
    executor_.post([this, push = std::move(push)]() {
        if (!acceptsRpc(push.identity)) return;
        runEffects(agent_.handleRankStateUpdate(push));
    });
}

void AgentHost::postViewUpdate(coro_rpc::context<ViewUpdateAck> ctx,
                               ViewUpdatePush push) {
    postRpc(std::move(ctx), std::move(push),
        [this](coro_rpc::context<ViewUpdateAck> ctx, ViewUpdatePush push) {
            auto result = agent_.handleViewUpdate(push);
            if (!result.has_value()) {
                replyRpc(std::move(ctx), push.identity,
                         ViewUpdateAck{push.view.group_id, push.view.epoch, false,
                                       result.error().message});
                return;
            }
            runEffects(result.value());
            replyRpc(std::move(ctx), push.identity,
                     ViewUpdateAck{push.view.group_id, push.view.epoch, true, {}});
        });
}

void AgentHost::postTransferEndpointUpdate(
    coro_rpc::context<TransferEndpointUpdateAck> ctx,
    TransferEndpointUpdatePush push) {
    postRpc(std::move(ctx), std::move(push),
        [this](coro_rpc::context<TransferEndpointUpdateAck> ctx,
               TransferEndpointUpdatePush push) {
            const auto request_id = next_transfer_endpoint_request_id_++;
            auto result = agent_.handleTransferEndpointUpdate(request_id, push);
            if (!result.has_value()) {
                LOG(ERROR) << result.error().message;
                replyRpc(std::move(ctx), push.identity,
                         TransferEndpointUpdateAck{push.snapshot.version, false,
                                                   std::nullopt});
                return;
            }
            pending_transfer_endpoint_resps_.emplace(request_id, std::move(ctx));
            runEffects(result.value());
        });
}

void AgentHost::enqueueTransferEndpointInstallation(
    const InstallTransferEndpoints& effect,
    coro_rpc::context<TransferEndpointUpdateAck> ctx) {
    auto submitted = transfer_endpoint_installer_.post([this, effect, ctx]() mutable {
        TransferEndpointUpdateAck ack{effect.snapshot.version, false, std::nullopt};
        if (acceptsRpc(effect.identity)) {
            auto result = installTransferEndpoints(device_transfer_service_, effect);
            if (result.has_value()) {
                ack.applied = true;
                auto endpoint = std::move(result).value();
                if (endpoint != *effect.snapshot.endpoints[effect.identity.rank])
                    ack.updated_endpoint = std::move(endpoint);
            } else {
                LOG(ERROR) << "Device endpoint installation failed: "
                           << result.error().message;
            }
        }
        executor_.post([this, ctx = std::move(ctx), identity = effect.identity,
                        ack = std::move(ack)]() mutable {
            replyRpc(std::move(ctx), identity, std::move(ack));
        });
    });
    if (!submitted.has_value()) {
        replyRpc(std::move(ctx), effect.identity,
                 TransferEndpointUpdateAck{effect.snapshot.version, false,
                                           std::nullopt});
    }
}

void AgentHost::startAgentRegistration() {
    if (shutdown_requested_.load(std::memory_order_acquire)) return;
    ++registration_id_;
    registration_request_.reset();
    registration_in_flight_ = false;
    runEffects(agent_.reset());
    link_manager_.stop();

    // Close the identity gate before draining installation work. A new identity
    // cannot use shared DTS resources until old operations have finished.
    transfer_endpoint_installer_.post([this, registration = registration_id_] {
        executor_.post([this, registration] {
            if (shutdown_requested_.load(std::memory_order_acquire) ||
                registration != registration_id_) return;
            agent_.setRegistrationPhase(
                AgentStateMachine::RegistrationPhase::Registering);
            sendRegistration();
        });
    });
}

void AgentHost::sendRegistration() {
    if (registration_in_flight_) return;
    if (!registration_request_) {
        RegisterAgentRequest request;
        request.rank = rank_;
        request.registration_id = registration_id_;
        request.agent_addr = rpc_server_->getListenAddr(host_ip_);
        request.te_server_name = link_manager_.localServerName();
        request.warmup_recv_addr = link_manager_.getWarmupRecvAddr();
#if MOONCAKE_PG_HAS_COLLECTIVE_V2
        if (device_transfer_service_)
            request.transfer_service_endpoint =
                device_transfer_service_->localEndpoint();
        if (device_collective_workspace_)
            request.collective_workspace_endpoint =
                device_collective_workspace_->localEndpoint();
#endif
        registration_request_ = std::move(request);
    }
    registration_in_flight_ = true;
    rpc_client_->callAsync<&CoordinatorRpcService::registerAgent>(
        coordinator_addr_, *registration_request_,
        [this, registration = registration_id_](
            PGResult<RegisterAgentResponse> result) {
            executor_.post([this, registration, result = std::move(result)]() mutable {
                if (shutdown_requested_.load(std::memory_order_acquire) ||
                    registration != registration_id_) return;
                registration_in_flight_ = false;
                next_registration_at_ =
                    std::chrono::steady_clock::now() + kHeartbeatInterval;
                if (!result.has_value() || !result.value().success) {
                    if (shouldLogAgentRegistrationError())
                        LOG(WARNING) << "Agent registration failed: "
                                     << (result.has_value() ? result.value().reject_reason
                                                           : result.error().message);
                    if (!result.has_value()) {
                        rpc_client_->tryReconnect(coordinator_addr_);
                    } else if (result.value().require_new_registration) {
                        startAgentRegistration();
                    }
                    return;
                }
                const auto identity = result.value().identity;
                if (!identity.valid() || identity.rank != rank_) {
                    LOG(ERROR) << "Coordinator returned an invalid rank identity";
                    startAgentRegistration();
                    return;
                }
                agent_.setIdentity(identity);
                sendConfirmation();
            });
        });
}

void AgentHost::sendConfirmation() {
    if (registration_in_flight_) return;
    registration_in_flight_ = true;
    ConfirmAgentRegistrationRequest request{registration_id_, agent_.identity()};
    rpc_client_->callAsync<&CoordinatorRpcService::confirmAgentRegistration>(
        coordinator_addr_, request,
        [this, request](PGResult<ConfirmAgentRegistrationResponse> result) {
            executor_.post([this, request, result = std::move(result)]() mutable {
                if (shutdown_requested_.load(std::memory_order_acquire) ||
                    request.registration_id != registration_id_ ||
                    !agent_.accepts(request.identity)) return;
                registration_in_flight_ = false;
                next_registration_at_ =
                    std::chrono::steady_clock::now() + kHeartbeatInterval;
                if (!result.has_value() || !result.value().success) {
                    if (shouldLogAgentRegistrationError())
                        LOG(WARNING) << "Agent confirmation failed: "
                                     << (result.has_value() ? result.value().reject_reason
                                                           : result.error().message);
                    if (!result.has_value()) {
                        rpc_client_->tryReconnect(coordinator_addr_);
                    } else if (result.value().require_new_registration) {
                        startAgentRegistration();
                    }
                    return;
                }
                auto effects = agent_.applyRegistrationSnapshot(result.value());
                if (!effects.has_value()) {
                    LOG(ERROR) << effects.error().message;
                    startAgentRegistration();
                    return;
                }
                runEffects(effects.value());
                agent_.setRegistrationPhase(
                    AgentStateMachine::RegistrationPhase::Registered);
                link_manager_.start(request.identity.epoch);
                for (auto& p : agent_registration_promises_) {
                    p->set_value();
                }
                agent_registration_promises_.clear();

                // Re-publish all local communicators' endpoints after (re-)reg.
                // Old session endpoints were cleared by Coordinator.
                forEachCommunicator([&](auto communicator) {
                    auto result = sendPublishEndpointRpc(
                        communicator->buildEndpointMetadata(), request.identity);
                    if (!result.has_value()) {
                        LOG(ERROR) << "AgentHost: failed to re-publish "
                                      "communicator endpoint: "
                                   << result.error().message;
                    }
                });
            });
        });
}

bool AgentHost::shouldLogAgentRegistrationError() {
    const auto now = std::chrono::steady_clock::now();
    if (last_agent_register_error_log_time_.time_since_epoch() !=
            std::chrono::steady_clock::duration{} &&
        now - last_agent_register_error_log_time_ <
            kAgentRegisterErrorLogInterval) {
        return false;
    }
    last_agent_register_error_log_time_ = now;
    return true;
}

void AgentHost::tick() {
    if (shutdown_requested_.load(std::memory_order_acquire)) return;
    if (!rpc_client_) return;
    const auto now = std::chrono::steady_clock::now();
    const auto phase = agent_.registrationPhase();
    if (phase != AgentStateMachine::RegistrationPhase::Registered) {
        if (registration_in_flight_ || now < next_registration_at_) return;
        if (phase == AgentStateMachine::RegistrationPhase::Registering) {
            sendRegistration();
        } else if (phase == AgentStateMachine::RegistrationPhase::Confirming) {
            sendConfirmation();
        }
        return;
    }
    if (now < next_heartbeat_at_) return;
    next_heartbeat_at_ = now + kHeartbeatInterval;
    // Link reports are idempotent by report_id. Retry the latest unacknowledged
    // snapshot with the heartbeat cadence when the request or its response is
    // lost.
    if (auto report = agent_.getLinkEventReport()) {
        sendLinkEventReport(std::move(*report));
    }
    callCoordinatorAsync<&CoordinatorRpcService::heartbeat>(
        HeartbeatRequest{agent_.identity()},
        [this](PGResult<HeartbeatResponse> result) {
            if (result.has_value() && result.value().require_new_registration)
                startAgentRegistration();
        });
}

void AgentHost::runEffects(const AgentApplyResult& effects) {
    for (const auto& effect : effects) {
        std::visit(
            overloaded{
                [this](const InstallTransferEndpoints& e) {
                    auto it =
                        pending_transfer_endpoint_resps_.find(e.request_id);
                    if (it != pending_transfer_endpoint_resps_.end()) {
                        auto ctx = std::move(it->second);
                        pending_transfer_endpoint_resps_.erase(it);
                        enqueueTransferEndpointInstallation(e, std::move(ctx));
                    }
                },
                [this](const InstallDeviceCollectiveWorkspaceEndpoint& e) {
#if MOONCAKE_PG_HAS_COLLECTIVE_V2
                    PG_ASSERT_OK(
                        device_collective_workspace_->installPeerEndpoint(
                            e.rank, e.endpoint));
#else
                    (void)e;
#endif
                },
                [this](const EnablePeerProbe& e) {
                    link_manager_.enablePeerProbe(e.rank, e.rank_epoch,
                                                  e.te_server_name,
                                                  e.warmup_recv_addr);
                },
                [this](const DisconnectLink& e) {
                    link_manager_.disconnect(e.peer);
                },
                [this](const RequestLinkHealthCheck& e) {
                    link_manager_.requestHealthCheck(e.peer);
                },
                [this](const SendLinkEventReport& e) {
                    sendLinkEventReport(e.report);
                },
                [this](const StopReconnect& e) {
                    link_manager_.stopReconnect(e.peer);
                },
                [this](const RefreshPeerLink& e) {
                    link_manager_.refreshPeerSegment(e.peer);
                },
                [this](const ResetPeerState& e) {
                    for (auto& [group_id, communicator] : communicators_) {
                        auto view = agent_.getGroupView(group_id);
                        for (int lr = 0;
                             lr < static_cast<int>(view.rank_order.size());
                             ++lr) {
                            if (view.rank_order[lr] == e.peer) {
                                communicator->onPeerLinkReset(lr);
                                break;
                            }
                        }
                    }
                },
                [this](const NotifyLinkRefreshed& e) {
                    for (auto& [group_id, communicator] : communicators_) {
                        auto view = agent_.getGroupView(group_id);
                        for (int lr = 0;
                             lr < static_cast<int>(view.rank_order.size());
                             ++lr) {
                            if (view.rank_order[lr] == e.peer) {
                                communicator->refreshSegmentID(lr);
                                break;
                            }
                        }
                    }
                },
                [this](const DisconnectAllLinks&) {
                    for (int i = 0; i < max_world_size_; ++i) {
                        if (i != rank_) {
                            link_manager_.disconnect(i);
                        }
                    }
                },
                [this](const ClearAllPeerMetadata&) {
                    for (int i = 0; i < max_world_size_; ++i) {
                        if (i != rank_) {
                            link_manager_.publishLinkDown(i);
                        }
                    }
                },
                [this](const ApplyGroupStateToCommunicator& e) {
                    withCommunicator(e.view.group_id, [&](auto communicator) {
                        communicator->applyGroupState(e.view, e.rank_states,
                                                      e.rank_epochs,
                                                      e.activatable);
                    });
                },
                [this](const ApplyRankStateToCommunicator& e) {
                    withCommunicator(e.group_id, [&](auto communicator) {
                        communicator->applyRankStateUpdate(
                            e.rank, e.in_group_rank, e.state, e.rank_epoch,
                            e.activatable);
                    });
                },
                [this](const NotifyGroupReady& e) {
                    auto it = group_ready_promises_.find(e.group_id);
                    if (it == group_ready_promises_.end()) return;
                    auto view = agent_.getGroupView(e.group_id);
                    for (auto& p : it->second) p->set_value(view);
                    group_ready_promises_.erase(it);
                },
                [this](const NotifyRanksActivated& e) {
                    auto it = rank_active_promises_.find(e.group_id);
                    if (it == rank_active_promises_.end()) return;
                    for (GlobalRank gr : e.ranks) {
                        auto rit = it->second.find(gr);
                        if (rit != it->second.end()) {
                            for (auto& p : rit->second) p->set_value();
                            it->second.erase(rit);
                        }
                    }
                    if (it->second.empty()) rank_active_promises_.erase(it);
                },
            },
            effect);
    }
}

}  // namespace mooncake
