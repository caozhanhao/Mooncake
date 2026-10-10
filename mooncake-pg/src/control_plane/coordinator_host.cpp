#include "control_plane/coordinator_host.h"

#include <chrono>
#include <glog/logging.h>

#include "control_plane/rpc.h"
#include "control_plane/rpc_runtime.h"
#include "pg_utils.h"

namespace mooncake {

void CoordinatorRpcServiceImpl::registerAgent(
    coro_rpc::context<RegisterAgentResponse> ctx, RegisterAgentRequest req) {
    host_.postRegisterAgent(std::move(ctx), std::move(req));
}

void CoordinatorRpcServiceImpl::confirmAgentRegistration(
    coro_rpc::context<ConfirmAgentRegistrationResponse> ctx,
    ConfirmAgentRegistrationRequest req) {
    host_.postConfirmAgentRegistration(std::move(ctx), std::move(req));
}

void CoordinatorRpcServiceImpl::heartbeat(
    coro_rpc::context<HeartbeatResponse> ctx, HeartbeatRequest req) {
    host_.postHeartbeat(std::move(ctx), std::move(req));
}

void CoordinatorRpcServiceImpl::unregisterAgent(
    coro_rpc::context<UnregisterAgentResponse> ctx,
    UnregisterAgentRequest req) {
    host_.postUnregisterAgent(std::move(ctx), std::move(req));
}

void CoordinatorRpcServiceImpl::registerGroup(
    coro_rpc::context<RegisterGroupResponse> ctx, RegisterGroupRequest req) {
    host_.postRegisterGroup(std::move(ctx), std::move(req));
}

void CoordinatorRpcServiceImpl::unregisterGroup(
    coro_rpc::context<UnregisterGroupResponse> ctx,
    UnregisterGroupRequest req) {
    host_.postUnregisterGroup(std::move(ctx), std::move(req));
}

void CoordinatorRpcServiceImpl::confirmReadyForActivation(
    coro_rpc::context<ConfirmReadyForActivationResponse> ctx,
    ConfirmReadyForActivationRequest req) {
    host_.postConfirmReadyForActivation(std::move(ctx), std::move(req));
}

void CoordinatorRpcServiceImpl::proposeViewUpdate(
    coro_rpc::context<ProposeViewUpdateResponse> ctx,
    ProposeViewUpdateRequest req) {
    host_.postProposeViewUpdate(std::move(ctx), std::move(req));
}
void CoordinatorRpcServiceImpl::publishEndpoint(
    coro_rpc::context<PublishEndpointResponse> ctx,
    PublishEndpointRequest req) {
    host_.postPublishEndpoint(std::move(ctx), std::move(req));
}

void CoordinatorRpcServiceImpl::reportLinkEvent(
    coro_rpc::context<LinkEventReportAck> ctx, LinkEventReport req) {
    host_.postLinkEventReport(std::move(ctx), std::move(req));
}

void CoordinatorRpcServiceImpl::syncAfterFailure(
    coro_rpc::context<SyncAfterFailureResponse> ctx,
    SyncAfterFailureRequest req) {
    host_.postSyncAfterFailure(std::move(ctx), std::move(req));
}

CoordinatorHost::CoordinatorHost(
    const std::string& host_ip, int max_world_size,
    int64_t fault_reconciliation_window_us,
    std::optional<DeviceAllReduceAlgorithm> all_reduce_algorithm)
    : state_machine_(max_world_size,
                     std::chrono::microseconds(fault_reconciliation_window_us),
                     all_reduce_algorithm),
      executor_("CoordinatorHost"),
      host_ip_(host_ip),
      max_world_size_(max_world_size),
      rpc_client_(std::make_unique<RpcClient>()) {}

CoordinatorHost::~CoordinatorHost() { shutdown(); }

PGResult<void> CoordinatorHost::setFaultReconciliationWindow(
    int64_t timeout_us) {
    return executor_.postAndWait([this, timeout_us] {
        state_machine_.setFaultReconciliationWindow(
            std::chrono::microseconds(timeout_us));
    });
}

PGResult<void> CoordinatorHost::start() {
    PG_VALIDATE_STATE(!shutdown_requested_.load(std::memory_order_acquire),
                      "CoordinatorHost cannot start after shutdown");

    rpc_server_ = std::make_unique<RpcServer>(/*port=*/0, /*thread_num=*/2);
    rpc_impl_ = std::make_unique<CoordinatorRpcServiceImpl>(*this);
    rpc_server_
        ->registerHandler<&CoordinatorRpcService::registerAgent,
                          &CoordinatorRpcService::confirmAgentRegistration,
                          &CoordinatorRpcService::heartbeat,
                          &CoordinatorRpcService::unregisterAgent,
                          &CoordinatorRpcService::registerGroup,
                          &CoordinatorRpcService::unregisterGroup,
                          &CoordinatorRpcService::confirmReadyForActivation,
                          &CoordinatorRpcService::proposeViewUpdate,
                          &CoordinatorRpcService::publishEndpoint,
                          &CoordinatorRpcService::reportLinkEvent,
                          &CoordinatorRpcService::syncAfterFailure>(
            rpc_impl_.get());

    bool server_started = rpc_server_->start();
    if (!server_started) {
        rpc_impl_.reset();
        rpc_server_.reset();
        return makePGError(PGErrorCode::SystemError,
                           "CoordinatorHost failed to start RPC server");
    }

    listen_addr_ = rpc_server_->getListenAddr(host_ip_);

    executor_.setTickCallback([this]() {
        auto result = state_machine_.tick();
        runEffects(result.effects);
    });

    executor_.start();
    return {};
}

void CoordinatorHost::shutdown() {
    if (shutdown_requested_.exchange(true, std::memory_order_acq_rel)) return;

    if (rpc_server_) {
        auto shutdown_confirmation = shutdown_confirmation_.get_future();
        auto post_result = executor_.postAndWait([this]() {
            auto result = state_machine_.requestShutdown();
            runEffects(result.effects);
        });

        if (!post_result.has_value()) {
            LOG(WARNING) << "[COORD] failed to request shutdown: "
                         << post_result.error().message;
        } else if (shutdown_confirmation.wait_for(kShutdownDrainTimeout) !=
                   std::future_status::ready) {
            LOG(WARNING) << "[COORD] shutdown drain timed out";
        }
        rpc_server_->shutdown();
    }

    // Keep the executor alive while outbound callbacks finish; callbacks may
    // still post their final state-machine work during client draining.
    if (rpc_client_) rpc_client_->shutdown();
    executor_.shutdown();
}

void CoordinatorHost::postRegisterAgent(
    coro_rpc::context<RegisterAgentResponse> ctx, RegisterAgentRequest req) {
    executor_.post(
        [this, ctx = std::move(ctx), req = std::move(req)]() mutable {
            auto r = state_machine_.handleRegisterAgent(req);
            runEffects(r.effects);
            ctx.response_msg(std::move(r.response));
        });
}

void CoordinatorHost::postConfirmAgentRegistration(
    coro_rpc::context<ConfirmAgentRegistrationResponse> ctx,
    ConfirmAgentRegistrationRequest req) {
    executor_.post([this, ctx = std::move(ctx), req = std::move(req)]() mutable {
        auto result = state_machine_.handleConfirmAgentRegistration(req);
        ctx.response_msg(std::move(result.response));
        runEffects(result.effects);
    });
}

void CoordinatorHost::postHeartbeat(coro_rpc::context<HeartbeatResponse> ctx,
                                    HeartbeatRequest req) {
    postRpc<&CentralizedCoordinatorStateMachine::handleHeartbeat>(
        std::move(ctx), std::move(req));
}

void CoordinatorHost::postUnregisterAgent(
    coro_rpc::context<UnregisterAgentResponse> ctx,
    UnregisterAgentRequest req) {
    postRpc<&CentralizedCoordinatorStateMachine::handleUnregisterAgent>(
        std::move(ctx), std::move(req), /*allow_closed=*/true);
}

void CoordinatorHost::postRegisterGroup(
    coro_rpc::context<RegisterGroupResponse> ctx, RegisterGroupRequest req) {
    postRpc<&CentralizedCoordinatorStateMachine::handleRegisterGroup>(
        std::move(ctx), std::move(req));
}

void CoordinatorHost::postUnregisterGroup(
    coro_rpc::context<UnregisterGroupResponse> ctx,
    UnregisterGroupRequest req) {
    postRpc<&CentralizedCoordinatorStateMachine::handleUnregisterGroup>(
        std::move(ctx), std::move(req));
}

void CoordinatorHost::postConfirmReadyForActivation(
    coro_rpc::context<ConfirmReadyForActivationResponse> ctx,
    ConfirmReadyForActivationRequest req) {
    postRpc<&CentralizedCoordinatorStateMachine::handleConfirmReadyForActivation>(
        std::move(ctx), std::move(req));
}

void CoordinatorHost::postProposeViewUpdate(
    coro_rpc::context<ProposeViewUpdateResponse> ctx,
    ProposeViewUpdateRequest req) {
    postDeferredRpc<&CentralizedCoordinatorStateMachine::handleProposeViewUpdate>(
        std::move(ctx), std::move(req), next_propose_id_, pending_proposal_resps_);
}

void CoordinatorHost::postPublishEndpoint(
    coro_rpc::context<PublishEndpointResponse> ctx,
    PublishEndpointRequest req) {
    postRpc<&CentralizedCoordinatorStateMachine::handlePublishEndpoint>(
        std::move(ctx), std::move(req));
}

void CoordinatorHost::postLinkEventReport(
    coro_rpc::context<LinkEventReportAck> ctx, LinkEventReport req) {
    postRpc<&CentralizedCoordinatorStateMachine::handleLinkEventReport>(
        std::move(ctx), std::move(req));
}

void CoordinatorHost::postSyncAfterFailure(
    coro_rpc::context<SyncAfterFailureResponse> ctx,
    SyncAfterFailureRequest req) {
    postDeferredRpc<&CentralizedCoordinatorStateMachine::handleSyncAfterFailure>(
        std::move(ctx), std::move(req), next_sync_id_, pending_sync_resps_);
}

void CoordinatorHost::runEffects(
    const std::vector<CoordinatorEffect>& effects) {
    for (const auto& effect : effects) {
        std::visit(
            overloaded{
                [this](const BroadcastRankState& e) {
                    for (int i = 0; i < max_world_size_; ++i) {
                        if (state_machine_.getRankState(i) !=
                            RankState::Offline) {
                            pushToAgent<&AgentRpcService::onRankStateUpdate>(
                                state_machine_.getIdentity(i), e.push);
                        }
                    }
                },
                [this](const PushViewUpdate& e) { pushViewUpdate(e); },
                [this](const PushTransferEndpointUpdate& e) {
                    pushTransferEndpointUpdate(e);
                },
                [this](const ReplyProposal& e) {
                    finishReply(pending_proposal_resps_, e.propose_id, e.response);
                },
                [this](const ReplySync& e) {
                    finishReply(pending_sync_resps_, e.sync_id, e.response);
                },
                [this](const BroadcastPeerJoined& e) {
                    for (int i = 0; i < max_world_size_; ++i) {
                        if (i != e.push.rank && state_machine_.getRankState(
                                                    i) != RankState::Offline) {
                            pushToAgent<&AgentRpcService::onPeerJoined>(
                                state_machine_.getIdentity(i), e.push);
                        }
                    }
                },
                [this](const ShutdownCoordinatorHost&) {
                    shutdown_confirmation_.set_value();
                },
            },
            effect);
    }
}

void CoordinatorHost::pushViewUpdate(const PushViewUpdate& effect) {
    for (int32_t rank = 0; rank < max_world_size_; ++rank) {
        if (!effect.view.members[rank].isMember()) continue;
        callAgent<&AgentRpcService::onViewUpdate>(
            state_machine_.getIdentity(rank), ViewUpdatePush{{}, effect.view},
            [this, group_id = effect.view.group_id, epoch = effect.view.epoch](
                RankIdentity identity, ViewUpdateAck ack) {
                if (ack.group_id != group_id || ack.epoch != epoch) return;
                runEffects(state_machine_.handleViewUpdateAck(
                    group_id, identity.rank, epoch, ack.applied).effects);
            });
    }
}

void CoordinatorHost::pushTransferEndpointUpdate(
    const PushTransferEndpointUpdate& effect) {
    for (auto rank : effect.push.snapshot.participants) {
        RankIdentity identity{rank, effect.push.snapshot.rank_epochs[rank]};
        callAgent<&AgentRpcService::onTransferEndpointUpdate>(
            identity, effect.push,
            [this, version = effect.push.snapshot.version](
                RankIdentity identity, TransferEndpointUpdateAck ack) {
                if (ack.version != version) return;
                runEffects(state_machine_.handleTransferEndpointUpdateAck(
                    identity, ack).effects);
            });
    }
}

}  // namespace mooncake
