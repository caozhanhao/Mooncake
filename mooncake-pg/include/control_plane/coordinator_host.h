#ifndef MOONCAKE_PG_COORDINATOR_HOST_H
#define MOONCAKE_PG_COORDINATOR_HOST_H

#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>

#include "coordinator.h"
#include "rpc.h"
#include "rpc_runtime.h"
#include "serialized_executor.h"
#include "error_types.h"

namespace mooncake {

class RpcServer;
class RpcClient;

// =========================================================================
// Control Plane Architecture (Coordinator side)
// =========================================================================
//
// The CentralizedCoordinator runs inside Rank 0's process.  It is the
// authoritative source of truth for rank health (RankState) and group
// membership (GroupView).
//
//   Agent (any rank)                 CoordinatorHost (Rank 0)
//   +-----------------+              +---------------------------+
//   | registerAgent   |--- RPC ----->| postRegisterAgent()       |
//   | heartbeat       |--- RPC ----->| postHeartbeat()           |
//   | proposeViewUpd  |--- RPC ----->| postProposeViewUpdate()   |
//   | publishEndpoint |--- RPC ----->| postPublishEndpoint()     |
//   | reportLinkEvent |--- RPC ----->| postLinkEventReport()     |
//   +-----------------+              +---------------------------+
//                                            |
//                                    SerializedExecutor
//                                            |
//                              +------------------------------------+
//                              | CentralizedCoordinatorStateMachine |
//                              |  (pure state machine, no I/O)      |
//                              +------------------------------------+
//                                            |
//                                    returns Effect list
//                                            |
//                              +-----------------------------------+
//                              | runEffects()                      |
//                              |  BroadcastRankState -> broadcast  |
//                              |  PushViewUpdate -> callAsync      |
//                              |  ReplyProposal -> reply           |
//                              +-----------------------------------+
//

class CoordinatorHost;

// CoordinatorRpcServiceImpl - thin RPC handler that forwards all calls
// to CoordinatorHost::post*().
class CoordinatorRpcServiceImpl : public CoordinatorRpcService {
   public:
    explicit CoordinatorRpcServiceImpl(CoordinatorHost& host) : host_(host) {}

    void registerAgent(coro_rpc::context<RegisterAgentResponse> ctx,
                       RegisterAgentRequest req) override;

    void confirmAgentRegistration(
        coro_rpc::context<ConfirmAgentRegistrationResponse> ctx,
        ConfirmAgentRegistrationRequest req) override;

    void heartbeat(coro_rpc::context<HeartbeatResponse> ctx,
                   HeartbeatRequest req) override;

    void unregisterAgent(coro_rpc::context<UnregisterAgentResponse> ctx,
                         UnregisterAgentRequest req) override;

    void registerGroup(coro_rpc::context<RegisterGroupResponse> ctx,
                       RegisterGroupRequest req) override;

    void unregisterGroup(coro_rpc::context<UnregisterGroupResponse> ctx,
                         UnregisterGroupRequest req) override;

    void confirmReadyForActivation(
        coro_rpc::context<ConfirmReadyForActivationResponse> ctx,
        ConfirmReadyForActivationRequest req) override;

    void publishEndpoint(coro_rpc::context<PublishEndpointResponse> ctx,
                         PublishEndpointRequest req) override;

    void proposeViewUpdate(coro_rpc::context<ProposeViewUpdateResponse> ctx,
                           ProposeViewUpdateRequest req) override;

    void reportLinkEvent(coro_rpc::context<LinkEventReportAck> ctx,
                         LinkEventReport req) override;

    void syncAfterFailure(coro_rpc::context<SyncAfterFailureResponse> ctx,
                          SyncAfterFailureRequest req) override;

   private:
    CoordinatorHost& host_;
};

// CoordinatorHost - execution host for the Coordinator state machine.
class CoordinatorHost {
   public:
    CoordinatorHost(const std::string& host_ip, int max_world_size,
                    int64_t fault_reconciliation_window_us,
                    std::optional<DeviceAllReduceAlgorithm>
                        all_reduce_algorithm = std::nullopt);

    ~CoordinatorHost();

    PGResult<void> start();
    void shutdown();
    PGResult<void> setFaultReconciliationWindow(int64_t timeout_us);

    const std::string& getListenAddr() const { return listen_addr_; }

    void postRegisterAgent(coro_rpc::context<RegisterAgentResponse> ctx,
                           RegisterAgentRequest req);

    void postConfirmAgentRegistration(
        coro_rpc::context<ConfirmAgentRegistrationResponse> ctx,
        ConfirmAgentRegistrationRequest req);

    void postHeartbeat(coro_rpc::context<HeartbeatResponse> ctx,
                       HeartbeatRequest req);

    void postUnregisterAgent(coro_rpc::context<UnregisterAgentResponse> ctx,
                             UnregisterAgentRequest req);

    void postRegisterGroup(coro_rpc::context<RegisterGroupResponse> ctx,
                           RegisterGroupRequest req);

    void postUnregisterGroup(coro_rpc::context<UnregisterGroupResponse> ctx,
                             UnregisterGroupRequest req);

    void postConfirmReadyForActivation(
        coro_rpc::context<ConfirmReadyForActivationResponse> ctx,
        ConfirmReadyForActivationRequest req);

    void postProposeViewUpdate(coro_rpc::context<ProposeViewUpdateResponse> ctx,
                               ProposeViewUpdateRequest req);

    void postPublishEndpoint(coro_rpc::context<PublishEndpointResponse> ctx,
                             PublishEndpointRequest req);

    void postLinkEventReport(coro_rpc::context<LinkEventReportAck> ctx,
                             LinkEventReport req);

    void postSyncAfterFailure(coro_rpc::context<SyncAfterFailureResponse> ctx,
                              SyncAfterFailureRequest req);

   private:
    CentralizedCoordinatorStateMachine state_machine_;
    SerializedExecutor executor_;

    std::string host_ip_;
    std::string listen_addr_;
    int max_world_size_;

    // RPC infrastructure.
    std::unique_ptr<RpcServer> rpc_server_;
    std::unique_ptr<RpcClient> rpc_client_;
    std::unique_ptr<CoordinatorRpcServiceImpl> rpc_impl_;

    // Host only maintains deferred response context mapping.
    // Related states is inside CentralizedCoordinatorStateMachine;

    template <typename Response>
    struct PendingReply {
        RankIdentity identity;
        coro_rpc::context<Response> context;
    };

    uint64_t next_propose_id_{1};
    std::unordered_map<uint64_t, PendingReply<ProposeViewUpdateResponse>>
        pending_proposal_resps_;

    uint64_t next_sync_id_{1};
    std::unordered_map<uint64_t, PendingReply<SyncAfterFailureResponse>>
        pending_sync_resps_;

    static constexpr auto kShutdownDrainTimeout = std::chrono::seconds(30);

    // The Host requests shutdown from the state machine and stops the RPC
    // server once all sessions in the shutdown snapshot have unregistered.
    std::atomic<bool> shutdown_requested_{false};
    std::promise<void> shutdown_confirmation_;

    void runEffects(const std::vector<CoordinatorEffect>& effects);
    void pushViewUpdate(const PushViewUpdate& effect);
    void pushTransferEndpointUpdate(const PushTransferEndpointUpdate& effect);

    template <typename Response>
    void replyRpc(coro_rpc::context<Response> ctx, RankIdentity identity,
                  Response response, bool allow_closed = false) {
        if (state_machine_.accepts(identity) ||
            (allow_closed && state_machine_.hasIdentity(identity))) {
            ctx.response_msg(std::move(response));
        } else if constexpr (std::is_same_v<Response, HeartbeatResponse>) {
            ctx.response_msg(HeartbeatResponse{true});
        } else {
            ctx.response_error(coro_rpc::errc::invalid_rpc_arguments,
                               "stale agent identity");
        }
    }

    // All ordinary ingress runs through this gate on the state-machine executor.
    template <auto Handler, typename Response, typename Request>
    void postRpc(coro_rpc::context<Response> ctx, Request req,
                 bool allow_closed = false) {
        executor_.post([this, ctx = std::move(ctx), req = std::move(req),
                        allow_closed]() mutable {
            Response response{};
            if (state_machine_.accepts(req.identity) ||
                (allow_closed && state_machine_.hasIdentity(req.identity))) {
                auto result = (state_machine_.*Handler)(req);
                response = std::move(result.response);
                runEffects(result.effects);
            }
            replyRpc(std::move(ctx), req.identity, std::move(response),
                     allow_closed);
        });
    }

    template <auto Handler, typename Response, typename Request>
    void postDeferredRpc(
        coro_rpc::context<Response> ctx, Request req, uint64_t& next_id,
        std::unordered_map<uint64_t, PendingReply<Response>>& pending) {
        executor_.post([this, ctx = std::move(ctx), req = std::move(req),
                        &next_id, &pending]() mutable {
            if (!state_machine_.accepts(req.identity)) {
                replyRpc(std::move(ctx), req.identity, Response{});
                return;
            }
            const auto id = next_id++;
            pending.emplace(
                id, PendingReply<Response>{req.identity, std::move(ctx)});
            runEffects((state_machine_.*Handler)(id, req).effects);
        });
    }

    template <typename Response>
    void finishReply(
        std::unordered_map<uint64_t, PendingReply<Response>>& pending,
        uint64_t id, const Response& response) {
        auto it = pending.find(id);
        if (it == pending.end()) return;
        auto& reply = it->second;
        replyRpc(std::move(reply.context), reply.identity, response);
        pending.erase(it);
    }

    template <auto Method, typename Push>
    void pushToAgent(RankIdentity identity, Push push) {
        if (!state_machine_.accepts(identity)) return;
        push.identity = identity;
        rpc_client_->send<Method>(state_machine_.getAgentAddr(identity.rank),
                                  std::move(push));
    }

    template <auto Method, typename Push, typename Callback>
    void callAgent(RankIdentity identity, Push push, Callback callback) {
        if (!state_machine_.accepts(identity)) return;
        push.identity = identity;
        using Reply = decltype(coro_rpc::get_return_type<Method>());
        rpc_client_->callAsync<Method>(
            state_machine_.getAgentAddr(identity.rank), std::move(push),
            [this, identity, callback = std::move(callback)](
                PGResult<Reply> result) mutable {
                executor_.post([this, identity, callback = std::move(callback),
                                result = std::move(result)]() mutable {
                    if (!result.has_value() || !state_machine_.accepts(identity))
                        return;
                    callback(identity, std::move(result).value());
                });
            });
    }
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_COORDINATOR_HOST_H
