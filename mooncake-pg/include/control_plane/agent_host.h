#ifndef MOONCAKE_PG_AGENT_HOST_H
#define MOONCAKE_PG_AGENT_HOST_H

#include <chrono>
#include <future>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <torch/csrc/distributed/c10d/Store.hpp>

#include "agent.h"
#include "rpc.h"
#include "serialized_executor.h"
#include "link_manager.h"

#include "pg_utils.h"

namespace mooncake {

// Forward declarations.
class RpcServer;
class RpcClient;
class MooncakeBackend;

// =========================================================================
// AgentInterface — control-plane service interface exposed to MooncakeBackend.
//
// AgentHost is the sole implementation.  The interface exists so that
// MooncakeBackend can receive an AgentInterface& via dependency injection
// without depending on the full AgentHost (RPC, executor, TE manager).
// =========================================================================

class AgentInterface {
   public:
    virtual ~AgentInterface() = default;

    // ---- Bootstrap (blocking) ----

    virtual bool waitUntilRegistered(std::chrono::milliseconds timeout) = 0;

    virtual GroupView waitUntilGroupReady(
        GroupId group_id, std::chrono::milliseconds timeout) = 0;

    // ---- Group management ----

    virtual void registerGroup(GroupDeclaration declaration,
                               MooncakeBackend* backend) = 0;

    virtual void unregisterGroup(GroupId group_id) = 0;

    virtual GroupView getGroupView(GroupId group_id) = 0;

    virtual void publishLocalEndpoint(GroupEndpointPublication endpoint) = 0;

    // ---- Membership proposals (synchronous, blocking on caller thread) ----

    virtual ProposeViewUpdateResponse proposeActivate(
        GroupId group_id, const std::vector<GlobalRank>& ranks) = 0;

    virtual ProposeViewUpdateResponse proposeDeactivate(
        GroupId group_id, const std::vector<GlobalRank>& ranks) = 0;

    // ---- Thread-safe async interface (called from worker threads) ----

    virtual void pushTransferObservation(
        GroupId group_id, std::vector<uint8_t> attempted_ranks,
        std::vector<uint8_t> failed_ranks,
        std::vector<uint8_t> succeeded_ranks) = 0;
};

// =========================================================================
// AgentRpcServiceImpl — thin RPC handler for Coordinator→Agent pushes.
// =========================================================================

class AgentHost;

class AgentRpcServiceImpl : public AgentRpcService {
   public:
    explicit AgentRpcServiceImpl(AgentHost& host) : host_(host) {}

    void onPeerJoined(PeerJoinedPush push) override;
    void onRankStateUpdate(RankStateUpdatePush push) override;
    void onViewUpdate(coro_rpc::context<ViewUpdateAck> ctx,
                      ViewUpdatePush push) override;

   private:
    AgentHost& host_;
};

// =========================================================================
// AgentHost — execution host for the AgentStateMachine state machine.
//
// Implements the AgentInterface for MooncakeBackend dependency injection.
//
// Owns:
//   - AgentStateMachine (pure state machine)
//   - SerializedExecutor (single-threaded event loop)
//   - Agent RPC server (receives Coordinator pushes)
//   - Coordinator RPC client (sends register/heartbeat/propose/etc.)
//   - LinkManager event callback
//   - Transfer observation queue (thread-safe)
//
// Bootstrap: waitUntilRegistered() / waitUntilGroupReady() block the caller
// thread using std::promise/std::future until the executor thread completes
// the initial registration handshake.
// =========================================================================

class AgentHost : public AgentInterface {
   public:
    static constexpr auto kCoordinatorAddrTimeout = std::chrono::seconds(30);
    static constexpr auto kCoordinatorAddrPollInterval =
        std::chrono::milliseconds(100);

    AgentHost(c10::intrusive_ptr<c10d::Store> store, const std::string& host_ip,
              GlobalRank rank, int max_world_size, LinkManager& link_manager);

    ~AgentHost() override;

    // ---- Lifecycle ----

    void start();
    void shutdown();

    // ---- Agent interface implementation ----

    bool waitUntilRegistered(std::chrono::milliseconds timeout) override;
    GroupView waitUntilGroupReady(GroupId group_id,
                                  std::chrono::milliseconds timeout) override;

    void registerGroup(GroupDeclaration declaration,
                       MooncakeBackend* backend) override;
    void unregisterGroup(GroupId group_id) override;
    GroupView getGroupView(GroupId group_id) override;
    void publishLocalEndpoint(GroupEndpointPublication endpoint) override;

    ProposeViewUpdateResponse proposeActivate(
        GroupId group_id, const std::vector<GlobalRank>& ranks) override;

    ProposeViewUpdateResponse proposeDeactivate(
        GroupId group_id, const std::vector<GlobalRank>& ranks) override;

    void pushTransferObservation(GroupId group_id,
                                 std::vector<uint8_t> attempted_ranks,
                                 std::vector<uint8_t> failed_ranks,
                                 std::vector<uint8_t> succeeded_ranks) override;

    // ---- RPC push callbacks (called from AgentRpcServiceImpl) ----

    void postPeerJoined(PeerJoinedPush push);
    void postRankStateUpdate(RankStateUpdatePush push);
    void postViewUpdate(coro_rpc::context<ViewUpdateAck> ctx,
                        ViewUpdatePush push);
    // ---- LinkManager event callback ----

    void postTELinkEvent(TELinkEvent event);

   private:
    AgentStateMachine agent_;      // pure state machine
    SerializedExecutor executor_;  // serialized event loop

    // Process-level TE link manager (non-owning, owned by ProcessContext).
    LinkManager& link_manager_;

    c10::intrusive_ptr<c10d::Store> store_;
    std::string host_ip_;
    GlobalRank rank_;
    int max_world_size_;

    std::string coordinator_addr_;
    uint64_t agent_session_epoch_ = 0;

    // RPC infrastructure.
    std::unique_ptr<RpcServer> rpc_server_;
    std::unique_ptr<RpcClient> rpc_client_;
    std::unique_ptr<AgentRpcServiceImpl> rpc_impl_;

    // Bootstrap synchronization: condition_variable so callers can retry
    // waitUntilRegistered() after a timeout without double-get_future() UB.
    mutable std::mutex registration_mutex_;
    std::condition_variable registration_cv_;
    bool registration_done_ = false;

    // group_ready_promises_ is fulfilled when declareGroup returns and
    // the GroupView is applied.
    std::unordered_map<GroupId,
                       std::vector<std::shared_ptr<std::promise<GroupView>>>>
        group_ready_promises_;

    // Backend registry: for view application and link reset fanout.
    // Accessed only from the executor thread.
    std::unordered_map<GroupId, MooncakeBackend*> backends_;

    // Transfer observation queue: worker thread → executor.
    ThreadSafeQueue<TransferObservationEvent> observation_queue_;

    // ---- Internal methods (run on executor thread) ----

    void startRegisterRpc();
    void tick();

    void runEffects(const AgentApplyResult& effects);
    template <typename F>
    void forEachBackend(F&& func) {
        for (auto& [group_id, backend] : backends_) {
            func(backend);
        }
    }
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_AGENT_HOST_H
