#ifndef MOONCAKE_PG_AGENT_HOST_H
#define MOONCAKE_PG_AGENT_HOST_H

#include <chrono>
#include <future>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <c10/util/intrusive_ptr.h>
#include <torch/csrc/distributed/c10d/Store.hpp>

#include "agent.h"
#include "rpc.h"
#include "serialized_executor.h"
#include "link_manager.h"

#include "pg_utils.h"
#include "mooncake_backend.h"

namespace mooncake {

class RpcServer;
class RpcClient;
class MooncakeBackend;

// =========================================================================
// Control Plane Architecture (Agent side)
// =========================================================================
//
// Each rank runs one AgentHost.  It owns the AgentStateMachine (pure state
// machine) and drives it via a SerializedExecutor.
//
//     MooncakeBackend                         AgentHost
//   +-----------------+              +---------------------------+
//   | proposeActivate |-> (sync) --->| call Coordinator RPC      |
//   | joinGroup       |-> post() --->| agent_.joinGroup()        |
//   | pushObservation |-> enqueue -->| observation_queue_        |
//   +-----------------+              +---------------------------+
//                                            |
//                                    SerializedExecutor (tick)
//                                            |
//                              +--------------------------------+
//                              | AgentStateMachine              |
//                              |  (pure state machine, no I/O)  |
//                              +--------------------------------+
//                                            |
//                                    returns Effect list
//                                            |
//                              +--------------------------------+
//                              | runEffects()                   |
//                              |  EnablePeerProbe -> LinkManager|
//                              |  SendObservation -> RPC        |
//                              |  ApplyViewToBackend -> backend |
//                              |  NotifyTEUnreachable -> fanout |
//                              |              ...               |
//                              +--------------------------------+
//
//   Coordinator pushes:                LinkManager events:
//   onPeerJoined -> postPeerJoined()   LinkUp/LinkDown -> postTELinkEvent()
//   onRankStateUpdate -> post...()     (both post to executor)
//   onViewUpdate -> post...()
//
// The Agent never makes autonomous decisions about health or membership.
// It strictly follows the Coordinator's authoritative broadcasts.

// AgentInterface - control-plane service interface exposed to MooncakeBackend.
class AgentInterface {
   public:
    virtual ~AgentInterface() = default;

    virtual bool waitUntilRegistered(std::chrono::milliseconds timeout) = 0;

    virtual GroupView waitUntilGroupReady(
        GroupId group_id, std::chrono::milliseconds timeout) = 0;

    virtual void waitUntilRankActive(GroupId group_id, GlobalRank rank,
                                     std::chrono::milliseconds timeout) = 0;

    virtual void joinGroup(const GroupView& group, bool auto_deactivate,
                           c10::intrusive_ptr<MooncakeBackend> backend) = 0;

    virtual void leaveGroup(GroupId group_id) = 0;

    virtual uint64_t getAgentSessionEpoch() = 0;

    virtual GroupView getGroupView(GroupId group_id) = 0;

    virtual void publishLocalEndpoint(GroupEndpointPublication endpoint) = 0;

    virtual ProposeViewUpdateResponse proposeActivate(
        GroupId group_id, const std::vector<GlobalRank>& ranks) = 0;

    virtual ProposeViewUpdateResponse proposeDeactivate(
        GroupId group_id, const std::vector<GlobalRank>& ranks) = 0;

    virtual void pushTransferObservation(GroupId group_id,
                                         std::vector<uint8_t> attempted_ranks,
                                         std::vector<uint8_t> failed_ranks_hint,
                                         std::vector<uint8_t> succeeded_ranks,
                                         bool local_success) = 0;
};

class AgentHost;

// AgentRpcServiceImpl  - thin RPC handler for Coordinator->Agent pushes.
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

// AgentHost - execution host for the agent state machine.
class AgentHost : public AgentInterface {
   public:
    static constexpr auto kCoordinatorAddrTimeout = std::chrono::seconds(30);
    static constexpr auto kCoordinatorAddrPollInterval =
        std::chrono::milliseconds(100);

    AgentHost(c10::intrusive_ptr<c10d::Store> store, const std::string& host_ip,
              GlobalRank rank, int max_world_size, LinkManager& link_manager);

    ~AgentHost() override;

    void start();
    void shutdown();

    bool waitUntilRegistered(std::chrono::milliseconds timeout) override;
    GroupView waitUntilGroupReady(GroupId group_id,
                                  std::chrono::milliseconds timeout) override;
    void waitUntilRankActive(GroupId group_id, GlobalRank rank,
                             std::chrono::milliseconds timeout) override;

    void joinGroup(const GroupView& group, bool auto_deactivate,
                   c10::intrusive_ptr<MooncakeBackend> backend) override;
    void leaveGroup(GroupId group_id) override;
    uint64_t getAgentSessionEpoch() override {
        return agent_.getAgentSessionEpoch();
    }
    GroupView getGroupView(GroupId group_id) override;
    void publishLocalEndpoint(GroupEndpointPublication endpoint) override;

    ProposeViewUpdateResponse proposeActivate(
        GroupId group_id, const std::vector<GlobalRank>& ranks) override;

    ProposeViewUpdateResponse proposeDeactivate(
        GroupId group_id, const std::vector<GlobalRank>& ranks) override;

    void pushTransferObservation(GroupId group_id,
                                 std::vector<uint8_t> attempted_ranks,
                                 std::vector<uint8_t> failed_ranks_hint,
                                 std::vector<uint8_t> succeeded_ranks,
                                 bool local_success) override;

    void postPeerJoined(PeerJoinedPush push);
    void postRankStateUpdate(RankStateUpdatePush push);
    void postViewUpdate(coro_rpc::context<ViewUpdateAck> ctx,
                        ViewUpdatePush push);

    void postTELinkEvent(TELinkEvent event);

   private:
    AgentStateMachine agent_;
    SerializedExecutor executor_;

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

    // Bootstrap synchronization: one-shot latch with executor-managed promises.
    bool registration_done_ = false;
    std::vector<std::shared_ptr<std::promise<void>>> registration_promises_;

    // group_ready_promises_ is fulfilled when joinGroup returns and
    // the GroupView is applied.
    std::unordered_map<GroupId,
                       std::vector<std::shared_ptr<std::promise<GroupView>>>>
        group_ready_promises_;

    // rank_active_promises_[group_id][rank] is fulfilled when a ViewUpdate
    // push activates `rank` in `group_id`.  Used by extension/replacement
    // ranks to block in joinGroup() until recover_ranks activates them.
    std::unordered_map<
        GroupId,
        std::unordered_map<GlobalRank,
                           std::vector<std::shared_ptr<std::promise<void>>>>>
        rank_active_promises_;

    // Backend registry: for view application and link reset.
    // Accessed only from the executor thread.
    std::unordered_map<GroupId, c10::intrusive_ptr<MooncakeBackend>> backends_;

    // Transfer observation queue: worker thread -> executor.
    ThreadSafeQueue<TransferObservationEvent> observation_queue_;

    void startRegisterRpc();
    void tick();

    void doJoinGroup(GroupView group, bool auto_deactivate,
                     c10::intrusive_ptr<MooncakeBackend> backend);
    void doPublishLocalEndpoint(GroupEndpointPublication endpoint);

    void runEffects(const AgentApplyResult& effects);
    template <typename F>
    void forEachBackend(F&& func) {
        for (auto& [group_id, backend] : backends_) {
            func(backend);
        }
    }
    template <typename F>
    void withBackend(GroupId group_id, F&& func) {
        auto it = backends_.find(group_id);
        if (it != backends_.end()) {
            func(it->second);
        }
    }
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_AGENT_HOST_H
