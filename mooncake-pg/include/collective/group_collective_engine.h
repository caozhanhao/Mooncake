#ifndef MOONCAKE_PG_COLLECTIVE_GROUP_COLLECTIVE_ENGINE_H
#define MOONCAKE_PG_COLLECTIVE_GROUP_COLLECTIVE_ENGINE_H

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>

#include <cuda_alike.h>

#include "collective/allreduce/allreduce.h"
#include "collective/endpoint.h"
#include "error_types.h"

namespace mooncake {

class CollectiveBufferPool;
class HostTransferExecutor;
class CollectiveChannels;
class DeviceLinkManager;
class CollectiveRuntime;
class LinkManager;
class TransferEngine;
struct GroupView;
template <typename Plan>
class DevicePlan;

// Complete group-scoped owner for the planned collective path. Process-level
// pools, the Host transfer executor and link managers are borrowed services.
// The communicator only dispatches into this object and coordinates its final
// close with legacy communicator teardown.
class GroupCollectiveEngine {
   public:
    static PGResult<std::unique_ptr<GroupCollectiveEngine>> create(
        CollectiveBufferPool& buffer_pool,
        HostTransferExecutor& host_transfer_executor,
        DeviceLinkManager& device_links, LinkManager& host_links,
        TransferEngine* transfer_engine, DeviceId device,
        InGroupRank self_in_group_rank, std::string te_location,
        size_t collective_timeout_us,
        std::function<PGResult<void>(InGroupRank)> report_failure);
    ~GroupCollectiveEngine() noexcept;

    const GroupEndpointV2& endpoint() const { return endpoint_; }

    bool shouldUsePlannedAllReduce(DataType datatype, ReduceOp op) const;
    PGResult<void> applyView(const GroupView& view);
    PGResult<void> allReduce(const AllReduceRequest& request,
                             cudaStream_t stream, int32_t* failed_ranks_hint,
                             size_t failed_ranks_hint_count);

    // Terminal execution shutdown. It drains eager work, stops monitoring,
    // and leaves endpoint/peer-signal storage alive until close() follows the
    // control-plane group unregister.
    bool stop(std::chrono::milliseconds timeout);
    bool close(bool resources_safe);

    GroupCollectiveEngine(const GroupCollectiveEngine&) = delete;
    GroupCollectiveEngine& operator=(const GroupCollectiveEngine&) = delete;

   private:
    GroupCollectiveEngine(CollectiveBufferPool& buffer_pool,
                          DeviceLinkManager& device_links,
                          LinkManager& host_links,
                          TransferEngine* transfer_engine, DeviceId device,
                          InGroupRank self_in_group_rank,
                          std::string te_location)
        : buffer_pool_(buffer_pool),
          device_links_(device_links),
          host_links_(host_links),
          transfer_engine_(transfer_engine),
          device_(device),
          self_in_group_rank_(self_in_group_rank),
          te_location_(std::move(te_location)) {}

    CollectiveBufferPool& buffer_pool_;
    DeviceLinkManager& device_links_;
    LinkManager& host_links_;
    TransferEngine* transfer_engine_ = nullptr;
    DeviceId device_ = kInvalidDeviceId;
    InGroupRank self_in_group_rank_ = -1;
    std::string te_location_;

    std::unique_ptr<CollectiveChannels> channels_;
    std::unique_ptr<DevicePlan<AllReducePlan>> allreduce_plan_;
    std::unique_ptr<CollectiveRuntime> runtime_;
    GroupEndpointV2 endpoint_;
    AllReduceProtocol allreduce_protocol_ = AllReduceProtocol::Legacy;
    uint32_t next_channel_index_ = 0;
    bool closed_ = false;
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_COLLECTIVE_GROUP_COLLECTIVE_ENGINE_H
