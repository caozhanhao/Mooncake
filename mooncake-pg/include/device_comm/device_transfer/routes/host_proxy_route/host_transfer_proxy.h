#ifndef MOONCAKE_PG_DEVICE_COMM_DEVICE_TRANSFER_ROUTES_HOST_PROXY_ROUTE_HOST_TRANSFER_PROXY_H
#define MOONCAKE_PG_DEVICE_COMM_DEVICE_TRANSFER_ROUTES_HOST_PROXY_ROUTE_HOST_TRANSFER_PROXY_H

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <transfer_engine.h>

#include "device_comm/device_transfer/routes/host_proxy_route/host_proxy_types.cuh"
#include "device_comm/device_transfer/transfer_endpoint.h"
#include "error_types.h"

namespace mooncake {

// Transfer-service-owned consumer for commands that cannot use a direct
// device path. The proxy knows the host TransferEngine but no caller-specific
// algorithm or metadata.
class HostTransferProxy {
   public:
    HostTransferProxy(TransferEngine& engine, uint32_t peer_capacity);
    ~HostTransferProxy() noexcept;

    HostTransferProxy(const HostTransferProxy&) = delete;
    HostTransferProxy& operator=(const HostTransferProxy&) = delete;

    PGResult<void> start();

    // Create the fixed command-slot set used by one transfer-service device.
    PGResult<HostProxyCommandSlot*> addDevice(int device_index);

    PGResult<void> installPeerEndpoint(
        uint32_t peer_index, const std::optional<HostProxyEndpoint>& endpoint);

    PGResult<void> waitUntilIdle(int device_index);
    PGResult<void> waitUntilIdle(int device_index,
                                 std::chrono::milliseconds timeout);
    PGResult<void> shutdown();

   private:
    struct Lane;
    struct LaneSet;

    enum class BatchPollResult : uint8_t {
        InFlight,
        Succeeded,
        Failed,
    };

    // These helpers inspect collections protected by mutex_.
    LaneSet* findLaneSet(int device_index) const noexcept;
    void releaseLaneSets() noexcept;

    static uint64_t loadSubmitted(const Lane& lane);
    static uint64_t loadCompleted(const Lane& lane);
    static bool laneIdle(const Lane& lane);
    static bool laneSetIdle(const LaneSet& lane_set);
    bool allLaneSetsIdle() const;
    std::optional<TransferMetadata::SegmentID> resolvePeer(uint32_t peer_index);
    void closePeerSegments() noexcept;

    void finishCommand(Lane& lane, HostProxyCommandResult result);
    void releaseBatch(Lane& lane);
    bool submitBatch(Lane& lane, const TransferRequest& request);
    BatchPollResult pollBatch(Lane& lane);
    bool tryStartCommand(Lane& lane);
    void startPayloadTransfer(Lane& lane);
    void startSignalRead(Lane& lane);
    void startSignalWrite(Lane& lane);
    bool stepPayloadTransfer(Lane& lane);
    bool stepSignalRead(Lane& lane);
    bool stepSignalWrite(Lane& lane);
    bool step(Lane& lane);
    void run() noexcept;
    void forceStop() noexcept;

    TransferEngine& engine_;
    struct Peer {
        std::string te_server_name;
        std::optional<TransferMetadata::SegmentID> segment_id;
    };
    std::vector<Peer> peers_;
    mutable std::mutex mutex_;
    std::condition_variable state_changed_;
    std::vector<std::unique_ptr<LaneSet>> lane_sets_;
    std::thread worker_;
    bool started_ = false;
    bool stop_requested_ = false;
    bool worker_failed_ = false;
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_DEVICE_COMM_DEVICE_TRANSFER_ROUTES_HOST_PROXY_ROUTE_HOST_TRANSFER_PROXY_H
