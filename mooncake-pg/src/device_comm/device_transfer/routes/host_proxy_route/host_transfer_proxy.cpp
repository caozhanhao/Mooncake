#include "device_comm/device_transfer/routes/host_proxy_route/host_transfer_proxy.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <glog/logging.h>
#include <transfer_engine.h>

#include "device_comm/device_transfer/transfer_types.cuh"
#include "gpu_runtime.h"
#include "memory_location.h"

namespace mooncake {
namespace {

constexpr auto kCommandPollInterval = std::chrono::microseconds(50);

static_assert(std::atomic_ref<uint64_t>::is_always_lock_free,
              "GPU/host command sequences require lock-free uint64 atomics");

bool isTerminalFailure(TransferStatusEnum status) {
    return status == TransferStatusEnum::INVALID ||
           status == TransferStatusEnum::CANCELED ||
           status == TransferStatusEnum::TIMEOUT ||
           status == TransferStatusEnum::FAILED;
}

}  // namespace

struct HostTransferProxy::Lane {
    // `batch_id` belongs to exactly the transfer named by the current state:
    //
    //   WaitingForCommand
    //        -> PayloadTransferInFlight (unless the payload is empty)
    //        -> SignalReadInFlight
    //        -> SignalWriteInFlight
    //        -> WaitingForCommand
    //
    // Only finishCommand() returns the command slot to the GPU producer.
    enum class State : uint8_t {
        WaitingForCommand,
        PayloadTransferInFlight,
        SignalReadInFlight,
        SignalWriteInFlight,
    };

    ~Lane() noexcept {
        if (batch_id != INVALID_BATCH_ID) {
            LOG(ERROR) << "Destroying a host-proxy lane with an outstanding "
                          "TE batch";
        }
    }

    HostProxyCommandSlot* host_slot = nullptr;
    uint64_t* signal_staging = nullptr;
    // The state-machine fields below belong exclusively to the worker thread.
    // Lifecycle methods inspect only this atomic bit and the shared slot
    // sequences, avoiding races with command and BatchID updates.
    std::atomic<bool> busy = false;

    State state = State::WaitingForCommand;
    uint64_t sequence = 0;
    HostProxyCommand command;
    TransferMetadata::SegmentID target_segment_id = 0;
    BatchID batch_id = INVALID_BATCH_ID;
};

struct HostTransferProxy::LaneSet {
    int device_index = -1;
    HostProxyCommandSlot* host_slots = nullptr;
    HostProxyCommandSlot* device_slots = nullptr;
    std::unique_ptr<std::array<uint64_t, kTransferLaneCount>> signal_staging;
    std::array<Lane, kTransferLaneCount> lanes;
    bool signal_staging_registered = false;
};

HostTransferProxy::LaneSet* HostTransferProxy::findLaneSet(
    int device_index) const noexcept {
    for (const auto& lane_set : lane_sets_) {
        if (lane_set->device_index == device_index) {
            return lane_set.get();
        }
    }
    return nullptr;
}

void HostTransferProxy::releaseLaneSets() noexcept {
    for (auto& lane_set : lane_sets_) {
        if (lane_set->signal_staging_registered) {
            const int result =
                engine_.unregisterLocalMemory(lane_set->signal_staging->data());
            if (result != 0) {
                LOG(ERROR) << "Failed to unregister host-proxy signal "
                              "staging for CUDA device "
                           << lane_set->device_index << ", rc=" << result;
                // TE may still refer to this memory. Leak only the registered
                // staging array; the command slots are unrelated to TE.
                lane_set->signal_staging.release();
            } else {
                lane_set->signal_staging_registered = false;
            }
        }
        lane_set->signal_staging.reset();

        if (lane_set->host_slots) {
            std::destroy_n(lane_set->host_slots, kTransferLaneCount);
            const auto result = cudaFreeHost(lane_set->host_slots);
            if (result != cudaSuccess) {
                LOG(ERROR) << "Failed to free host-proxy slots for CUDA "
                              "device "
                           << lane_set->device_index << ": "
                           << cudaGetErrorString(result);
            }
            lane_set->host_slots = nullptr;
            lane_set->device_slots = nullptr;
        }
    }
    lane_sets_.clear();
}

uint64_t HostTransferProxy::loadSubmitted(const Lane& lane) {
    return std::atomic_ref(lane.host_slot->submitted_sequence)
        .load(std::memory_order_acquire);
}

uint64_t HostTransferProxy::loadCompleted(const Lane& lane) {
    return std::atomic_ref(lane.host_slot->completed_sequence)
        .load(std::memory_order_acquire);
}

bool HostTransferProxy::laneIdle(const Lane& lane) {
    return !lane.busy.load(std::memory_order_acquire) &&
           loadSubmitted(lane) == loadCompleted(lane);
}

bool HostTransferProxy::laneSetIdle(const LaneSet& lane_set) {
    for (const auto& lane : lane_set.lanes) {
        if (!laneIdle(lane)) return false;
    }
    return true;
}

bool HostTransferProxy::allLaneSetsIdle() const {
    for (const auto& lane_set : lane_sets_) {
        if (!laneSetIdle(*lane_set)) return false;
    }
    return true;
}

std::optional<TransferMetadata::SegmentID> HostTransferProxy::resolvePeer(
    uint32_t peer_index) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (peer_index >= peers_.size()) return std::nullopt;
    auto& peer = peers_[peer_index];
    if (peer.te_server_name.empty()) return std::nullopt;
    if (peer.segment_id) return peer.segment_id;

    const auto segment_id = engine_.openSegment(peer.te_server_name);
    if (segment_id == static_cast<TransferMetadata::SegmentID>(-1)) {
        LOG(WARNING) << "HostTransferProxy failed to open TE segment for "
                     << "peer " << peer_index << ": " << peer.te_server_name;
        return std::nullopt;
    }
    peer.segment_id = segment_id;
    return segment_id;
}

void HostTransferProxy::closePeerSegments() noexcept {
    for (uint32_t peer_index = 0; peer_index < peers_.size(); ++peer_index) {
        auto& peer = peers_[peer_index];
        if (!peer.segment_id) continue;
        const int result = engine_.closeSegment(*peer.segment_id);
        if (result != 0) {
            LOG(WARNING) << "HostTransferProxy leaked TE segment for peer "
                         << peer_index << ", rc=" << result;
        }
        peer.segment_id.reset();
    }
}

void HostTransferProxy::finishCommand(Lane& lane,
                                      HostProxyCommandResult result) {
    lane.host_slot->result = result;
    std::atomic_ref(lane.host_slot->completed_sequence)
        .store(lane.sequence, std::memory_order_release);
    lane.state = Lane::State::WaitingForCommand;
    lane.sequence = 0;
    lane.command = {};
    lane.target_segment_id = 0;
    lane.busy.store(false, std::memory_order_release);
    state_changed_.notify_all();
}

void HostTransferProxy::releaseBatch(Lane& lane) {
    if (lane.batch_id == INVALID_BATCH_ID) return;
    const auto batch_id = lane.batch_id;
    lane.batch_id = INVALID_BATCH_ID;
    const auto status = engine_.freeBatchID(batch_id);
    if (!status.ok()) {
        // The initial implementation does not retain and drain failed TE
        // batches. Forget the BatchID after reporting the leak.
        LOG(WARNING) << "Host-proxy TE BatchID leaked after "
                        "freeBatchID failed: "
                     << status.message();
    }
}

bool HostTransferProxy::submitBatch(Lane& lane,
                                    const TransferRequest& request) {
    lane.batch_id = engine_.allocateBatchID(1);
    if (lane.batch_id == INVALID_BATCH_ID) return false;

    const auto status = engine_.submitTransfer(lane.batch_id, {request});
    if (status.ok()) return true;

    LOG(WARNING) << "Host-proxy TE submission failed: " << status.message();
    releaseBatch(lane);
    return false;
}

HostTransferProxy::BatchPollResult HostTransferProxy::pollBatch(Lane& lane) {
    TransferStatus transfer_status{};
    const auto status =
        engine_.getTransferStatus(lane.batch_id, 0, transfer_status);
    if (!status.ok()) {
        LOG(WARNING) << "Host-proxy TE status query failed: "
                     << status.message();
        releaseBatch(lane);
        return BatchPollResult::Failed;
    }
    if (transfer_status.s == TransferStatusEnum::COMPLETED) {
        releaseBatch(lane);
        return BatchPollResult::Succeeded;
    }
    if (isTerminalFailure(transfer_status.s)) {
        releaseBatch(lane);
        return BatchPollResult::Failed;
    }
    return BatchPollResult::InFlight;
}

bool HostTransferProxy::tryStartCommand(Lane& lane) {
    const uint64_t completed = loadCompleted(lane);
    const uint64_t submitted = loadSubmitted(lane);
    if (submitted == completed) return false;

    lane.busy.store(true, std::memory_order_release);
    lane.sequence = submitted;
    // loadSubmitted() acquired the GPU's publication. Snapshot the shared
    // command before starting asynchronous TE work.
    lane.command = lane.host_slot->command;

    // A producer may only publish the next sequence after observing the
    // previous completion. Anything else means the single-slot protocol
    // was violated; fail the observed command instead of guessing.
    if (completed == UINT64_MAX || submitted != completed + 1 ||
        lane.command.signal_remote_addr == 0 ||
        lane.command.signal_delta == 0 ||
        lane.command.signal_delta >= (uint64_t{1} << 63) ||
        lane.command.target_peer_index == UINT32_MAX) {
        finishCommand(lane, HostProxyCommandResult::Failed);
        return true;
    }

    const auto segment = resolvePeer(lane.command.target_peer_index);
    if (!segment) {
        finishCommand(lane, HostProxyCommandResult::Failed);
        return true;
    }
    lane.target_segment_id = *segment;

    if (lane.command.size == 0) {
        startSignalRead(lane);
        return true;
    }
    if (lane.command.local_addr == 0 || lane.command.remote_addr == 0) {
        finishCommand(lane, HostProxyCommandResult::Failed);
        return true;
    }

    startPayloadTransfer(lane);
    return true;
}

void HostTransferProxy::startPayloadTransfer(Lane& lane) {
    lane.state = Lane::State::PayloadTransferInFlight;
    if (!submitBatch(lane, TransferRequest{
                               .opcode = TransferRequest::WRITE,
                               .source = reinterpret_cast<void*>(
                                   lane.command.local_addr),
                               .target_id = lane.target_segment_id,
                               .target_offset = lane.command.remote_addr,
                               .length = static_cast<size_t>(lane.command.size),
                           })) {
        finishCommand(lane, HostProxyCommandResult::Failed);
    }
}

void HostTransferProxy::startSignalRead(Lane& lane) {
    // TE exposes READ/WRITE but no remote atomic add. The single-publisher
    // contract lets the fallback route implement add as a remote read followed
    // by a write through this lane's registered staging word.
    lane.state = Lane::State::SignalReadInFlight;
    if (!submitBatch(lane, TransferRequest{
                               .opcode = TransferRequest::READ,
                               .source = lane.signal_staging,
                               .target_id = lane.target_segment_id,
                               .target_offset = lane.command.signal_remote_addr,
                               .length = sizeof(uint64_t),
                           })) {
        finishCommand(lane, HostProxyCommandResult::Failed);
    }
}

void HostTransferProxy::startSignalWrite(Lane& lane) {
    *lane.signal_staging += lane.command.signal_delta;
    lane.state = Lane::State::SignalWriteInFlight;
    if (!submitBatch(lane, TransferRequest{
                               .opcode = TransferRequest::WRITE,
                               .source = lane.signal_staging,
                               .target_id = lane.target_segment_id,
                               .target_offset = lane.command.signal_remote_addr,
                               .length = sizeof(uint64_t),
                           })) {
        finishCommand(lane, HostProxyCommandResult::Failed);
    }
}

bool HostTransferProxy::stepPayloadTransfer(Lane& lane) {
    const auto result = pollBatch(lane);
    if (result == BatchPollResult::InFlight) return false;
    if (result == BatchPollResult::Failed) {
        finishCommand(lane, HostProxyCommandResult::Failed);
        return true;
    }
    startSignalRead(lane);
    return true;
}

bool HostTransferProxy::stepSignalRead(Lane& lane) {
    const auto result = pollBatch(lane);
    if (result == BatchPollResult::InFlight) return false;
    if (result == BatchPollResult::Failed) {
        finishCommand(lane, HostProxyCommandResult::Failed);
        return true;
    }
    startSignalWrite(lane);
    return true;
}

bool HostTransferProxy::stepSignalWrite(Lane& lane) {
    const auto result = pollBatch(lane);
    if (result == BatchPollResult::InFlight) return false;
    finishCommand(lane, result == BatchPollResult::Succeeded
                            ? HostProxyCommandResult::Succeeded
                            : HostProxyCommandResult::Failed);
    return true;
}

bool HostTransferProxy::step(Lane& lane) {
    switch (lane.state) {
        case Lane::State::WaitingForCommand:
            return tryStartCommand(lane);
        case Lane::State::PayloadTransferInFlight:
            return stepPayloadTransfer(lane);
        case Lane::State::SignalReadInFlight:
            return stepSignalRead(lane);
        case Lane::State::SignalWriteInFlight:
            return stepSignalWrite(lane);
    }
    return false;
}

void HostTransferProxy::run() noexcept {
    try {
        std::vector<LaneSet*> snapshot;
        for (;;) {
            {
                std::unique_lock<std::mutex> lock(mutex_);
                if (stop_requested_) break;
                snapshot.clear();
                snapshot.reserve(lane_sets_.size());
                for (const auto& lane_set : lane_sets_) {
                    snapshot.push_back(lane_set.get());
                }
            }

            bool progressed = false;
            for (auto* lane_set : snapshot) {
                for (auto& lane : lane_set->lanes) {
                    progressed |= step(lane);
                }
            }

            if (!progressed) {
                std::unique_lock<std::mutex> lock(mutex_);
                state_changed_.wait_for(lock, kCommandPollInterval,
                                        [this] { return stop_requested_; });
            }
        }
    } catch (const std::exception& error) {
        LOG(ERROR) << "HostTransferProxy worker stopped after an "
                      "exception: "
                   << error.what();
        std::lock_guard<std::mutex> lock(mutex_);
        worker_failed_ = true;
        state_changed_.notify_all();
    } catch (...) {
        LOG(ERROR) << "HostTransferProxy worker stopped after an unknown "
                      "exception";
        std::lock_guard<std::mutex> lock(mutex_);
        worker_failed_ = true;
        state_changed_.notify_all();
    }
}

void HostTransferProxy::forceStop() noexcept {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_requested_ = true;
    }
    state_changed_.notify_all();
    if (worker_.joinable()) worker_.join();
}

HostTransferProxy::HostTransferProxy(TransferEngine& engine,
                                     uint32_t peer_capacity)
    : engine_(engine), peers_(peer_capacity) {}

HostTransferProxy::~HostTransferProxy() noexcept {
    const auto result = shutdown();
    if (!result.has_value()) {
        LOG(ERROR) << "HostTransferProxy shutdown failed during destruction: "
                   << result.error().message;
        forceStop();
        // A parked GPU kernel can still reference a device's fixed command
        // slots. Leave their raw mapped allocations registered and alive.
    }
}

PGResult<void> HostTransferProxy::start() {
    std::lock_guard<std::mutex> lock(mutex_);
    PG_VALIDATE_STATE(!stop_requested_, "HostTransferProxy is shut down");
    if (started_) return {};
    try {
        worker_ = std::thread([this] { run(); });
    } catch (const std::exception& error) {
        return makePGError(
            PGErrorCode::SystemError,
            std::string("failed to start HostTransferProxy: ") + error.what());
    }
    started_ = true;
    return {};
}

PGResult<HostProxyCommandSlot*> HostTransferProxy::addDevice(int device_index) {
    PG_VALIDATE_ARG(device_index >= 0, "invalid host-proxy CUDA device");

    PG_TRY(auto device_guard, GpuDeviceGuard::create(device_index));
    std::lock_guard<std::mutex> lock(mutex_);
    PG_VALIDATE_STATE(started_ && !stop_requested_ && !worker_failed_,
                      "HostTransferProxy is not running");
    if (const auto* existing = findLaneSet(device_index)) {
        return existing->device_slots;
    }

    std::unique_ptr<LaneSet> lane_set;
    try {
        lane_sets_.reserve(lane_sets_.size() + 1);
        lane_set = std::make_unique<LaneSet>();
        lane_set->signal_staging =
            std::make_unique<std::array<uint64_t, kTransferLaneCount>>();
    } catch (const std::exception& error) {
        return makePGError(
            PGErrorCode::SystemError,
            std::string("failed to allocate fixed host-proxy lanes: ") +
                error.what());
    }

    lane_set->device_index = device_index;
    constexpr size_t slot_count = kTransferLaneCount;
    constexpr size_t slot_size = slot_count * sizeof(HostProxyCommandSlot);
    const auto allocation =
        cudaHostAlloc(reinterpret_cast<void**>(&lane_set->host_slots),
                      slot_size, cudaHostAllocMapped | cudaHostAllocPortable);
    if (allocation != cudaSuccess) {
        return makePGError(
            PGErrorCode::SystemError,
            std::string("failed to allocate fixed host-proxy slots: ") +
                cudaGetErrorString(allocation));
    }
    std::uninitialized_value_construct_n(lane_set->host_slots, slot_count);

    const auto map_result = cudaHostGetDevicePointer(
        reinterpret_cast<void**>(&lane_set->device_slots), lane_set->host_slots,
        0);
    if (map_result != cudaSuccess) {
        std::destroy_n(lane_set->host_slots, slot_count);
        const auto released = cudaFreeHost(lane_set->host_slots);
        if (released != cudaSuccess) {
            LOG(ERROR) << "Failed to release unmapped host-proxy slots: "
                       << cudaGetErrorString(released);
        }
        lane_set->host_slots = nullptr;
        return makePGError(PGErrorCode::SystemError,
                           std::string("cudaHostGetDevicePointer failed: ") +
                               cudaGetErrorString(map_result));
    }

    const int registered = engine_.registerLocalMemory(
        lane_set->signal_staging->data(), sizeof(*lane_set->signal_staging),
        kWildcardLocation,
        /*remote_accessible=*/false);
    if (registered != 0) {
        std::destroy_n(lane_set->host_slots, slot_count);
        const auto released = cudaFreeHost(lane_set->host_slots);
        if (released != cudaSuccess) {
            LOG(ERROR) << "Failed to release host-proxy slots after signal "
                          "staging registration failed: "
                       << cudaGetErrorString(released);
        }
        lane_set->host_slots = nullptr;
        lane_set->device_slots = nullptr;
        return makePGError(PGErrorCode::TransferEngineError,
                           "failed to register host-proxy signal staging, rc=" +
                               std::to_string(registered));
    }
    lane_set->signal_staging_registered = true;

    for (uint32_t index = 0; index < kTransferLaneCount; ++index) {
        auto& lane = lane_set->lanes[index];
        lane.host_slot = lane_set->host_slots + index;
        lane.signal_staging = lane_set->signal_staging->data() + index;
    }
    auto* const device_slots = lane_set->device_slots;
    lane_sets_.push_back(std::move(lane_set));
    state_changed_.notify_all();
    return device_slots;
}

PGResult<void> HostTransferProxy::installPeerEndpoint(
    uint32_t peer_index, const std::optional<HostProxyEndpoint>& endpoint) {
    std::lock_guard<std::mutex> lock(mutex_);
    PG_VALIDATE_ARG(peer_index < peers_.size(),
                    "host-proxy peer index is out of range");
    auto& peer = peers_[peer_index];
    const std::string next_name = endpoint ? endpoint->te_server_name : "";
    if (peer.te_server_name == next_name) return {};

    PG_VALIDATE_STATE(allLaneSetsIdle(),
                      "cannot replace a host-proxy endpoint while transfers "
                      "are in flight");
    if (peer.segment_id) {
        const int result = engine_.closeSegment(*peer.segment_id);
        if (result != 0) {
            LOG(WARNING) << "HostTransferProxy leaked the replaced TE segment "
                         << "for peer " << peer_index << ", rc=" << result;
        }
        peer.segment_id.reset();
    }
    peer.te_server_name = next_name;
    return {};
}

PGResult<void> HostTransferProxy::waitUntilIdle(int device_index) {
    std::unique_lock<std::mutex> lock(mutex_);
    const auto* lane_set = findLaneSet(device_index);
    PG_VALIDATE_ARG(lane_set, "host-proxy CUDA device is not registered");
    state_changed_.wait(lock, [this, lane_set] {
        return worker_failed_ || laneSetIdle(*lane_set);
    });
    PG_VALIDATE_STATE(!worker_failed_, "HostTransferProxy worker has failed");
    return {};
}

PGResult<void> HostTransferProxy::waitUntilIdle(
    int device_index, std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    const auto* lane_set = findLaneSet(device_index);
    PG_VALIDATE_ARG(lane_set, "host-proxy CUDA device is not registered");
    const bool ready = state_changed_.wait_for(lock, timeout, [this, lane_set] {
        return worker_failed_ || laneSetIdle(*lane_set);
    });
    if (!ready) {
        return makePGError(PGErrorCode::Timeout,
                           "host-proxy device did not become idle in time");
    }
    PG_VALIDATE_STATE(!worker_failed_, "HostTransferProxy worker has failed");
    return {};
}

PGResult<void> HostTransferProxy::shutdown() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stop_requested_) return {};
        PG_VALIDATE_STATE(allLaneSetsIdle(),
                          "HostTransferProxy still has in-flight commands");
        stop_requested_ = true;
    }
    state_changed_.notify_all();
    if (worker_.joinable()) worker_.join();

    std::lock_guard<std::mutex> lock(mutex_);
    closePeerSegments();
    releaseLaneSets();
    return {};
}

}  // namespace mooncake
