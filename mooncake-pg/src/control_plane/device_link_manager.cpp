#include "control_plane/device_link_manager.h"

#include <algorithm>
#include <exception>
#include <utility>

#include <glog/logging.h>

#include <cuda_alike.h>
#include <transport/device/device_transport.h>

#include "collective/buffer/collective_buffer_pool.h"
#include "gpu_runtime.h"

namespace mooncake {
namespace {

void appendChanged(std::vector<GlobalRank>& changed, GlobalRank peer) {
    if (std::find(changed.begin(), changed.end(), peer) == changed.end()) {
        changed.push_back(peer);
    }
}

}  // namespace

struct DeviceLinkManager::DeviceArenaState {
    DeviceId device = kInvalidDeviceId;
    uint64_t generation = 0;
    void* base = nullptr;
    uint64_t bytes = 0;
    cudaStream_t bootstrap_stream = nullptr;
    std::unique_ptr<device::RdmaTransport> rdma;
    std::optional<RdmaArenaBinding> published_rdma;
    std::vector<bool> rdma_connected;

    ~DeviceArenaState() noexcept {
        try {
            const GpuDeviceGuard guard(device);
            rdma.reset();
            if (bootstrap_stream) {
                const auto error = cudaStreamDestroy(bootstrap_stream);
                if (error != cudaSuccess) {
                    LOG(WARNING) << "Failed to destroy device-link bootstrap "
                                    "stream: "
                                 << cudaGetErrorString(error);
                }
            }
        } catch (const std::exception& error) {
            LOG(WARNING) << "Device arena cleanup failed: " << error.what();
        } catch (...) {
            LOG(WARNING) << "Device arena cleanup failed";
        }
    }
};

DeviceLinkManager::~DeviceLinkManager() noexcept { shutdown(); }

void DeviceLinkManager::shutdown() noexcept {
    {
        std::lock_guard<std::mutex> lock(event_callback_mutex_);
        event_callback_ = {};
    }
    try {
        std::lock_guard<std::mutex> lock(peers_mutex_);
        arenas_.clear();
        device_peers_.clear();
        target_rank_epochs_.clear();
        rank_ = kInvalidGlobalRank;
        max_world_size_ = 0;
    } catch (const std::exception& error) {
        LOG(WARNING) << "DeviceLinkManager cleanup failed: " << error.what();
    } catch (...) {
        LOG(WARNING) << "DeviceLinkManager cleanup failed";
    }
}

void DeviceLinkManager::init(GlobalRank rank, int max_world_size) {
    rank_ = rank;
    max_world_size_ = max_world_size;
    target_rank_epochs_.assign(max_world_size_, 0);
}

void DeviceLinkManager::bindCollectiveArena(const CollectiveArenaView& view) {
    std::lock_guard<std::mutex> lock(peers_mutex_);
    if (const auto found = arenas_.find(view.device);
        found != arenas_.end() &&
        found->second->generation == view.generation) {
        return;
    }

    const GpuDeviceGuard guard(view.device);
    auto arena = std::make_shared<DeviceArenaState>();
    arena->device = view.device;
    arena->generation = view.generation;
    arena->base = view.base;
    arena->bytes = view.bytes;
    arena->rdma_connected.assign(max_world_size_, false);
    arena->rdma = device::createIbgdaDeviceTransport();

    bool rdma_ready = arena->rdma != nullptr;
    if (rdma_ready) {
        rdma_ready =
            arena->rdma->initialize("", max_world_size_, max_world_size_) ==
                0 &&
            arena->rdma->registerMemory(arena->base, arena->bytes) == 0 &&
            arena->rdma->allocateControlBuffer() == 0;
    }
    if (rdma_ready) {
        rdma_ready =
            cudaStreamCreateWithFlags(&arena->bootstrap_stream,
                                      cudaStreamNonBlocking) == cudaSuccess;
    }
    if (rdma_ready) {
        rdma_ready =
            arena->rdma->createQueuePairs(arena->bootstrap_stream) == 0;
    }

    if (rdma_ready) {
        const auto metadata = arena->rdma->localMetadata();
        rdma_ready =
            metadata.raddr != 0 && metadata.rkey != 0 &&
            metadata.qpns.size() >= static_cast<size_t>(max_world_size_) &&
            metadata.lids.size() >= static_cast<size_t>(max_world_size_);
        if (rdma_ready) {
            arena->published_rdma = RdmaArenaBinding{
                .remote_access_address = static_cast<uint64_t>(metadata.raddr),
                .remote_key = static_cast<uint32_t>(metadata.rkey),
                .subnet_prefix = static_cast<uint64_t>(metadata.subnet_prefix),
                .interface_id = static_cast<uint64_t>(metadata.interface_id),
                .qpns = metadata.qpns,
                .lids = metadata.lids,
                .is_roce = arena->rdma->isRoce(),
            };
        }
    }

    if (!rdma_ready) {
        arena->rdma.reset();
        if (arena->bootstrap_stream) {
            cudaStreamDestroy(arena->bootstrap_stream);
            arena->bootstrap_stream = nullptr;
        }
        LOG(INFO) << "TE device RDMA is unavailable for collective device "
                  << view.device;
    }

    arenas_[view.device] = std::move(arena);
}

void DeviceLinkManager::populateEndpoint(GroupEndpointV2& endpoint) const {
    if (endpoint.device < 0) return;
    std::lock_guard<std::mutex> lock(peers_mutex_);
    const auto arena = arenas_.find(endpoint.device);
    if (arena == arenas_.end() || !arena->second->published_rdma.has_value()) {
        return;
    }
    if (arena->second->generation != endpoint.arena_generation) {
        return;
    }
    endpoint.device_rdma = arena->second->published_rdma;
}

void DeviceLinkManager::observeGroupView(
    const GroupView& view, const std::vector<uint64_t>& rank_epochs) {
    if (!rankInRange(rank_)) return;
    const auto& self = view.members[rank_].endpoint_v2;
    if (!self.has_value() || self->device < 0) return;

    const GpuDeviceGuard guard(self->device);
    std::vector<GlobalRank> changed;
    {
        std::lock_guard<std::mutex> lock(peers_mutex_);
        auto& observations = device_peers_[self->device];
        if (observations.empty()) observations.resize(max_world_size_);
        for (const auto peer : view.rank_order) {
            if (peer == rank_ || !rankInRange(peer)) continue;
            const auto& member = view.members[peer];
            if (!member.isMember() || !member.endpoint_v2.has_value()) continue;

            auto& observation = observations[peer];
            const auto& endpoint = *member.endpoint_v2;
            const auto old_epoch = target_rank_epochs_[peer];
            const bool old_available = observation.p2p_mapping != nullptr;
            target_rank_epochs_[peer] = rank_epochs[peer];

            if (!endpoint.device_p2p.has_value()) {
                observation.p2p_mapping.reset();
                observation.p2p_target_rank_epoch = 0;
                observation.p2p_arena_generation = 0;
            } else {
                const auto& binding = *endpoint.device_p2p;
                const bool needs_import =
                    observation.p2p_target_rank_epoch != rank_epochs[peer] ||
                    observation.p2p_arena_generation !=
                        endpoint.arena_generation ||
                    !observation.p2p_mapping;
                if (needs_import) {
                    auto mapping =
                        device::importP2pPeerBuffer(binding.opaque_handle);
                    const bool available =
                        mapping &&
                        (mapping->mappedBytes() == 0 ||
                         mapping->mappedBytes() >= endpoint.arena_bytes);
                    observation.p2p_mapping =
                        available ? std::move(mapping) : nullptr;
                    observation.p2p_arena_generation =
                        endpoint.arena_generation;
                }
                observation.p2p_target_rank_epoch = rank_epochs[peer];
            }
            if (old_epoch != target_rank_epochs_[peer] ||
                old_available != (observation.p2p_mapping != nullptr)) {
                appendChanged(changed, peer);
            }
        }

        const auto arena = arenas_.find(self->device);
        if (arena != arenas_.end() && arena->second->rdma) {
            connectRdmaPeers(*arena->second, observations, view, rank_epochs,
                             changed);
        }
    }

    for (const auto peer : changed) {
        PeerLinkUpdate update;
        {
            std::lock_guard<std::mutex> lock(peers_mutex_);
            update = makeLinkUpdate(peer);
        }
        emit(std::move(update));
    }
}

void DeviceLinkManager::connectRdmaPeers(
    DeviceArenaState& arena, std::vector<PeerObservation>& observations,
    const GroupView& view, const std::vector<uint64_t>& rank_epochs,
    std::vector<GlobalRank>& changed) {
    std::vector<int64_t> remote_addrs(max_world_size_);
    std::vector<int32_t> remote_keys(max_world_size_);
    std::vector<int32_t> remote_qpns(max_world_size_);
    std::vector<int32_t> remote_lids(max_world_size_);
    std::vector<int64_t> subnet_prefixes(max_world_size_);
    std::vector<int64_t> interface_ids(max_world_size_);
    std::vector<int> active(max_world_size_);
    std::vector<GlobalRank> connecting;

    for (const auto peer : view.rank_order) {
        const auto& member = view.members[peer];
        if (!member.isMember() || !member.endpoint_v2.has_value() ||
            !member.endpoint_v2->device_rdma.has_value()) {
            continue;
        }
        const auto& remote = *member.endpoint_v2->device_rdma;
        if (remote.is_roce != arena.rdma->isRoce() ||
            remote.qpns.size() <= static_cast<size_t>(rank_) ||
            remote.lids.size() <= static_cast<size_t>(rank_)) {
            continue;
        }

        auto& observation = observations[peer];
        if (peer != rank_ && arena.rdma_connected[peer] &&
            (observation.rdma_target_rank_epoch != rank_epochs[peer] ||
             observation.rdma_arena_generation !=
                 member.endpoint_v2->arena_generation)) {
            if (observation.rdma_available) appendChanged(changed, peer);
            observation.rdma_available = false;
            observation.rdma_target_rank_epoch = rank_epochs[peer];
            observation.rdma_arena_generation =
                member.endpoint_v2->arena_generation;
            continue;
        }
        if (arena.rdma_connected[peer]) continue;

        remote_addrs[peer] = static_cast<int64_t>(remote.remote_access_address);
        remote_keys[peer] = static_cast<int32_t>(remote.remote_key);
        remote_qpns[peer] = remote.qpns[rank_];
        remote_lids[peer] = remote.lids[rank_];
        subnet_prefixes[peer] = static_cast<int64_t>(remote.subnet_prefix);
        interface_ids[peer] = static_cast<int64_t>(remote.interface_id);
        active[peer] = 1;
        connecting.push_back(peer);
    }

    if (connecting.empty()) return;
    if (arena.rdma->connectPeers(rank_, arena.rdma->isRoce(), remote_addrs,
                                 remote_keys, remote_qpns, remote_lids,
                                 subnet_prefixes, interface_ids, active) != 0) {
        LOG(WARNING) << "Failed to connect collective device RDMA peers";
        return;
    }

    for (const auto peer : connecting) {
        arena.rdma_connected[peer] = true;
        if (peer == rank_) continue;
        const auto& remote = *view.members[peer].endpoint_v2->device_rdma;
        auto& observation = observations[peer];
        target_rank_epochs_[peer] = rank_epochs[peer];
        observation.rdma_target_rank_epoch = rank_epochs[peer];
        observation.rdma_arena_generation =
            view.members[peer].endpoint_v2->arena_generation;
        observation.remote_rdma_arena_address = remote.remote_access_address;
        observation.rdma_available = true;
        appendChanged(changed, peer);
    }
}

std::optional<DeviceP2pHandle> DeviceLinkManager::resolveP2p(
    DeviceId source_device, GlobalRank peer, uint64_t arena_generation) const {
    if (!rankInRange(peer)) return std::nullopt;
    std::lock_guard<std::mutex> lock(peers_mutex_);
    const auto device = device_peers_.find(source_device);
    if (device == device_peers_.end()) return std::nullopt;
    const auto& observation = device->second[peer];
    if (!observation.p2p_mapping ||
        observation.p2p_target_rank_epoch != target_rank_epochs_[peer] ||
        observation.p2p_arena_generation != arena_generation) {
        return std::nullopt;
    }
    return DeviceP2pHandle{observation.p2p_mapping,
                           observation.p2p_mapping->mappedBase()};
}

std::optional<DeviceRdmaHandle> DeviceLinkManager::resolveRdma(
    DeviceId source_device, GlobalRank peer, uint64_t arena_generation) const {
    if (!rankInRange(peer)) return std::nullopt;
    std::lock_guard<std::mutex> lock(peers_mutex_);
    const auto device = device_peers_.find(source_device);
    if (device == device_peers_.end()) return std::nullopt;
    const auto& observation = device->second[peer];
    const auto arena = arenas_.find(source_device);
    if (!observation.rdma_available ||
        observation.rdma_target_rank_epoch != target_rank_epochs_[peer] ||
        observation.rdma_arena_generation != arena_generation ||
        arena == arenas_.end() || !arena->second->rdma) {
        return std::nullopt;
    }
    return DeviceRdmaHandle{
        .keepalive = arena->second,
        .qp_contexts = arena->second->rdma->qpDevCtxsPtr(),
        .remote_keys =
            static_cast<const uint32_t*>(arena->second->rdma->rkeysPtr()),
        .local_peer_index = rank_,
        .peer_index = peer,
        .qps_per_peer = 1,
        .remote_arena_address = observation.remote_rdma_arena_address,
    };
}

void DeviceLinkManager::disconnect(GlobalRank peer) {
    if (peer == rank_ || !rankInRange(peer)) return;
    std::optional<PeerLinkUpdate> update;
    {
        std::lock_guard<std::mutex> lock(peers_mutex_);
        bool was_available = false;
        for (auto& [_, observations] : device_peers_) {
            auto& observation = observations[peer];
            was_available |= observation.p2p_mapping != nullptr ||
                             observation.rdma_available;
            observation = PeerObservation{};
        }
        if (was_available) update = makeLinkUpdate(peer);
    }
    if (update.has_value()) emit(std::move(*update));
}

void DeviceLinkManager::clear() {
    std::lock_guard<std::mutex> lock(peers_mutex_);
    device_peers_.clear();
    std::fill(target_rank_epochs_.begin(), target_rank_epochs_.end(), 0);
}

void DeviceLinkManager::setEventCallback(EventCallback callback) {
    std::lock_guard<std::mutex> lock(event_callback_mutex_);
    event_callback_ = std::move(callback);
}

PeerLinkUpdate DeviceLinkManager::makeLinkUpdate(GlobalRank peer) const {
    bool reachable = false;
    for (const auto& [_, observations] : device_peers_) {
        const auto& observation = observations[peer];
        reachable |=
            (observation.p2p_mapping &&
             observation.p2p_target_rank_epoch == target_rank_epochs_[peer]) ||
            (observation.rdma_available &&
             observation.rdma_target_rank_epoch == target_rank_epochs_[peer]);
    }
    return PeerLinkUpdate{
        .peer = peer,
        .target_rank_epoch = target_rank_epochs_[peer],
        .provider = LinkProvider::Device,
        .reachable = reachable,
    };
}

void DeviceLinkManager::emit(PeerLinkUpdate update) {
    std::lock_guard<std::mutex> lock(event_callback_mutex_);
    if (event_callback_) event_callback_(std::move(update));
}

}  // namespace mooncake
