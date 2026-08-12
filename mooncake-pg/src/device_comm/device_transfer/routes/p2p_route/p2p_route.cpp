#include "device_comm/device_transfer/routes/p2p_route/p2p_route.h"

#include <transport/device/device_transport.h>

#include "gpu_runtime.h"

namespace mooncake {

P2pRoute::P2pRoute(device::P2pTransport& transport, void* local_region,
                   int device_index, uint32_t self_peer_index,
                   uint32_t peer_capacity)
    : transport_(transport),
      local_region_(local_region),
      device_index_(device_index),
      self_peer_index_(self_peer_index),
      peer_capacity_(peer_capacity),
      peers_(peer_capacity) {}

std::vector<int32_t> P2pRoute::localHandle() const {
    return transport_.exportIpcHandle(local_region_);
}

void P2pRoute::installPeerHandle(uint32_t peer_index,
                                 const std::vector<int32_t>& ipc_handle) {
    if (peer_index >= peer_capacity_) return;

    auto& mapping = peers_[peer_index];
    if (mapping.ipc_handle == ipc_handle) return;

    mapping.ipc_handle = ipc_handle;
    mapping.remote_region_address = 0;
    mapping.disabled = false;
    refresh_needed_ = true;
}

PGResult<void> P2pRoute::refreshMappings() {
    if (!refresh_needed_) return {};

    std::vector<std::vector<int32_t>> handles(peer_capacity_);
    std::vector<int> active(peer_capacity_, 0);
    for (uint32_t peer = 0; peer < peer_capacity_; ++peer) {
        if (peers_[peer].ipc_handle.empty() || peers_[peer].disabled) continue;
        handles[peer] = peers_[peer].ipc_handle;
        active[peer] = 1;
    }
    handles[self_peer_index_] = localHandle();
    active[self_peer_index_] = 1;

    PG_TRY(auto device_guard, GpuDeviceGuard::create(device_index_));
    transport_.importPeerHandles(local_region_, self_peer_index_,
                                 peer_capacity_, handles, active);

    std::vector<int32_t> available(peer_capacity_, 0);
    std::vector<void*> peer_bases(peer_capacity_, nullptr);
    PG_TRY_CUDA(cudaMemcpy(available.data(), transport_.availableTablePtr(),
                           available.size() * sizeof(int32_t),
                           cudaMemcpyDeviceToHost));
    PG_TRY_CUDA(cudaMemcpy(peer_bases.data(), transport_.peerPtrsTablePtr(),
                           peer_bases.size() * sizeof(void*),
                           cudaMemcpyDeviceToHost));

    // A full-snapshot import may remap even an unchanged handle. Rebuild the
    // complete cache instead of retaining any address from the old snapshot.
    for (auto& mapping : peers_) mapping.remote_region_address = 0;
    for (uint32_t peer = 0; peer < peer_capacity_; ++peer) {
        if (!available[peer] || !peer_bases[peer]) {
            continue;
        }
        peers_[peer].remote_region_address =
            reinterpret_cast<uint64_t>(peer_bases[peer]);
    }
    refresh_needed_ = false;
    return {};
}

std::optional<uint64_t> P2pRoute::resolve(uint32_t peer_index) const {
    if (peer_index >= peer_capacity_) return std::nullopt;
    if (peer_index == self_peer_index_) {
        return reinterpret_cast<uint64_t>(local_region_);
    }

    const auto& mapping = peers_[peer_index];
    if (mapping.remote_region_address == 0) {
        return std::nullopt;
    }
    return mapping.remote_region_address;
}

void P2pRoute::invalidate(uint32_t peer_index) {
    if (peer_index >= peer_capacity_) return;
    peers_[peer_index].remote_region_address = 0;
    // Keep a failed route disabled when an unrelated peer later causes a
    // full-snapshot refresh. A new IPC handle re-enables it when installed.
    peers_[peer_index].disabled = true;
}

}  // namespace mooncake
