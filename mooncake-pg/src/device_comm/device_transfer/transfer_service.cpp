#include "device_comm/device_transfer/transfer_service.h"

#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include <glog/logging.h>
#include <transfer_engine.h>
#include <transport/device/device_transport.h>

#include "device_comm/device_transfer/transfer_types.cuh"
#include "device_comm/device_transfer/routes/host_proxy_route/host_proxy_route.h"
#include "device_comm/device_transfer/routes/host_proxy_route/host_transfer_proxy.h"
#include "device_comm/device_transfer/routes/p2p_route/p2p_route.h"
#include "gpu_runtime.h"
#include "memory_location.h"

namespace mooncake {
namespace {

constexpr uint64_t kServiceResourceAlignment = 256;

bool addOverflows(uint64_t left, uint64_t right) {
    return right > std::numeric_limits<uint64_t>::max() - left;
}

std::optional<uint64_t> alignUp(uint64_t value, uint64_t alignment) {
    if (alignment == 0) return std::nullopt;
    const auto remainder = value % alignment;
    if (remainder == 0) return value;
    const auto increment = alignment - remainder;
    if (addOverflows(value, increment)) return std::nullopt;
    return value + increment;
}

bool validEndpoint(const DeviceTransferEndpoint& endpoint) {
    if (endpoint.region_address == 0 ||
        endpoint.region_address % alignof(uint64_t) != 0 ||
        endpoint.region_size == 0 ||
        addOverflows(endpoint.region_address, endpoint.region_size)) {
        return false;
    }
    if (endpoint.p2p && endpoint.p2p->ipc_handle.empty()) return false;
    if (endpoint.host_proxy && endpoint.host_proxy->te_server_name.empty()) {
        return false;
    }
    return endpoint.p2p.has_value() || endpoint.host_proxy.has_value();
}

struct DeviceMetadataLayout {
    uint64_t size = 0;
    uint64_t handle_offset = 0;
    uint64_t peer_routes_offset = 0;
    uint64_t lane_results_offset = 0;
};

DeviceMetadataLayout makeDeviceMetadataLayout(uint32_t peer_capacity) {
    DeviceMetadataLayout layout;
    uint64_t cursor = 0;
    auto reserve = [&](uint64_t size, uint64_t alignment) -> uint64_t {
        const auto aligned = alignUp(cursor, alignment);
        PG_ASSERT(aligned && !addOverflows(*aligned, size),
                  "device-transfer service layout overflows");
        const uint64_t result = *aligned;
        cursor = result + size;
        return result;
    };

    layout.handle_offset = reserve(sizeof(DeviceTransferHandle),
                                   alignof(DeviceTransferHandle));
    layout.peer_routes_offset =
        reserve(static_cast<uint64_t>(peer_capacity) * sizeof(DevicePeerRoute),
                alignof(DevicePeerRoute));
    layout.lane_results_offset = reserve(
        kTransferLaneCount * sizeof(uint64_t), alignof(uint64_t));
    layout.size = reserve(0, kServiceResourceAlignment);
    return layout;
}

}  // namespace

struct DeviceTransferService::DeviceState {
    // Remotely addressable device memory owned through TE's P2P transport.
    struct RegisteredRegion {
        void* addr = nullptr;
        size_t size = 0;
        bool registered = false;
    };

    // One local-only CUDA allocation and its typed device subviews.
    struct DeviceMetadata {
        void* allocation = nullptr;
        DeviceTransferHandle* handle = nullptr;
        DevicePeerRoute* peer_routes = nullptr;
        uint64_t* lane_results = nullptr;
    };

    static PGResult<std::unique_ptr<DeviceState>> create(
        int device_index, uint32_t self_peer_index, uint32_t peer_capacity,
        size_t region_size, TransferEngine& engine,
        HostTransferProxy& host_proxy) {
        auto* p2p_transport =
            engine.getOrCreateP2pTransport(static_cast<int>(peer_capacity));
        PG_VALIDATE_STATE(p2p_transport,
                          "CUDA P2P device transport is unavailable");
        PG_TRY(auto device_guard, GpuDeviceGuard::create(device_index));
        PG_TRY(auto route_stream, GpuStream::createNonBlocking(device_index));
        auto state = std::unique_ptr<DeviceState>(new DeviceState(
            device_index, self_peer_index, peer_capacity, engine,
            *p2p_transport, host_proxy, std::move(route_stream)));

        // The current TE P2P API exports IPC handles for buffers allocated by
        // its P2pTransport. Keep that constraint inside the Service: callers
        // still see one Service-owned region shared by every route.
        state->registered_region.size = region_size;
        state->registered_region.addr =
            p2p_transport->allocateBuffer(region_size);
        PG_VALIDATE_STATE(state->registered_region.addr,
                          "failed to allocate transfer-service memory");

        const auto location = GPU_PREFIX + std::to_string(device_index);
        const int registration = engine.registerLocalMemory(
            state->registered_region.addr, region_size, location);
        if (registration != 0) {
            return makePGError(
                PGErrorCode::TransferEngineError,
                "failed to register transfer-service memory, rc=" +
                    std::to_string(registration));
        }
        state->registered_region.registered = true;
        state->p2p_route = std::make_unique<P2pRoute>(
            *p2p_transport, state->registered_region.addr, device_index,
            self_peer_index, peer_capacity);

        // Route selection and lane results are local device metadata, not
        // remotely addressed memory, so they do not consume registered-region
        // capacity.
        const auto layout = makeDeviceMetadataLayout(peer_capacity);
        PG_TRY_CUDA(
            cudaMalloc(&state->device_metadata.allocation, layout.size));
        PG_TRY_CUDA(
            cudaMemset(state->device_metadata.allocation, 0, layout.size));

        auto* const service_base =
            static_cast<char*>(state->device_metadata.allocation);
        state->device_metadata.handle = reinterpret_cast<DeviceTransferHandle*>(
            service_base + layout.handle_offset);
        state->device_metadata.peer_routes = reinterpret_cast<DevicePeerRoute*>(
            service_base + layout.peer_routes_offset);
        state->device_metadata.lane_results = reinterpret_cast<uint64_t*>(
            service_base + layout.lane_results_offset);

        const size_t route_table_size =
            static_cast<size_t>(peer_capacity) * sizeof(DevicePeerRoute);
        // Recovery publishes this image while a failed kernel is parked. Keep
        // the source pinned so the copy can use the independent update stream.
        PG_TRY_CUDA(
            cudaHostAlloc(reinterpret_cast<void**>(&state->host_route_image),
                          route_table_size, cudaHostAllocPortable));
        std::uninitialized_value_construct_n(state->host_route_image,
                                             peer_capacity);

        PG_TRY(state->host_proxy_route.initialize(device_index));
        const DeviceTransferHandle handle_image{
            .local_region = state->registered_region.addr,
            .local_region_size = region_size,
            .peer_routes = state->device_metadata.peer_routes,
            .lane_results = state->device_metadata.lane_results,
            .host_proxy_command_slots =
                state->host_proxy_route.deviceCommandSlots(),
            .peer_capacity = peer_capacity,
        };
        PG_TRY_CUDA(cudaMemcpy(state->device_metadata.handle, &handle_image,
                               sizeof(handle_image), cudaMemcpyHostToDevice));

        const auto p2p_handle = state->p2p_route->localHandle();
        const auto te_server_name = engine.getLocalIpAndPort();
        state->local_endpoint = DeviceTransferEndpoint{
            .region_address =
                reinterpret_cast<uint64_t>(state->registered_region.addr),
            .region_size = region_size,
            .p2p = p2p_handle.empty() ? std::nullopt
                                      : std::optional<P2pEndpoint>(P2pEndpoint{
                                            .ipc_handle = p2p_handle}),
            .host_proxy =
                te_server_name.empty()
                    ? std::nullopt
                    : std::optional<HostProxyEndpoint>(
                          HostProxyEndpoint{.te_server_name = te_server_name}),
        };
        PG_VALIDATE_STATE(validEndpoint(state->local_endpoint),
                          "local transfer-service endpoint is invalid");

        state->peer_endpoints.resize(peer_capacity);
        state->installPeerEndpoint(self_peer_index, state->local_endpoint);
        state->host_route_image[self_peer_index] =
            state->resolveRoute(self_peer_index);
        PG_TRY(state->publishRoutes());
        return state;
    }

    DeviceState(int device_index, uint32_t self_peer_index,
                uint32_t peer_capacity, TransferEngine& engine,
                device::P2pTransport& transport, HostTransferProxy& host_proxy,
                GpuStream route_stream)
        : device_index(device_index),
          self_peer_index(self_peer_index),
          peer_capacity(peer_capacity),
          engine(engine),
          p2p_transport(transport),
          route_stream(std::move(route_stream)),
          host_proxy_route(host_proxy) {}

    ~DeviceState() noexcept {
        auto result = shutdown();
        if (!result.has_value()) {
            LOG(ERROR) << "DeviceTransferService device shutdown failed: "
                       << result.error().message;
        }
    }

    DeviceState(const DeviceState&) = delete;
    DeviceState& operator=(const DeviceState&) = delete;

    void installPeerEndpoint(
        uint32_t peer_index,
        const std::optional<DeviceTransferEndpoint>& endpoint) {
        PG_ASSERT(peer_index < peer_capacity,
                  "transfer peer index is out of range");
        auto& installed = peer_endpoints[peer_index];
        if (installed == endpoint) return;

        host_proxy_route.installPeerEndpoint(
            peer_index, endpoint ? endpoint->host_proxy
                                 : std::optional<HostProxyEndpoint>{});
        const std::vector<int32_t> empty_handle;
        p2p_route->installPeerHandle(peer_index, endpoint && endpoint->p2p
                                                     ? endpoint->p2p->ipc_handle
                                                     : empty_handle);
        installed = endpoint;
    }

    DevicePeerRoute resolveRoute(uint32_t peer_index) const {
        DevicePeerRoute route;
        if (peer_index >= peer_capacity) return route;
        const auto& endpoint = peer_endpoints[peer_index];
        if (!endpoint) return route;

        route.region_size = endpoint->region_size;
        // One invocation sees one stable route selected from the currently
        // installed endpoint snapshot.
        if (const auto p2p = p2p_route->resolve(peer_index)) {
            route.kind = DeviceRouteKind::P2p;
            route.p2p.region_addr = *p2p;
            return route;
        }
        if (endpoint->host_proxy) {
            route.kind = DeviceRouteKind::HostProxy;
            route.host_proxy.region_addr = endpoint->region_address;
        }
        return route;
    }

    PGResult<void> publishRoutes() {
        PG_TRY(auto device_guard, GpuDeviceGuard::create(device_index));
        PG_TRY_CUDA(cudaMemcpyAsync(
            device_metadata.peer_routes, host_route_image,
            static_cast<size_t>(peer_capacity) * sizeof(DevicePeerRoute),
            cudaMemcpyHostToDevice, route_stream.get()));
        return route_stream.synchronize();
    }

    PGResult<void> shutdown() {
        if (closed) return {};

        // This is the only retryable preflight. Once teardown starts below,
        // every owned resource is visited exactly once and failures are
        // reported after the remaining cleanup has run.
        auto route_shutdown = host_proxy_route.shutdown();
        if (!route_shutdown.has_value()) return route_shutdown;

        std::optional<PGError> first_error;
        const auto record_error = [&](PGError error) {
            if (!first_error) first_error = std::move(error);
        };

        std::optional<GpuDeviceGuard> device_guard;
        auto guard_result = GpuDeviceGuard::create(device_index);
        if (guard_result.has_value()) {
            device_guard.emplace(std::move(guard_result).value());
        } else {
            record_error(std::move(guard_result).error());
        }

        if (host_route_image) {
            if (device_guard) {
                const auto result = cudaFreeHost(host_route_image);
                if (result != cudaSuccess) {
                    record_error(PGError{
                        PGErrorCode::SystemError,
                        std::string("cudaFreeHost(host_route_image) failed: ") +
                            cudaGetErrorString(result),
                    });
                }
            } else {
                LOG(ERROR) << "Leaking transfer-service host route image "
                              "because its CUDA device could not be selected";
            }
            host_route_image = nullptr;
        }
        if (device_metadata.allocation) {
            if (device_guard) {
                const auto result = cudaFree(device_metadata.allocation);
                if (result != cudaSuccess) {
                    record_error(PGError{
                        PGErrorCode::SystemError,
                        std::string("cudaFree(device metadata) failed: ") +
                            cudaGetErrorString(result),
                    });
                }
            } else {
                LOG(ERROR) << "Leaking transfer-service device metadata "
                              "because its CUDA device could not be selected";
            }
            device_metadata = {};
        }
        p2p_route.reset();

        bool may_free_registered_region = device_guard.has_value();
        if (registered_region.registered && registered_region.addr) {
            const int result =
                engine.unregisterLocalMemory(registered_region.addr);
            if (result != 0) {
                record_error(PGError{
                    PGErrorCode::TransferEngineError,
                    "failed to unregister transfer-service memory, rc=" +
                        std::to_string(result),
                });
                // TE may still refer to the allocation. Deliberately leak it
                // instead of freeing memory behind a live registration.
                may_free_registered_region = false;
            }
        }
        if (registered_region.addr) {
            if (may_free_registered_region) {
                p2p_transport.freeBuffer(registered_region.addr);
            } else {
                LOG(ERROR) << "Leaking registered transfer-service region "
                              "after cleanup failure";
            }
        }
        registered_region = {};
        closed = true;
        if (first_error) return makePGError(std::move(*first_error));
        return {};
    }

    int device_index;
    uint32_t self_peer_index;
    uint32_t peer_capacity;
    TransferEngine& engine;
    device::P2pTransport& p2p_transport;
    RegisteredRegion registered_region;
    GpuStream route_stream;
    std::unique_ptr<P2pRoute> p2p_route;
    HostProxyRoute host_proxy_route;
    DeviceMetadata device_metadata;
    // Pinned host image copied into device_metadata.peer_routes.
    DevicePeerRoute* host_route_image = nullptr;
    DeviceTransferEndpoint local_endpoint;
    std::vector<std::optional<DeviceTransferEndpoint>> peer_endpoints;
    bool closed = false;
};

DeviceTransferService::DeviceTransferService() = default;

DeviceTransferService::~DeviceTransferService() noexcept {
    auto result = shutdown();
    if (!result.has_value()) {
        LOG(ERROR) << "DeviceTransferService shutdown failed during "
                      "destruction: "
                   << result.error().message;
    }
}

PGResult<void> DeviceTransferService::initialize(
    uint32_t self_peer_index, uint32_t peer_capacity, int device_index,
    TransferEngine& transfer_engine, size_t region_size) {
    std::lock_guard<std::mutex> lock(mutex_);
    PG_VALIDATE_STATE(!shutdown_, "DeviceTransferService is shut down");
    PG_VALIDATE_STATE(!device_ && !host_proxy_,
                      "DeviceTransferService is already initialized");
    PG_VALIDATE_ARG(peer_capacity != 0,
                    "device transfer peer capacity is zero");
    PG_VALIDATE_ARG(self_peer_index < peer_capacity,
                    "device transfer self peer is out of range");
    PG_VALIDATE_ARG(device_index >= 0, "invalid transfer CUDA device");
    PG_VALIDATE_ARG(region_size != 0, "transfer-service region is empty");

    auto host_proxy =
        std::make_unique<HostTransferProxy>(transfer_engine, peer_capacity);
    PG_TRY(host_proxy->start());

    PG_TRY(auto device,
           DeviceState::create(device_index, self_peer_index, peer_capacity,
                               region_size, transfer_engine, *host_proxy));

    host_proxy_ = std::move(host_proxy);
    device_ = std::move(device);
    return {};
}

DeviceTransferService::DeviceState& DeviceTransferService::deviceState() {
    PG_ASSERT(!shutdown_ && device_,
              "DeviceTransferService is not initialized");
    return *device_;
}

const DeviceTransferHandle* DeviceTransferService::deviceHandle() {
    std::lock_guard<std::mutex> lock(mutex_);
    return deviceState().device_metadata.handle;
}

const DeviceTransferEndpoint& DeviceTransferService::localEndpoint()
    const noexcept {
    PG_ASSERT(device_, "DeviceTransferService is not initialized");
    return device_->local_endpoint;
}

void* DeviceTransferService::regionAddr() const noexcept {
    PG_ASSERT(device_, "DeviceTransferService is not initialized");
    return device_->registered_region.addr;
}

size_t DeviceTransferService::regionSize() const noexcept {
    PG_ASSERT(device_, "DeviceTransferService is not initialized");
    return device_->registered_region.size;
}

PGResult<void> DeviceTransferService::installPeerEndpoint(
    uint32_t peer_index, const DeviceTransferEndpoint& endpoint) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& state = deviceState();
    PG_VALIDATE_ARG(peer_index < state.peer_capacity,
                    "transfer peer index is out of range");
    PG_VALIDATE_ARG(peer_index != state.self_peer_index,
                    "cannot replace the local transfer endpoint");
    PG_VALIDATE_ARG(validEndpoint(endpoint),
                    "transfer-service endpoint is invalid");

    state.installPeerEndpoint(peer_index, endpoint);

    PG_TRY(state.p2p_route->refreshMappings());
    for (uint32_t rank = 0; rank < state.peer_capacity; ++rank) {
        state.host_route_image[rank] = state.resolveRoute(rank);
    }
    return state.publishRoutes();
}

PGResult<void> DeviceTransferService::waitUntilIdle() {
    std::lock_guard<std::mutex> lock(mutex_);
    return deviceState().host_proxy_route.waitUntilIdle();
}

PGResult<void> DeviceTransferService::shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (shutdown_) return {};

    std::optional<PGError> first_error;
    if (device_) {
        auto result = device_->shutdown();
        if (!result.has_value()) {
            first_error = std::move(result).error();
        }
        if (device_->closed) device_.reset();
    }

    // A failed preflight means commands are still in flight and the proxy must
    // remain alive for a later retry. A closed DeviceState no longer exposes
    // command slots to kernels, so proxy teardown can proceed even if another
    // cleanup operation above was only best-effort.
    if (!device_ && host_proxy_) {
        auto result = host_proxy_->shutdown();
        if (!result.has_value()) {
            if (!first_error) first_error = std::move(result).error();
        } else {
            host_proxy_.reset();
        }
    }

    shutdown_ = !device_ && !host_proxy_;
    if (first_error) return makePGError(std::move(*first_error));
    return {};
}

}  // namespace mooncake
