// Copyright 2026 KVCache.AI
// Licensed under the Apache License, Version 2.0.

#include "transport/device/p2p_memory.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <glog/logging.h>

#include <cstring>
#include <limits>
#include <utility>

#include "local_fd_exchange.h"

namespace mooncake::device {
namespace {

using detail::LocalFdExchange;
using detail::OwnedFd;
constexpr int32_t kVersion = 1;
constexpr int32_t kInlineHandle = 0;
constexpr int32_t kFdHandle = 1;
constexpr size_t kHeaderWords = 2;

struct FdMetadata {
    uint64_t bytes;
    char endpoint[64];
    LocalFdExchange::Token token;
};
static_assert(sizeof(FdMetadata) == 88);

// Kept separate from the mapping implementation shared by the EP transport.
struct FdMapping : P2pMapping {
    ~FdMapping() override {
        int previous = -1;
        if (cudaGetDevice(&previous) != cudaSuccess ||
            (previous != device && cudaSetDevice(device) != cudaSuccess)) {
            LOG(ERROR) << "[P2P] cannot select device for FD mapping cleanup";
            return;
        }
        if (mapped) cuMemUnmap(ptr, bytes);
        if (ptr) cuMemAddressFree(ptr, bytes);
        if (handle) cuMemRelease(handle);
        if (previous != device && cudaSetDevice(previous) != cudaSuccess)
            LOG(ERROR)
                << "[P2P] cannot restore device after FD mapping cleanup";
    }

    void* address() const override { return reinterpret_cast<void*>(ptr); }
    int device = -1;
    CUdeviceptr ptr = 0;
    size_t bytes = 0;
    CUmemGenericAllocationHandle handle = 0;
    bool mapped = false;
};

OwnedFd exportFd(void* ptr, size_t bytes) {
    if (!ptr || !bytes) return {};
    cudaPointerAttributes attributes{};
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaPointerGetAttributes(&attributes, ptr) != cudaSuccess ||
        attributes.type != cudaMemoryTypeDevice || attributes.device != device)
        return {};
    CUmemGenericAllocationHandle handle = 0;
    if (cuMemRetainAllocationHandle(&handle, ptr) != CUDA_SUCCESS) return {};
    CUmemAllocationProp properties{};
    size_t granularity = 0;
    int fd = -1;
    if (cuMemGetAllocationPropertiesFromHandle(&properties, handle) ==
            CUDA_SUCCESS &&
        properties.location.type == CU_MEM_LOCATION_TYPE_DEVICE &&
        properties.location.id == device &&
        (properties.requestedHandleTypes &
         CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) &&
        cuMemGetAllocationGranularity(&granularity, &properties,
                                      CU_MEM_ALLOC_GRANULARITY_MINIMUM) ==
            CUDA_SUCCESS &&
        granularity && reinterpret_cast<uintptr_t>(ptr) % granularity == 0 &&
        bytes % granularity == 0) {
        const auto result = cuMemExportToShareableHandle(
            &fd, handle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0);
        if (result != CUDA_SUCCESS) {
            LOG(WARNING) << "[P2P] POSIX FD export failed: " << result;
            fd = -1;
        }
    }
    cuMemRelease(handle);
    return OwnedFd(fd);
}

std::unique_ptr<P2pMapping> importFd(const FdMetadata& metadata) {
    const auto endpoint_length =
        strnlen(metadata.endpoint, sizeof(metadata.endpoint));
    if (!metadata.bytes ||
        metadata.bytes > std::numeric_limits<size_t>::max() ||
        endpoint_length == 0 || endpoint_length == sizeof(metadata.endpoint))
        return nullptr;
    OwnedFd fd(LocalFdExchange::request(
        {std::string(metadata.endpoint, endpoint_length), metadata.token}));
    if (fd.get() < 0) {
        VLOG(1) << "[P2P] FD endpoint is unavailable or export was revoked";
        return nullptr;
    }

    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess) return nullptr;
    auto mapping = std::make_unique<FdMapping>();
    mapping->device = device;
    mapping->bytes = static_cast<size_t>(metadata.bytes);
    CUmemGenericAllocationHandle handle = 0;
    auto result = cuMemImportFromShareableHandle(
        &handle, reinterpret_cast<void*>(static_cast<uintptr_t>(fd.get())),
        CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
    if (result != CUDA_SUCCESS) return nullptr;
    mapping->handle = handle;
    fd.reset();

    CUmemAllocationProp properties{};
    size_t granularity = 0;
    if (cuMemGetAllocationPropertiesFromHandle(&properties, mapping->handle) !=
            CUDA_SUCCESS ||
        cuMemGetAllocationGranularity(&granularity, &properties,
                                      CU_MEM_ALLOC_GRANULARITY_MINIMUM) !=
            CUDA_SUCCESS ||
        !granularity || mapping->bytes % granularity != 0)
        return nullptr;
    CUdeviceptr address = 0;
    result = cuMemAddressReserve(&address, mapping->bytes, 0, 0, 0);
    if (result != CUDA_SUCCESS) return nullptr;
    mapping->ptr = address;
    result = cuMemMap(mapping->ptr, mapping->bytes, 0, mapping->handle, 0);
    if (result != CUDA_SUCCESS) return nullptr;
    mapping->mapped = true;
    CUmemAccessDesc access{};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = device;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    if (cuMemSetAccess(mapping->ptr, mapping->bytes, &access, 1) !=
        CUDA_SUCCESS)
        return nullptr;
    return mapping;
}

}  // namespace

struct P2pExport::Impl {
    std::vector<int32_t> metadata;
    std::unique_ptr<LocalFdExchange::Registration> registration;
};

struct P2pMemoryExchange::Impl {
    LocalFdExchange fd_exchange;
};

P2pExport::P2pExport(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
P2pExport::~P2pExport() = default;
const std::vector<int32_t>& P2pExport::metadata() const {
    return impl_->metadata;
}
P2pMemoryExchange::P2pMemoryExchange() : impl_(std::make_unique<Impl>()) {}
P2pMemoryExchange::~P2pMemoryExchange() = default;

std::unique_ptr<P2pExport> P2pMemoryExchange::exportMemory(void* ptr,
                                                           size_t bytes) {
    auto exported = std::make_unique<P2pExport::Impl>();
    auto inline_handle = exportP2pMemory(ptr, bytes);
    if (!inline_handle.empty()) {
        exported->metadata = {kVersion, kInlineHandle};
        exported->metadata.insert(exported->metadata.end(),
                                  inline_handle.begin(), inline_handle.end());
    } else {
        auto fd = exportFd(ptr, bytes);
        if (fd.get() < 0) return nullptr;
        exported->registration = impl_->fd_exchange.publish(fd.get());
        if (!exported->registration) {
            LOG(WARNING) << "[P2P] cannot publish FD export";
            return nullptr;
        }
        const auto& reference = exported->registration->reference();
        FdMetadata payload{};
        payload.bytes = bytes;
        if (reference.endpoint.size() >= sizeof(payload.endpoint))
            return nullptr;
        memcpy(payload.endpoint, reference.endpoint.data(),
               reference.endpoint.size());
        payload.token = reference.token;
        exported->metadata.resize(kHeaderWords +
                                  sizeof(payload) / sizeof(int32_t));
        exported->metadata[0] = kVersion;
        exported->metadata[1] = kFdHandle;
        memcpy(exported->metadata.data() + kHeaderWords, &payload,
               sizeof(payload));
    }
    return std::unique_ptr<P2pExport>(new P2pExport(std::move(exported)));
}

std::unique_ptr<P2pMapping> P2pMemoryExchange::importMemory(
    const std::vector<int32_t>& metadata) {
    if (metadata.size() <= kHeaderWords || metadata[0] != kVersion)
        return nullptr;
    if (metadata[1] == kInlineHandle)
        return importP2pMemory(
            {metadata.begin() + kHeaderWords, metadata.end()});
    if (metadata[1] != kFdHandle ||
        metadata.size() != kHeaderWords + sizeof(FdMetadata) / sizeof(int32_t))
        return nullptr;
    FdMetadata payload{};
    memcpy(&payload, metadata.data() + kHeaderWords, sizeof(payload));
    return importFd(payload);
}

}  // namespace mooncake::device
