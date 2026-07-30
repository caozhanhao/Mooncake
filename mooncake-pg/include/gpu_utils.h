#ifndef MOONCAKE_PG_GPU_UTILS_H
#define MOONCAKE_PG_GPU_UTILS_H

#pragma once

#include <cstdio>
#include <exception>
#include <utility>

#include <cuda_alike.h>

#include "pg_utils.h"

namespace mooncake {

inline void checkCuda(cudaError_t error, const char* operation) {
    PG_CHECK(error == cudaSuccess, operation, ": ", cudaGetErrorString(error));
}

namespace detail {

inline void warnCuda(cudaError_t error, const char* operation) noexcept {
    if (error == cudaSuccess) return;
    std::fprintf(stderr, "Mooncake PG failed to %s: %s\n", operation,
                 cudaGetErrorString(error));
}

inline void warnCudaCleanupException(const char* operation,
                                     const char* error) noexcept {
    std::fprintf(stderr, "Mooncake PG failed to %s: %s\n", operation, error);
}

}  // namespace detail

class GpuDeviceGuard {
   public:
    explicit GpuDeviceGuard(int device) {
        PG_CHECK(device >= 0, "invalid CUDA device index");

        checkCuda(cudaGetDevice(&previous_device_),
                  "get current CUDA device");
        if (previous_device_ == device) return;

        checkCuda(cudaSetDevice(device), "set CUDA device");
        restore_device_ = true;
    }

    ~GpuDeviceGuard() noexcept {
        if (!restore_device_) return;
        detail::warnCuda(cudaSetDevice(previous_device_),
                         "restore collective CUDA device");
    }

    GpuDeviceGuard(const GpuDeviceGuard&) = delete;
    GpuDeviceGuard& operator=(const GpuDeviceGuard&) = delete;

   private:
    int previous_device_ = -1;
    bool restore_device_ = false;
};

class GpuStream {
   public:
    GpuStream() noexcept = default;

    ~GpuStream() noexcept { reset(); }

    GpuStream(const GpuStream&) = delete;
    GpuStream& operator=(const GpuStream&) = delete;

    GpuStream(GpuStream&& other) noexcept { moveFrom(std::move(other)); }

    GpuStream& operator=(GpuStream&& other) noexcept {
        if (this != &other) {
            reset();
            moveFrom(std::move(other));
        }
        return *this;
    }

    static GpuStream createNonBlocking(int device) {
        const GpuDeviceGuard device_guard(device);
        cudaStream_t stream = nullptr;
        checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                  "create stream");
        return GpuStream(stream, device, true);
    }

    static GpuStream borrow(cudaStream_t stream, int device) {
        PG_CHECK(device >= 0, "invalid CUDA device index");
        return GpuStream(stream, device, false);
    }

    operator cudaStream_t() const noexcept { return stream_; }

    int deviceIndex() const noexcept { return device_index_; }

    bool isCapturing() const {
        cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
        checkCuda(cudaStreamIsCapturing(stream_, &capture_status),
                  "cudaStreamIsCapturing");
        return capture_status != cudaStreamCaptureStatusNone;
    }

   private:
    GpuStream(cudaStream_t stream, int device, bool owns_stream) noexcept
        : stream_(stream),
          device_index_(device),
          owns_stream_(owns_stream) {}

    void reset() noexcept {
        if (owns_stream_ && stream_) {
            try {
                const GpuDeviceGuard device_guard(device_index_);
                detail::warnCuda(cudaStreamDestroy(stream_),
                                 "destroy stream");
            } catch (const std::exception& error) {
                detail::warnCudaCleanupException(
                    "destroy stream", error.what());
            } catch (...) {
                detail::warnCudaCleanupException(
                    "destroy stream", "unknown error");
            }
        }
        stream_ = nullptr;
        device_index_ = -1;
        owns_stream_ = false;
    }

    void moveFrom(GpuStream&& other) noexcept {
        stream_ = other.stream_;
        device_index_ = other.device_index_;
        owns_stream_ = other.owns_stream_;
        other.stream_ = nullptr;
        other.device_index_ = -1;
        other.owns_stream_ = false;
    }

    cudaStream_t stream_ = nullptr;
    int device_index_ = -1;
    bool owns_stream_ = false;
};

class GpuEvent {
   public:
    explicit GpuEvent(unsigned int flags = cudaEventDisableTiming) noexcept
        : flags_(flags) {}

    ~GpuEvent() noexcept { reset(); }

    GpuEvent(const GpuEvent&) = delete;
    GpuEvent& operator=(const GpuEvent&) = delete;

    GpuEvent(GpuEvent&& other) noexcept { moveFrom(std::move(other)); }

    GpuEvent& operator=(GpuEvent&& other) noexcept {
        if (this != &other) {
            reset();
            moveFrom(std::move(other));
        }
        return *this;
    }

    void record(const GpuStream& stream) {
        if (!is_created_) create(stream.deviceIndex());
        PG_CHECK(device_index_ == stream.deviceIndex(),
                 "CUDA event device does not match recording stream device");

        const GpuDeviceGuard device_guard(device_index_);
        checkCuda(cudaEventRecord(event_, stream),
                  "record CUDA event");
    }

    void block(const GpuStream& stream) const {
        if (!is_created_) return;

        const GpuDeviceGuard device_guard(stream.deviceIndex());
        checkCuda(cudaStreamWaitEvent(stream, event_, 0),
                  "wait for CUDA event");
    }

   private:
    void create(int device) {
        const GpuDeviceGuard device_guard(device);
        checkCuda(cudaEventCreateWithFlags(&event_, flags_),
                  "create CUDA event");
        device_index_ = device;
        is_created_ = true;
    }

    void reset() noexcept {
        if (is_created_) {
            try {
                const GpuDeviceGuard device_guard(device_index_);
                detail::warnCuda(cudaEventDestroy(event_),
                                 "destroy CUDA event");
            } catch (const std::exception& error) {
                detail::warnCudaCleanupException(
                    "destroy CUDA event", error.what());
            } catch (...) {
                detail::warnCudaCleanupException(
                    "destroy CUDA event", "unknown error");
            }
        }
        event_ = nullptr;
        device_index_ = -1;
        is_created_ = false;
    }

    void moveFrom(GpuEvent&& other) noexcept {
        flags_ = other.flags_;
        event_ = other.event_;
        device_index_ = other.device_index_;
        is_created_ = other.is_created_;
        other.event_ = nullptr;
        other.device_index_ = -1;
        other.is_created_ = false;
    }

    unsigned int flags_ = cudaEventDisableTiming;
    cudaEvent_t event_ = nullptr;
    int device_index_ = -1;
    bool is_created_ = false;
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_GPU_UTILS_H
