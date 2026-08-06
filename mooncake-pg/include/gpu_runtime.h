#ifndef MOONCAKE_PG_GPU_RUNTIME_H
#define MOONCAKE_PG_GPU_RUNTIME_H

#include <cstdint>

#include <cuda_alike.h>

#include "error_types.h"

namespace mooncake {

struct GraphCaptureState {
    bool active = false;
    uint64_t id = 0;
    cudaGraph_t graph = nullptr;
};

PGResult<GraphCaptureState> queryGraphCapture(cudaStream_t stream);

// Graph-held user object facade. The current CUDA implementation owns the
// initial reference returned by cudaUserObjectCreate. moveTo() transfers that
// reference to a graph; destruction releases it when the transfer did not
// happen. The runtime owns the payload through its destructor callback once
// create() succeeds.
class GpuGraphUserObject {
   public:
    static PGResult<GpuGraphUserObject> create(void* payload,
                                               cudaHostFn_t destructor);
    ~GpuGraphUserObject() noexcept;

    GpuGraphUserObject(const GpuGraphUserObject&) = delete;
    GpuGraphUserObject& operator=(const GpuGraphUserObject&) = delete;
    GpuGraphUserObject(GpuGraphUserObject&& other) noexcept;
    GpuGraphUserObject& operator=(GpuGraphUserObject&& other) noexcept;

    PGResult<void> moveTo(cudaGraph_t graph);

   private:
    explicit GpuGraphUserObject(cudaUserObject_t object) : object_(object) {}

    void reset() noexcept;

    cudaUserObject_t object_ = nullptr;
};

class GpuDeviceGuard {
   public:
    explicit GpuDeviceGuard(int device);
    ~GpuDeviceGuard() noexcept;

    GpuDeviceGuard(const GpuDeviceGuard&) = delete;
    GpuDeviceGuard& operator=(const GpuDeviceGuard&) = delete;

   private:
    int previous_device_ = -1;
    bool restore_device_ = false;
};

class GpuEvent;

class GpuStream {
   public:
    GpuStream() = delete;
    ~GpuStream() noexcept;

    GpuStream(const GpuStream&) = delete;
    GpuStream& operator=(const GpuStream&) = delete;

    GpuStream(GpuStream&& other) noexcept;
    GpuStream& operator=(GpuStream&& other) noexcept;

    [[nodiscard]] static GpuStream createNonBlocking(int device);
    [[nodiscard]] static GpuStream borrow(cudaStream_t stream, int device);

    [[nodiscard]] cudaStream_t get() const noexcept { return stream_; }

    [[nodiscard]] int deviceIndex() const noexcept { return device_index_; }

    [[nodiscard]] bool isCapturing() const;

    void waitEvent(const GpuEvent& event) const;

   private:
    GpuStream(cudaStream_t stream, int device, bool owns_stream) noexcept;

    void reset() noexcept;
    void moveFrom(GpuStream&& other) noexcept;

    cudaStream_t stream_ = nullptr;
    int device_index_ = -1;
    bool owns_stream_ = false;
};

class GpuEvent {
   public:
    explicit GpuEvent(int device, unsigned int flags = cudaEventDisableTiming);
    ~GpuEvent() noexcept;

    GpuEvent(const GpuEvent&) = delete;
    GpuEvent& operator=(const GpuEvent&) = delete;

    GpuEvent(GpuEvent&& other) noexcept;
    GpuEvent& operator=(GpuEvent&& other) noexcept;

    void record(const GpuStream& stream);

   private:
    friend class GpuStream;

    void reset() noexcept;
    void moveFrom(GpuEvent&& other) noexcept;

    cudaEvent_t event_ = nullptr;
    int device_index_ = -1;
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_GPU_RUNTIME_H
