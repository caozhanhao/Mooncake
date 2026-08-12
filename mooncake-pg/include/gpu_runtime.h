#ifndef MOONCAKE_PG_GPU_RUNTIME_H
#define MOONCAKE_PG_GPU_RUNTIME_H

#include <cstddef>
#include <cstdint>

#include <cuda_alike.h>

#include "error_types.h"

namespace mooncake {

struct GpuCaptureInfo {
    bool active = false;
    cudaStream_t origin = nullptr;
    cudaGraph_t graph = nullptr;
    uint64_t graph_id = 0;
};

class GpuDeviceGuard {
   public:
    [[nodiscard]] static PGResult<GpuDeviceGuard> create(int device);

    ~GpuDeviceGuard() noexcept;

    GpuDeviceGuard(const GpuDeviceGuard&) = delete;
    GpuDeviceGuard& operator=(const GpuDeviceGuard&) = delete;

    GpuDeviceGuard(GpuDeviceGuard&& other) noexcept;
    GpuDeviceGuard& operator=(GpuDeviceGuard&& other) noexcept;

   private:
    GpuDeviceGuard(int previous_device, bool restore_device) noexcept
        : previous_device_(previous_device), restore_device_(restore_device) {}

    void reset() noexcept;
    void moveFrom(GpuDeviceGuard&& other) noexcept;

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

    [[nodiscard]] static PGResult<GpuStream> createNonBlocking(int device);
    // Borrowing does not call the CUDA runtime and therefore cannot report a
    // runtime failure. `device` is an internal object invariant.
    [[nodiscard]] static GpuStream borrow(cudaStream_t stream, int device);

    [[nodiscard]] cudaStream_t get() const noexcept { return stream_; }

    [[nodiscard]] int deviceIndex() const noexcept { return device_index_; }

    [[nodiscard]] PGResult<GpuCaptureInfo> captureInfo() const;
    [[nodiscard]] PGResult<cudaStreamCaptureStatus> captureStatus() const;

    // During capture, waiting for an event recorded in the same capture adds
    // the event's saved Graph nodes to the dependencies of this stream's next
    // captured operation. CUDA turns that into static intra-Graph edges.
    [[nodiscard]] PGResult<void> waitEvent(const GpuEvent& event) const;
    // During capture, emit a runtime event-wait node. Each replay uses it to
    // import completion published outside this Graph; it does not order two
    // otherwise independent branches inside the Graph.
    [[nodiscard]] PGResult<void> waitExternalEvent(const GpuEvent& event) const;

    [[nodiscard]] PGResult<void> synchronize() const;

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
    [[nodiscard]] static PGResult<GpuEvent> create(
        int device, unsigned int flags = cudaEventDisableTiming);

    ~GpuEvent() noexcept;

    GpuEvent(const GpuEvent&) = delete;
    GpuEvent& operator=(const GpuEvent&) = delete;

    GpuEvent(GpuEvent&& other) noexcept;
    GpuEvent& operator=(GpuEvent&& other) noexcept;

    [[nodiscard]] cudaEvent_t get() const noexcept { return event_; }

    [[nodiscard]] int deviceIndex() const noexcept { return device_index_; }

    // Outside capture this enqueues an ordinary CUDA event record. During
    // capture it saves the Graph node(s) on which the stream's next operation
    // currently depends; a same-capture wait can transfer those dependencies
    // to another captured stream as static edges.
    [[nodiscard]] PGResult<void> record(const GpuStream& stream);
    // During capture, emit an event-record node which runs on every replay and
    // publishes completion outside this Graph. Unlike record(), the event is
    // deliberately kept as a runtime Graph node rather than folded into an
    // intra-Graph dependency.
    [[nodiscard]] PGResult<void> recordExternal(const GpuStream& stream);
    [[nodiscard]] PGResult<bool> query() const;

   private:
    friend class GpuStream;

    GpuEvent(cudaEvent_t event, int device) noexcept
        : event_(event), device_index_(device) {}

    void reset() noexcept;
    void moveFrom(GpuEvent&& other) noexcept;

    cudaEvent_t event_ = nullptr;
    int device_index_ = -1;
};

// Add `stream` to an active capture without making its first captured work
// depend on work already captured on the origin stream. This establishes only
// capture membership; the caller must install the stream's real first
// dependency, normally an external wait on a cross-replay event.
PGResult<void> joinCaptureWithoutDependencies(const GpuCaptureInfo& capture,
                                              const GpuStream& stream);

// A move-only reference that can be transferred to a CUDA Graph. CUDA owns the
// payload after create() succeeds and invokes `destructor` when the final graph
// or graph-exec reference is released.
class GpuGraphUserObject {
   public:
    static PGResult<GpuGraphUserObject> create(int device, void* payload,
                                               cudaHostFn_t destructor);

    ~GpuGraphUserObject() noexcept;

    GpuGraphUserObject(const GpuGraphUserObject&) = delete;
    GpuGraphUserObject& operator=(const GpuGraphUserObject&) = delete;

    GpuGraphUserObject(GpuGraphUserObject&& other) noexcept;
    GpuGraphUserObject& operator=(GpuGraphUserObject&& other) noexcept;

    PGResult<void> moveTo(const GpuCaptureInfo& capture);

   private:
    GpuGraphUserObject(cudaUserObject_t object, int device) noexcept
        : object_(object), device_index_(device) {}

    void reset() noexcept;
    void moveFrom(GpuGraphUserObject&& other) noexcept;

    cudaUserObject_t object_ = nullptr;
    int device_index_ = -1;
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_GPU_RUNTIME_H
