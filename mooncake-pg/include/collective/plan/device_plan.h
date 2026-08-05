#ifndef MOONCAKE_PG_COLLECTIVE_PLAN_DEVICE_PLAN_H
#define MOONCAKE_PG_COLLECTIVE_PLAN_DEVICE_PLAN_H

#include <memory>
#include <type_traits>

#include <cuda_alike.h>

#include "error_types.h"
#include "gpu_runtime.h"

namespace mooncake {

// Communicator-scoped publication with a stable device address. View updates
// overwrite the host-mapped value only at the collective-quiescent boundary;
// captured kernels therefore observe the current plan without recapture.
template <typename Plan>
class DevicePlan {
    static_assert(std::is_trivially_copyable_v<Plan>);

   public:
    static PGResult<std::unique_ptr<DevicePlan>> create(DeviceId device) {
        const GpuDeviceGuard guard(device);

        void* plan_host = nullptr;
        void* plan_device = nullptr;
        PG_TRY_CUDA(cudaHostAlloc(&plan_host, sizeof(Plan),
                                  cudaHostAllocMapped | cudaHostAllocPortable));
        HostAllocation plan_memory(plan_host);
        PG_TRY_CUDA(
            cudaHostGetDevicePointer(&plan_device, plan_memory.get(), 0));

        auto result = std::unique_ptr<DevicePlan>(
            new DevicePlan(static_cast<Plan*>(plan_memory.release()),
                           static_cast<Plan*>(plan_device)));
        *result->host_plan_ = Plan{};
        return result;
    }

    ~DevicePlan() noexcept {
        if (retained_) return;
        if (host_plan_) (void)cudaFreeHost(host_plan_);
    }

    PGResult<const Plan*> devicePlan() const {
        PG_VALIDATE_STATE(ready_, "collective plan is not ready");
        return device_plan_;
    }

    void publish(const Plan& plan) {
        *host_plan_ = plan;
        ready_ = true;
    }

    // Captured kernels must stop using the preceding view even when local
    // materialization fails. Publish the typed invalid value, but reject new
    // host submissions until a later view produces a valid plan.
    void publishInvalid(const Plan& plan) {
        *host_plan_ = plan;
        ready_ = false;
    }

    void retainForProcessLifetime() { retained_ = true; }

    DevicePlan(const DevicePlan&) = delete;
    DevicePlan& operator=(const DevicePlan&) = delete;

   private:
    struct HostAllocationDeleter {
        void operator()(void* memory) const noexcept {
            if (memory) (void)cudaFreeHost(memory);
        }
    };
    using HostAllocation = std::unique_ptr<void, HostAllocationDeleter>;

    DevicePlan(Plan* host_plan, Plan* device_plan)
        : host_plan_(host_plan), device_plan_(device_plan) {}

    Plan* host_plan_ = nullptr;
    Plan* device_plan_ = nullptr;
    bool ready_ = false;
    bool retained_ = false;
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_COLLECTIVE_PLAN_DEVICE_PLAN_H
