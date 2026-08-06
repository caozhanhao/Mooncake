#ifndef MOONCAKE_PG_COLLECTIVE_RUNTIME_COLLECTIVE_SUBMISSION_H
#define MOONCAKE_PG_COLLECTIVE_RUNTIME_COLLECTIVE_SUBMISSION_H

#include <cstdint>
#include <utility>

#include "collective/buffer/collective_buffer_pool.h"
#include "collective/device_context.cuh"
#include "collective/runtime/collective_channels.h"

namespace mooncake {

// Owns the resources prepared by an operation for one device submission.
// Runtime only retains this object until eager completion or graph teardown;
// it does not decide which resources an operation needs. Before submission,
// member RAII performs normal rollback. After submission, destruction defaults
// to quarantine unless retire() has proved safe reuse.
class CollectiveSubmission {
   public:
    CollectiveSubmission(CollectiveChannelLease&& channel,
                         CollectiveBufferLease&& buffer,
                         CollectiveKernelResources kernel_resources)
        : channel_(std::move(channel)),
          buffer_(std::move(buffer)),
          kernel_resources_(kernel_resources) {}
    ~CollectiveSubmission() noexcept;

    CollectiveSubmission(const CollectiveSubmission&) = delete;
    CollectiveSubmission& operator=(const CollectiveSubmission&) = delete;

    CollectiveKernelArgs kernelArgs(uint64_t failure_target_id) const;
    CollectiveControlBlock& hostControl() const {
        return *channel_.host_control;
    }

    void markSubmitted() noexcept { submitted_ = true; }
    // Called only after stream completion or graph teardown proves that the
    // device no longer uses this submission. A transport still in flight is
    // quarantined instead of returned to either owner.
    bool retire() noexcept;

   private:
    CollectiveChannelLease channel_;
    CollectiveBufferLease buffer_;
    CollectiveKernelResources kernel_resources_;
    bool submitted_ = false;
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_COLLECTIVE_RUNTIME_COLLECTIVE_SUBMISSION_H
