#ifndef MOONCAKE_PG_COLLECTIVE_RUNTIME_COLLECTIVE_SUBMISSION_H
#define MOONCAKE_PG_COLLECTIVE_RUNTIME_COLLECTIVE_SUBMISSION_H

#include <cstdint>
#include <memory>
#include <utility>

#include "collective/buffer/collective_buffer_pool.h"
#include "collective/device_context.cuh"
#include "collective/runtime/collective_channels.h"

namespace mooncake {

// Owns the resources prepared by an operation for one device submission.
// Runtime only retains this object until eager completion or graph teardown;
// it does not decide which resources an operation needs.
class CollectiveSubmission {
   public:
    CollectiveSubmission(CollectiveBufferPool& buffer_pool,
                         CollectiveChannels& channels,
                         CollectiveChannel channel,
                         std::unique_ptr<CollectiveBufferLease> buffer,
                         CollectiveKernelResources kernel_resources)
        : buffer_pool_(&buffer_pool),
          channels_(&channels),
          channel_(channel),
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
    void release() noexcept;
    void abandon() noexcept;

    CollectiveBufferPool* buffer_pool_ = nullptr;
    CollectiveChannels* channels_ = nullptr;
    CollectiveChannel channel_;
    std::unique_ptr<CollectiveBufferLease> buffer_;
    CollectiveKernelResources kernel_resources_;
    bool submitted_ = false;
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_COLLECTIVE_RUNTIME_COLLECTIVE_SUBMISSION_H
