#ifndef MOONCAKE_PG_COLLECTIVE_RUNTIME_COLLECTIVE_INVOCATION_H
#define MOONCAKE_PG_COLLECTIVE_RUNTIME_COLLECTIVE_INVOCATION_H

#include <cstdint>

#include <cuda_alike.h>

#include "collective/runtime/kernel_context.cuh"
#include "collective/types.h"

namespace mooncake {

// Per-call operation plugin. Collective-independent runtime code owns lane,
// capture lifetime, progress and failure reporting. A plugin only translates
// operation arguments into its typed device executor ABI.
class CollectiveInvocation {
   public:
    virtual ~CollectiveInvocation() = default;

    virtual CollectiveBindingId bindingId() const = 0;
    virtual void launch(const CollectiveKernelContext& context,
                        cudaStream_t stream) const = 0;
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_COLLECTIVE_RUNTIME_COLLECTIVE_INVOCATION_H
