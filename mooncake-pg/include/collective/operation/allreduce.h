#ifndef MOONCAKE_PG_COLLECTIVE_OPERATION_ALLREDUCE_H
#define MOONCAKE_PG_COLLECTIVE_OPERATION_ALLREDUCE_H

#include <cstddef>
#include <cstdint>

#include "collective/executor/allreduce.cuh"
#include "collective/runtime/collective_invocation.h"
#include "comm_types.h"
#include "error_types.h"

namespace mooncake {

class AllReduceInvocation final : public CollectiveInvocation {
   public:
    static bool supports(DataType datatype, ReduceOp op);
    static PGResult<AllReduceInvocation> create(CollectiveBindingId binding_id,
                                                const void* input, void* output,
                                                size_t element_count,
                                                DataType datatype, ReduceOp op);

    CollectiveBindingId bindingId() const override { return binding_id_; }
    void launch(const CollectiveKernelContext& context,
                cudaStream_t stream) const override;

   private:
    AllReduceInvocation(CollectiveBindingId binding_id, const void* input,
                        void* output, uint64_t element_count, DataType datatype)
        : binding_id_(binding_id),
          input_(input),
          output_(output),
          element_count_(element_count),
          datatype_(datatype) {}

    CollectiveBindingId binding_id_ = kInvalidCollectiveBindingId;
    const void* input_ = nullptr;
    void* output_ = nullptr;
    uint64_t element_count_ = 0;
    DataType datatype_ = DataType::Float16;
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_COLLECTIVE_OPERATION_ALLREDUCE_H
