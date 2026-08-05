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
    static PGResult<AllReduceInvocation> create(const void* input, void* output,
                                                size_t element_count,
                                                DataType datatype, ReduceOp op);

    void launch(const CollectiveKernelArgs& common,
                cudaStream_t stream) const override;

   private:
    AllReduceInvocation(const void* input, void* output, uint64_t element_count,
                        DataType datatype)
        : input_(input),
          output_(output),
          element_count_(element_count),
          datatype_(datatype) {}

    const void* input_ = nullptr;
    void* output_ = nullptr;
    uint64_t element_count_ = 0;
    DataType datatype_ = DataType::Float16;
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_COLLECTIVE_OPERATION_ALLREDUCE_H
