#include "collective/operation/allreduce.h"

#include <limits>

namespace mooncake {

bool AllReduceInvocation::supports(DataType datatype, ReduceOp op) {
    return op == ReduceOp::Sum &&
           (datatype == DataType::Float16 || datatype == DataType::Bfloat16 ||
            datatype == DataType::Float32);
}

PGResult<AllReduceInvocation> AllReduceInvocation::create(
    CollectiveBindingId binding_id, const void* input, void* output,
    size_t element_count, DataType datatype, ReduceOp op) {
    PG_VALIDATE_ARG(supports(datatype, op),
                    "planned AllReduce signature is not supported");
    const uint64_t bytes_per_element = elementSize(datatype);
    PG_VALIDATE_ARG(element_count <= std::numeric_limits<uint64_t>::max() /
                                         bytes_per_element,
                    "planned AllReduce element count overflows uint64_t");
    return AllReduceInvocation(binding_id, input, output,
                               static_cast<uint64_t>(element_count), datatype);
}

void AllReduceInvocation::launch(const CollectiveKernelContext& context,
                                 void* retry_input, cudaStream_t stream) const {
    launchAllReduceExecutor(
        AllReduceExecutorArgs{
            .input = input_,
            .output = output_,
            .retry_input = retry_input,
            .context = context,
            .element_count = element_count_,
            .datatype = datatype_,
        },
        stream);
}

}  // namespace mooncake
