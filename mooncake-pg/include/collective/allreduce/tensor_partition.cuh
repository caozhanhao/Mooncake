#ifndef MOONCAKE_PG_COLLECTIVE_ALLREDUCE_TENSOR_PARTITION_CUH
#define MOONCAKE_PG_COLLECTIVE_ALLREDUCE_TENSOR_PARTITION_CUH

#include <cstdint>

namespace mooncake {

struct TensorShard {
    uint64_t offset_elements = 0;
    uint64_t length_elements = 0;
};

#if defined(__CUDACC__)
#define MOONCAKE_COLLECTIVE_HOST_DEVICE __host__ __device__
#else
#define MOONCAKE_COLLECTIVE_HOST_DEVICE
#endif

// Callers consume Coordinator-owned state, which already establishes
// shard_count != 0 and shard < shard_count.
MOONCAKE_COLLECTIVE_HOST_DEVICE inline TensorShard tensorShardUnchecked(
    uint64_t total_elements, uint32_t shard_count, uint32_t shard) {
    const uint64_t base = total_elements / shard_count;
    const uint64_t remainder = total_elements % shard_count;
    return TensorShard{
        static_cast<uint64_t>(shard) * base +
            (shard < remainder ? shard : remainder),
        base + (shard < remainder ? 1 : 0),
    };
}

#undef MOONCAKE_COLLECTIVE_HOST_DEVICE

}  // namespace mooncake

#endif  // MOONCAKE_PG_COLLECTIVE_ALLREDUCE_TENSOR_PARTITION_CUH
