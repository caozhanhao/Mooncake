#include <cuda_runtime.h>

#ifdef TENT_NVLINK_USE_COPY_KERNEL

__global__ void nvlinkCopyKernel(char* __restrict__ dst,
                                 const char* __restrict__ src,
                                 size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    // Grid-stride over 16-byte chunks. Use __builtin_memcpy to avoid
    // alignment requirements for ulonglong2 vector loads/stores.
    // The compiler optimizes this to LDG.128 / STG.128 when addresses
    // are aligned at runtime.
    size_t num_elems = n / sizeof(ulonglong2);
    for (size_t i = idx; i < num_elems; i += stride) {
        size_t offset = i * sizeof(ulonglong2);
        ulonglong2 val;
        __builtin_memcpy(&val, src + offset, sizeof(val));
        __builtin_memcpy(dst + offset, &val, sizeof(val));
    }

    // Cooperative tail copy: all threads participate
    size_t tail_start = num_elems * sizeof(ulonglong2);
    for (size_t i = tail_start + idx; i < n; i += stride) {
        dst[i] = src[i];
    }
}

#endif  // TENT_NVLINK_USE_COPY_KERNEL

extern "C" cudaError_t nvlinkMemcpyAsync(void* dst, const void* src, size_t n,
                                           cudaMemcpyKind kind,
                                           cudaStream_t stream) {
#ifdef TENT_NVLINK_USE_COPY_KERNEL
    if (kind == cudaMemcpyDeviceToDevice) {
        if (n == 0) return cudaSuccess;

        int device;
        cudaError_t err = cudaGetDevice(&device);
        if (err != cudaSuccess) return err;

        int sm_count;
        err = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount,
                                     device);
        if (err != cudaSuccess) return err;

        constexpr int block_size = 256;
        int grid_size = sm_count * 8;
        int num_threads_total = grid_size * block_size;
        int num_elems =
            static_cast<int>((n + sizeof(ulonglong2) - 1) / sizeof(ulonglong2));
        if (num_elems < num_threads_total) {
            grid_size = (num_elems + block_size - 1) / block_size;
        }
        if (grid_size == 0) grid_size = 1;

        nvlinkCopyKernel<<<grid_size, block_size, 0, stream>>>(
            static_cast<char*>(dst), static_cast<const char*>(src), n);

        return cudaPeekAtLastError();
    }
#endif  // TENT_NVLINK_USE_COPY_KERNEL

    return cudaMemcpyAsync(dst, src, n, kind, stream);
}
