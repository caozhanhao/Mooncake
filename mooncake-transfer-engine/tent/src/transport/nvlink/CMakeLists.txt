if (CUDAToolkit_FOUND)
    option(TENT_NVLINK_USE_COPY_KERNEL "Use CUDA copy kernel instead of cudaMemcpyAsync for NVLink transport" ON)

    set(XPORT_SOURCES nvlink_transport.cpp)

    if (TENT_NVLINK_USE_COPY_KERNEL)
        enable_language(CUDA)
        list(APPEND XPORT_SOURCES nvlink_copy_kernel.cu)
    endif()

    add_library(tent_xport_nvlink STATIC ${XPORT_SOURCES})
    target_link_libraries(tent_xport_nvlink PUBLIC CUDA::cuda_driver CUDA::cudart platform_cuda tent_common)

    if (TENT_NVLINK_USE_COPY_KERNEL)
        target_compile_definitions(tent_xport_nvlink PRIVATE TENT_NVLINK_USE_COPY_KERNEL)
    endif()
endif()