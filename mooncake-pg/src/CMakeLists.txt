set(_pg_device_sources
    "${CMAKE_CURRENT_SOURCE_DIR}/mooncake_worker.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/collective/executor/allreduce.cu")
if(USE_CUDA)
  # Preserve the existing sm_80/sm_90 coverage unless the caller selects
  # architectures through CMake's standard variable or CUDAARCHS environment.
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND
     "$ENV{CUDAARCHS}" STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES 80-real 90-real)
  endif()
  enable_language(CUDA)
  set(_pg_device_inputs ${_pg_device_sources})
elseif(USE_MUSA OR USE_MACA)
  set(_pg_device_flags
      -std=c++20
      -O3
      --compiler-options=-fPIC
      -MMD
      "-I${CMAKE_CURRENT_SOURCE_DIR}/../include"
      "-I${CMAKE_CURRENT_SOURCE_DIR}/../../mooncake-transfer-engine/include")
  set(_pg_device_dependencies
      "${CMAKE_CURRENT_SOURCE_DIR}/../include/mooncake_worker_kernels.cuh"
      "${CMAKE_CURRENT_SOURCE_DIR}/../include/error_types.h"
      "${CMAKE_CURRENT_SOURCE_DIR}/../../mooncake-transfer-engine/include/common.h"
      "${CMAKE_CURRENT_SOURCE_DIR}/../../mooncake-transfer-engine/include/cuda_alike.h"
      "${CMAKE_CURRENT_SOURCE_DIR}/../../mooncake-transfer-engine/include/transfer_engine.h")

  if(USE_MUSA)
    set(_pg_platform_name MUSA)
    set(_pg_device_compiler_name mcc)
    if(DEFINED ENV{MUSA_HOME} AND NOT "$ENV{MUSA_HOME}" STREQUAL "")
      set(_pg_device_compiler_hint "$ENV{MUSA_HOME}/bin")
    else()
      set(_pg_device_compiler_hint /usr/local/musa/bin)
    endif()
    set(_pg_platform_device_flags
        --cuda-gpu-arch=mp_21
        --cuda-gpu-arch=mp_31)
  elseif(USE_MACA)
    set(_pg_platform_name MACA)
    set(_pg_device_compiler_name mxcc)
    set(_pg_device_compiler_hint "${MACA_ROOT}/bin")
    set(_pg_platform_device_flags)
  endif()

  find_program(_pg_device_compiler NAMES "${_pg_device_compiler_name}"
               HINTS "${_pg_device_compiler_hint}")
  if(NOT _pg_device_compiler)
    message(FATAL_ERROR
            "USE_${_pg_platform_name}=ON requires the "
            "${_pg_platform_name} compiler (${_pg_device_compiler_name})")
  endif()
  # Transfer Engine's device-side portability headers currently use the
  # MOONCAKE_EP_USE_* selector for non-CUDA compilers.
  list(APPEND _pg_device_flags
       "-DUSE_${_pg_platform_name}"
       "-DMOONCAKE_EP_USE_${_pg_platform_name}"
       ${_pg_platform_device_flags})

  set(_pg_device_inputs)
  foreach(_pg_device_source IN LISTS _pg_device_sources)
    get_filename_component(_pg_device_stem "${_pg_device_source}" NAME_WE)
    set(_pg_device_object
        "${CMAKE_CURRENT_BINARY_DIR}/${_pg_device_stem}_device.o")
    set(_pg_device_depfile
        "${CMAKE_CURRENT_BINARY_DIR}/${_pg_device_stem}_device.d")
    set(_pg_source_device_flags
        ${_pg_device_flags}
        -MT "${_pg_device_object}"
        -MF "${_pg_device_depfile}")
    set(_pg_depfile_argument)
    if(CMAKE_GENERATOR MATCHES "^Ninja" OR
       (CMAKE_GENERATOR MATCHES "Makefiles" AND
        CMAKE_VERSION VERSION_GREATER_EQUAL 3.20))
      set(_pg_depfile_argument DEPFILE "${_pg_device_depfile}")
    endif()

    add_custom_command(
      OUTPUT "${_pg_device_object}"
      COMMAND "${_pg_device_compiler}" ${_pg_source_device_flags}
              -c "${_pg_device_source}" -o "${_pg_device_object}"
      DEPENDS "${_pg_device_source}" ${_pg_device_dependencies}
      ${_pg_depfile_argument}
      COMMENT
        "Compiling Mooncake PG ${_pg_platform_name} device source ${_pg_device_stem}"
      COMMAND_EXPAND_LISTS
      VERBATIM)
    set_source_files_properties(
      "${_pg_device_object}"
      PROPERTIES GENERATED TRUE EXTERNAL_OBJECT TRUE)
    list(APPEND _pg_device_inputs "${_pg_device_object}")
  endforeach()
else()
  message(FATAL_ERROR
          "Mooncake PG currently requires one of USE_CUDA, USE_MUSA, or USE_MACA")
endif()

# Keep the GPU fatbin out of the host core so auditwheel can safely repair the
# core while the device library is added to the wheel afterwards.
add_library(mooncake_pg_device SHARED ${_pg_device_inputs})
target_include_directories(
  mooncake_pg_device
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include
          ${CMAKE_CURRENT_SOURCE_DIR}/../../mooncake-transfer-engine/include)

if(USE_CUDA)
  set_target_properties(
    mooncake_pg_device
    PROPERTIES CUDA_STANDARD 20
               CUDA_STANDARD_REQUIRED ON
               CUDA_EXTENSIONS OFF
               CUDA_RUNTIME_LIBRARY Shared)
  target_compile_options(
    mooncake_pg_device PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
  target_link_libraries(mooncake_pg_device PRIVATE CUDA::cudart)
elseif(USE_MUSA)
  set_target_properties(mooncake_pg_device PROPERTIES LINKER_LANGUAGE CXX)
  target_link_libraries(mooncake_pg_device PRIVATE musa musart rt)
elseif(USE_MACA)
  if(NOT DEFINED MACA_RUNTIME_LIBS)
    set(MACA_RUNTIME_LIBS mcruntime mxc-runtime64 rt)
  endif()
  set_target_properties(mooncake_pg_device PROPERTIES LINKER_LANGUAGE CXX)
  target_link_libraries(mooncake_pg_device PRIVATE ${MACA_RUNTIME_LIBS})
endif()

set_target_properties(
  mooncake_pg_device
  PROPERTIES POSITION_INDEPENDENT_CODE ON
             BUILD_RPATH "$ORIGIN"
             BUILD_RPATH_USE_ORIGIN YES)

add_library(mooncake_pg SHARED
    collective/plan/allreduce_plan_builder.cpp
    collective/plan/active_group_ranks.cpp
    collective/plan/collective_plan_registry.cpp
    collective/route/group_peer_routes.cpp
    collective/buffer/collective_buffer_pool.cpp
    collective/operation/allreduce.cpp
    collective/runtime/collective_failure.cpp
    collective/runtime/collective_lane_pool.cpp
    collective/runtime/collective_runtime.cpp
    collective/runtime/control_pool.cpp
    collective/runtime/host_progress.cpp
    collective/runtime/resource_pool.cpp
    collective/transport/host_transfer_proxy.cpp
    control_plane/link_manager.cpp
    control_plane/device_link_manager.cpp
    control_plane/collective_policy.cpp
    control_plane/coordinator.cpp
    control_plane/coordinator_host.cpp
    control_plane/agent.cpp
    control_plane/agent_host.cpp
    control_plane/rpc_runtime.cpp
    gpu_runtime.cpp
    mooncake_communicator.cpp
    mooncake_pg.cpp
    mooncake_worker_host.cpp
    mooncake_worker_thread.cpp
    p2p_proxy.cpp
)

target_compile_features(mooncake_pg PRIVATE cxx_std_20)
target_include_directories(
  mooncake_pg
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
         $<INSTALL_INTERFACE:include>)
target_link_libraries(mooncake_pg PRIVATE transfer_engine mooncake_pg_device)

set_target_properties(
  mooncake_pg
  PROPERTIES POSITION_INDEPENDENT_CODE ON
             BUILD_RPATH "$ORIGIN"
             BUILD_RPATH_USE_ORIGIN YES)
