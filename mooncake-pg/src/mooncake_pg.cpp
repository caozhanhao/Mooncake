#include <mooncake_pg.h>

#include <mooncake_communicator.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <new>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include "comm_types.h"
#include "error_types.h"

using namespace mooncake;

struct mooncakePgContext {
    std::unique_ptr<MooncakePGContext> impl;
};

struct mooncakePgComm {
    explicit mooncakePgComm(MooncakePGContext& context) : context(&context) {}

    ~mooncakePgComm() {
        impl.reset();
        if (context_use_counted) context->decrementCommUseCount();
    }

    MooncakePGContext* context;
    bool context_use_counted = false;
    std::unique_ptr<MooncakeCommunicator> impl;
};

struct mooncakePgCompletion {
    std::unique_ptr<WorkCompletion> impl;
};

namespace {

struct LastError {
    mooncakePgResult_t result = mooncakePgSuccess;
    // Keep catch paths allocation-free: assigning to a std::string here could
    // throw and let an exception escape the C API boundary.
    std::array<char, MOONCAKE_PG_MAX_ERROR_STRING> message{};
};

thread_local LastError g_last_error;

mooncakePgResult_t setLastError(mooncakePgResult_t result,
                                const char* message) {
    g_last_error.result = result;
    std::snprintf(g_last_error.message.data(), g_last_error.message.size(),
                  "%s", message ? message : "");
    return result;
}

mooncakePgResult_t setLastError(const PGError& error) {
    const auto set = [&](mooncakePgResult_t result) {
        return setLastError(result, error.message.c_str());
    };
    switch (error.code) {
        case PGErrorCode::InvalidArgument:
            return set(mooncakePgInvalidArgument);
        case PGErrorCode::InvalidState:
            return set(mooncakePgInvalidState);
        case PGErrorCode::NotSupported:
            return set(mooncakePgNotSupported);
        case PGErrorCode::Timeout:
            return set(mooncakePgTimeout);
        case PGErrorCode::ResourceBusy:
            return set(mooncakePgResourceBusy);
        case PGErrorCode::TransferEngineError:
            return set(mooncakePgTransferEngineError);
        case PGErrorCode::RpcError:
            return set(mooncakePgRpcError);
        case PGErrorCode::SystemError:
            return set(mooncakePgSystemError);
        case PGErrorCode::InternalError:
            return set(mooncakePgInternalError);
    }
    PG_ASSERT(false, "unknown PGErrorCode: ", static_cast<int>(error.code));
}

template <typename Function>
mooncakePgResult_t asCApiResult(Function&& function) {
    static_assert(
        std::is_same_v<std::invoke_result_t<Function>, PGResult<void>>);
    try {
        const auto result = function();
        return result.has_value() ? mooncakePgSuccess
                                  : setLastError(result.error());
    } catch (const PGAssertionException& error) {
        return setLastError(mooncakePgInternalError, error.what());
    } catch (const std::bad_alloc& error) {
        return setLastError(mooncakePgSystemError, error.what());
    } catch (const std::exception& error) {
        return setLastError(mooncakePgInternalError, error.what());
    } catch (...) {
        return setLastError(mooncakePgInternalError,
                            "unknown Mooncake PG error");
    }
}

PGResult<void> parseCommConfig(const mooncakePgCommConfig_t* config,
                               MooncakeCommunicatorConfig& output) {
    PG_VALIDATE_ARG(config, "communicator config is null");
    PG_VALIDATE_ARG(config->structSize >= sizeof(*config) &&
                        config->magic == MOONCAKE_PG_COMM_CONFIG_MAGIC &&
                        config->version == MOONCAKE_PG_COMM_CONFIG_VERSION,
                    "invalid communicator config header");
    PG_VALIDATE_ARG(config->groupId, "communicator group ID is null");
    PG_VALIDATE_ARG(config->globalRanks, "communicator global ranks are null");
    PG_VALIDATE_ARG(config->globalRankCount != 0 &&
                        config->globalRankCount <= MOONCAKE_PG_MAX_RANKS,
                    "invalid communicator global rank count");
    PG_VALIDATE_ARG(
        config->activeRanksMirror || config->activeRanksMirrorCount == 0,
        "active-ranks mirror is null");

    MooncakeCommunicatorConfig internal;
    internal.rank = config->rank;
    internal.size = config->size;
    internal.max_group_size =
        config->maxGroupSize == MOONCAKE_PG_CONFIG_UNDEF_INT
            ? config->size
            : config->maxGroupSize;
    internal.global_ranks.assign(config->globalRanks,
                                 config->globalRanks + config->globalRankCount);
    internal.group_bootstrap_id = config->groupId;
    switch (config->deviceType) {
        case mooncakePgDeviceCpu:
            internal.is_cpu = true;
            break;
        case mooncakePgDeviceGpu:
            internal.is_cpu = false;
            break;
        default:
            return makePGError(PGErrorCode::InvalidArgument,
                               "invalid communicator device type");
    }
    internal.device_index = config->deviceIndex == MOONCAKE_PG_CONFIG_UNDEF_INT
                                ? -1
                                : config->deviceIndex;
    switch (config->idResolvePolicy) {
        case mooncakePgIdResolveCreateOrAttach:
            internal.group_resolve_policy =
                GroupBootstrapIdResolvePolicy::CreateOrAttach;
            break;
        case mooncakePgIdResolveAttachOrExtend:
            internal.group_resolve_policy =
                GroupBootstrapIdResolvePolicy::AttachOrExtend;
            break;
        default:
            return makePGError(PGErrorCode::InvalidArgument,
                               "invalid communicator group resolve policy");
    }
    internal.auto_deactivate_on_failure = config->autoDeactivateOnFailure != 0;
    internal.auto_sync_on_failure = config->autoSyncOnFailure != 0;
    internal.active_ranks_mirror = config->activeRanksMirror;
    internal.active_ranks_mirror_count = config->activeRanksMirrorCount;
    internal.active_ranks_mirror_is_device =
        config->activeRanksMirrorIsDevice != 0;
    if (internal.active_ranks_mirror_is_device) {
        internal.active_ranks_mirror_device_index =
            config->activeRanksMirrorDeviceIndex == MOONCAKE_PG_CONFIG_UNDEF_INT
                ? -1
                : config->activeRanksMirrorDeviceIndex;
    }
    output = std::move(internal);
    return {};
}

PGResult<void> convertDataType(mooncakePgDataType_t data_type,
                               DataType& output) {
    switch (data_type) {
        case mooncakePgInt8:
            output = DataType::Int8;
            break;
        case mooncakePgUint8:
            output = DataType::Uint8;
            break;
        case mooncakePgInt16:
            output = DataType::Int16;
            break;
        case mooncakePgUint16:
            output = DataType::Uint16;
            break;
        case mooncakePgInt32:
            output = DataType::Int32;
            break;
        case mooncakePgUint32:
            output = DataType::Uint32;
            break;
        case mooncakePgInt64:
            output = DataType::Int64;
            break;
        case mooncakePgUint64:
            output = DataType::Uint64;
            break;
        case mooncakePgFloat16:
            output = DataType::Float16;
            break;
        case mooncakePgFloat32:
            output = DataType::Float32;
            break;
        case mooncakePgFloat64:
            output = DataType::Float64;
            break;
        case mooncakePgBfloat16:
            output = DataType::Bfloat16;
            break;
        case mooncakePgBool:
            output = DataType::Bool;
            break;
        case mooncakePgFloat8e4m3fn:
            output = DataType::Float8e4m3fn;
            break;
        case mooncakePgFloat8e5m2:
            output = DataType::Float8e5m2;
            break;
        case mooncakePgFloat8e4m3fnuz:
            output = DataType::Float8e4m3fnuz;
            break;
        case mooncakePgFloat8e5m2fnuz:
            output = DataType::Float8e5m2fnuz;
            break;
        case mooncakePgFloat8e8m0fnu:
            output = DataType::Float8e8m0fnu;
            break;
        default:
            return makePGError(PGErrorCode::InvalidArgument,
                               "unsupported Mooncake PG datatype");
    }
    return {};
}

PGResult<void> convertReduceOp(mooncakePgReduceOp_t reduce_op,
                               ReduceOp& output) {
    switch (reduce_op) {
        case mooncakePgSum:
            output = ReduceOp::Sum;
            break;
        case mooncakePgAvg:
            output = ReduceOp::Avg;
            break;
        case mooncakePgProduct:
            output = ReduceOp::Product;
            break;
        case mooncakePgMin:
            output = ReduceOp::Min;
            break;
        case mooncakePgMax:
            output = ReduceOp::Max;
            break;
        default:
            return makePGError(PGErrorCode::InvalidArgument,
                               "unsupported Mooncake PG reduction operation");
    }
    return {};
}

cudaStream_t convertStream(mooncakePgStream_t stream) {
    return reinterpret_cast<cudaStream_t>(stream);
}

template <typename Launch>
mooncakePgResult_t invokeCommOpWithCompletion(
    mooncakePgComm_t comm, mooncakePgCompletion_t* output_completion,
    Launch&& launch) {
    return asCApiResult([&]() -> PGResult<void> {
        PG_VALIDATE_ARG(output_completion, "completion output is null");
        *output_completion = nullptr;
        PG_VALIDATE_ARG(comm && comm->impl, "invalid communicator");

        auto output = std::make_unique<mooncakePgCompletion>();
        auto result = launch();
        PG_TRY(result);
        auto completion = std::move(result).value();
        PG_ASSERT(completion, "operation returned no completion");

        output->impl = std::move(completion);
        *output_completion = output.release();
        return {};
    });
}

template <typename Launch>
mooncakePgResult_t invokeCommOp(mooncakePgComm_t comm, Launch&& launch) {
    return asCApiResult([&]() -> PGResult<void> {
        PG_VALIDATE_ARG(comm && comm->impl, "invalid communicator");
        return launch();
    });
}

PGResult<void> parseRanks(const int32_t* ranks, size_t rank_count,
                          std::vector<int>& output) {
    PG_VALIDATE_ARG(rank_count <= MOONCAKE_PG_MAX_RANKS,
                    "rank count exceeds maximum");
    PG_VALIDATE_ARG(rank_count == 0 || ranks, "ranks are null");
    if (rank_count == 0) {
        output.clear();
    } else {
        output.assign(ranks, ranks + rank_count);
    }
    return {};
}

mooncakePgProposalResponse_t convertProposalResponse(
    const ProposeViewUpdateResponse& response) {
    mooncakePgProposalResponse_t converted{};
    converted.status = static_cast<mooncakePgProposalStatus_t>(response.status);
    converted.newEpoch = response.new_epoch;
    converted.droppedRankCount =
        std::min(response.dropped_ranks.size(),
                 static_cast<size_t>(MOONCAKE_PG_MAX_RANKS));
    std::copy_n(response.dropped_ranks.begin(), converted.droppedRankCount,
                converted.droppedRanks);
    std::snprintf(converted.rejectReason, sizeof(converted.rejectReason), "%s",
                  response.reject_reason.c_str());
    return converted;
}

}  // namespace

const char* mooncakePgGetErrorString(mooncakePgResult_t result) {
    switch (result) {
        case mooncakePgSuccess:
            return "success";
        case mooncakePgInvalidArgument:
            return "invalid argument";
        case mooncakePgInvalidState:
            return "invalid state";
        case mooncakePgNotSupported:
            return "operation not supported";
        case mooncakePgTimeout:
            return "operation timed out";
        case mooncakePgResourceBusy:
            return "resource busy";
        case mooncakePgTransferEngineError:
            return "transfer engine error";
        case mooncakePgRpcError:
            return "RPC error";
        case mooncakePgSystemError:
            return "system error";
        case mooncakePgInternalError:
            return "internal error";
    }
    return "unknown result";
}

const char* mooncakePgGetLastError(void) { return g_last_error.message.data(); }

mooncakePgResult_t mooncakePgContextCreate(mooncakePgContext_t* context) {
    return asCApiResult([&]() -> PGResult<void> {
        PG_VALIDATE_ARG(context, "context output is null");
        *context = nullptr;

        auto output = std::make_unique<mooncakePgContext>();
        output->impl = std::make_unique<MooncakePGContext>();
        *context = output.release();
        return {};
    });
}

mooncakePgResult_t mooncakePgContextInitialize(mooncakePgContext_t context,
                                               int global_rank,
                                               int max_world_size) {
    return asCApiResult([&]() -> PGResult<void> {
        PG_VALIDATE_ARG(context && context->impl, "invalid context");
        return context->impl->initialize(global_rank, max_world_size);
    });
}

mooncakePgResult_t mooncakePgContextLaunchCoordinator(
    mooncakePgContext_t context, char* coordinator_address_buf,
    size_t coordinator_address_buf_size) {
    return asCApiResult([&]() -> PGResult<void> {
        PG_VALIDATE_ARG(context && context->impl, "invalid context");
        PG_VALIDATE_ARG(coordinator_address_buf,
                        "coordinator-address output is null");
        auto result = context->impl->launchCoordinator();
        PG_TRY(result);
        const auto& value = result.value();
        PG_VALIDATE_ARG(value.size() < coordinator_address_buf_size,
                        "coordinator-address output is too small");
        std::memcpy(coordinator_address_buf, value.c_str(), value.size() + 1);
        return {};
    });
}

mooncakePgResult_t mooncakePgContextConnectCoordinator(
    mooncakePgContext_t context, const char* coordinator_address) {
    return asCApiResult([&]() -> PGResult<void> {
        PG_VALIDATE_ARG(context && context->impl, "invalid context");
        PG_VALIDATE_ARG(coordinator_address && coordinator_address[0] != '\0',
                        "coordinator address is null or empty");
        PG_TRY(context->impl->connectCoordinator(coordinator_address));
        return {};
    });
}

mooncakePgResult_t mooncakePgContextSetHostIp(mooncakePgContext_t context,
                                              const char* host_ip) {
    return asCApiResult([&]() -> PGResult<void> {
        PG_VALIDATE_ARG(context && context->impl, "invalid context");
        PG_VALIDATE_ARG(host_ip, "host IP is null");
        return context->impl->setHostIp(host_ip);
    });
}

mooncakePgResult_t mooncakePgContextSetTransferEngine(
    mooncakePgContext_t context, void* transfer_engine) {
    return asCApiResult([&]() -> PGResult<void> {
        PG_VALIDATE_ARG(context && context->impl, "invalid context");
        return context->impl->setExternalEngine(
            static_cast<TransferEngine*>(transfer_engine));
    });
}

mooncakePgResult_t mooncakePgContextSetDeviceFilter(mooncakePgContext_t context,
                                                    const char* const* filters,
                                                    size_t filter_count) {
    return asCApiResult([&]() -> PGResult<void> {
        PG_VALIDATE_ARG(context && context->impl, "invalid context");
        PG_VALIDATE_ARG(filter_count == 0 || filters,
                        "device filters are null");
        std::vector<std::string> values;
        values.reserve(filter_count);
        for (size_t index = 0; index < filter_count; ++index) {
            PG_VALIDATE_ARG(filters[index], "device filter is null");
            values.emplace_back(filters[index]);
        }
        return context->impl->setDeviceFilter(std::move(values));
    });
}

mooncakePgResult_t mooncakePgContextSetCollectiveTimeout(
    mooncakePgContext_t context, size_t timeout_us) {
    return asCApiResult([&]() -> PGResult<void> {
        PG_VALIDATE_ARG(context && context->impl, "invalid context");
        return context->impl->setCollectiveTimeout(timeout_us);
    });
}

mooncakePgResult_t mooncakePgContextSetP2PTimeout(mooncakePgContext_t context,
                                                  int64_t timeout_us) {
    return asCApiResult([&]() -> PGResult<void> {
        PG_VALIDATE_ARG(context && context->impl, "invalid context");
        return context->impl->setP2PTimeout(timeout_us);
    });
}

mooncakePgResult_t mooncakePgContextSetFaultReconciliationWindow(
    mooncakePgContext_t context, int64_t timeout_us) {
    return asCApiResult([&]() -> PGResult<void> {
        PG_VALIDATE_ARG(context && context->impl, "invalid context");
        return context->impl->setFaultReconciliationWindow(timeout_us);
    });
}

mooncakePgResult_t mooncakePgContextDestroy(mooncakePgContext_t context) {
    return asCApiResult([&]() -> PGResult<void> {
        if (!context) return {};
        PG_VALIDATE_ARG(context->impl, "invalid context");
        auto result = context->impl->shutdown();
        if (result.has_value()) delete context;
        return result;
    });
}

mooncakePgResult_t mooncakePgCommCreate(mooncakePgContext_t context,
                                        const mooncakePgCommConfig_t* config,
                                        mooncakePgComm_t* comm) {
    return asCApiResult([&]() -> PGResult<void> {
        PG_VALIDATE_ARG(comm, "communicator output is null");
        *comm = nullptr;
        PG_VALIDATE_ARG(context && context->impl, "invalid context");
        MooncakeCommunicatorConfig internal;
        PG_TRY(parseCommConfig(config, internal));

        auto output = std::make_unique<mooncakePgComm>(*context->impl);
        auto use_count_result = context->impl->incrementCommUseCount();
        PG_TRY(use_count_result);
        output->context_use_counted = true;
        auto communicator_result =
            MooncakeCommunicator::create(*context->impl, std::move(internal));
        PG_TRY(communicator_result);
        output->impl = std::move(communicator_result.value());
        *comm = output.release();
        return {};
    });
}

mooncakePgResult_t mooncakePgCommDestroy(mooncakePgComm_t comm) {
    return asCApiResult([&]() -> PGResult<void> {
        std::unique_ptr<mooncakePgComm> holder(comm);
        if (holder && holder->impl) {
            return holder->impl->shutdown();
        }
        return {};
    });
}

mooncakePgResult_t mooncakePgCommGetRank(mooncakePgComm_t comm, int* rank) {
    return asCApiResult([&]() -> PGResult<void> {
        PG_VALIDATE_ARG(comm && comm->impl, "invalid communicator");
        PG_VALIDATE_ARG(rank, "rank output is null");
        *rank = comm->impl->getRank();
        return {};
    });
}

mooncakePgResult_t mooncakePgCommGetSize(mooncakePgComm_t comm, int* size) {
    return asCApiResult([&]() -> PGResult<void> {
        PG_VALIDATE_ARG(comm && comm->impl, "invalid communicator");
        PG_VALIDATE_ARG(size, "size output is null");
        *size = comm->impl->getSize();
        return {};
    });
}

mooncakePgResult_t mooncakePgCommGetMaxGroupSize(mooncakePgComm_t comm,
                                                 int* max_group_size) {
    return asCApiResult([&]() -> PGResult<void> {
        PG_VALIDATE_ARG(comm && comm->impl, "invalid communicator");
        PG_VALIDATE_ARG(max_group_size, "maximum group-size output is null");
        *max_group_size = comm->impl->getMaxGroupSize();
        return {};
    });
}

mooncakePgResult_t mooncakePgBroadcastGpu(const void* send_buffer,
                                          void* recv_buffer, size_t count,
                                          mooncakePgDataType_t data_type,
                                          int root, mooncakePgComm_t comm,
                                          mooncakePgStream_t stream,
                                          int32_t* failed_ranks_hint,
                                          size_t failed_ranks_hint_count) {
    return invokeCommOp(comm, [&]() -> PGResult<void> {
        DataType converted_data_type;
        PG_TRY(convertDataType(data_type, converted_data_type));
        return comm->impl->broadcastGpu(
            send_buffer, recv_buffer, count, converted_data_type, root,
            convertStream(stream), failed_ranks_hint, failed_ranks_hint_count);
    });
}

mooncakePgResult_t mooncakePgAllReduceGpu(
    const void* send_buffer, void* recv_buffer, size_t count,
    mooncakePgDataType_t data_type, mooncakePgReduceOp_t reduce_op,
    mooncakePgComm_t comm, mooncakePgStream_t stream,
    int32_t* failed_ranks_hint, size_t failed_ranks_hint_count) {
    return invokeCommOp(comm, [&]() -> PGResult<void> {
        DataType converted_data_type;
        PG_TRY(convertDataType(data_type, converted_data_type));
        ReduceOp converted_reduce_op;
        PG_TRY(convertReduceOp(reduce_op, converted_reduce_op));
        return comm->impl->allReduceGpu(
            send_buffer, recv_buffer, count, converted_data_type,
            converted_reduce_op, convertStream(stream), failed_ranks_hint,
            failed_ranks_hint_count);
    });
}

mooncakePgResult_t mooncakePgAllGatherGpu(const void* send_buffer,
                                          void* recv_buffer, size_t count,
                                          mooncakePgDataType_t data_type,
                                          mooncakePgComm_t comm,
                                          mooncakePgStream_t stream,
                                          int32_t* failed_ranks_hint,
                                          size_t failed_ranks_hint_count) {
    return invokeCommOp(comm, [&]() -> PGResult<void> {
        DataType converted_data_type;
        PG_TRY(convertDataType(data_type, converted_data_type));
        return comm->impl->allGatherGpu(
            send_buffer, recv_buffer, count, converted_data_type,
            convertStream(stream), failed_ranks_hint, failed_ranks_hint_count);
    });
}

mooncakePgResult_t mooncakePgReduceScatterGpu(
    const void* send_buffer, void* recv_buffer, size_t count,
    mooncakePgDataType_t data_type, mooncakePgReduceOp_t reduce_op,
    mooncakePgComm_t comm, mooncakePgStream_t stream,
    int32_t* failed_ranks_hint, size_t failed_ranks_hint_count) {
    return invokeCommOp(comm, [&]() -> PGResult<void> {
        DataType converted_data_type;
        PG_TRY(convertDataType(data_type, converted_data_type));
        ReduceOp converted_reduce_op;
        PG_TRY(convertReduceOp(reduce_op, converted_reduce_op));
        return comm->impl->reduceScatterGpu(
            send_buffer, recv_buffer, count, converted_data_type,
            converted_reduce_op, convertStream(stream), failed_ranks_hint,
            failed_ranks_hint_count);
    });
}

mooncakePgResult_t mooncakePgAllToAllGpu(const void* send_buffer,
                                         void* recv_buffer, size_t count,
                                         mooncakePgDataType_t data_type,
                                         mooncakePgComm_t comm,
                                         mooncakePgStream_t stream,
                                         int32_t* failed_ranks_hint,
                                         size_t failed_ranks_hint_count) {
    return invokeCommOp(comm, [&]() -> PGResult<void> {
        DataType converted_data_type;
        PG_TRY(convertDataType(data_type, converted_data_type));
        return comm->impl->allToAllGpu(
            send_buffer, recv_buffer, count, converted_data_type,
            convertStream(stream), failed_ranks_hint, failed_ranks_hint_count);
    });
}

mooncakePgResult_t mooncakePgReduceGpu(
    const void* send_buffer, void* recv_buffer, size_t count,
    mooncakePgDataType_t data_type, mooncakePgReduceOp_t reduce_op, int root,
    mooncakePgComm_t comm, mooncakePgStream_t stream,
    int32_t* failed_ranks_hint, size_t failed_ranks_hint_count) {
    return invokeCommOp(comm, [&]() -> PGResult<void> {
        DataType converted_data_type;
        PG_TRY(convertDataType(data_type, converted_data_type));
        ReduceOp converted_reduce_op;
        PG_TRY(convertReduceOp(reduce_op, converted_reduce_op));
        return comm->impl->reduceGpu(
            send_buffer, recv_buffer, count, converted_data_type,
            converted_reduce_op, root, convertStream(stream), failed_ranks_hint,
            failed_ranks_hint_count);
    });
}

mooncakePgResult_t mooncakePgGatherGpu(const void* send_buffer,
                                       void* recv_buffer, size_t count,
                                       mooncakePgDataType_t data_type, int root,
                                       mooncakePgComm_t comm,
                                       mooncakePgStream_t stream,
                                       int32_t* failed_ranks_hint,
                                       size_t failed_ranks_hint_count) {
    return invokeCommOp(comm, [&]() -> PGResult<void> {
        DataType converted_data_type;
        PG_TRY(convertDataType(data_type, converted_data_type));
        return comm->impl->gatherGpu(
            send_buffer, recv_buffer, count, converted_data_type, root,
            convertStream(stream), failed_ranks_hint, failed_ranks_hint_count);
    });
}

mooncakePgResult_t mooncakePgScatterGpu(const void* send_buffer,
                                        void* recv_buffer, size_t count,
                                        mooncakePgDataType_t data_type,
                                        int root, mooncakePgComm_t comm,
                                        mooncakePgStream_t stream,
                                        int32_t* failed_ranks_hint,
                                        size_t failed_ranks_hint_count) {
    return invokeCommOp(comm, [&]() -> PGResult<void> {
        DataType converted_data_type;
        PG_TRY(convertDataType(data_type, converted_data_type));
        return comm->impl->scatterGpu(
            send_buffer, recv_buffer, count, converted_data_type, root,
            convertStream(stream), failed_ranks_hint, failed_ranks_hint_count);
    });
}

mooncakePgResult_t mooncakePgBarrierGpu(mooncakePgComm_t comm,
                                        mooncakePgStream_t stream,
                                        int32_t* failed_ranks_hint,
                                        size_t failed_ranks_hint_count) {
    return invokeCommOp(comm, [&]() -> PGResult<void> {
        return comm->impl->barrierGpu(convertStream(stream), failed_ranks_hint,
                                      failed_ranks_hint_count);
    });
}

mooncakePgResult_t mooncakePgBroadcastCpu(const void* send_buffer,
                                          void* recv_buffer, size_t count,
                                          mooncakePgDataType_t data_type,
                                          int root, mooncakePgComm_t comm,
                                          int32_t* failed_ranks_hint,
                                          size_t failed_ranks_hint_count,
                                          mooncakePgCompletion_t* completion) {
    return invokeCommOpWithCompletion(
        comm, completion, [&]() -> PGResult<std::unique_ptr<WorkCompletion>> {
            DataType converted_data_type;
            PG_TRY(convertDataType(data_type, converted_data_type));
            return comm->impl->broadcastCpu(
                send_buffer, recv_buffer, count, converted_data_type, root,
                failed_ranks_hint, failed_ranks_hint_count);
        });
}

mooncakePgResult_t mooncakePgAllReduceCpu(
    const void* send_buffer, void* recv_buffer, size_t count,
    mooncakePgDataType_t data_type, mooncakePgReduceOp_t reduce_op,
    mooncakePgComm_t comm, int32_t* failed_ranks_hint,
    size_t failed_ranks_hint_count, mooncakePgCompletion_t* completion) {
    return invokeCommOpWithCompletion(
        comm, completion, [&]() -> PGResult<std::unique_ptr<WorkCompletion>> {
            DataType converted_data_type;
            PG_TRY(convertDataType(data_type, converted_data_type));
            ReduceOp converted_reduce_op;
            PG_TRY(convertReduceOp(reduce_op, converted_reduce_op));
            return comm->impl->allReduceCpu(
                send_buffer, recv_buffer, count, converted_data_type,
                converted_reduce_op, failed_ranks_hint,
                failed_ranks_hint_count);
        });
}

mooncakePgResult_t mooncakePgAllGatherCpu(const void* send_buffer,
                                          void* recv_buffer, size_t count,
                                          mooncakePgDataType_t data_type,
                                          mooncakePgComm_t comm,
                                          int32_t* failed_ranks_hint,
                                          size_t failed_ranks_hint_count,
                                          mooncakePgCompletion_t* completion) {
    return invokeCommOpWithCompletion(
        comm, completion, [&]() -> PGResult<std::unique_ptr<WorkCompletion>> {
            DataType converted_data_type;
            PG_TRY(convertDataType(data_type, converted_data_type));
            return comm->impl->allGatherCpu(
                send_buffer, recv_buffer, count, converted_data_type,
                failed_ranks_hint, failed_ranks_hint_count);
        });
}

mooncakePgResult_t mooncakePgReduceScatterCpu(
    const void* send_buffer, void* recv_buffer, size_t count,
    mooncakePgDataType_t data_type, mooncakePgReduceOp_t reduce_op,
    mooncakePgComm_t comm, int32_t* failed_ranks_hint,
    size_t failed_ranks_hint_count, mooncakePgCompletion_t* completion) {
    return invokeCommOpWithCompletion(
        comm, completion, [&]() -> PGResult<std::unique_ptr<WorkCompletion>> {
            DataType converted_data_type;
            PG_TRY(convertDataType(data_type, converted_data_type));
            ReduceOp converted_reduce_op;
            PG_TRY(convertReduceOp(reduce_op, converted_reduce_op));
            return comm->impl->reduceScatterCpu(
                send_buffer, recv_buffer, count, converted_data_type,
                converted_reduce_op, failed_ranks_hint,
                failed_ranks_hint_count);
        });
}

mooncakePgResult_t mooncakePgAllToAllCpu(const void* send_buffer,
                                         void* recv_buffer, size_t count,
                                         mooncakePgDataType_t data_type,
                                         mooncakePgComm_t comm,
                                         int32_t* failed_ranks_hint,
                                         size_t failed_ranks_hint_count,
                                         mooncakePgCompletion_t* completion) {
    return invokeCommOpWithCompletion(
        comm, completion, [&]() -> PGResult<std::unique_ptr<WorkCompletion>> {
            DataType converted_data_type;
            PG_TRY(convertDataType(data_type, converted_data_type));
            return comm->impl->allToAllCpu(
                send_buffer, recv_buffer, count, converted_data_type,
                failed_ranks_hint, failed_ranks_hint_count);
        });
}

mooncakePgResult_t mooncakePgReduceCpu(
    const void* send_buffer, void* recv_buffer, size_t count,
    mooncakePgDataType_t data_type, mooncakePgReduceOp_t reduce_op, int root,
    mooncakePgComm_t comm, int32_t* failed_ranks_hint,
    size_t failed_ranks_hint_count, mooncakePgCompletion_t* completion) {
    return invokeCommOpWithCompletion(
        comm, completion, [&]() -> PGResult<std::unique_ptr<WorkCompletion>> {
            DataType converted_data_type;
            PG_TRY(convertDataType(data_type, converted_data_type));
            ReduceOp converted_reduce_op;
            PG_TRY(convertReduceOp(reduce_op, converted_reduce_op));
            return comm->impl->reduceCpu(
                send_buffer, recv_buffer, count, converted_data_type,
                converted_reduce_op, root, failed_ranks_hint,
                failed_ranks_hint_count);
        });
}

mooncakePgResult_t mooncakePgGatherCpu(const void* send_buffer,
                                       void* recv_buffer, size_t count,
                                       mooncakePgDataType_t data_type, int root,
                                       mooncakePgComm_t comm,
                                       int32_t* failed_ranks_hint,
                                       size_t failed_ranks_hint_count,
                                       mooncakePgCompletion_t* completion) {
    return invokeCommOpWithCompletion(
        comm, completion, [&]() -> PGResult<std::unique_ptr<WorkCompletion>> {
            DataType converted_data_type;
            PG_TRY(convertDataType(data_type, converted_data_type));
            return comm->impl->gatherCpu(
                send_buffer, recv_buffer, count, converted_data_type, root,
                failed_ranks_hint, failed_ranks_hint_count);
        });
}

mooncakePgResult_t mooncakePgScatterCpu(const void* send_buffer,
                                        void* recv_buffer, size_t count,
                                        mooncakePgDataType_t data_type,
                                        int root, mooncakePgComm_t comm,
                                        int32_t* failed_ranks_hint,
                                        size_t failed_ranks_hint_count,
                                        mooncakePgCompletion_t* completion) {
    return invokeCommOpWithCompletion(
        comm, completion, [&]() -> PGResult<std::unique_ptr<WorkCompletion>> {
            DataType converted_data_type;
            PG_TRY(convertDataType(data_type, converted_data_type));
            return comm->impl->scatterCpu(
                send_buffer, recv_buffer, count, converted_data_type, root,
                failed_ranks_hint, failed_ranks_hint_count);
        });
}

mooncakePgResult_t mooncakePgBarrierCpu(mooncakePgComm_t comm,
                                        int32_t* failed_ranks_hint,
                                        size_t failed_ranks_hint_count,
                                        mooncakePgCompletion_t* completion) {
    return invokeCommOpWithCompletion(
        comm, completion, [&]() -> PGResult<std::unique_ptr<WorkCompletion>> {
            return comm->impl->barrierCpu(failed_ranks_hint,
                                          failed_ranks_hint_count);
        });
}

mooncakePgResult_t mooncakePgSendGpu(const void* send_buffer, size_t count,
                                     mooncakePgDataType_t data_type, int peer,
                                     mooncakePgComm_t comm,
                                     mooncakePgStream_t stream,
                                     int32_t* failed_ranks_hint,
                                     size_t failed_ranks_hint_count,
                                     mooncakePgCompletion_t* completion) {
    return invokeCommOpWithCompletion(
        comm, completion, [&]() -> PGResult<std::unique_ptr<WorkCompletion>> {
            DataType converted_data_type;
            PG_TRY(convertDataType(data_type, converted_data_type));
            return comm->impl->sendGpu(send_buffer, count, converted_data_type,
                                       peer, convertStream(stream),
                                       failed_ranks_hint,
                                       failed_ranks_hint_count);
        });
}

mooncakePgResult_t mooncakePgRecvGpu(void* recv_buffer, size_t count,
                                     mooncakePgDataType_t data_type, int peer,
                                     mooncakePgComm_t comm,
                                     mooncakePgStream_t stream,
                                     int32_t* failed_ranks_hint,
                                     size_t failed_ranks_hint_count,
                                     mooncakePgCompletion_t* completion) {
    return invokeCommOpWithCompletion(
        comm, completion, [&]() -> PGResult<std::unique_ptr<WorkCompletion>> {
            DataType converted_data_type;
            PG_TRY(convertDataType(data_type, converted_data_type));
            return comm->impl->recvGpu(recv_buffer, count, converted_data_type,
                                       peer, convertStream(stream),
                                       failed_ranks_hint,
                                       failed_ranks_hint_count);
        });
}

mooncakePgResult_t mooncakePgSendCpu(const void* send_buffer, size_t count,
                                     mooncakePgDataType_t data_type, int peer,
                                     mooncakePgComm_t comm,
                                     int32_t* failed_ranks_hint,
                                     size_t failed_ranks_hint_count,
                                     mooncakePgCompletion_t* completion) {
    return invokeCommOpWithCompletion(
        comm, completion, [&]() -> PGResult<std::unique_ptr<WorkCompletion>> {
            DataType converted_data_type;
            PG_TRY(convertDataType(data_type, converted_data_type));
            return comm->impl->sendCpu(send_buffer, count, converted_data_type,
                                       peer, failed_ranks_hint,
                                       failed_ranks_hint_count);
        });
}

mooncakePgResult_t mooncakePgRecvCpu(void* recv_buffer, size_t count,
                                     mooncakePgDataType_t data_type, int peer,
                                     mooncakePgComm_t comm,
                                     int32_t* failed_ranks_hint,
                                     size_t failed_ranks_hint_count,
                                     mooncakePgCompletion_t* completion) {
    return invokeCommOpWithCompletion(
        comm, completion, [&]() -> PGResult<std::unique_ptr<WorkCompletion>> {
            DataType converted_data_type;
            PG_TRY(convertDataType(data_type, converted_data_type));
            return comm->impl->recvCpu(recv_buffer, count, converted_data_type,
                                       peer, failed_ranks_hint,
                                       failed_ranks_hint_count);
        });
}

mooncakePgResult_t mooncakePgCompletionIsCompleted(
    mooncakePgCompletion_t completion, int* completed) {
    return asCApiResult([&]() -> PGResult<void> {
        PG_VALIDATE_ARG(completion && completion->impl, "invalid completion");
        PG_VALIDATE_ARG(completed, "completed output is null");
        *completed = completion->impl->isCompleted() ? 1 : 0;
        return {};
    });
}

mooncakePgResult_t mooncakePgCompletionWait(mooncakePgCompletion_t completion,
                                            int64_t timeout_us) {
    return asCApiResult([&]() -> PGResult<void> {
        PG_VALIDATE_ARG(completion && completion->impl, "invalid completion");
        if (!completion->impl->wait(std::chrono::microseconds(timeout_us))) {
            return makePGError(PGErrorCode::Timeout,
                               "completion wait timed out");
        }
        return {};
    });
}

mooncakePgResult_t mooncakePgCompletionDestroy(
    mooncakePgCompletion_t completion) {
    return asCApiResult([&]() -> PGResult<void> {
        delete completion;
        return {};
    });
}

mooncakePgResult_t mooncakePgCommGetActiveRanks(mooncakePgComm_t comm,
                                                int32_t* active_ranks,
                                                size_t rank_count) {
    return asCApiResult([&]() -> PGResult<void> {
        PG_VALIDATE_ARG(comm && comm->impl, "invalid communicator");
        const auto ranks = comm->impl->getActiveRanks();
        PG_VALIDATE_ARG(rank_count >= ranks.size(),
                        "active-ranks output is too small");
        PG_VALIDATE_ARG(ranks.empty() || active_ranks,
                        "active-ranks output is null");
        if (!ranks.empty()) {
            std::copy(ranks.begin(), ranks.end(), active_ranks);
        }
        return {};
    });
}

mooncakePgResult_t mooncakePgCommGetPeerState(mooncakePgComm_t comm,
                                              const int32_t* ranks,
                                              size_t rank_count,
                                              int32_t* peer_states) {
    return asCApiResult([&]() -> PGResult<void> {
        PG_VALIDATE_ARG(comm && comm->impl, "invalid communicator");
        PG_VALIDATE_ARG(rank_count == 0 || peer_states,
                        "peer-states output is null");
        std::vector<int> parsed_ranks;
        PG_TRY(parseRanks(ranks, rank_count, parsed_ranks));
        auto result = comm->impl->getPeerState(parsed_ranks);
        PG_TRY(result);
        const auto& states = result.value();
        for (size_t index = 0; index < states.size(); ++index) {
            peer_states[index] = states[index] ? 1 : 0;
        }
        return {};
    });
}

mooncakePgResult_t mooncakePgCommActivateRanks(
    mooncakePgComm_t comm, const int32_t* ranks, size_t rank_count,
    mooncakePgProposalResponse_t* response) {
    return asCApiResult([&]() -> PGResult<void> {
        PG_VALIDATE_ARG(comm && comm->impl, "invalid communicator");
        PG_VALIDATE_ARG(response, "proposal response is null");
        std::vector<int> parsed_ranks;
        PG_TRY(parseRanks(ranks, rank_count, parsed_ranks));
        auto result = comm->impl->activateRanks(parsed_ranks);
        PG_TRY(result);
        *response = convertProposalResponse(result.value());
        return {};
    });
}

mooncakePgResult_t mooncakePgCommDeactivateRanks(
    mooncakePgComm_t comm, const int32_t* ranks, size_t rank_count,
    mooncakePgProposalResponse_t* response) {
    return asCApiResult([&]() -> PGResult<void> {
        PG_VALIDATE_ARG(comm && comm->impl, "invalid communicator");
        PG_VALIDATE_ARG(response, "proposal response is null");
        std::vector<int> parsed_ranks;
        PG_TRY(parseRanks(ranks, rank_count, parsed_ranks));
        auto result = comm->impl->deactivateRanks(parsed_ranks);
        PG_TRY(result);
        *response = convertProposalResponse(result.value());
        return {};
    });
}

mooncakePgResult_t mooncakePgCommJoin(mooncakePgComm_t comm) {
    return asCApiResult([&]() -> PGResult<void> {
        PG_VALIDATE_ARG(comm && comm->impl, "invalid communicator");
        return comm->impl->joinGroup();
    });
}

mooncakePgResult_t mooncakePgCommSyncAfterFailure(
    mooncakePgComm_t comm, mooncakePgSyncAfterFailureResponse_t* response) {
    return asCApiResult([&]() -> PGResult<void> {
        PG_VALIDATE_ARG(comm && comm->impl, "invalid communicator");
        PG_VALIDATE_ARG(response, "sync response is null");
        auto result = comm->impl->syncAfterFailure();
        PG_TRY(result);
        const auto& sync_response = result.value();
        std::memset(response, 0, sizeof(*response));
        response->status = static_cast<mooncakePgSyncAfterFailureStatus_t>(
            sync_response.status);
        std::snprintf(response->rejectReason, sizeof(response->rejectReason),
                      "%s", sync_response.reject_reason.c_str());
        return {};
    });
}

mooncakePgResult_t mooncakePgCommGetEpoch(mooncakePgComm_t comm,
                                          uint64_t* epoch) {
    return asCApiResult([&]() -> PGResult<void> {
        PG_VALIDATE_ARG(comm && comm->impl, "invalid communicator");
        PG_VALIDATE_ARG(epoch, "epoch output is null");
        *epoch = comm->impl->getCurrentEpoch();
        return {};
    });
}

mooncakePgResult_t mooncakePgCommGetNumSyncedRanks(mooncakePgComm_t comm,
                                                   int* num_synced_ranks) {
    return asCApiResult([&]() -> PGResult<void> {
        PG_VALIDATE_ARG(comm && comm->impl, "invalid communicator");
        PG_VALIDATE_ARG(num_synced_ranks, "num-synced-ranks output is null");
        *num_synced_ranks = comm->impl->getNumSyncedRanks();
        return {};
    });
}

mooncakePgResult_t mooncakePgCommGetPreferredHca(mooncakePgComm_t comm,
                                                 const char* location,
                                                 char* hca_buf,
                                                 size_t hca_buf_size) {
    return asCApiResult([&]() -> PGResult<void> {
        PG_VALIDATE_ARG(comm && comm->impl, "invalid communicator");
        PG_VALIDATE_ARG(location && hca_buf && hca_buf_size != 0,
                        "invalid preferred-HCA output arguments");
        auto result = comm->impl->getPreferredHca(location);
        PG_TRY(result);
        const auto& value = result.value();
        PG_VALIDATE_ARG(value.size() < hca_buf_size,
                        "preferred-HCA output is too small");
        std::memcpy(hca_buf, value.c_str(), value.size() + 1);
        return {};
    });
}
