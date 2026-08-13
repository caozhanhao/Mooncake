#include "device_comm/device_collective/device_collective.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <exception>
#include <limits>
#include <mutex>
#include <new>
#include <string>
#include <type_traits>
#include <utility>

#include <glog/logging.h>

#include "device_comm/device_collective/device_collective_recovery.h"
#include "gpu_runtime.h"
#include "device_comm/device_collective/strong_stream.h"

namespace mooncake {
namespace {

constexpr uint64_t kGroupResourceAlignment = 256;
constexpr size_t kTargetChannelDataSize = 4ull << 20;
static_assert(kMaxDeviceCollectiveChannels == 4);

uint32_t chooseChannelCount(size_t size) {
    // Add parallel CTAs only when each receives a useful amount of tensor
    // work. The service then divides its full 16 MiB buffer pair by the same
    // 1/2/4 count, yielding 16/8/4 MiB transfer units respectively.
    if (size <= kTargetChannelDataSize) return 1;
    if (size <= 2 * kTargetChannelDataSize) return 2;
    return kMaxDeviceCollectiveChannels;
}

// cudaMemcpyAsync must not borrow a stack/pageable source while a failed
// collective kernel is parked. This small pinned image is reused only while
// the runtime mutex is held, and publishPlan() drains control_stream before it
// returns.
struct PlanUpdateStaging {
    DeviceAllReducePlanImage plan;
    std::array<uint64_t, kMaxDeviceCollectiveChannels> initial_step_sequences =
        {1, 1, 1, 1};
};

PGResult<uint64_t> timeoutTicks(int device_index, size_t timeout_us) {
    if (timeout_us == 0) return uint64_t{0};
    PG_TRY(auto device_guard, GpuDeviceGuard::create(device_index));
    int clock_rate_khz_value = 0;
    PG_TRY_CUDA(cudaDeviceGetAttribute(&clock_rate_khz_value,
                                       cudaDevAttrClockRate, device_index));
    const uint64_t clock_rate_khz = static_cast<uint64_t>(clock_rate_khz_value);
    if (clock_rate_khz == 0) return uint64_t{0};
    if (timeout_us > std::numeric_limits<uint64_t>::max() / clock_rate_khz) {
        return uint64_t{std::numeric_limits<uint64_t>::max()};
    }
    return uint64_t{std::max<uint64_t>(1, timeout_us * clock_rate_khz / 1000)};
}

PGResult<void> validateAllReduceDataType(DataType datatype) {
    switch (datatype) {
        case DataType::Float16:
        case DataType::Bfloat16:
        case DataType::Float32:
            return {};
        default:
            return makePGError(
                PGErrorCode::NotSupported,
                "device AllReduce supports fp16, bf16, and fp32");
    }
}

bool rangesOverlap(const void* left, const void* right, size_t size) {
    if (size == 0 || left == right) return false;
    const auto left_begin = reinterpret_cast<uintptr_t>(left);
    const auto right_begin = reinterpret_cast<uintptr_t>(right);
    if (size > std::numeric_limits<uintptr_t>::max() - left_begin ||
        size > std::numeric_limits<uintptr_t>::max() - right_begin) {
        return true;
    }
    return left_begin < right_begin + size && right_begin < left_begin + size;
}

bool validDeviceCollectiveControlRange(const DeviceCollectiveEndpoint& endpoint,
                                       uint64_t expected_control_size) {
    return endpoint.control_offset >= kDeviceCollectiveWorkspaceSize &&
           endpoint.control_size == expected_control_size &&
           endpoint.control_offset <=
               std::numeric_limits<uint64_t>::max() - endpoint.control_size;
}

void assertValidPlanImage(const DeviceAllReducePlanImage& plan,
                          uint32_t max_group_size) {
    PG_ASSERT(plan.status == DeviceCollectivePlanStatus::Ready ||
                  plan.status == DeviceCollectivePlanStatus::Unavailable,
              "device collective Plan has an invalid status");

    if (plan.status == DeviceCollectivePlanStatus::Unavailable) return;

    PG_ASSERT(
        plan.self_rank >= 0 &&
            static_cast<uint32_t>(plan.self_rank) < max_group_size,
        "device collective Plan has an invalid local group rank");

    PG_ASSERT(
        plan.participant_count != 0 && plan.participant_count <= max_group_size,
        "device collective Plan has an invalid participant count");
    PG_ASSERT(
        plan.self_active_index >= 0 &&
            static_cast<uint32_t>(plan.self_active_index) <
                plan.participant_count,
        "device collective Plan has an invalid local active index");
    PG_ASSERT(
        plan.predecessor_rank >= 0 &&
            static_cast<uint32_t>(plan.predecessor_rank) < max_group_size,
        "device collective Plan has an invalid predecessor rank");
    PG_ASSERT(
        plan.successor_rank >= 0 &&
            static_cast<uint32_t>(plan.successor_rank) < max_group_size,
        "device collective Plan has an invalid successor rank");
}

#ifdef USE_CUDA
PGResult<void> validateDeviceBuffer(const void* buffer, size_t size,
                                    int device_index, const char* name) {
    if (size == 0) return {};
    cudaPointerAttributes attributes{};
    const auto result = cudaPointerGetAttributes(&attributes, buffer);
    if (result != cudaSuccess) {
        // Pointer queries set CUDA's thread-local last error. Consume it so a
        // validation failure cannot poison the later kernel launch check.
        cudaGetLastError();
        return makePGError(PGErrorCode::InvalidArgument,
                           std::string(name) + " is not CUDA-accessible: " +
                               cudaGetErrorString(result));
    }
    PG_VALIDATE_ARG(attributes.type == cudaMemoryTypeDevice ||
                        attributes.type == cudaMemoryTypeManaged,
                    std::string(name) + " must be device or managed memory");
    PG_VALIDATE_ARG(attributes.type == cudaMemoryTypeManaged ||
                        attributes.device == device_index,
                    std::string(name) + " belongs to a different CUDA device");
    return {};
}
#endif

struct GraphUsePayload {
    std::atomic<size_t>* live_uses = nullptr;
};

void releaseGraphUse(void* opaque) {
    auto* payload = static_cast<GraphUsePayload*>(opaque);
    payload->live_uses->fetch_sub(1, std::memory_order_acq_rel);
    delete payload;
}

class GpuStreamDrainGuard {
   public:
    explicit GpuStreamDrainGuard(GpuStream& stream) : stream_(stream) {}

    ~GpuStreamDrainGuard() noexcept {
        if (!active_) return;
        try {
            auto result = stream_.synchronize();
            if (!result.has_value()) {
                LOG(ERROR)
                    << "Failed to drain device collective control stream: "
                    << result.error().message;
            }
        } catch (const std::exception& error) {
            LOG(ERROR) << "Failed to drain device collective control stream: "
                       << error.what();
        } catch (...) {
            LOG(ERROR) << "Failed to drain device collective control stream";
        }
    }

    PGResult<void> finish() {
        PG_VALIDATE_STATE(active_, "control stream was already drained");
        auto result = stream_.synchronize();
        active_ = false;
        return result;
    }

   private:
    GpuStream& stream_;
    bool active_ = true;
};

}  // namespace

DeviceCollectiveRuntime::ControlSliceLayout
DeviceCollectiveRuntime::ControlSliceLayout::make(uint32_t max_group_size) {
    ControlSliceLayout layout;
    uint64_t cursor = 0;

    auto reserve = [&](uint64_t size, uint64_t alignment) -> uint64_t {
        PG_ASSERT(alignment != 0,
                  "device collective control-slice alignment is zero");
        const uint64_t remainder = cursor % alignment;
        const uint64_t padding = remainder == 0 ? 0 : alignment - remainder;
        PG_ASSERT(
            padding <= std::numeric_limits<uint64_t>::max() - cursor,
            "device collective control-slice layout alignment overflows");
        cursor += padding;
        const uint64_t result = cursor;
        PG_ASSERT(size <= std::numeric_limits<uint64_t>::max() - cursor,
                  "device collective control-slice layout size overflows");
        cursor += size;
        return result;
    };

    layout.peers_offset =
        reserve(static_cast<uint64_t>(max_group_size) *
                    sizeof(DeviceCollectivePeerBinding),
                alignof(DeviceCollectivePeerBinding));
    layout.plan_offset = reserve(sizeof(DeviceAllReducePlanImage),
                                 alignof(DeviceAllReducePlanImage));
    layout.next_step_sequences_offset =
        reserve(kMaxDeviceCollectiveChannels * sizeof(uint64_t),
                alignof(uint64_t));
    layout.invocation_offset = reserve(sizeof(DeviceCollectiveInvocationState),
                                       alignof(DeviceCollectiveInvocationState));
    layout.signal_slots_offset =
        reserve(static_cast<uint64_t>(kMaxDeviceCollectiveChannels) *
                    max_group_size * sizeof(uint64_t),
                alignof(uint64_t));
    layout.consumed_ack_slots_offset =
        reserve(static_cast<uint64_t>(kMaxDeviceCollectiveChannels) *
                    max_group_size * sizeof(uint64_t),
                alignof(uint64_t));
    layout.size = reserve(0, kGroupResourceAlignment);

    layout.max_group_size = max_group_size;
    return layout;
}

DeviceCollectiveKernelResources
DeviceCollectiveRuntime::ControlSliceLayout::bind(
    void* control_addr, uint64_t control_region_offset,
    const DeviceTransferHandle* transfer_handle,
    DeviceCollectiveTransferBuffer send_buffer,
    DeviceCollectiveTransferBuffer recv_buffer,
    uint64_t timeout_ticks) const noexcept {
    // Resolve the byte layout exactly once. From this point onward host code
    // and kernels use typed pointers and table descriptors rather than
    // reconstructing the control-slice layout.
    auto* const control_base = static_cast<char*>(control_addr);
    return DeviceCollectiveKernelResources{
        .transfer_handle = transfer_handle,
        .peers =
            DeviceCollectivePeerTable{
                .entries = reinterpret_cast<DeviceCollectivePeerBinding*>(
                    control_base + peers_offset),
            },
        .send_buffer = send_buffer,
        .recv_buffer = recv_buffer,
        .timeout_ticks = timeout_ticks,
        .all_reduce_plan = reinterpret_cast<DeviceAllReducePlanImage*>(
            control_base + plan_offset),
        .next_step_sequences = reinterpret_cast<uint64_t*>(
            control_base + next_step_sequences_offset),
        .invocation = reinterpret_cast<DeviceCollectiveInvocationState*>(
            control_base + invocation_offset),
        .signal_slots =
            DeviceCollectiveSignalTable{
                .slots = reinterpret_cast<uint64_t*>(control_base +
                                                     signal_slots_offset),
                .local_region_offset =
                    control_region_offset + signal_slots_offset,
                .control_offset = signal_slots_offset,
                .max_group_size = max_group_size,
            },
        .consumed_ack_slots =
            DeviceCollectiveSignalTable{
                .slots = reinterpret_cast<uint64_t*>(control_base +
                                                     consumed_ack_slots_offset),
                .local_region_offset =
                    control_region_offset + consumed_ack_slots_offset,
                .control_offset = consumed_ack_slots_offset,
                .max_group_size = max_group_size,
            },
    };
}

struct alignas(64) DeviceCollectiveRuntime::HostControl {
    // The host and device directly share this member through the mapped
    // allocation.
    DeviceCollectiveRecoveryMailbox recovery_mailbox;

    // The remaining members are pinned host sources for control-stream copies.
    PlanUpdateStaging plan_staging;
    std::array<DeviceCollectivePeerBinding, kMaxNumRanks> peer_binding_staging;
};

DeviceCollectiveRuntime::DeviceCollectiveRuntime(
    DeviceTransferService& transfer_service, int device_index,
    GlobalRank self_rank, uint64_t timeout_ticks, ControlSliceLayout layout,
    DeviceArenaSlice control_slice,
    DeviceCollectiveKernelResources kernel_resources,
    StrongStream& strong_stream, DeviceCollectiveEndpoint endpoint,
    GpuStream control_stream, GpuEvent handoff_event)
    : transfer_service_(transfer_service),
      device_index_(device_index),
      self_rank_(self_rank),
      timeout_ticks_(timeout_ticks),
      layout_(std::move(layout)),
      control_slice_(std::move(control_slice)),
      strong_stream_(strong_stream),
      endpoint_(std::move(endpoint)),
      kernel_resources_(kernel_resources),
      control_stream_(std::move(control_stream)),
      handoff_event_(std::move(handoff_event)) {}

PGResult<void> DeviceCollectiveRuntime::initializeHostControl() {
    static_assert(std::is_standard_layout_v<HostControl>);
    PG_TRY(auto device_guard, GpuDeviceGuard::create(device_index_));
    PG_TRY_CUDA(cudaHostAlloc(reinterpret_cast<void**>(&host_control_),
                              sizeof(HostControl),
                              cudaHostAllocMapped | cudaHostAllocPortable));
    std::construct_at(host_control_);

    void* device_control = nullptr;
    PG_TRY_CUDA(cudaHostGetDevicePointer(&device_control, host_control_, 0));
    kernel_resources_.recovery =
        reinterpret_cast<DeviceCollectiveRecoveryMailbox*>(
            static_cast<char*>(device_control) +
            offsetof(HostControl, recovery_mailbox));
    return {};
}

void DeviceCollectiveRuntime::releaseHostControl() noexcept {
    if (!host_control_) return;
    std::destroy_at(host_control_);
    const auto result = cudaFreeHost(host_control_);
    if (result != cudaSuccess) {
        LOG(ERROR) << "Failed to free device collective host control: "
                   << cudaGetErrorString(result);
    }
    host_control_ = nullptr;
    kernel_resources_.recovery = nullptr;
}

PGResult<void> DeviceCollectiveRuntime::publishPlan(
    DeviceAllReducePlanImage plan, bool reset_protocol_state) {
    assertValidPlanImage(plan, layout_.max_group_size);
    PG_TRY(auto device_guard, GpuDeviceGuard::create(device_index_));
    auto& update = host_control_->plan_staging;
    update.plan = plan;
    GpuStreamDrainGuard drain(control_stream_);
    if (reset_protocol_state) {
        const size_t slot_count =
            static_cast<size_t>(kMaxDeviceCollectiveChannels) *
            layout_.max_group_size;
        PG_TRY_CUDA(cudaMemsetAsync(kernel_resources_.signal_slots.slots, 0,
                                    slot_count * sizeof(uint64_t),
                                    control_stream_.get()));
        PG_TRY_CUDA(cudaMemsetAsync(kernel_resources_.consumed_ack_slots.slots,
                                    0, slot_count * sizeof(uint64_t),
                                    control_stream_.get()));
        PG_TRY_CUDA(cudaMemcpyAsync(
            kernel_resources_.next_step_sequences,
            update.initial_step_sequences.data(),
            update.initial_step_sequences.size() * sizeof(uint64_t),
            cudaMemcpyHostToDevice, control_stream_.get()));
    }
    PG_TRY_CUDA(cudaMemcpyAsync(kernel_resources_.all_reduce_plan,
                                &update.plan, sizeof(update.plan),
                                cudaMemcpyHostToDevice, control_stream_.get()));
    PG_TRY(drain.finish());
    host_plan_ = plan;
    return {};
}

PGResult<void> DeviceCollectiveRuntime::attachGraphUse(
    const GpuCaptureInfo& capture) {
    if (!capture.active) return {};

    auto* payload = new (std::nothrow) GraphUsePayload{&live_graph_uses_};
    if (!payload) {
        return makePGError(PGErrorCode::ResourceBusy,
                           "failed to allocate CUDA Graph use token");
    }
    live_graph_uses_.fetch_add(1, std::memory_order_acq_rel);
    auto object_result =
        GpuGraphUserObject::create(device_index_, payload, releaseGraphUse);
    if (!object_result.has_value()) {
        live_graph_uses_.fetch_sub(1, std::memory_order_acq_rel);
        delete payload;
        return makePGError(std::move(object_result).error());
    }
    auto object = std::move(object_result).value();
    return object.moveTo(capture);
}

PGResult<void> DeviceCollectiveRuntime::recoverFailure(uint64_t generation) {
    auto& recovery = host_control_->recovery_mailbox;

    // The peer signal may time out while an outgoing HostProxy command is
    // still in flight. The parked kernel owns the shared buffers, so this
    // dedicated worker can wait here without introducing a retry state.
    PG_TRY(transfer_service_.waitUntilIdle());

    const auto failed_rank = recovery.failed_rank;
    PG_ASSERT(
        failed_rank >= 0 &&
            static_cast<uint32_t>(failed_rank) < layout_.max_group_size,
        "device collective transfer failure has an invalid peer rank");

    if (recovery.failed_hint_address != 0) {
        auto* hint = reinterpret_cast<int32_t*>(recovery.failed_hint_address);
        hint[failed_rank] = 1;
    }

    auto recovered = recovery_handler_(failed_rank);

    if (!recovered.has_value()) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto unavailable_plan = host_plan_;
        unavailable_plan.status = DeviceCollectivePlanStatus::Unavailable;
        auto unavailable = publishPlan(unavailable_plan, false);
        if (!unavailable.has_value()) {
            return makePGError(
                PGErrorCode::SystemError,
                "device collective recovery failed and an unavailable Plan "
                "could not be published: " +
                    recovered.error().message + "; " +
                    unavailable.error().message);
        }
        LOG(ERROR) << "device collective recovery failed; Plan was made "
                      "unavailable: "
                   << recovered.error().message;
    }

    std::atomic_ref(recovery.ready_generation)
        .store(generation, std::memory_order_release);
    return {};
}

PGResult<std::unique_ptr<DeviceCollectiveRuntime>>
DeviceCollectiveRuntime::create(DeviceTransferService& transfer_service,
                                DeviceArena& arena,
                                const DeviceArenaSlice& workspace,
                                StrongStream& strong_stream, int device_index,
                                GlobalRank self_rank, uint32_t max_group_size,
                                size_t collective_timeout_us) {
    PG_VALIDATE_ARG(device_index >= 0,
                    "device collective requires a CUDA device");
    PG_VALIDATE_ARG(max_group_size > 0 && max_group_size <= kMaxNumRanks,
                    "device collective group capacity is out of range");
    PG_VALIDATE_ARG(self_rank >= 0, "device collective global rank is invalid");
    PG_VALIDATE_ARG(arena.deviceIndex() == device_index,
                    "device collective arena belongs to another device");
    PG_VALIDATE_ARG(
        workspace.valid() && workspace.size() == kDeviceCollectiveWorkspaceSize,
        "device collective workspace has the wrong size");

    PG_TRY(auto timeout_ticks,
           timeoutTicks(device_index, collective_timeout_us));
    auto layout = ControlSliceLayout::make(max_group_size);
    const auto* transfer_handle = transfer_service.deviceHandle();
    PG_TRY(auto control_slice,
           arena.allocate(layout.size, kGroupResourceAlignment));
    {
        PG_TRY(auto device_guard, GpuDeviceGuard::create(device_index));
        PG_TRY_CUDA(cudaMemset(control_slice.addr(), 0, control_slice.size()));
    }
    PG_TRY(auto control_stream, GpuStream::createNonBlocking(device_index));
    PG_TRY(auto handoff_event, GpuEvent::create(device_index));
    const DeviceCollectiveTransferBuffer send_buffer{
        .addr = workspace.addr(),
        .region_offset = workspace.offset(),
        .size = kDeviceCollectiveTransferBufferSize,
    };
    const DeviceCollectiveTransferBuffer recv_buffer{
        .addr = static_cast<char*>(workspace.addr()) +
                kDeviceCollectiveTransferBufferSize,
        .region_offset =
            workspace.offset() + kDeviceCollectiveTransferBufferSize,
        .size = kDeviceCollectiveTransferBufferSize,
    };
    const auto kernel_resources =
        layout.bind(control_slice.addr(), control_slice.offset(),
                    transfer_handle, send_buffer, recv_buffer, timeout_ticks);
    DeviceCollectiveEndpoint endpoint{
        .control_offset = control_slice.offset(),
        .control_size = control_slice.size(),
    };
    auto runtime =
        std::unique_ptr<DeviceCollectiveRuntime>(new DeviceCollectiveRuntime(
            transfer_service, device_index, self_rank, timeout_ticks,
            std::move(layout), std::move(control_slice), kernel_resources,
            strong_stream, std::move(endpoint), std::move(control_stream),
            std::move(handoff_event)));
    PG_TRY(runtime->initializeHostControl());
    PG_TRY(runtime->publishPlan(DeviceAllReducePlanImage{}, true));
    return runtime;
}

DeviceCollectiveRuntime::~DeviceCollectiveRuntime() noexcept {
    auto result = shutdown(std::chrono::seconds(5));
    if (!result.has_value()) {
        // The public destroy path keeps the complete communicator alive when
        // shutdown is retryable. Reaching this destructor with live Graph
        // references or unfinished device work is therefore an internal
        // lifetime violation; freeing only part of the runtime would leave
        // captured kernels with dangling device addresses.
        LOG(FATAL) << "DeviceCollectiveRuntime destroyed before shutdown "
                      "could complete: "
                   << result.error().message;
    }
    releaseHostControl();
}

const DeviceCollectiveEndpoint& DeviceCollectiveRuntime::localEndpoint() const {
    return endpoint_;
}

PGResult<void> DeviceCollectiveRuntime::useLocalOnly(InGroupRank self_rank) {
    std::lock_guard<std::mutex> lock(mutex_);
    PG_VALIDATE_STATE(!shutdown_requested_ && !shutdown_complete_,
                      "device collective runtime is shutting down");

    PG_ASSERT(self_rank >= 0 && static_cast<uint32_t>(self_rank) <
                                        layout_.max_group_size,
              "local-only group rank is out of range");

    // Gate host submission before changing any device-visible state. If either
    // publication below fails, enqueueAllReduce() must not reuse the previous
    // Ready Plan with partially updated control data.
    auto unavailable = host_plan_;
    unavailable.status = DeviceCollectivePlanStatus::Unavailable;
    host_plan_.status = DeviceCollectivePlanStatus::Unavailable;
    PG_TRY(publishPlan(unavailable, false));

    return publishPlan(
        DeviceAllReducePlanImage{
            .status = DeviceCollectivePlanStatus::Ready,
            .self_rank = self_rank,
            .self_active_index = 0,
            .participant_count = 1,
            .predecessor_rank = self_rank,
            .successor_rank = self_rank,
        },
        true);
}

PGResult<void> DeviceCollectiveRuntime::materializeGroupView(
    const GroupView& view) {
    std::lock_guard<std::mutex> lock(mutex_);
    PG_VALIDATE_STATE(!shutdown_requested_ && !shutdown_complete_,
                      "device collective runtime is shutting down");

    PG_ASSERT(view.rank_order.size() <= layout_.max_group_size,
              "GroupView exceeds device collective capacity");

    // Keep the old Plan unavailable throughout the update. Peer bindings and
    // protocol state are published separately, so a failed update must never
    // leave the old Ready Plan paired with a partial new view.
    auto unavailable = host_plan_;
    unavailable.status = DeviceCollectivePlanStatus::Unavailable;
    host_plan_.status = DeviceCollectivePlanStatus::Unavailable;
    PG_TRY(publishPlan(unavailable, false));

    std::fill_n(host_control_->peer_binding_staging.data(),
                layout_.max_group_size, DeviceCollectivePeerBinding{});

    // The caller has already installed the Agent's rank-scoped endpoint
    // snapshot in the transfer service. GroupView contributes only the mapping
    // from group rank to service peer index and the peer runtime's control
    // range.
    for (size_t peer_rank = 0; peer_rank < view.rank_order.size();
         ++peer_rank) {
        const auto global_rank = view.rank_order[peer_rank];
        PG_ASSERT(global_rank >= 0 && static_cast<size_t>(global_rank) <
                                          view.members.size(),
                  "GroupView contains an invalid global rank");
        const auto& member = view.members[global_rank];
        const bool valid_binding =
            member.endpoint && member.endpoint->device_collective &&
            validDeviceCollectiveControlRange(
                *member.endpoint->device_collective, layout_.size);
        PG_ASSERT(!member.isActive() || valid_binding,
                  "active device collective peer has no valid control range");
        if (valid_binding) {
            const auto& endpoint = *member.endpoint->device_collective;
            host_control_->peer_binding_staging[peer_rank] =
                DeviceCollectivePeerBinding{
                    .peer_idx = static_cast<uint32_t>(global_rank),
                    .remote_control_offset = endpoint.control_offset,
                };
        }
    }

    {
        PG_TRY(auto device_guard, GpuDeviceGuard::create(device_index_));
        GpuStreamDrainGuard drain(control_stream_);
        PG_TRY_CUDA(cudaMemcpyAsync(
            kernel_resources_.peers.entries,
            host_control_->peer_binding_staging.data(),
            static_cast<size_t>(layout_.max_group_size) *
                sizeof(DeviceCollectivePeerBinding),
            cudaMemcpyHostToDevice, control_stream_.get()));
        PG_TRY(drain.finish());
    }

    InGroupRank self_rank = kInvalidInGroupRank;
    std::vector<InGroupRank> active_ranks;
    active_ranks.reserve(view.rank_order.size());
    for (size_t group_rank = 0; group_rank < view.rank_order.size();
         ++group_rank) {
        const auto global_rank = view.rank_order[group_rank];
        if (global_rank == self_rank_) {
            self_rank = static_cast<InGroupRank>(group_rank);
        }
        if (global_rank >= 0 &&
            static_cast<size_t>(global_rank) < view.members.size() &&
            view.members[global_rank].isActive()) {
            active_ranks.push_back(static_cast<InGroupRank>(group_rank));
        }
    }
    PG_ASSERT(self_rank >= 0, "local rank is absent from GroupView");

    const auto self =
        std::find(active_ranks.begin(), active_ranks.end(), self_rank);
    if (self == active_ranks.end()) {
        return publishPlan(
            DeviceAllReducePlanImage{
                .status = DeviceCollectivePlanStatus::Unavailable,
                .self_rank = self_rank,
            },
            true);
    }

    const auto active_index =
        static_cast<size_t>(std::distance(active_ranks.begin(), self));
    const auto participant_count = active_ranks.size();
    const auto predecessor =
        active_ranks[(active_index + participant_count - 1) %
                     participant_count];
    const auto successor = active_ranks[(active_index + 1) % participant_count];
    return publishPlan(
        DeviceAllReducePlanImage{
            .status = DeviceCollectivePlanStatus::Ready,
            .self_rank = self_rank,
            .self_active_index = static_cast<int32_t>(active_index),
            .participant_count = static_cast<uint32_t>(participant_count),
            .predecessor_rank = predecessor,
            .successor_rank = successor,
        },
        true);
}

PGResult<void> DeviceCollectiveRuntime::enableRecovery(
    DeviceCollectiveRecoveryWorker& worker, RecoveryHandler handler) {
    PG_VALIDATE_ARG(static_cast<bool>(handler),
                    "device collective recovery handler is empty");
    std::lock_guard<std::mutex> lock(mutex_);
    PG_VALIDATE_STATE(!shutdown_requested_ && !shutdown_complete_,
                      "device collective runtime is shutting down");
    PG_VALIDATE_STATE(!recovery_worker_,
                      "device collective recovery is already enabled");
    recovery_handler_ = std::move(handler);
    auto added = worker.addMailbox(
        &host_control_->recovery_mailbox,
        [this](uint64_t generation) { return recoverFailure(generation); });
    if (!added.has_value()) {
        recovery_handler_ = {};
        return makePGError(std::move(added).error());
    }
    recovery_worker_ = &worker;
    return {};
}

PGResult<void> DeviceCollectiveRuntime::enqueueAllReduce(
    const void* send_buffer, void* recv_buffer, size_t count, DataType datatype,
    ReduceOp op, cudaStream_t user_stream_handle,
    int32_t* failed_ranks_hint) {
#ifndef USE_CUDA
    (void)send_buffer;
    (void)recv_buffer;
    (void)count;
    (void)datatype;
    (void)op;
    (void)user_stream_handle;
    (void)failed_ranks_hint;
    return makePGError(PGErrorCode::NotSupported,
                       "device collectives initially require CUDA");
#else
    PG_VALIDATE_ARG(send_buffer || count == 0, "send buffer is null");
    PG_VALIDATE_ARG(recv_buffer || count == 0, "recv buffer is null");
    PG_VALIDATE_ARG(op == ReduceOp::Sum,
                    "device AllReduce initially supports sum only");
    PG_TRY(validateAllReduceDataType(datatype));

    const size_t element_size = elementSize(datatype);
    PG_VALIDATE_ARG(count <= std::numeric_limits<size_t>::max() / element_size,
                    "device AllReduce byte count overflows size_t");
    const size_t buffer_size = count * element_size;
    PG_TRY(validateDeviceBuffer(send_buffer, buffer_size, device_index_,
                                "send buffer"));
    PG_TRY(validateDeviceBuffer(recv_buffer, buffer_size, device_index_,
                                "recv buffer"));
    PG_VALIDATE_ARG(
        !rangesOverlap(send_buffer, recv_buffer, buffer_size),
        "device AllReduce buffers must be identical or non-overlapping");

    // Concurrent host submission is not part of the API contract. Holding the
    // runtime mutex across this short enqueue scope makes a racing shutdown
    // wait until stream handoff and StrongStream publication are complete.
    std::unique_lock<std::mutex> enqueue_lock(mutex_);
    PG_VALIDATE_STATE(!shutdown_requested_ && !shutdown_complete_,
                      "device collective runtime is shutting down");
    PG_VALIDATE_STATE(host_plan_.status == DeviceCollectivePlanStatus::Ready,
                      "device collective Plan is not ready");

    auto user_stream = GpuStream::borrow(user_stream_handle, device_index_);
    PG_TRY(auto capture, user_stream.captureInfo());
    PG_TRY(attachGraphUse(capture));
    PG_TRY(auto order_lease, strong_stream_.acquire(capture));
    const auto& order_stream = order_lease.stream();

    cudaError_t launch_error = cudaSuccess;
    auto submitted = [&]() -> PGResult<void> {
        // Keep the kernel on the user-provided stream.
        PG_TRY(handoff_event_.record(order_stream));
        PG_TRY(user_stream.waitEvent(handoff_event_));

        launch_error = launchDeviceAllReduceKernel(
            DeviceAllReduceKernelArgs{
                .send_buffer = send_buffer,
                .recv_buffer = recv_buffer,
                .count = static_cast<uint64_t>(count),
                .datatype = datatype,
                .channel_count = chooseChannelCount(buffer_size),
                // A captured node can outlive the Work object that owns this raw
                // host pointer. V1 deliberately keeps Graph failure reporting in
                // the PG-owned mailbox instead of retaining caller memory.
                .failed_ranks_hint = capture.active ? nullptr
                                                    : failed_ranks_hint,
            },
            kernel_resources_, user_stream.get());

        PG_TRY(handoff_event_.record(user_stream));
        PG_TRY(order_stream.waitEvent(handoff_event_));
        return {};
    }();

    // release() clears StrongStream's acquire state even when publishing its
    // final event fails. Always call it before propagating a submission error.
    auto released = order_lease.release();
    if (!submitted.has_value()) {
        auto error = std::move(submitted).error();
        if (!released.has_value()) {
            error.message += "; StrongStream release also failed: " +
                             released.error().message;
        }
        return makePGError(std::move(error));
    }
    PG_TRY(released);
    PG_TRY_CUDA(launch_error);
    return {};
#endif
}

PGResult<void> DeviceCollectiveRuntime::shutdown(
    std::chrono::milliseconds eager_timeout) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (shutdown_complete_) return {};
        if (live_graph_uses_.load(std::memory_order_acquire) != 0) {
            return makePGError(
                PGErrorCode::ResourceBusy,
                "CUDA Graph/GraphExec still references this communicator");
        }
        PG_VALIDATE_STATE(!shutdown_requested_,
                          "device collective runtime is already shutting down");
        shutdown_requested_ = true;
    }

    auto synchronized = strong_stream_.waitUntilIdle(eager_timeout);
    if (!synchronized.has_value()) {
        std::lock_guard<std::mutex> lock(mutex_);
        shutdown_requested_ = false;
        return makePGError(std::move(synchronized).error());
    }

    DeviceCollectiveRecoveryWorker* recovery_worker = nullptr;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        recovery_worker = std::exchange(recovery_worker_, nullptr);
    }

    // Removal may wait for the worker's current recovery. Do not hold the
    // Runtime mutex, and keep the mailbox and control slice alive until it
    // returns.
    if (recovery_worker) {
        recovery_worker->removeMailbox(&host_control_->recovery_mailbox);
    }

    DeviceArenaSlice control_slice;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        recovery_handler_ = {};
        control_slice = std::move(control_slice_);
        shutdown_complete_ = true;
        shutdown_requested_ = false;
    }
    control_slice.reset();
    return {};
}

}  // namespace mooncake
