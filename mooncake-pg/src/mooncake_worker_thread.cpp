#include <cuda_alike.h>
#include <thread>
#include <mooncake_worker.cuh>
#include <mooncake_backend.h>
#include <glog/logging.h>
#include <transfer_engine.h>
#include "pg_utils.h"
#include "control_plane/te_link_manager.h"

namespace mooncake {

enum WorkerTaskStatus {
    IDLE = 0,
    TRANSFERRED_1 = 1,
    SIGNALED_1 = 2,
    DONE = 3,
};

static constexpr size_t kInvalidTaskId = static_cast<size_t>(-1);

void MooncakeWorker::Start() {
    bool expected = false;
    if (started_.compare_exchange_strong(expected, true)) {
        startWorker();
    }
}

bool MooncakeWorker::drainTasks(const TransferGroupMeta* meta) const {
    BackoffWaiter waiter;
    return waiter.wait_for(
        std::chrono::milliseconds(kDrainTasksTimeoutMs), [this, meta] {
            for (size_t i = 0; i < kNumTasks_; ++i) {
                if (tasks_[i].active && tasks_[i].transferGroupMeta == meta)
                    return false;
            }
            return true;
        });
}

bool MooncakeWorker::waitUntilTasksSubmitted(
    const std::vector<CudaTaskSubmissionToken>& tasks,
    std::chrono::milliseconds timeout) const {
    if (tasks.empty()) {
        return true;
    }

    auto submitted = [this, &tasks] {
        for (const auto& task : tasks) {
            if (task.task_id >= kNumTasks_) {
                LOG(ERROR) << "Invalid task id.";
                return true;
            }
            if (submitted_task_sequence_[task.task_id].load(
                    std::memory_order_acquire) < task.sequence) {
                return false;
            }
        }
        return true;
    };

    BackoffWaiter waiter(
        BackoffWaiterConfig::constantSleep(std::chrono::microseconds(10)));
    if (timeout == kNoTimeout) {
        waiter.wait(submitted);
        return true;
    }
    return waiter.wait_for(timeout, submitted);
}

void MooncakeWorker::startWorker() {
    running_ = true;
    worker_thread_ = std::thread([this] {
        if (cuda_device_index_ >= 0) {
            cudaSetDevice(cuda_device_index_);
        }
        std::atomic<WorkerTaskStatus> task_status[kNumTasks_];
        using clock = std::chrono::high_resolution_clock;
        clock::time_point activeTime[kNumTasks_];
        size_t rankToTaskId[kNumTasks_][kMaxNumRanks];
        while (running_) {
            PAUSE();
            for (size_t i = 0; i < kNumTasks_; ++i) {
                auto& task = tasks_[i];
                if (!task.active) {
                    task_status[i].store(IDLE, std::memory_order_release);
                    continue;
                }

                auto group = (TransferGroupMeta*)task.transferGroupMeta;
                bool skipTransfer =
                    ((c10d::OpType)task.opType == c10d::OpType::BROADCAST &&
                     group->rank != task.broadcastRoot) ||
                    ((c10d::OpType)task.opType == c10d::OpType::SCATTER &&
                     group->rank != task.broadcastRoot) ||
                    (c10d::OpType)task.opType == c10d::OpType::BARRIER;

                // Snapshot rankOrder under RCU for consistent iteration.
                std::shared_ptr<const std::vector<GlobalRank>> rankOrderPtr;
                bool hasView = false;
#if !defined(__MUSA__)
                hasView = group->groupEpoch.load(std::memory_order_acquire) !=
                          kInvalidEpoch;
                if (hasView && group->backend) {
                    rankOrderPtr = group->backend->getRankOrder();
                }
#endif

                if (task_status[i].load(std::memory_order_acquire) == IDLE) {
                    const auto submit_sequence = task.submitSequence;
                    if (skipTransfer) {
                        submitted_task_sequence_[i].store(
                            submit_sequence, std::memory_order_release);
                        task_status[i].store(TRANSFERRED_1,
                                             std::memory_order_release);
                        continue;
                    }
                    for (size_t j = 0; j < kMaxNumRanks; ++j) {
                        rankToTaskId[i][j] = kInvalidTaskId;
                    }
                    std::vector<TransferRequest> entries;

                    if (hasView && group->backend) {
                        // New path: iterate over rankOrder, map
                        // InGroupRank→GlobalRank.
                        auto view_ptr = group->backend->getGroupView();
                        auto& view = *view_ptr;
                        auto& local_ep = group->backend->getLocalEndpointInfo();

                        for (int ig = 0;
                             ig < static_cast<int>((*rankOrderPtr).size());
                             ++ig) {
                            GlobalRank peer = (*rankOrderPtr)[ig];
                            if (peer < 0 ||
                                peer >= static_cast<int>(view.members.size()))
                                continue;
                            if (!view.members[peer].active) continue;
                            if (((c10d::OpType)task.opType ==
                                     c10d::OpType::GATHER ||
                                 (c10d::OpType)task.opType ==
                                     c10d::OpType::REDUCE) &&
                                ig != task.broadcastRoot) {
                                continue;
                            }

                            if (task.attemptedRanksHost) {
                                task.attemptedRanksHost[ig] = 1;
                            }

                            auto handle =
                                group->backend->getProcessContext()
                                    .te_link_manager.resolvePeer(peer);
                            if (!handle) {
                                task.failedRanksHost[ig] = 1;
                                continue;
                            }

                            uint64_t source =
                                local_ep.send_buffer[task.bufferOffset];
                            uint64_t target_offset =
                                view.members[peer]
                                    .endpoint_info
                                    .recv_buffer[task.bufferOffset];

                            switch ((c10d::OpType)task.opType) {
                                case c10d::OpType::ALLTOALL_BASE:
                                case c10d::OpType::ALLTOALL:
                                case c10d::OpType::_REDUCE_SCATTER_BASE:
                                case c10d::OpType::SCATTER:
                                    source += ig * task.tensorSize;
                                    break;
                                default:
                                    break;
                            }

                            switch ((c10d::OpType)task.opType) {
                                case c10d::OpType::BROADCAST:
                                case c10d::OpType::SCATTER:
                                    break;
                                case c10d::OpType::ALLREDUCE:
                                case c10d::OpType::ALLGATHER:
                                case c10d::OpType::_ALLGATHER_BASE:
                                case c10d::OpType::ALLTOALL_BASE:
                                case c10d::OpType::ALLTOALL:
                                case c10d::OpType::_REDUCE_SCATTER_BASE:
                                case c10d::OpType::REDUCE:
                                case c10d::OpType::GATHER:
                                    target_offset +=
                                        group->rank * task.tensorSize;
                                    break;
                                default:
                                    break;
                            }

                            rankToTaskId[i][ig] = entries.size();
                            entries.push_back(TransferRequest{
                                .opcode = TransferRequest::WRITE,
                                .source = (void*)source,
                                .target_id = handle->target_id,
                                .target_offset = target_offset,
                                .length = task.tensorSize,
                            });
                        }
                    } else {
                        // Legacy fallback: iterate over size with activeRanks.
                        for (int j = 0; j < group->size; ++j) {
                            if (!group->activeRanks[j]) continue;
                            if (((c10d::OpType)task.opType ==
                                     c10d::OpType::GATHER ||
                                 (c10d::OpType)task.opType ==
                                     c10d::OpType::REDUCE) &&
                                j != task.broadcastRoot) {
                                continue;
                            }
                            if (task.attemptedRanksHost) {
                                task.attemptedRanksHost[j] = 1;
                            }
                            auto handle = group->backend->getProcessContext()
                                              .te_link_manager.resolvePeer(j);
                            if (!handle) {
                                task.failedRanksHost[j] = 1;
                                continue;
                            }
                            rankToTaskId[i][j] = entries.size();
                            entries.push_back(TransferRequest{
                                .opcode = TransferRequest::WRITE,
                                .source = nullptr,
                                .target_id = handle->target_id,
                                .target_offset = 0,
                                .length = task.tensorSize,
                            });
                        }
                    }

                    task.batchID =
                        group->engine->allocateBatchID(entries.size());
                    group->engine->submitTransfer(task.batchID, entries);
                    submitted_task_sequence_[i].store(
                        submit_sequence, std::memory_order_release);
                    activeTime[i] = clock::now();
                    task_status[i].store(TRANSFERRED_1,
                                         std::memory_order_release);
                } else if (task_status[i].load(std::memory_order_acquire) ==
                           TRANSFERRED_1) {
                    bool batch_done = true;
                    TransferStatus status;

                    if (!skipTransfer) {
                        auto now = clock::now();
                        auto diff = std::chrono::duration_cast<
                            std::chrono::microseconds>(now - activeTime[i]);
                        int iter_count =
                            hasView ? static_cast<int>((*rankOrderPtr).size())
                                    : group->size;
                        for (int j = 0; j < iter_count; ++j) {
                            if (rankToTaskId[i][j] == kInvalidTaskId) continue;
                            group->engine->getTransferStatus(
                                task.batchID, rankToTaskId[i][j], status);
                            if (status.s != TransferStatusEnum::COMPLETED) {
                                bool timed_out =
                                    j != group->rank &&
                                    diff.count() > *group->collectiveTimeoutUs;
                                if (status.s == TransferStatusEnum::FAILED ||
                                    timed_out) {
                                    task.failedRanksHost[j] = 1;
                                    LOG(ERROR) << "Rank " << group->rank
                                               << " transfer to peer " << j
                                               << " failed for op "
                                               << (int)task.opType;
                                } else {
                                    batch_done = false;
                                    break;
                                }
                            }
                        }
                    }

                    if (!batch_done) continue;

                    if (!skipTransfer) {
                        auto s = group->engine->freeBatchID(task.batchID);
                        if (!s.ok()) {
                            LOG(WARNING)
                                << "BatchID leaked due to freeBatchID "
                                   "failure (likely caused by a timeout): "
                                << s.message();
                        }
                    }

                    // Build sync signal transfers.
                    auto* source_ptr = (int32_t*)group->local_endpoint_info
                                           .send_sync[task.bufferOffset];

                    for (size_t j = 0; j < kMaxNumRanks; ++j) {
                        rankToTaskId[i][j] = kInvalidTaskId;
                    }
                    std::vector<TransferRequest> entries;

                    if (hasView && group->backend) {
                        auto view_ptr = group->backend->getGroupView();
                        auto& view = *view_ptr;
                        for (int ig = 0;
                             ig < static_cast<int>((*rankOrderPtr).size());
                             ++ig) {
                            GlobalRank peer = (*rankOrderPtr)[ig];
                            if (peer < 0 ||
                                peer >= static_cast<int>(view.members.size()))
                                continue;
                            if (!view.members[peer].active) continue;

                            auto handle =
                                group->backend->getProcessContext()
                                    .te_link_manager.resolvePeer(peer);
                            if (!handle) continue;

                            *source_ptr = 1;
                            rankToTaskId[i][ig] = entries.size();
                            entries.push_back(TransferRequest{
                                .opcode = TransferRequest::WRITE,
                                .source = (void*)source_ptr,
                                .target_id = handle->target_id,
                                .target_offset =
                                    view.members[peer]
                                        .endpoint_info
                                        .recv_sync[task.bufferOffset] +
                                    group->rank * sizeof(int32_t),
                                .length = sizeof(int32_t),
                            });
                        }
                    } else {
                        // Legacy fallback: use activeRanks.
                        for (int j = 0; j < group->size; ++j) {
                            if (!group->activeRanks[j]) continue;
                            auto handle = group->backend->getProcessContext()
                                              .te_link_manager.resolvePeer(j);
                            if (!handle) continue;
                            *source_ptr = 1;
                            rankToTaskId[i][j] = entries.size();
                            entries.push_back(TransferRequest{
                                .opcode = TransferRequest::WRITE,
                                .source = (void*)source_ptr,
                                .target_id = handle->target_id,
                                .target_offset = 0,
                                .length = sizeof(int32_t),
                            });
                        }
                    }

                    task.batchID =
                        group->engine->allocateBatchID(entries.size());
                    group->engine->submitTransfer(task.batchID, entries);
                    activeTime[i] = clock::now();
                    task_status[i].store(SIGNALED_1, std::memory_order_release);
                } else if (task_status[i].load(std::memory_order_acquire) ==
                           SIGNALED_1) {
                    bool task_done = true;
                    auto* signal_ptr = (int32_t*)group->local_endpoint_info
                                           .recv_sync[task.bufferOffset];

                    auto now = clock::now();
                    auto diff =
                        std::chrono::duration_cast<std::chrono::microseconds>(
                            now - activeTime[i]);

                    int iter_count =
                        hasView ? static_cast<int>((*rankOrderPtr).size())
                                : group->size;

                    TransferStatus status;
                    for (int j = 0; j < iter_count; ++j) {
                        if (rankToTaskId[i][j] == kInvalidTaskId) continue;
                        group->engine->getTransferStatus(
                            task.batchID, rankToTaskId[i][j], status);

                        GlobalRank peer = hasView ? (*rankOrderPtr)[j]
                                                  : static_cast<GlobalRank>(j);
                        bool signal_received = (signal_ptr[peer] == 1);

                        if (!signal_received ||
                            status.s != TransferStatusEnum::COMPLETED) {
                            bool timed_out =
                                j != group->rank &&
                                diff.count() > *group->collectiveTimeoutUs;
                            if (status.s == TransferStatusEnum::FAILED ||
                                timed_out) {
                                task.failedRanksHost[j] = 1;
                                LOG(ERROR)
                                    << "Rank " << group->rank
                                    << " sync to peer " << j
                                    << " failed for op " << (int)task.opType;
                            } else {
                                task_done = false;
                                break;
                            }
                        }
                    }
                    if (diff.count() > *group->collectiveTimeoutUs) {
                        activeTime[i] = clock::now();
                    }
                    if (task_done) {
                        // Clear sync signals.
                        if (hasView && group->backend) {
                            for (int ig = 0;
                                 ig < static_cast<int>((*rankOrderPtr).size());
                                 ++ig) {
                                GlobalRank peer = (*rankOrderPtr)[ig];
                                if (peer >= 0 && peer < kMaxNumRanks) {
                                    signal_ptr[peer] = 0;
                                }
                            }
                        } else {
                            for (int j = 0; j < group->size; ++j) {
                                signal_ptr[j] = 0;
                            }
                        }

                        // Push transfer observation via backend's Agent.
                        if (task.attemptedRanksHost && group->backend) {
                            bool has_any_attempted = false;
                            for (int j = 0; j < iter_count; ++j) {
                                if (task.attemptedRanksHost[j]) {
                                    has_any_attempted = true;
                                    break;
                                }
                            }
                            if (has_any_attempted) {
                                std::vector<uint8_t> attempted(kMaxNumRanks, 0);
                                std::vector<uint8_t> failed(kMaxNumRanks, 0);
                                std::vector<uint8_t> succeeded(kMaxNumRanks, 0);
                                for (int j = 0; j < iter_count; ++j) {
                                    GlobalRank peer =
                                        hasView ? (*rankOrderPtr)[j]
                                                : static_cast<GlobalRank>(j);
                                    if (peer >= 0 && peer < kMaxNumRanks) {
                                        attempted[peer] =
                                            task.attemptedRanksHost[j] ? 1 : 0;
                                        failed[peer] =
                                            task.failedRanksHost[j] ? 1 : 0;
                                        succeeded[peer] =
                                            (attempted[peer] && !failed[peer])
                                                ? 1
                                                : 0;
                                    }
                                }
                                group->backend->getAgent()
                                    .pushTransferObservation(
                                        group->group_id, std::move(attempted),
                                        std::move(failed),
                                        std::move(succeeded));
                            }
                        }

                        task_status[i].store(DONE, std::memory_order_release);
                        task.active = false;
                        if (hasCallback_[i]) {
                            auto callback = std::move(callbacks_[i]);
                            hasCallback_[i] = false;
                            callback();
                        }
                        auto s = group->engine->freeBatchID(task.batchID);
                        if (!s.ok()) {
                            LOG(WARNING)
                                << "BatchID leaked due to freeBatchID "
                                   "failure (likely caused by a timeout): "
                                << s.message();
                        }
                    }
                }
            }
        }
    });
}

std::shared_ptr<MooncakeWorker> MooncakeWorkerManager::GetWorker(
    int worker_id) {
    std::lock_guard<std::mutex> lock(manager_mutex_);
    auto it = workers_.find(worker_id);
    if (it != workers_.end()) {
        return it->second;
    }
    auto worker = std::make_shared<MooncakeWorker>(worker_id);
    workers_[worker_id] = worker;
    return worker;
}

std::shared_ptr<MooncakeWorker> MooncakeWorkerManager::GetCPUWorker() {
    return GetWorker(CPUWorkerID);
}

std::shared_ptr<MooncakeWorker> MooncakeWorkerManager::GetCUDAWorker(
    int cuda_device_index) {
    return GetWorker(cuda_device_index);
}

}  // namespace mooncake
