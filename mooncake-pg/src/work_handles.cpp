#include <work_handles.h>
#include <mooncake_backend.h>
#include <mooncake_worker.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include "pg_utils.h"

namespace mooncake {

FailedRanksHint FailedRanksHint::allocate(int n, bool isCpu) {
    // Always use plain heap memory.  The GPU reduce kernel no longer reads
    // these bitmaps, so cudaHostAllocMapped is unnecessary.  Avoiding CUDA
    // runtime allocation calls on the hot path prevents lock contention /
    // deadlocks with other CUDA operations (kernel launches, memcpys in TE).
    if (isCpu) {
        return {torch::zeros({n}, torch::kInt32),
                torch::zeros({n}, torch::kInt32)};
    }
    auto alloc_plain = [n](at::Tensor& tensor) {
        int* ptr = new int[n]();  // zero-initialized
        auto deleter = [](void* p) { delete[] static_cast<int*>(p); };
        tensor = torch::from_blob(ptr, {n}, deleter,
                                  torch::TensorOptions().dtype(torch::kInt32));
    };

    at::Tensor failed_tensor;
    alloc_plain(failed_tensor);
    at::Tensor attempted_tensor;
    alloc_plain(attempted_tensor);

    return {failed_tensor, attempted_tensor};
}

bool MooncakeWorkCpu::wait(std::chrono::milliseconds timeout) {
    future_->wait();
    bool ok = future_->completed() && !future_->hasError();
    if (ok) {
        bool all_ok = true;
        int* data = failedRanksHint_.data();
        for (int i = 0; i < meta_->size; ++i) {
            if (data[i] != 0) {
                all_ok = false;
                break;
            }
        }
        failedRanksHint_.local_success = all_ok;
    } else {
        failedRanksHint_.local_success = false;
    }

    // Implicit sync-after-failure: if a peer failure was detected locally,
    // notify the Coordinator and block until a membership decision is made.
    if (!failedRanksHint_.local_success && meta_->backend) {
        auto& ctx = meta_->backend->getProcessContext();
        if (ctx.implicit_sync_after_failure) {
            LOG(INFO) << "Local failure detected on cpu work, triggering implicit syncAfterFailure";
            meta_->backend->syncAfterFailure();
        }
    }

    return ok;
}

bool MooncakeWorkCuda::wait(std::chrono::milliseconds timeout) {
    // Wait until the task has been submitted to TransferEngine:
    // This tries to ensure that the CUDA kernels required for the transfer
    // have been launched by the time `waitUntilTasksSubmitted` returns.
    //
    // Why is this needed? PyTorch documentation implies that collective
    // operations should be enqueued when `wait()` returns. In practice, we
    // found that violating this causes hangs.
    //
    // Our current hypothesis for the hang is: PyTorch assumes the kernels
    // needed for the transfer are already launched when `wait` returns
    // true. It may then launch subsequent operations after the collective
    // (e.g., `.cpu()`). Such operations may acquire a process-wide lock in
    // the CUDA runtime. Also, they may rely on the data produced by the
    // collective, thus causing a synchronization on enq_stream. However,
    // holding that runtime lock prevents cudaMemcpy(Async) in TE/TENT from
    // launching. This means the transfer can't finish, and enq_stream won't
    // complete. Thus, a deadlock occurs.
    // (In practice, we found that replacing all cudaMemcpyAsync in TENT
    // with cuMemcpyAsync actually alleviates this, which further suggests a
    // deadlock in the CUDA runtime. However, that change is too invasive
    // for TE/TENT, so we do not adopt it here.)
    //
    // Strictly speaking, the wait is needed for another reason: The current
    // stream will be blocked on the event below. Any subsequent work on
    // `current_stream` will wait on that event, which effectively waits for
    // the task to be done. Therefore, we must ensure all kernels needed for
    // the transfer task are launched BEFORE blocking the current stream, in
    // case TE/TENT use `current_stream` to launch those kernels (though it
    // is rare).
    //
    // Please note that this logic relies on the assumption that TE/TENT
    // will launch all CUDA operations in `submitTransfer`.
    // Unfortunately, TcpTransport in TE and TENT currently violates this
    // assumption (cudaMemcpy(Async) may be called later from a callback),
    // which can cause hangs in PG when a CUDA operation such as
    // `x.cpu().item()` follows the collective. For TE's TcpTransport, the
    // use of cudaMemcpy on the default stream may also contribute to the
    // hang.
    //
    // Besides, for CPU-only transports (like RdmaTransport),
    // waitUntilTasksSubmitted is totally unnecessary, but we keep it for
    // uniform behavior to avoid invasive changes to TE/TENT.
    bool submitted = true;
    if (at::cuda::currentStreamCaptureStatus() ==
        c10::cuda::CaptureStatus::None) {
        // Normal execution: block until tasks are submitted.
        submitted = worker_->waitUntilTasksSubmitted(submitted_tasks_, timeout);
    } else {
        // During CUDA graph capture, kernels are recorded but not actually
        // executed. The enqueueTaskKernel would never run, so
        // waitUntilTasksSubmitted would hang because the CPU worker thread
        // never sees task.active == true.
        //
        // Note that this also means NvlinkTransport (and TcpTransport too,
        // of course) won't work with CUDA Graphs: Kernels launched inside
        // TE/TENT can't be captured by the graph, and during replay they
        // are not ordered with the graph execution. This may trigger the
        // same deadlock described above.
    }
    if (!submitted) return false;

    // Once all tasks have been submitted, use the event to synchronize
    // the current stream and the enqueue stream, but do not wait on this
    // event.
    //
    // See PyTorch docs for more details:
    // https://docs.pytorch.org/docs/stable/distributed.html#synchronous-and-asynchronous-collective-operations
    //   "wait() - in the case of CPU collectives, will block the process
    //    until the operation is completed. In the case of CUDA collectives,
    //    will block the currently active CUDA stream until the operation
    //    is completed (but will not block the CPU)."
    auto current_stream = at::cuda::getCurrentCUDAStream();
    event_->block(current_stream);

    // Compute local_success: true iff no peer failed in this operation.
    bool all_ok = true;
    int* data = failedRanksHint_.data();
    for (int i = 0; i < meta_->size; ++i) {
        if (data[i] != 0) {
            all_ok = false;
            break;
        }
    }
    failedRanksHint_.local_success = all_ok;

    // Implicit sync-after-failure: same rationale as MooncakeWorkCpu::wait.
    if (!all_ok && meta_->backend) {
        auto& ctx = meta_->backend->getProcessContext();
        if (ctx.implicit_sync_after_failure) {
            LOG(INFO) << "Local failure detected on cuda work, triggering implicit syncAfterFailure";
            meta_->backend->syncAfterFailure();
        }
    }

    return true;
}

bool MooncakeBarrierWorkCuda::wait(std::chrono::milliseconds timeout) {
    // Skip host-side synchronization during CUDA graph capture.
    // cudaEventSynchronize is not permitted while a stream is capturing.
    if (at::cuda::currentStreamCaptureStatus() !=
        c10::cuda::CaptureStatus::None) {
        // We still need stream-level synchronization so that subsequent
        // operations on the capture stream are ordered after the barrier
        // task on the enqueue stream.
        auto current_stream = at::cuda::getCurrentCUDAStream();
        event_->block(current_stream);
        return true;
    }

    if (timeout == kNoTimeout) {
        event_->synchronize();

        // Implicit sync-after-failure: same rationale as MooncakeWorkCpu.
        bool all_ok = true;
        int* data = failedRanksHint_.data();
        for (int i = 0; i < meta_->size; ++i) {
            if (data[i] != 0) {
                all_ok = false;
                break;
            }
        }
        failedRanksHint_.local_success = all_ok;
        if (!all_ok && meta_->backend) {
            auto& ctx = meta_->backend->getProcessContext();
            if (ctx.implicit_sync_after_failure) {
                meta_->backend->syncAfterFailure();
            }
        }
        return true;
    }

    BackoffWaiter waiter(
        BackoffWaiterConfig::constantSleep(std::chrono::microseconds(10)));
    bool ok = waiter.wait_for(timeout, [this] { return event_->query(); });

    if (ok) {
        bool all_ok = true;
        int* data = failedRanksHint_.data();
        for (int i = 0; i < meta_->size; ++i) {
            if (data[i] != 0) {
                all_ok = false;
                break;
            }
        }
        failedRanksHint_.local_success = all_ok;
        if (!all_ok && meta_->backend) {
            auto& ctx = meta_->backend->getProcessContext();
            if (ctx.implicit_sync_after_failure) {
                meta_->backend->syncAfterFailure();
            }
        }
    }

    return ok;
}

at::Tensor MooncakeWorkCuda::getFailedRanksHint() const {
    // Ensure the worker thread has completed the task and set
    // failedRanksHintHost before returning the tensor.
    if (event_ && at::cuda::currentStreamCaptureStatus() ==
                      c10::cuda::CaptureStatus::None) {
        event_->synchronize();
    }
    return failedRanksHint_.tensor;
}

bool MooncakeWorkCuda::getLocalSuccess() const {
    return failedRanksHint_.local_success;
}

bool MooncakeP2PWork::isCompleted() {
    return status_->load(std::memory_order_acquire) !=
           MooncakeP2PWork::Status::kPending;
}

bool MooncakeP2PWork::isSuccess() const {
    return status_->load(std::memory_order_acquire) ==
           MooncakeP2PWork::Status::kSuccess;
}

bool MooncakeP2PWork::wait(std::chrono::milliseconds timeout) {
    BackoffWaiterConfig cfg{};
    cfg.max_sleep = std::chrono::microseconds(10);
    BackoffWaiter waiter(cfg);

    bool done = false;
    if (timeout.count() > 0) {
        done = waiter.wait_for(timeout, [this] {
            return status_->load(std::memory_order_acquire) !=
                   MooncakeP2PWork::Status::kPending;
        });
    } else {
        waiter.wait([this] {
            return status_->load(std::memory_order_acquire) !=
                   MooncakeP2PWork::Status::kPending;
        });
        done = true;
    }

    if (!done) {
        return false;
    }

    return true;
}

}  // namespace mooncake
