#ifndef MOONCAKE_PG_SERIALIZED_EXECUTOR_H
#define MOONCAKE_PG_SERIALIZED_EXECUTOR_H

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace mooncake {

// Generic single-threaded serialized executor.
//
// All tasks posted to this executor are processed sequentially on a single
// dedicated thread.  The executor itself does no I/O and holds no locks
// beyond the task queue — it is purely a concurrency primitive.
//
// The optional tick callback fires after every batch of tasks (including
// empty batches, thanks to the 50 ms wait timeout).  Hosts use this callback
// for periodic work such as heartbeat timeout detection.
//
// Thread safety:
//   - post() may be called from any thread.
//   - The tick callback and all tasks run on the executor thread.
class SerializedExecutor {
   public:
    explicit SerializedExecutor(std::string name) : name_(std::move(name)) {}

    ~SerializedExecutor() { shutdown(); }

    // Start the executor thread.  Idempotent — second call is a no-op.
    void start() {
        if (running_.exchange(true, std::memory_order_acq_rel)) return;
        thread_ = std::thread([this] { loop(); });
    }

    // Gracefully stop the executor thread.  Blocks until the thread exits.
    // Idempotent — safe to call multiple times.
    //
    // Drains any tasks that were still in the queue when the loop exited so
    // that RPC contexts (held inside lambda captures) receive a response
    // rather than being leaked.
    void shutdown() {
        if (!running_.exchange(false, std::memory_order_acq_rel)) return;
        cv_.notify_one();
        if (thread_.joinable()) thread_.join();
        // Execute any tasks that were posted between the last loop iteration
        // and now (e.g. callbacks from the RPC background thread).  These run
        // on the caller's thread.
        std::vector<std::function<void()>> leftover;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            leftover.swap(queue_);
        }
        for (auto& task : leftover) {
            task();
        }
    }

    // Post a task to the executor.  May be called from any thread.
    // Tasks are processed in FIFO order on the executor thread.
    void post(std::function<void()> task) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push_back(std::move(task));
        }
        cv_.notify_one();
    }

    // Set a callback that fires after every batch of tasks, even empty
    // batches (the wait_for timeout ensures it fires roughly every 50 ms).
    // Only call this before start() or from the executor thread itself.
    void setTickCallback(std::function<void()> cb) {
        tick_callback_ = std::move(cb);
    }

   private:
    void loop() {
        while (running_.load(std::memory_order_acquire)) {
            std::vector<std::function<void()>> batch;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                // Wait up to 50 ms for work.  This also guarantees tick
                // fires at least every ~50 ms even when idle.
                cv_.wait_for(lock, std::chrono::milliseconds(50), [this] {
                    return !queue_.empty() ||
                           !running_.load(std::memory_order_acquire);
                });
                batch.swap(queue_);
            }
            for (auto& task : batch) {
                task();
            }
            if (tick_callback_) {
                tick_callback_();
            }
        }
    }

    std::string name_;
    std::thread thread_;
    std::atomic<bool> running_{false};
    std::mutex mutex_;
    std::condition_variable cv_;
    std::vector<std::function<void()>> queue_;
    std::function<void()> tick_callback_;
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_SERIALIZED_EXECUTOR_H
