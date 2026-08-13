#include "device_comm/device_collective/device_collective_recovery.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <exception>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <glog/logging.h>

namespace mooncake {
struct DeviceCollectiveRecoveryWorker::MailboxState {
    DeviceCollectiveRecoveryMailbox* mailbox = nullptr;
    Handler handler;
    bool removed = false;
};

namespace {

constexpr auto kRecoveryCheckInterval = std::chrono::milliseconds(1);

static_assert(std::atomic_ref<uint64_t>::is_always_lock_free,
              "device/host recovery generations require lock-free atomics");

}  // namespace

void DeviceCollectiveRecoveryWorker::runLoop() {
    for (;;) {
        MailboxState* pending = nullptr;
        uint64_t generation = 0;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            if (stop_requested_) break;
            for (const auto& state : mailboxes_) {
                if (state->removed) continue;
                const uint64_t observed =
                    std::atomic_ref(state->mailbox->failure_generation)
                        .load(std::memory_order_acquire);
                const uint64_t ready =
                    std::atomic_ref(state->mailbox->ready_generation)
                        .load(std::memory_order_acquire);
                if (observed <= ready) continue;
                pending = state.get();
                generation = observed;
                active_mailbox_ = state->mailbox;
                break;
            }
            if (!pending) {
                changed_.wait_for(lock, kRecoveryCheckInterval);
                continue;
            }
        }

        auto recovered = pending->handler(generation);
        if (!recovered.has_value()) {
            LOG(ERROR) << "Device collective recovery failed; the kernel "
                          "remains "
                          "parked: "
                       << recovered.error().message;
            std::lock_guard<std::mutex> lock(mutex_);
            active_mailbox_ = nullptr;
            worker_failed_ = true;
            changed_.notify_all();
            return;
        }

        const uint64_t ready =
            std::atomic_ref(pending->mailbox->ready_generation)
                .load(std::memory_order_acquire);
        PG_ASSERT(ready >= generation,
                  "device collective recovery returned without releasing its "
                  "parked kernel");

        std::unique_lock<std::mutex> lock(mutex_);
        active_mailbox_ = nullptr;
        changed_.notify_all();
    }
}

void DeviceCollectiveRecoveryWorker::run() noexcept {
    try {
        runLoop();
    } catch (const std::exception& error) {
        LOG(ERROR) << "DeviceCollectiveRecoveryWorker stopped after an "
                      "exception: "
                   << error.what();
        std::lock_guard<std::mutex> lock(mutex_);
        active_mailbox_ = nullptr;
        worker_failed_ = true;
        changed_.notify_all();
    } catch (...) {
        LOG(ERROR) << "DeviceCollectiveRecoveryWorker stopped after an "
                      "unknown exception";
        std::lock_guard<std::mutex> lock(mutex_);
        active_mailbox_ = nullptr;
        worker_failed_ = true;
        changed_.notify_all();
    }
}

DeviceCollectiveRecoveryWorker::DeviceCollectiveRecoveryWorker() = default;

DeviceCollectiveRecoveryWorker::~DeviceCollectiveRecoveryWorker() noexcept {
    shutdown();
}

PGResult<void> DeviceCollectiveRecoveryWorker::start() {
    std::lock_guard<std::mutex> lock(mutex_);
    PG_VALIDATE_STATE(!stop_requested_,
                      "DeviceCollectiveRecoveryWorker is shut down");
    if (started_) return {};
    try {
        worker_ = std::thread([this] { run(); });
    } catch (const std::exception& error) {
        return makePGError(
            PGErrorCode::SystemError,
            std::string("failed to start DeviceCollectiveRecoveryWorker: ") +
                error.what());
    }
    started_ = true;
    return {};
}

PGResult<void> DeviceCollectiveRecoveryWorker::addMailbox(
    DeviceCollectiveRecoveryMailbox* mailbox, Handler handler) {
    PG_VALIDATE_ARG(mailbox, "device collective recovery mailbox is null");
    PG_VALIDATE_ARG(static_cast<bool>(handler),
                    "device collective recovery handler is empty");

    auto state = std::make_unique<MailboxState>();
    state->mailbox = mailbox;
    state->handler = std::move(handler);

    {
        std::lock_guard<std::mutex> lock(mutex_);
        PG_VALIDATE_STATE(started_ && !stop_requested_ && !worker_failed_,
                          "DeviceCollectiveRecoveryWorker is not running");
        const auto existing =
            std::find_if(mailboxes_.begin(), mailboxes_.end(),
                         [mailbox](const auto& current) {
                             return current->mailbox == mailbox;
                         });
        PG_VALIDATE_STATE(existing == mailboxes_.end(),
                          "device collective recovery mailbox is already "
                          "added");
        mailboxes_.push_back(std::move(state));
    }
    changed_.notify_all();
    return {};
}

void DeviceCollectiveRecoveryWorker::removeMailbox(
    DeviceCollectiveRecoveryMailbox* mailbox) noexcept {
    if (!mailbox) return;
    std::unique_lock<std::mutex> lock(mutex_);
    const auto selected = std::find_if(
        mailboxes_.begin(), mailboxes_.end(),
        [mailbox](const auto& current) { return current->mailbox == mailbox; });
    if (selected == mailboxes_.end()) return;
    auto* const state = selected->get();
    state->removed = true;
    changed_.notify_all();
    changed_.wait(lock, [this, mailbox] { return active_mailbox_ != mailbox; });
    std::erase_if(mailboxes_, [state](const auto& current) {
        return current.get() == state;
    });
}

void DeviceCollectiveRecoveryWorker::shutdown() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stop_requested_) return;
        if (!mailboxes_.empty() || active_mailbox_) {
            LOG(ERROR)
                << "DeviceCollectiveRecoveryWorker is shutting down with "
                   "mailboxes still added";
        }
        stop_requested_ = true;
    }
    changed_.notify_all();
    if (worker_.joinable()) worker_.join();
    std::lock_guard<std::mutex> lock(mutex_);
    mailboxes_.clear();
}

}  // namespace mooncake
