// Copyright 2026 KVCache.AI
// Licensed under the Apache License, Version 2.0.

#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <unistd.h>

namespace mooncake::device::detail {

class OwnedFd {
   public:
    OwnedFd() = default;
    explicit OwnedFd(int fd) : fd_(fd) {}
    ~OwnedFd() { reset(); }
    OwnedFd(const OwnedFd&) = delete;
    OwnedFd& operator=(const OwnedFd&) = delete;
    OwnedFd(OwnedFd&& other) noexcept : fd_(other.release()) {}
    OwnedFd& operator=(OwnedFd&& other) noexcept {
        reset(other.release());
        return *this;
    }
    int get() const { return fd_; }
    int release() { return std::exchange(fd_, -1); }
    void reset(int fd = -1) {
        if (fd_ >= 0) close(fd_);
        fd_ = fd;
    }

   private:
    int fd_ = -1;
};

// Linux-only implementation detail. Construct after spawning worker processes.
// Publication starts one CPU worker lazily; no GPU or external progress engine
// is involved. Registrations keep the worker alive even if the exchange dies.
class LocalFdExchange {
    struct State;

   public:
    using Token = std::array<uint64_t, 2>;
    struct Reference {
        std::string
            endpoint;  // Abstract AF_UNIX name, without the leading NUL.
        Token token{};
    };

    class Registration {
       public:
        ~Registration();
        Registration(const Registration&) = delete;
        Registration& operator=(const Registration&) = delete;
        const Reference& reference() const { return reference_; }

       private:
        friend class LocalFdExchange;
        Registration(std::shared_ptr<State> state, Reference reference)
            : state_(std::move(state)), reference_(std::move(reference)) {}
        std::shared_ptr<State> state_;
        Reference reference_;
        bool registered_ = false;
    };

    LocalFdExchange();
    ~LocalFdExchange();
    LocalFdExchange(const LocalFdExchange&) = delete;
    LocalFdExchange& operator=(const LocalFdExchange&) = delete;

    // Duplicates fd; the caller retains ownership of its original descriptor.
    // Destroying the registration rejects subsequent requests for this token.
    std::unique_ptr<Registration> publish(int fd);

    // Returns a caller-owned descriptor, or -1 on rejection/timeout/error.
    static int request(
        const Reference& reference,
        std::chrono::milliseconds timeout = std::chrono::milliseconds(1000));

   private:
    std::shared_ptr<State> state_;
};

}  // namespace mooncake::device::detail
