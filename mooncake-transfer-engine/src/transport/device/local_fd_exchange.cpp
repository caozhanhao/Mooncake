// Copyright 2026 KVCache.AI
// Licensed under the Apache License, Version 2.0.

#include "local_fd_exchange.h"

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <map>
#include <mutex>
#include <poll.h>
#include <sys/eventfd.h>
#include <sys/random.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <system_error>
#include <thread>

namespace mooncake::device::detail {
namespace {

struct Packet {
    LocalFdExchange::Token token{};
    int32_t error = 0;
    uint32_t version = 1;
};

bool randomToken(LocalFdExchange::Token& token) {
    auto* out = reinterpret_cast<char*>(token.data());
    size_t remaining = sizeof(token);
    while (remaining) {
        const auto count = getrandom(out, remaining, GRND_NONBLOCK);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) return false;
        out += count;
        remaining -= count;
    }
    return true;
}

socklen_t makeAddress(const std::string& name, sockaddr_un& address) {
    if (name.empty() || name.size() >= sizeof(address.sun_path) ||
        name.find('\0') != std::string::npos)
        return 0;
    address = {};
    address.sun_family = AF_UNIX;
    memcpy(address.sun_path + 1, name.data(), name.size());
    return offsetof(sockaddr_un, sun_path) + 1 + name.size();
}

OwnedFd boundSocket(std::string& name) {
    LocalFdExchange::Token token;
    if (!randomToken(token)) return {};
    constexpr char hex[] = "0123456789abcdef";
    name = "mc-p2p-";
    const auto* bytes = reinterpret_cast<const unsigned char*>(token.data());
    for (size_t i = 0; i < sizeof(token); ++i) {
        const auto byte = bytes[i];
        name += hex[byte >> 4];
        name += hex[byte & 15];
    }
    OwnedFd fd(socket(AF_UNIX, SOCK_DGRAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0));
    int enabled = 1;
    sockaddr_un address{};
    const auto length = makeAddress(name, address);
    if (fd.get() < 0 ||
        setsockopt(fd.get(), SOL_SOCKET, SO_PASSCRED, &enabled,
                   sizeof(enabled)) != 0 ||
        bind(fd.get(), reinterpret_cast<sockaddr*>(&address), length) != 0)
        return {};
    return fd;
}

// Always collect/close ancillary descriptors, including malformed packets.
// MSG_CMSG_CLOEXEC closes the exec race on received descriptors.
bool receivePacket(int socket, Packet& packet, OwnedFd& received_fd,
                   sockaddr_un& address, socklen_t& address_length) {
    alignas(cmsghdr) char
        control[CMSG_SPACE(sizeof(ucred)) + CMSG_SPACE(sizeof(int))]{};
    iovec payload{&packet, sizeof(packet)};
    msghdr message{};
    message.msg_name = &address;
    message.msg_namelen = sizeof(address);
    message.msg_iov = &payload;
    message.msg_iovlen = 1;
    message.msg_control = control;
    message.msg_controllen = sizeof(control);
    const auto count =
        recvmsg(socket, &message, MSG_CMSG_CLOEXEC | MSG_DONTWAIT);
    if (count < 0) return false;
    address_length = message.msg_namelen;
    bool same_user = false;
    int fd_count = 0;
    for (auto* cmsg = CMSG_FIRSTHDR(&message); cmsg;
         cmsg = CMSG_NXTHDR(&message, cmsg)) {
        if (cmsg->cmsg_level != SOL_SOCKET) continue;
        if (cmsg->cmsg_type == SCM_CREDENTIALS &&
            cmsg->cmsg_len == CMSG_LEN(sizeof(ucred))) {
            ucred credentials{};
            memcpy(&credentials, CMSG_DATA(cmsg), sizeof(credentials));
            same_user = credentials.uid == getuid();
        } else if (cmsg->cmsg_type == SCM_RIGHTS) {
            const auto bytes = cmsg->cmsg_len - CMSG_LEN(0);
            for (size_t offset = 0; offset + sizeof(int) <= bytes;
                 offset += sizeof(int)) {
                int fd;
                memcpy(&fd, CMSG_DATA(cmsg) + offset, sizeof(fd));
                received_fd.reset(fd);
                ++fd_count;
            }
        }
    }
    return count == sizeof(packet) && same_user && packet.version == 1 &&
           fd_count <= 1 && !(message.msg_flags & (MSG_TRUNC | MSG_CTRUNC));
}

void reply(int socket, const Packet& packet, int fd, sockaddr_un& address,
           socklen_t address_length) {
    alignas(cmsghdr) char control[CMSG_SPACE(sizeof(int))]{};
    iovec payload{const_cast<Packet*>(&packet), sizeof(packet)};
    msghdr message{};
    message.msg_name = &address;
    message.msg_namelen = address_length;
    message.msg_iov = &payload;
    message.msg_iovlen = 1;
    if (fd >= 0) {
        message.msg_control = control;
        message.msg_controllen = sizeof(control);
        auto* cmsg = CMSG_FIRSTHDR(&message);
        cmsg->cmsg_level = SOL_SOCKET;
        cmsg->cmsg_type = SCM_RIGHTS;
        cmsg->cmsg_len = CMSG_LEN(sizeof(int));
        memcpy(CMSG_DATA(cmsg), &fd, sizeof(fd));
    }
    // A peer that stops receiving must not block other imports or shutdown.
    sendmsg(socket, &message, MSG_DONTWAIT | MSG_NOSIGNAL);
}

bool waitReady(int fd, short events,
               std::chrono::steady_clock::time_point deadline) {
    for (;;) {
        const auto remaining =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                deadline - std::chrono::steady_clock::now())
                .count();
        if (remaining <= 0) return false;
        const int timeout = static_cast<int>(
            std::min<int64_t>(remaining, std::numeric_limits<int>::max()));
        pollfd poll_fd{fd, events, 0};
        const auto result = poll(&poll_fd, 1, timeout);
        if (result < 0 && errno == EINTR) continue;
        return result > 0 && (poll_fd.revents & events);
    }
}

}  // namespace

struct LocalFdExchange::State {
    ~State() {
        if (worker.joinable()) {
            uint64_t signal = 1;
            while (write(stop.get(), &signal, sizeof(signal)) < 0 &&
                   errno == EINTR) {
            }
            worker.join();
        }
    }

    // Called under mutex, only on the first publication.
    bool start() {
        if (worker.joinable()) return true;
        socket = boundSocket(endpoint);
        stop.reset(eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK));
        if (socket.get() < 0 || stop.get() < 0) {
            socket.reset();
            stop.reset();
            return false;
        }
        try {
            worker = std::thread([this] { serve(); });
        } catch (const std::system_error&) {
            socket.reset();
            stop.reset();
            return false;
        }
        return true;
    }

    void serve() {
        for (;;) {
            pollfd fds[]{{socket.get(), POLLIN, 0}, {stop.get(), POLLIN, 0}};
            const auto ready = poll(fds, 2, -1);
            if (ready < 0 && errno == EINTR) continue;
            if (ready <= 0 || fds[1].revents) return;
            if (!(fds[0].revents & POLLIN)) return;
            Packet packet{};
            OwnedFd unexpected_fd;
            sockaddr_un address{};
            socklen_t length = 0;
            if (!receivePacket(socket.get(), packet, unexpected_fd, address,
                               length) ||
                unexpected_fd.get() >= 0 || packet.error != 0 ||
                length <= offsetof(sockaddr_un, sun_path) + 1 ||
                length > sizeof(address))
                continue;
            std::lock_guard<std::mutex> lock(mutex);
            const auto found = entries.find(packet.token);
            packet.error = found == entries.end() ? ENOENT : 0;
            // Keep the registration alive until sendmsg has taken its
            // reference.
            reply(socket.get(), packet,
                  found == entries.end() ? -1 : found->second.get(), address,
                  length);
        }
    }

    std::mutex mutex;
    std::map<Token, OwnedFd> entries;
    std::string endpoint;
    OwnedFd socket;
    OwnedFd stop;
    std::thread worker;
};

LocalFdExchange::LocalFdExchange() : state_(std::make_shared<State>()) {}
LocalFdExchange::~LocalFdExchange() = default;

LocalFdExchange::Registration::~Registration() {
    if (!registered_) return;
    std::lock_guard<std::mutex> lock(state_->mutex);
    state_->entries.erase(reference_.token);
}

std::unique_ptr<LocalFdExchange::Registration> LocalFdExchange::publish(
    int fd) {
    OwnedFd duplicate(fcntl(fd, F_DUPFD_CLOEXEC, 0));
    Reference reference;
    if (duplicate.get() < 0 || !randomToken(reference.token)) return nullptr;
    auto registration = std::unique_ptr<Registration>(
        new Registration(state_, std::move(reference)));
    std::lock_guard<std::mutex> lock(state_->mutex);
    if (!state_->start()) return nullptr;
    registration->reference_.endpoint = state_->endpoint;
    if (!state_->entries
             .emplace(registration->reference_.token, std::move(duplicate))
             .second)
        return nullptr;
    registration->registered_ = true;
    return registration;
}

int LocalFdExchange::request(const Reference& reference,
                             std::chrono::milliseconds timeout) {
    sockaddr_un remote{};
    const auto length = makeAddress(reference.endpoint, remote);
    if (length == 0 || timeout.count() <= 0) return -1;
    std::string local_name;
    auto socket = boundSocket(local_name);
    if (socket.get() < 0 ||
        connect(socket.get(), reinterpret_cast<sockaddr*>(&remote), length) !=
            0)
        return -1;
    Packet packet{};
    packet.token = reference.token;
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    for (;;) {
        if (std::chrono::steady_clock::now() >= deadline) return -1;
        if (send(socket.get(), &packet, sizeof(packet), MSG_NOSIGNAL) ==
            sizeof(packet))
            break;
        if (errno == EINTR) continue;
        if ((errno != EAGAIN && errno != EWOULDBLOCK) ||
            !waitReady(socket.get(), POLLOUT, deadline))
            return -1;
    }
    OwnedFd received;
    sockaddr_un sender{};
    socklen_t sender_length = 0;
    if (!waitReady(socket.get(), POLLIN, deadline) ||
        !receivePacket(socket.get(), packet, received, sender, sender_length) ||
        packet.token != reference.token || packet.error != 0)
        return -1;
    return received.release();
}

}  // namespace mooncake::device::detail
