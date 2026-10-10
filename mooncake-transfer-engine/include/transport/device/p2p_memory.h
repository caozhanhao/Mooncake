// Copyright 2026 KVCache.AI
// Licensed under the Apache License, Version 2.0.

#pragma once

#include "transport/device/device_transport.h"

namespace mooncake::device {

// Keeps an export available to new importers. The backing allocation remains
// caller-owned and must outlive all GPU accesses through imported mappings.
class P2pExport {
   public:
    ~P2pExport();
    P2pExport(const P2pExport&) = delete;
    P2pExport& operator=(const P2pExport&) = delete;

    const std::vector<int32_t>& metadata() const;

   private:
    friend class P2pMemoryExchange;
    struct Impl;
    explicit P2pExport(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

// CUDA memory sharing independent of the EP buffer/peer-table transport.
// Uses the existing IPC/FABRIC helpers, and a lazy, internal Linux UDS worker
// for POSIX FD VMM. No RPC, launcher cooperation or external progress needed.
// FD peers must share a Linux network namespace and UID (abstract UDS).
// Construct after spawning workers. Exports may outlive this exchange.
class P2pMemoryExchange {
   public:
    P2pMemoryExchange();
    ~P2pMemoryExchange();
    P2pMemoryExchange(const P2pMemoryExchange&) = delete;
    P2pMemoryExchange& operator=(const P2pMemoryExchange&) = delete;

    // ptr must be an allocation base on the current device. For VMM, bytes
    // must be granularity-aligned and covered by that allocation's mapping.
    // Returns nullptr when memory cannot be exported. Keep the result alive
    // while peers may import its metadata (including elastic replacements).
    std::unique_ptr<P2pExport> exportMemory(void* ptr, size_t bytes);

    // Import onto the current device. FD exchange has a bounded wait; an
    // inaccessible, revoked or malformed handle returns nullptr. Metadata is
    // versioned separately from the EP P2pTransport handle format.
    std::unique_ptr<P2pMapping> importMemory(
        const std::vector<int32_t>& metadata);

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace mooncake::device
