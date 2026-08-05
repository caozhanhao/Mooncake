#ifndef MOONCAKE_PG_COLLECTIVE_TRANSPORT_LINK_H
#define MOONCAKE_PG_COLLECTIVE_TRANSPORT_LINK_H

#include <cstdint>
#include <limits>

namespace mooncake {

// Stable process-local reference to a host Transfer Engine peer. The proxy
// resolves the current SegmentID immediately before submission. Reconnects in
// the same peer incarnation are therefore transparent, while a replacement
// process invalidates the captured handle through target_rank_epoch.
struct HostLinkHandle {
    uint32_t slot = std::numeric_limits<uint32_t>::max();
    uint64_t target_rank_epoch = std::numeric_limits<uint64_t>::max();

    bool operator==(const HostLinkHandle&) const = default;
};

inline constexpr HostLinkHandle kInvalidHostLinkHandle{};

}  // namespace mooncake

#endif  // MOONCAKE_PG_COLLECTIVE_TRANSPORT_LINK_H
