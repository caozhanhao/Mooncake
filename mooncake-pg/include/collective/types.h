#ifndef MOONCAKE_PG_COLLECTIVE_TYPES_H
#define MOONCAKE_PG_COLLECTIVE_TYPES_H

#include <cstdint>

namespace mooncake {

using DeviceId = int32_t;

inline constexpr DeviceId kCpuDeviceId = -1;
inline constexpr DeviceId kInvalidDeviceId = -2;

// Group-scoped identity used below the control-plane binding boundary.
// GlobalRank must not leak into collective executors or transports.
using InGroupRank = int32_t;

using CollectiveBindingId = uint32_t;
inline constexpr CollectiveBindingId kInvalidCollectiveBindingId =
    ~CollectiveBindingId{0};

struct BufferSpan {
    uint64_t offset = 0;
    uint64_t bytes = 0;

    bool operator==(const BufferSpan&) const = default;
};

enum class CollectiveRoute : uint8_t {
    DevP2p = 0,
    DevRdma,
    Host,
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_COLLECTIVE_TYPES_H
