#ifndef MOONCAKE_PG_CONTROL_PLANE_TYPES_H
#define MOONCAKE_PG_CONTROL_PLANE_TYPES_H

#include <cstdint>
#include <vector>

#include "control_plane/strong_id.h"

// mooncake-pg uses two rank namespaces that are easy to confuse:
//
//   * GlobalRank  - process-wide identifier, range 0 .. max_world_size-1.
//                   Used for process-level link state, segment IDs, and
//                   GroupView member slots.
//   * InGroupRank - group-local identifier, range 0 .. group_size-1.
//                   Used inside a single process group and mapped to a
//                   GlobalRank through GroupView::rank_order.
//
// Because both were previously plain int32_t, the codebase had subtle bugs
// where an InGroupRank was used where a GlobalRank was expected (or vice
// versa).  We now represent them as distinct StrongId types so that the
// compiler rejects accidental mixing.  IndexedVector is used for arrays whose
// index namespace matters, and it is made transparent to struct_pack so it
// can be used directly in RPC structs.  At explicit boundaries (device POD,
// C-style APIs, device-visible arrays such as TransferGroupMeta) the
// underlying integer is still extracted with toUnderlying().

namespace mooncake {

// Rank tags for StrongId.  The tag types are empty; they exist only to make
// GlobalRank and InGroupRank distinct, incompatible types.
struct GlobalRankTag {};
struct InGroupRankTag {};

// GlobalRank: process-wide rank identifier, namespace 0 .. max_world_size-1.
// InGroupRank: group-local rank identifier, namespace 0 .. group_size-1.
//
// Both wrap int32_t but cannot be mixed up accidentally.  They are aggregates,
// so construct with braces (GlobalRank{5}) and use toUnderlying() or
// static_cast<int32_t>() only at explicit boundaries (device POD, C APIs).
using GlobalRank = StrongId<GlobalRankTag, int32_t>;
using InGroupRank = StrongId<InGroupRankTag, int32_t>;

using GroupId = int32_t;  // process group ID (= backendIndex)

constexpr GlobalRank kInvalidGlobalRank{-1};
constexpr GroupId kInvalidGroupId = -1;
constexpr int kMaxNumRanks = 64;

// Convenience range factories for the common "iterate over every valid rank"
// pattern.  These keep the rank namespace explicit without repeating
// GlobalRank{size} / InGroupRank{size} at every loop site.
//
//   for (GlobalRank r : globalRankRange(max_world_size_)) { ... }
//   for (InGroupRank r : inGroupRankRange(group_size)) { ... }
inline IndexRange<GlobalRankTag, int32_t> globalRankRange(int32_t end) {
    return makeIndexRange<GlobalRank>(0, end);
}
inline IndexRange<InGroupRankTag, int32_t> inGroupRankRange(int32_t end) {
    return makeIndexRange<InGroupRank>(0, end);
}

// Epoch sentinels.  All epochs start at kInvalidEpoch (0) and only increase
// from there.  A value of kInvalidEpoch means "not yet initialised" / "stale".
// The first real epoch value is >= 1.
constexpr uint64_t kInvalidEpoch = 0;
constexpr uint64_t kInitialEndpointEpoch = 1;

// Process-level state for a rank.  All transitions are driven by the
// Coordinator (authoritative); the Agent never self-demotes.
//
//   registerAgent()         Coordinator computes TE HealthySet
//        |                  (heartbeat + transfer observations)
//        v                         |
//   OFFLINE -> SYNCED ---------> HEALTHY
//        ^                         |
//        |   heartbeat timeout     |   excluded from HealthySet
//        +--- or disconnect -------+-----> SYNCED
//
enum class RankState : uint8_t {
    OFFLINE = 0,
    SYNCED = 1,
    HEALTHY = 2,
};

// Group-level, per-(group_id, rank) buffer/sync/P2P addresses.
struct GroupEndpointInfo {
    // collective
    uint64_t send_buffer[2] = {};
    uint64_t recv_buffer[2] = {};
    uint64_t send_sync[2] = {};
    uint64_t recv_sync[2] = {};
    // p2p
    uint64_t p2p_credit_region = 0;
    uint64_t p2p_ack_region = 0;
};

// Membership state of one rank inside a single GroupView.
enum class GroupMemberStatus : uint8_t {
    kNone = 0,      // slot has never belonged to this group
    kLeft = 1,      // rank explicitly left the group (called destroy_group)
    kInactive = 2,  // deactivated
    kActive = 3,    // rank is an active member
};

// Rank state inside a single GroupView.
//
// endpoint_epoch is monotonically increasing and never reset.  kInvalidEpoch
// means "no endpoint published yet".  Validity requires agent_session_epoch
// to match the rank's current session AND endpoint_epoch != kInvalidEpoch.
struct GroupMember {
    GroupMemberStatus status = GroupMemberStatus::kNone;
    uint64_t agent_session_epoch = kInvalidEpoch;
    uint64_t endpoint_epoch = kInvalidEpoch;
    GroupEndpointInfo endpoint_info;

    bool isActive() const { return status == GroupMemberStatus::kActive; }
    bool isMember() const {
        return status == GroupMemberStatus::kActive ||
               status == GroupMemberStatus::kInactive ||
               status == GroupMemberStatus::kLeft;
    }
    bool hasLeft() const { return status == GroupMemberStatus::kLeft; }
};

// Group lifecycle status.
//
//   joinGroup()
//       |
//       v
//   Bootstrapping  -- all active ranks HEALTHY + have endpoints -->
//   BootstrapSyncing
//       (waiting for       (Coordinator broadcasts ViewUpdate,
//        publishEndpoint     waits for ACKs from all active ranks)
//        calls)                        |
//                                      v  (all ACKs received)
//                                    Ready
//                               (group usable for data-plane transfers)
//
//   Bootstrapping      - collecting endpoints and waiting for all active ranks
//                        to become HEALTHY with valid endpoints.
//   BootstrapSyncing   - Coordinator initiated 2PC barrier; waiting for all
//                        active ranks to ACK the initial ViewUpdate.
//                        If a peer dies here, waitUntilGroupReady() hangs
//                        until its timeout.
//   Ready              - barrier complete; all ranks ready for data-plane
//                        transfers.
enum class GroupStatus : uint8_t {
    Bootstrapping = 0,
    BootstrapSyncing = 1,
    Ready = 2,
};

// Runtime state for a group.  rank_order maps InGroupRank → GlobalRank.
// epoch starts at 0 and monotonically increases; the first real epoch after
// bootstrap completion is 1.
struct GroupView {
    GroupId group_id = 0;
    GroupStatus status = GroupStatus::Bootstrapping;
    uint64_t epoch = 0;
    IndexedVector<GlobalRank, InGroupRankTag>
        rank_order;  // InGroupRank → GlobalRank
    IndexedVector<GroupMember, GlobalRankTag> members;  // indexed by GlobalRank

    // Strongly-typed accessor for members (GlobalRank-indexed).  Use these
    // instead of members[...] to keep the index namespace explicit.
    GroupMember& member(GlobalRank r) { return members[r]; }
    const GroupMember& member(GlobalRank r) const { return members[r]; }

    // Strongly-typed accessor for rank_order (InGroupRank → GlobalRank).
    GlobalRank globalRank(InGroupRank local) const { return rank_order[local]; }
};

// TransferObservationEvent, worker thread -> Agent queue.
// attempted_ranks / failed_ranks / succeeded_ranks are bit-vectors indexed by
// GlobalRank (size kMaxNumRanks).  Producers must translate InGroupRank peers
// to GlobalRank via rank_order before setting bits.
struct TransferObservationEvent {
    GroupId group_id = 0;
    IndexedVector<uint8_t, GlobalRankTag> attempted_ranks;
    IndexedVector<uint8_t, GlobalRankTag> failed_ranks;
    IndexedVector<uint8_t, GlobalRankTag> succeeded_ranks;
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_CONTROL_PLANE_TYPES_H
