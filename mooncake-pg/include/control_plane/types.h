#ifndef MOONCAKE_PG_CONTROL_PLANE_TYPES_H
#define MOONCAKE_PG_CONTROL_PLANE_TYPES_H

#include <cstdint>
#include <string>
#include <vector>

// mooncake-pg uses two rank namespaces that are easy to confuse:
//
//   * GlobalRank  - process-wide identifier, range 0 .. max_world_size-1.
//                   Used for process-level link state, segment IDs, and
//                   GroupView member slots.
//   * InGroupRank - group-local identifier, range 0 .. group_size-1.
//                   Used inside a single process group and mapped to a
//                   GlobalRank through GroupView::rank_order.
//
// Both are plain int32_t.  Callers must take care not to mix them up.

namespace mooncake {

// Rank types.  Both alias int32_t; the names are the documentation.
using GlobalRank = int32_t;
using InGroupRank = int32_t;

using GroupId = std::string;  // process group ID (from PyTorch
                              // DistributedBackendOptions::group_id)

constexpr GlobalRank kInvalidGlobalRank = -1;
inline const GroupId kInvalidGroupId = "";
constexpr int kMaxNumRanks = 64;

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
               status == GroupMemberStatus::kInactive;
    }
    bool hasLeft() const { return status == GroupMemberStatus::kLeft; }
};

// Group lifecycle status.
//
//   registerGroup()
//          |
//          v
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
    GroupId group_id;
    GroupStatus status = GroupStatus::Bootstrapping;
    uint64_t epoch = 0;
    std::vector<GlobalRank> rank_order;  // InGroupRank → GlobalRank
    std::vector<GroupMember> members;    // indexed by GlobalRank
};

// TransferObservationEvent, worker thread -> Agent queue.
// attempted_ranks / failed_ranks_hint / succeeded_ranks are bit-vectors
// indexed by GlobalRank (size kMaxNumRanks).  Producers must translate
// InGroupRank peers to GlobalRank via rank_order before setting bits.
struct TransferObservationEvent {
    GroupId group_id;
    std::vector<uint8_t> attempted_ranks;
    std::vector<uint8_t> failed_ranks_hint;
    std::vector<uint8_t> succeeded_ranks;
    bool local_success =
        false;  // true iff ALL attempted peers succeeded locally
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_CONTROL_PLANE_TYPES_H
