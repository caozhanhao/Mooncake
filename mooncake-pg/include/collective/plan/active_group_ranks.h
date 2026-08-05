#ifndef MOONCAKE_PG_COLLECTIVE_PLAN_ACTIVE_GROUP_RANKS_H
#define MOONCAKE_PG_COLLECTIVE_PLAN_ACTIVE_GROUP_RANKS_H

#include <cstdint>
#include <optional>
#include <vector>

#include "collective/types.h"

namespace mooncake {

struct GroupView;

// Ordered active ranks projected from Coordinator-authoritative membership.
// Collective plan builders use this value instead of inspecting GroupView.
struct ActiveGroupRanks {
    std::vector<InGroupRank> ordered;
    std::optional<uint32_t> self_ordinal;
};

ActiveGroupRanks deriveActiveGroupRanks(const GroupView& view,
                                        InGroupRank self_in_group_rank);

}  // namespace mooncake

#endif  // MOONCAKE_PG_COLLECTIVE_PLAN_ACTIVE_GROUP_RANKS_H
