#include "collective/plan/active_group_ranks.h"

#include <cstddef>

#include "control_plane/control_types.h"

namespace mooncake {

ActiveGroupRanks deriveActiveGroupRanks(const GroupView& view,
                                        InGroupRank self_in_group_rank) {
    ActiveGroupRanks active_ranks;
    active_ranks.ordered.reserve(view.rank_order.size());
    for (size_t index = 0; index < view.rank_order.size(); ++index) {
        const auto global_rank = view.rank_order[index];
        if (!view.members[global_rank].isActive()) continue;
        if (static_cast<InGroupRank>(index) == self_in_group_rank) {
            active_ranks.self_ordinal =
                static_cast<uint32_t>(active_ranks.ordered.size());
        }
        active_ranks.ordered.push_back(static_cast<InGroupRank>(index));
    }
    return active_ranks;
}

}  // namespace mooncake
