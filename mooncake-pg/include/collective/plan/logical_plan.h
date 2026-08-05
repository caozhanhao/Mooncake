#ifndef MOONCAKE_PG_COLLECTIVE_PLAN_LOGICAL_PLAN_H
#define MOONCAKE_PG_COLLECTIVE_PLAN_LOGICAL_PLAN_H

#include <cstdint>
#include <limits>
#include <vector>

namespace mooncake {

// Coordinator-owned logical policy. Plans contain only choices that must be
// identical on every rank. Membership roles, routes, and local resources are
// resolved by each Agent from the authoritative GroupView.
// The first planned protocol implements Flat Ring for every size bucket. Add
// an algorithm choice here only when a second executable algorithm exists.
struct AllReducePlan {
    uint64_t max_message_bytes = std::numeric_limits<uint64_t>::max();

    bool operator==(const AllReducePlan&) const = default;
};

// Legacy and Planned are different wire protocols. Once a group selects the
// planned path, a temporarily unavailable plan must fail closed rather than
// silently sending one rank back through the legacy worker.
enum class AllReduceProtocol : uint8_t {
    Legacy = 0,
    Planned,
};

struct CollectivePlanSet {
    AllReduceProtocol allreduce_protocol = AllReduceProtocol::Legacy;
    std::vector<AllReducePlan> allreduce_plans;

    bool operator==(const CollectivePlanSet&) const = default;
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_COLLECTIVE_PLAN_LOGICAL_PLAN_H
