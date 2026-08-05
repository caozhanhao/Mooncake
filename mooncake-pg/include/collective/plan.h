#ifndef MOONCAKE_PG_COLLECTIVE_PLAN_H
#define MOONCAKE_PG_COLLECTIVE_PLAN_H

#include <cstdint>
#include <limits>
#include <variant>
#include <vector>

namespace mooncake {

// Coordinator-owned logical policy. Plans contain only choices that must be
// identical on every rank. Membership roles, routes, and local resources are
// materialized by each Agent from the authoritative GroupView.
struct FlatRingPlan {
    bool operator==(const FlatRingPlan&) const = default;
};

// The first vertical slice does not select this algorithm. Keeping the plan
// alternative exposes the algorithm boundary required by failure recovery and
// future topology-aware policy without adding an incomplete implementation.
struct HierarchicalPlan {
    bool operator==(const HierarchicalPlan&) const = default;
};

using AllReduceAlgorithmPlan = std::variant<FlatRingPlan, HierarchicalPlan>;

struct AllReducePlan {
    uint64_t max_message_bytes = std::numeric_limits<uint64_t>::max();
    AllReduceAlgorithmPlan algorithm;

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

#endif  // MOONCAKE_PG_COLLECTIVE_PLAN_H
