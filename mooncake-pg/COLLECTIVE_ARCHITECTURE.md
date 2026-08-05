# Collective architecture and review guide

This document describes the first vertical slice of the collective refactor.
Supported GPU AllReduce uses the new path. Other collectives, CPU AllReduce,
and unsupported AllReduce signatures continue to use the legacy worker until
later incremental PRs migrate them.

The Torch adapter is outside this refactor. It continues to call the
NCCL-shaped `mooncakePg*` C API; dispatch changes only inside
`MooncakeCommunicator`.

## Overall architecture

```text
 Existing call surface
 +---------------+     +--------------------+     +-------------------------+
 | Torch adapter | --> | mooncake_pg C API  | --> | MooncakeCommunicator    |
 | unchanged     |     | NCCL-shaped args   |     | incremental dispatch    |
 +---------------+     +--------------------+     +------------+------------+
                                                               |
                         supported GPU SUM + Planned protocol   |
                                                               v

 Coordinator            Per-group Agent preparation       Process link state
 +------------------+   +-------------------------------+  +------------------+
 | CollectivePolicy |-->| deriveActiveGroupRanks        |<-| Host LinkManager |
 | algo + buckets   |   | buildGroupPeerRoutes          |  | DeviceLinkManager|
 +------------------+   +---------------+---------------+  +------------------+
                                       |
                              active ranks + peer routes
                                       |
                                       v
                            +-----------------------+
                            | CollectivePlanBuilder |
                            | AllReduce today       |
                            | more plugins later    |
                            +-----------+-----------+
                          |
                          v
               +------------------------+
               | CollectivePlanPublisher|
               | graph-stable plan slots|
               +-----------+------------+
                           |
                           v
               +------------------------+
               | GroupCollectiveRuntime |
               | admission + launch     |
               +-----------+------------+
                           |
                           v
               +------------------------+
               | Executor / algorithm   |
               | Flat Ring AllReduce    |
               +-----------+------------+
                           |
                           v
               +------------------------+
               | PeerRoute transport    |
               | DevP2p/DevRdma/Host    |
               +------------------------+

 Unsupported signature or Legacy protocol -----------------> legacy worker
```

The Coordinator publishes only decisions that must be identical on every
rank. Each Agent turns the authoritative view into ordered active group ranks
and peer routes. Collective-specific builders then construct typed kernel
plans without seeing `GroupView`, `GlobalRank`, or either link manager.

## Responsibilities

| Area | Main modules | Owns | Must not own |
| --- | --- | --- | --- |
| Dispatch | C API, `MooncakeCommunicator` | Argument validation and Planned/Legacy split | Algorithm construction or transport selection |
| Control policy | `CollectivePolicyBuilder`, `CollectivePlanSet` | Canonical algorithm and message-size buckets | Per-rank neighbors, addresses, QPs, or transport choice |
| Active ranks | `ActiveGroupRanks` | Active `InGroupRank` order and local ordinal | Global ranks, endpoints, or routes |
| Peer routes | `GroupPeerRoutes`, host/device link managers | Endpoint-to-route resolution and device-link lifetime | Ring/tree roles or collective algorithms |
| Plan construction | `CollectivePlanBuilder`, typed builders | Algorithm-specific peer roles and typed kernel plans | `GroupView`, `GlobalRank`, or link-resource ownership |
| Plan publication | `CollectivePlanPublisher`, `CollectivePlanHandle` | Stable double-buffered publication visible to graph replay | Invocation buffers, streams, or algorithm execution |
| Runtime/operation | `GroupCollectiveRuntime`, `CollectiveInvocation` | Admission, resource acquisition, kernel launch, and graph retention | Membership, algorithm policy, or eager retirement |
| Host progress | `CollectiveHostProgress`, `CollectiveFailureHandler` | Failure report/sync/ack and eager completion retirement | Transport progress, graph ownership, or pool allocation policy |
| Executor | typed executor and algorithm routines | Reduction, dependency order, participant shards, transport tiling | CPU policy calls or local algorithm fallback |
| Transport | `PeerRoute`, device transfer, host proxy | One-sided put-and-signal through the selected route | Collective algorithm selection |

## GroupView projection and plan construction

There is no generic Group object between control and data plane. View
application uses two concrete projections:

```text
 GroupView
    |
    +--> deriveActiveGroupRanks()
    |        |
    |        `--> ActiveGroupRanks
    |             - active InGroupRank order
    |             - optional local ordinal
    |
    `--> buildGroupPeerRoutes() <---- Host/Device LinkManager
             |
             `--> GroupPeerRoutes[InGroupRank]
                  - PeerRoute kernel value
                  - typed owner of referenced device resources
```

`ActiveGroupRanks` is a small value shared by every collective builder.
It is the active projection of Coordinator-authoritative membership. An absent
local ordinal represents an inactive local rank, which a captured graph must
handle without host admission.

`GroupPeerRoutes` is built once for a communicator when a view is applied. It
translates `InGroupRank` to `GlobalRank` only while consulting endpoints and
process link managers. Everything below this boundary stays group-local. The
table owns imported P2P mappings and device RDMA transports referenced by its
POD `PeerRoute` entries. A published plan slot retains the route table, so its
raw kernel addresses cannot outlive their concrete owner.

This is local O(group size) work, not Coordinator-side per-rank planning and
not a global capability matrix. Builders use only the routes required by their
algorithm:

```text
 Flat Ring     predecessor + successor
 Tree          parent + children
 Hierarchical  local peers + leaders
 Broadcast     parent + children
```

The first slice registers `AllReducePlanBuilder`. Future AllGather and
Broadcast builders consume the same active-rank and route projections without
duplicating rank-space or transport code.

## Representations and lifetimes

```text
 Logical policy             Group-local routes          Kernel plan
 Coordinator-owned         Agent/communicator-owned    collective-specific
 ----------------------    --------------------------   ----------------------
 protocol                  active InGroupRank order     algorithm dispatch
 message-size buckets      endpoint-selected routes    peer roles
 algorithm                 device resource owners      typed executor fields

 same on every rank        rebuilt per GroupView       rebuilt per GroupView
```

Invocation state remains separate:

```text
 input/output + count/type/op + stream
                    |
                    v
       GroupCollectiveRuntime admission
                    |
                    v
       CollectiveKernelArgs + typed arguments
                    |
                    v
                   kernel
```

Tensor pointers and invocation buffer offsets never enter the published
kernel plan. A process-wide buffer pool may assign different offsets on each
rank, so the kernel exchanges invocation-local offsets through the common
peer-buffer protocol before running the selected algorithm.

## Transport agreement

Ranks agree on the collective algorithm and participant order. They do not
need identical physical transports for every directed edge: `DevP2p`,
`DevRdma`, and `Host` implement the same receiver-visible operation—write the
remote arena and then publish its signal.

Route selection therefore remains local to view application:

- `DevP2p` uses the device P2P API without inventing NVLink/MNNVL knowledge.
- `DevRdma` uses the device RDMA API.
- `Host` sends the same operation through the host proxy; Transfer Engine owns
  its installed physical transport.

The executor receives one already selected `PeerRoute`. It never switches
transport or algorithm after launch. If a required local route is unavailable,
the builder publishes an invalid plan and the planned path fails closed.

## Eager, CUDA Graph, and failure handling

Eager and CUDA Graph use the same invocation, runtime, executor, and transport
protocol. Their difference is resource lifetime: eager resources return to the
pool after completion, while a captured invocation retains its resources until
communicator teardown.

```text
                         observes failures
                                ^
                                |
                    CollectiveHostProgress
                       /                 \
            completion owner         failure observer
                    /                       \
           EagerSubmission             GraphResources
           resource + event       capture id + pinned resource
                    |                       |
             event completes        communicator teardown
                    |                       |
                    `----------> pool <-----'
```

`CollectiveHostProgress` does not move collective data. Device kernels and the
host-transfer proxy own transport progress. The host loop observes mapped
failure controls for both lifetimes, while only eager submissions enter its
completion queue. Resource retirement is performed after removing a completed
submission from the progress lock; the resource pools own the actual return
and reuse policy.

```text
 invocation / graph replay
          |
          +--> acquire or locate retained runtime resources
          +--> read current CollectivePlanHandle active slot
          +--> run current Coordinator-selected plan
                    |
             +------+------+
             |             |
          success       failure
             |             |
          complete      publish failed peer and wait
                           |
                  CollectiveHostProgress
                           |
                    report + syncAfterFailure
                           |
                    apply authoritative GroupView
                           |
                    acknowledge failed invocation
                           |
                    invocation returns failure
```

There is no collective-level retry. A failed invocation reports the failure
and finishes after sync-after-failure aligns the observed view. The application
owns any retry. A later eager invocation or graph replay reads the newly
published active plan slot without graph recapture.

Host admission rejects a new collective after self deactivation. Replay has no
host participation, so the published AllReduce plan carries an explicit local
participation state. An inactive replay performs a local identity operation and
does not touch peer transport.

Per-lane invocation sequences are monotonic within one view and reset before a
new-epoch plan is published. The view epoch and sequence form the wire-token
domain. Reapplying the same view does not reset the counters.

## First-PR scope

Implemented on the new path:

- GPU AllReduce SUM for float16, bfloat16, and float32;
- Flat Ring policy and execution;
- per-peer `DevP2p`, `DevRdma`, or `Host` routes;
- device API and host proxy paths;
- one resource and execution protocol for eager and CUDA Graph;
- failure publication and sync-after-failure;
- an explicit but unsupported Hierarchical plan/executor shell.

Kept on the legacy path:

- every other collective;
- CPU AllReduce;
- unsupported GPU AllReduce datatype/reduction signatures.

The signature check is an incremental migration boundary, not an error
fallback. Once the Coordinator selects Planned for a supported signature, a
local plan/runtime error cannot move only one rank back to the legacy wire
protocol.

## Deferred, not hidden

- Hierarchical topology/performance policy and implementation;
- AllGather, Broadcast, and a general hybrid dependency scheduler;
- graph-destruction callbacks and independently pinned concurrent graph lanes;
- device-RDMA reconnect/generation semantics after uncertain in-flight work;
- scalable device-RDMA endpoint exchange beyond the current process-rank
  table;
- cross-rank admission/lane agreement for concurrent submissions;
- collective-wide success/failure rendezvous;
- configurable process collective arena sizing;
- more precise failure evidence than invalidating both host and device
  contributions for a failed peer.

## Suggested review path

1. Coordinator-owned logical policy:
   `collective/plan/logical_plan.h`, `control_plane/collective_policy.*`.
2. Serializable endpoint descriptors and process link state:
   `collective/endpoint.h`, `control_plane/device_link_manager.*`,
   `control_plane/link_manager.*`.
3. The control-to-local boundary:
   `collective/plan/active_group_ranks.*`,
   `collective/route/group_peer_routes.*`, and
   `collective/transport/peer_route.h`.
4. Collective-specific plan construction and generic publication:
   `collective/plan/allreduce_plan_builder.*`,
   `collective/plan/collective_plan_registry.*`, and `plan_handle.cuh`.
5. Common invocation resources and failure progress:
   `collective/runtime/`, then `collective/operation/allreduce.*`.
6. Kernel dispatch, Flat Ring, and route transports:
   `collective/executor/allreduce.cu`,
   `collective/executor/algorithm/flat_ring.cuh`, and
   `collective/transport/`.
7. Finally review communicator initialization, view application, and the one
   incremental AllReduce dispatch point in `mooncake_communicator.cpp`.

No file under `mooncake-pg/torch/` belongs to this refactor.
