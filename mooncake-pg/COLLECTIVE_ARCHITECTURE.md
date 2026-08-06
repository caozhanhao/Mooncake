# Collective architecture and review guide

This is the first vertical slice of the collective refactor. Supported GPU
AllReduce uses the planned path. Other collectives, CPU AllReduce, and
unsupported AllReduce signatures continue to use the legacy worker.

The Torch adapter remains unchanged. Dispatch changes only inside
`MooncakeCommunicator`.

## Overall architecture

```text
 Torch adapter -> mooncake_pg C API -> MooncakeCommunicator
                                            |
                         supported planned GPU AllReduce
                                            |
                                            v
                                GroupCollectiveEngine
                                group-scoped owner
                              /                     \
                    applyView                       allReduce
                       |                                |
                       v                                v
          resolveCollectiveView          prepareAllReduceSubmission
          GroupView + LinkManagers        channel + transfer buffer
                       |                                |
                       v                                v
             buildAllReducePlan                CollectiveRuntime
                       |                     lifetime + launch
                       v                                |
              DevicePlan::publish                       v
                                               AllReduce executor
                       |                                |
                       `---------------------->     Flat Ring
                                                        |
                                                        v
                                              PeerRoute transport
                                           DevP2p / DevRdma / Host

 Legacy protocol or unsupported signature ----------------> legacy worker
```

`MooncakeCommunicator` owns one `GroupCollectiveEngine`. It no longer
coordinates plan publication, lanes, runtime, and endpoint state separately.
The engine owns their complete group lifecycle: creation, view application,
submission, execution stop, and close.

Process-level services remain shared across groups:

```text
 MooncakePGContext
 ├── CollectiveBufferPool          registered device arena per device
 ├── HostTransferExecutor          polls communicator command regions
 ├── DeviceLinkManager             process-owned device links
 └── LinkManager                   process-owned Host TE links

 GroupCollectiveEngine
 ├── GroupEndpointV2
 ├── CollectiveChannels            fixed signals, status and Host commands
 ├── DevicePlan<AllReducePlan>
 └── CollectiveRuntime
      └── CollectiveMonitor
```

Only the large registered transfer buffers are time-shared across groups.
Peer signals, mapped status blocks, Host commands, and invocation sequences
have stable communicator lifetime. `HostTransferExecutor` executes commands
from registered communicator regions but does not own their storage.

## Control policy and local resolution

The Coordinator publishes only decisions that must agree across ranks:

- planned versus legacy protocol;
- message-size buckets;
- the collective algorithm once more than one implementation exists.

It does not generate per-rank neighbors, addresses, QPs, or physical transport
choices. The first slice has only Flat Ring, so no unsupported algorithm value
is part of the policy or executable plan. Hierarchical policy and dispatch
will be added together with an executable implementation.

Each Agent resolves one authoritative `GroupView` into:

```text
 ResolvedCollectiveView
 ├── epoch
 ├── active_order                 InGroupRank only
 ├── self_ordinal
 └── peer_routes[InGroupRank]     borrowed from process LinkManagers
```

This is the boundary below which collective code does not see `GlobalRank`,
endpoints, `GroupView`, or link-manager APIs. Operation builders assign roles
such as predecessor/successor, parent/children, or local leader from this
shared projection.

Adding AllGather does not require another framework layer. It adds one
parallel feature module and explicit typed state to the existing group owner:

```text
 collective/allgather/
 ├── allgather.h                  request + executable plan + launch API
 ├── allgather.cpp                validation + plan construction
 ├── allgather.cu                 executor
 └── ring/tree algorithm routine

 GroupCollectiveEngine
 ├── DevicePlan<AllReducePlan> allreduce_plan
 ├── DevicePlan<AllGatherPlan> allgather_plan
 ├── allReduce(AllReduceRequest)
 └── allGather(AllGatherRequest)
```

`applyView()` resolves membership and routes once, then explicitly builds and
publishes each operation's typed plan. The finite list is intentionally visible
instead of hidden behind a virtual builder registry. `CollectiveRuntime`,
collective monitoring, and transport code remain shared. Each operation owns
the preparation of any algorithm-specific transfer memory it needs.

Ranks agree on algorithm and participant order. They do not need identical
physical transports for a directed edge: `DevP2p`, `DevRdma`, and `Host` all
provide the same receiver-visible put-and-signal operation. The executor uses
the already materialized route and never changes transport or algorithm after
launch.

## Typed plan publication and CUDA Graph

The Coordinator publishes `AllReducePolicy`; `buildAllReducePlan()` combines it
with the resolved local view and returns the executable `AllReducePlan`.
`DevicePlan<T>` owns only its stable device-visible value; it has no GroupView,
invocation state, or operation-building responsibilities. Physical channels
own the invocation sequence shared by every collective operation on that
channel, so
adding an operation cannot create an overlapping wire-token domain.

```text
 GroupView + LinkManagers
          |
          v
 ResolvedCollectiveView
          |
          v
 buildAllReducePlan
          |
          v
 DevicePlan<AllReducePlan>::publish
          |
          `------ stable device address ------> captured executor
```

View application overwrites the value at that address. Every later eager
invocation or graph replay reads the current plan, so membership roles and
routes are not frozen into captured kernel arguments. If local materialization
fails, an explicit invalid executable plan is published before host submission
is rejected; captured replay therefore cannot continue using the preceding
view.

This slice assumes view application is serialized with collective execution.
That is a temporary internal contract, not a user guarantee. A future
control-plane quiescing phase must stop host admission, hold graph replay at a
device-visible gate, wait for old plan/link consumers, publish the replacement
view, and release the group. No partial quiescing mechanism is implemented in
this PR.

## Submission and resource lifetime

The operation owns resource requirements. Current bulk AllReduce chooses its
registered staging/protocol layout and acquires a buffer from the process-level
pool. This keeps a future low-latency protocol free to use communicator-fixed
memory without adding branches to the common runtime.

The runtime retains only responsibilities shared by submitted operations:

- serialize host admission;
- ask operation code to prepare a submission only when one is needed;
- retain graph submissions by capture id;
- record eager completion;
- observe and acknowledge device failure reports.

There is no virtual `CollectiveInvocation`, resource registry, or common
allocator hierarchy.

```text
 GroupCollectiveEngine::allReduce
          |
          +--> prepareAllReduceSubmission
          |    ├── acquire exact communicator channel
          |    └── lease bulk transfer buffer
          |              |
          |              v
          |      CollectiveSubmission
          |      prepared kernel ABI + lifetime
          |              |
          v              v
 CollectiveRuntime::submit(prepare, launch)
          ├── launchAllReduce
          └── CollectiveMonitor
               ├── eager event retirement
               ├── graph user-object retirement
               └── failure claim / report / acknowledge
```

Eager and graph execution use the same prepared submission and device protocol.
Only retention differs: eager uses a completion event, while graph capture
attaches a GPU user object and keeps the submission until no graph or executable
can reference it. Communicator teardown remains terminal: callers must not
replay or destroy a communicator while its graph execution is in flight.

The pooled bulk buffer and its exact channel are retired together:

```text
 prepared --launch--> submitted --completion proven--> returned
     |                       `--safety unproven--------> quarantined
     `--launch rejected-------------------------------> returned
```

Only the large buffer returns to the process pool for cross-group reuse. A
quarantined buffer range and channel are never silently selected again.

## Failure boundary

There is no collective-level retry. A failed executor:

1. records its failed peer and error;
2. waits for Host acknowledgement;
3. lets `CollectiveMonitor` report the failure and invoke
   sync-after-failure;
4. finishes the failed invocation after the authoritative view is aligned.

The application owns any retry. A later eager invocation or graph replay reads
the newly published plan. Success/failure synchronization with ranks that did
not observe the failure is outside PG scope.

## First-PR scope

Implemented on the planned path:

- GPU AllReduce SUM for float16, bfloat16, and float32;
- Flat Ring plan construction and execution;
- per-peer `DevP2p`, `DevRdma`, or `Host` routes;
- device API and Host transfer execution paths;
- one eager/CUDA Graph execution protocol;
- communicator-fixed control channels and cross-group bulk-buffer reuse;
- failure publication and sync-after-failure.

Deferred:

- Hierarchical AllReduce and topology/performance policy;
- AllGather, Broadcast, and general hybrid dependency scheduling;
- low-latency communicator-fixed data paths and receiver-driven bulk paths;
- the control-plane quiescing phase and device replay gate;
- independently pinned concurrent multi-stream graph submissions;
- device-RDMA reconnect semantics after uncertain in-flight work;
- cross-rank concurrent lane admission;
- collective-wide success/failure rendezvous.

## Review path

1. Feature owner and complete happy paths:
   `collective/group_collective_engine.*`.
2. Control-to-local projection:
   `collective/resolved_collective_view.*`, then
   `collective/transport/peer_route.h`.
3. Typed AllReduce request and plan:
   `collective/allreduce/allreduce.*` and `collective/plan/device_plan.h`.
4. Submission lifetime and monitoring:
   `collective/runtime/runtime.*`, `collective_submission.*`,
   `collective_channels.*`, and `collective_monitor.*`.
5. Device execution:
   `collective/device_context.cuh`, `collective/allreduce/flat_ring.cuh`, then
   `collective/transport/`.
6. Incremental dispatch and shutdown integration:
   the planned-collective call sites in `mooncake_communicator.cpp`.

No file under `mooncake-pg/torch/` belongs to this refactor.
