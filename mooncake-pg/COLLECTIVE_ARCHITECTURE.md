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
          resolveCollectiveView                 CollectiveRuntime
          GroupView + LinkManagers              resources + launch
                       |                                |
                       v                                v
             buildAllReducePlan                AllReduce executor
                       |                                |
                       v                                v
              DevicePlan::publish                  Flat Ring
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
 ├── CollectiveControlPool         mapped host/device status blocks
 ├── HostTransferExecutor          mapped commands + Host TE execution
 ├── DeviceLinkManager             process-owned device links
 └── LinkManager                   process-owned Host TE links

 GroupCollectiveEngine
 ├── GroupEndpointV2
 ├── CollectiveLanePool            shared invocation order + peer signals
 ├── DevicePlan<AllReducePlan>
 └── CollectiveRuntime
```

Consequently, invocation buffers, status blocks, and Host commands are reused
across groups. Peer-signal spans and graph-retained invocation resources are
group scoped.

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
resource pools, collective monitoring, and transport code remain unchanged.

Ranks agree on algorithm and participant order. They do not need identical
physical transports for a directed edge: `DevP2p`, `DevRdma`, and `Host` all
provide the same receiver-visible put-and-signal operation. The executor uses
the already materialized route and never changes transport or algorithm after
launch.

## Typed plan publication and CUDA Graph

The Coordinator publishes `AllReducePolicy`; `buildAllReducePlan()` combines it
with the resolved local view and returns the executable `AllReducePlan`.
`DevicePlan<T>` owns only its stable device-visible value; it has no GroupView,
invocation state, or operation-building responsibilities. Physical lanes own
the invocation sequence shared by every collective operation on that lane, so
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

The runtime is collective-neutral because the following responsibilities are
shared by AllReduce, AllGather, Broadcast, and future operations:

- choose an invocation lane;
- acquire the resource bundle;
- retain graph resources by capture id;
- record eager completion;
- observe and acknowledge device failure reports.

Operation code supplies only a typed request, plan pointer, and launch callback.
There is no virtual `CollectiveInvocation` hierarchy.

```text
 GroupCollectiveEngine::allReduce
          |
          v
 CollectiveRuntime::submit(launch)
          |
          +--> CollectiveResourcePool
          |    ├── lane
          |    ├── registered device buffer
          |    ├── mapped status block
          |    └── Host transfer command
          |
          +--> launchAllReduce
          |
          `--> CollectiveMonitor
               ├── eager completion retirement
               └── failure claim / report / acknowledge
```

Eager and graph execution use the same resource and device protocol. Eager
resources are temporary and return after completion. A graph capture pins one
bundle until communicator teardown.

Pool entries have explicit lifecycle states:

```text
 Free -> Acquired -> Free
                 `-> Abandoned
```

`release()` means asynchronous use is proven complete and permits reuse.
`abandon()` means reuse safety is unknown; that entry and its backing arena are
retained for process lifetime. There is no ownerless `in_use/reusable` boolean
combination and no reuse-safety boolean propagated between pools.

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
- failure publication and sync-after-failure.

Deferred:

- Hierarchical AllReduce and topology/performance policy;
- AllGather, Broadcast, and general hybrid dependency scheduling;
- the control-plane quiescing phase and device replay gate;
- graph destruction callbacks and independently pinned concurrent graph lanes;
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
4. Common submission resources and monitoring:
   `collective/runtime/runtime.*`, `resource_pool.*`, and
   `collective_monitor.*`.
5. Device execution:
   `collective/device_context.cuh`, `collective/allreduce/flat_ring.cuh`, then
   `collective/transport/`.
6. Incremental dispatch and shutdown integration:
   the planned-collective call sites in `mooncake_communicator.cpp`.

No file under `mooncake-pg/torch/` belongs to this refactor.
