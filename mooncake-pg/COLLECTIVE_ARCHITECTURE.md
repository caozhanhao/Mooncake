# Collective architecture and review guide

This document describes the first vertical slice of the collective refactor.
It is intentionally narrower than the final design: supported GPU AllReduce
uses the new path, while every other collective and unsupported AllReduce
signature continues to use the legacy worker.

The Torch adapter is not part of this refactor. It keeps calling the
NCCL-shaped `mooncakePg*` C API; dispatch happens inside `MooncakeCommunicator`.

## Overall architecture

There is one call boundary followed by five internal layers. Each layer owns a
different kind of decision; the separation is meant to prevent either the
Coordinator or a captured kernel from becoming a per-rank God object.

```text
 Existing call surface
 +---------------+     +--------------------+     +-------------------------+
 | Torch adapter | --> | mooncake_pg C API  | --> | MooncakeCommunicator    |
 | (unchanged)   |     | raw NCCL-like args |     | incremental dispatch    |
 +---------------+     +--------------------+     +------------+------------+
                                                               |
                         supported GPU SUM + Planned protocol   |
                                                               v
 Control-plane policy                              Group-local data plane
 +-------------------------+     GroupView     +-----------------------------+
 | 1. Policy / Coordinator | ----------------> | 2. Binding                  |
 | per-group/per-size      |                   | GroupView -> stable local   |
 | algorithm choice only   |                   | execution state            |
 +------------^------------+                   +--------------+--------------+
              | endpoint/capability                           |
 +------------+------------+                                  v
 | Agent state machine     |                   +-----------------------------+
 | merges process link     |                   | 3. Runtime / operation      |
 | capability and events  |                   | admission, lane, snapshot,  |
 +------------^------------+                   | invocation ABI, recovery    |
              |                                +--------------+--------------+
 +------------+------------+                                  |
 | Process resources       |                                  v
 | LinkManager (host)      |                   +-----------------------------+
 | DeviceLinkManager       |                   | 4. Executor / algorithm     |
 | BufferManager / proxy   |                   | AllReduce + Flat Ring       |
 +-------------------------+                   | Hierarchical is a shell     |
                                                +--------------+--------------+
                                                               |
                                                               v
                                                +-----------------------------+
                                                | 5. Transport                |
                                                | DevP2p / DevRdma / Host     |
                                                | common put-and-signal       |
                                                +-----------------------------+

 Unsupported signature or Legacy protocol -----------------> legacy worker
```

## Layer responsibilities

| Layer | Main modules | Owns | Must not own |
| --- | --- | --- | --- |
| Dispatch | C API, `MooncakeCommunicator` | External argument validation and the incremental Planned/Legacy split | Rank-specific plans, routes, or algorithm execution |
| Control policy | `CollectivePolicyBuilder`, `CollectivePlanSet` | One canonical algorithm per group and message-size bucket | Per-rank predecessors, addresses, QPs, or transport choice |
| Binding | `GroupPeerResolver`, `CollectiveBindingMaterializer`, `GroupCollectiveBindings` | Convert an authoritative `GroupView` into group-local, typed, graph-stable execution state | Tensor pointers, streams, invocation lifetime, or algorithm execution |
| Runtime and operation | `GroupCollectiveRuntime`, `CollectiveProgressEngine`, `CollectiveInvocation` | Admission, snapshot allocation, eager lease versus graph pin, failure gating, and operation ABI translation | Membership policy or collective-specific scheduling |
| Executor | AllReduce executor and algorithm routines | Participant shards, transport tiles, reduction, and dependency order | `GroupView`, `GlobalRank`, local fallback policy, or CPU control-plane calls |
| Transport/resources | buffer and link managers, host proxy, device transfer | Process-shared memory/link resources and per-edge put-and-signal | Collective algorithm selection |

The five internal layers are policy, binding, runtime/operation, executor, and
transport/resources. `MooncakeCommunicator` is the compatibility boundary in
front of them, not a sixth planning layer.

Resource teardown follows the inverse ownership order: stop progress first,
destroy device mappings/QPs second, then unregister and release the process
collective arena. Materialized P2P and RDMA edges retain their process-level
resource owner, so replacing a view removes old links from resolution without
invalidating an in-flight invocation or captured graph.

## Three representations, three lifetimes

The design deliberately avoids a single executable-plan object.

```text
 Logical plan                  Materialized binding             Invocation
 Coordinator-owned            Agent/rank-owned                 Call-owned
 --------------------------    ------------------------------   -----------------
 protocol                     active participant order         input/output
 size buckets                 local predecessor/successor      count/type/op
 algorithm                    peer routes and handles          stream
                              stable double-buffered state      lane/failure cookie

 same on every rank           rebuilt for each GroupView       created per call
```

Only the logical plan is broadcast by the Coordinator. The Agent derives
neighbors from canonical active order and resolves process-shared links. This
keeps Coordinator work proportional to groups and size buckets rather than to
every rank's executable addresses.

`GroupPeerResolver` is the last place allowed to see `GlobalRank` and
`GroupView`. Everything below it addresses peers by `InGroupRank` and consumes
already materialized bindings.

## Transport agreement

Ranks must agree on the collective algorithm and participant order. They do
not need to use the same physical transport for every directed edge because
all routes implement the same receiver-visible operation: write the remote
arena, then publish its signal.

- `DevP2p` uses the device P2P API without pretending it is NVLink or MNNVL.
- `DevRdma` uses the device RDMA API.
- `Host` enqueues the same put-and-signal operation through the host proxy;
  Transfer Engine owns its installed physical transport choice.

Route selection therefore stays in Agent-side binding. It must fail closed if
a selected local binding cannot be materialized; an executor must never
silently choose another route or protocol.

`DeviceLinkManager` owns concrete observations by `(local device, global
peer)` and resolves a binding for the communicator's exact device. Only its
aggregate process-level Device reachability is reported to the Agent state
machine. A Host failure from the still-supported legacy data plane
conservatively invalidates both provider contributions, so Device reachability
cannot hide a failed legacy collective during the incremental migration.

Once a group has selected the Planned wire protocol, a joining rank must
publish its V2 endpoint before activation. A Legacy group may still admit a
legacy-only rank; it switches to Planned only after every active rank has
published the new endpoint.

## Eager, graph replay, and failure recovery

Eager and CUDA Graph execution use the same invocation and executor protocol.
Their only difference is resource lifetime: eager temporarily leases a lane,
while capture pins a lane at a stable address.

```text
 invocation
    |
    +-- acquire/locate lane (eager lease or graph pin)
    +-- allocate full input snapshot on the caller stream
    +-- launch the same executor
           |
           +-- enter stable failure gate
           +-- load current binding state
           +-- run Coordinator-selected algorithm
           +-- success: complete
           |
           +-- failure: publish evidence and park
                           |
                    CPU progress thread
                           |
                    report + syncAfterFailure
                           |
                    apply new GroupView/binding
                           |
             +-------------+-------------+
             | safe to retry             | not safe to reuse
             v                           v
          gate Open                   gate Closed
          newer view required         survive replay; no arena access
          reload + retry snapshot     later replay may request recovery again
```

The captured kernel sees stable state pointers, lane addresses, and a failure
gate. Membership, algorithm choice, peer addresses, and link resources remain
behind the double-buffered state and can change without graph recapture. The
kernel never infers a fallback algorithm from a failure; it retries only after
CPU has applied a newer, executable Coordinator-authoritative view.

Host admission rejects new collectives after self deactivation. A captured
replay cannot run that host check, so the published execution state carries an
explicit `self_participating` bit. An inactive replay is a transport-free
local identity: it restores the invocation snapshot to output and never
assumes another participant's ordinal. The restore also removes a partial
writeback left by an attempt that failed before self deactivation.

The per-lane sequence belongs to the invocation and remains fixed across its
retry attempts. Sequence counters are monotonic within one view and reset when
a new authoritative view is published, so a reactivated rank does not inherit
sequence history different from the surviving ranks. A newer `view_epoch`
separates the reset token domain, while an in-flight retry keeps the value it
already acquired. This avoids letting locally different CPU recovery timing
create different wire tokens on different ranks. The counters live in mapped
control memory and are reset before active-state publication; view application
does not enqueue GPU work behind the parked executor it is trying to recover.
An invocation-local retry-attempt field separates that parked retry from a
future invocation that starts at the reset sequence value. Reapplying the same
view does not reset counters or create a new token domain.

The full input snapshot is common recovery state, not a graph special case. A
failed attempt may already have partially modified the output. Retrying from
the snapshot avoids depending on partially committed output. Ring planning is
over full participant shards; the fixed registered arena appears only as
internal transport tiling, so copy-in, transfer, reduction, and copy-out can
overlap without exposing a plan-level `window` or `RingChunk` concept.

## First-PR scope

Implemented on the new path:

- GPU AllReduce SUM for float16, bfloat16, and float32;
- one Planned dispatch path for every configured PG GPU backend; compiler and
  runtime selection remains in the existing platform adapters;
- Flat Ring policy and execution;
- per-directed-edge `DevP2p`, `DevRdma`, or `Host` materialization;
- the device API and host proxy path;
- one execution protocol shared by eager calls and CUDA Graph replay;
- failure publication, `syncAfterFailure`, stable view replacement, and retry;
- an explicit but unsupported Hierarchical plan/binding/executor shell.

Kept on the legacy path:

- every other collective;
- CPU AllReduce;
- unsupported GPU AllReduce datatype/reduction signatures.

The signature check is a deterministic migration boundary, not an error
fallback. Once the Coordinator selects the Planned protocol for a supported
signature, local materialization/runtime failure cannot send one rank back to
the legacy wire protocol.

## Deferred, not hidden

The first slice intentionally leaves these follow-ups visible without solving
them prematurely:

- Hierarchical topology/performance policy and implementation;
- multiple independently pinned graph lanes, concurrent graph replay, and a
  graph-destruction callback for reclaiming retained resources;
- device-RDMA endpoint/QP generation and reconnect semantics after an
  uncertain in-flight operation;
- cross-backend build/runtime validation of the shared cuda-alike graph and
  device-source surface;
- a proper odd/even publication protocol for `LinkManager`'s host-link read
  model when segment refresh races a handle lookup;
- more precise failure evidence than conservatively invalidating both host
  and device provider contributions for a failed peer;
- per-replay reset/epoch semantics for caller-visible failed-rank hints in a
  retained CUDA Graph;
- scalable device-RDMA endpoint/QP exchange beyond the current fixed
  process-rank table;
- configurable/dynamic process collective arena sizing;
- a general cross-rank admission/lane agreement for concurrent submissions,
  local lane exhaustion, and control-plane view application racing a
  data-plane invocation;
- a collective-wide success/failure rendezvous that proves every surviving
  participant has parked before an in-flight invocation retries;
- AllGather, Broadcast, and a general hybrid dependency scheduler.

## Suggested review path

The commits are intentionally ordered by dependency. Reviewing them in this
order avoids starting in the Flat Ring kernel before its ownership boundaries
are established.

1. `caece66a` — logical schema and Coordinator policy:
   `collective/plan.h`, `collective/types.h`, and
   `control_plane/collective_policy.*`.
2. `b8315a9b` — process-scoped resources and endpoint publication:
   `collective/endpoint.h`, `collective/buffer/`, both link managers, and the
   host transfer proxy.
3. `427871a4` — the GlobalRank-to-InGroupRank boundary and graph-stable
   publication: `collective/binding/`.
4. `fb947322` — generic runtime/operation contract followed by the AllReduce
   executor: `collective/runtime/`, `collective/operation/`, then
   `collective/executor/algorithm/flat_ring.cuh`.
5. `a0ab94f3` — communicator initialization, view application, recovery, and
   the single incremental dispatch point in `mooncake_communicator.cpp`.
6. `362c6e46` — failure-gate resource safety across later graph replay.
7. `6deca315` — require a strictly newer authoritative binding before retry.
8. `b3e9bdae` — keep wire sequence invocation-scoped across retry attempts.
9. `9ac8dcab` — retain device link owners with materialized bindings and tear
   them down before the process arena.
10. `2d5dc217` — align replay across view changes: inactive ranks execute a
    local identity, view-scoped sequences reset safely, and retry attempts use
    a distinct wire-token domain.
11. `0e69ddd1` — key concrete device mappings and QPs by the communicator's
    local device while keeping one process-level reachability contribution.
12. `e229afa7` — preserve incremental compatibility by requiring V2 endpoints
    for Planned activation and preventing Device reachability from hiding a
    failed legacy Host path.

Finally, inspect the two Transfer Engine files changed for the narrow P2P peer
mapping API. No file under `mooncake-pg/torch/` should differ from the
decoupling baseline.
