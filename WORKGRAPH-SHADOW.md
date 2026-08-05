# WorkGraphV1 passive shadow adapter

`WorkGraphShadowTraceV1` compares a preregistered, digest-bound compiled WorkGraphV1 plan with
metadata-only observed lane events. It is an observer. It does not create tasks, dispatch agents,
acquire leases, retry work, rewrite dependencies, mutate BridgeDB, or enforce runtime transitions.

Machine output contract:
[`schema/workgraph-shadow-trace.schema.json`](schema/workgraph-shadow-trace.schema.json).
Machine input contract:
[`schema/workgraph-observed-events.schema.json`](schema/workgraph-observed-events.schema.json).
Prospective registration contract:
[`schema/workgraph-pilot-registration.schema.json`](schema/workgraph-pilot-registration.schema.json).

## Prospective evidence only

A workflow must be compiled and frozen before dispatch. Retrofitting a graph after agents have already
run is post-hoc evidence and is not an admissible pilot. A minimal prospective sequence is:

1. Compile the graph and freeze the exact compiled-plan SHA-256.
2. Freeze a separate registration receipt with its timestamp, source/plan digests,
   lane-to-agent assignments, dependencies, waves, and mutation modes.
3. Observe `started` plus one terminal event (`completed`, `failed`, or `blocked`) per lane.
4. Start synthesis only after all declared discovery dependencies terminate.
5. Start independent audit only after synthesis terminates.
6. Bind the event document to the exact registration SHA-256 and recompute the report
   from all three exact files.

The adapter verifies local digest and timestamp ordering, including that no event
predates registration. It does not provide an external timestamp authority or
cryptographic notarization; coordinated rewriting of all local evidence remains outside
the claim.

## Observed event contract

`WorkGraphObservedEventsV1` contains the graph ID, exact compiled-plan and registration
digests, and events with:

- stable event and lane IDs;
- agent path;
- `started`, `completed`, `failed`, or `blocked` transition;
- timezone-aware observation timestamp;
- evidence references;
- mutation observation;
- optional duplicate signature;
- optional token/cost indicators;
- retry count and explicit unknown reason.

No prompt, response, command output, or tool payload belongs in this document. Identifiers are opaque,
single-line metadata capped at 128 characters; agent paths, evidence references, and duplicate
signatures use an opaque reference grammar capped at 256 characters. The optional unknown reason and
other operator-facing metadata are single-line and capped at 512 characters. The Python validator and
JSON schemas enforce the same structural boundary and reject duplicate evidence references.

## Dispositions

- `GO`: every declared lane maps once, has exactly one start and terminal transition, dependencies are
  respected, read-only lanes do not mutate, and no duplicate-work or unsafe-transition finding exists.
- `NO_GO`: an observed failure, block, undeclared lane/mutation, dependency violation, mapping drift,
  duplicate work, or impossible timestamp is present.
- `UNKNOWN`: terminal coverage or terminal evidence is incomplete. Missing proof never renders green.

The adapter recomputes lane coverage, evidence completeness, wall time, same-wave serialization,
duplicate groups, failures, blocks, retries, and token/cost totals when complete. Missing token or cost
data remains `null` rather than being treated as zero.

```sh
plumbline workgraph-shadow compiled-plan.json registration.json observed-events.json --gate
```

`--gate` exits nonzero for both `NO_GO` and `UNKNOWN`.

## Acceptance and falsification

The pilot is falsified by an unregistered agent, missing terminal event, dependency start before its
prerequisite terminates, read-only mutation, duplicate signature, incomplete evidence, or digest drift.
One green pilot supports only the passive adapter contract and that workflow's observations. It cannot
prove schedule optimality, causal improvement, safe automatic dispatch, or production readiness.

## Rollback

Stop producing observed-event documents and remove the passive query from the workflow. The frozen
WorkGraph compiler and agent runtime are unchanged, so rollback requires no scheduler, lease, or data
migration.
