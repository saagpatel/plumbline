# Span-to-Test Generator

`plumbline span-to-test` reduces one failing Plumbline span subtree to a compact,
deterministic, synthetic replay fixture. The fixture is inert JSON: it preserves
the structural failure signal without retaining executable commands, provider
requests, credentials, or transcript content.

Machine contracts:

- [`SpanToTestGenerationRequestV1`](schema/span-to-test-generation-request.schema.json)
- [`SanitizedReplayFixtureV1`](schema/sanitized-replay-fixture.schema.json)
- [`SpanToTestReductionReceiptV1`](schema/span-to-test-reduction-receipt.schema.json)

## Claim ceiling

**Structural reproduction is not proof of provider or production reproduction.**

A green generated regression test proves that Plumbline's local parser,
validator, reduction algorithm, and structural assertion agree on the inert
fixture. It does not prove that a model provider, agent runtime, tool server,
network, credential, production environment, or real user workflow will
reproduce the source failure.

## Threat model

The source trace is untrusted and may contain private prompts, completions,
commands, tool arguments/results, home paths, environment values, URLs, emails,
tokens, exception text, binary material, oversized strings, arbitrary nested
attributes, malformed links, or cycles.

Controls:

- Source trace and span identifiers become deterministic, namespaced SHA-256
  pseudonyms. No raw identity map is emitted.
- Prompts, completions, rationales, summaries, exception messages, commands,
  tool values/results, unknown attributes, and run workspace/model data are
  removed. Each removed leaf gets a pseudonymous transformation entry.
- Tool calls become inert descriptors containing only operation, tool name,
  argument-key shape, and result kind. A fixture cannot carry an argument value.
- URLs, email addresses, home/environment paths, token shapes, binary-looking
  data, control bytes, and oversized strings are classified in the receipt and
  excluded from the fixture.
- Source timestamps are replaced with monotonically increasing source-order
  sequence numbers.
- Missing parents, missing causal targets, duplicate IDs, dependency cycles,
  absent failure signals, oversized source files, symlink inputs, and malformed
  JSON fail closed.
- Every file write needs an explicit path. Existing files are refused unless
  `--force` is supplied; final-component symlinks are always refused; output
  parents are never created implicitly; source/output symlink and hardlink
  aliases are refused.
- The optional pytest skeleton imports only Plumbline's local fixture loader and
  structural assertion evaluator. It contains no captured source value and can
  only refer to a sibling JSON fixture.

Non-goals: anonymizing arbitrary quasi-identifiers, replaying provider behavior,
executing a captured tool, recreating production state, or proving that a source
trace was complete or truthful. A receipt records deterministic transformations;
it is not an independent privacy review.

## Reduction algorithm

1. Validate the request and source trace; index unique span IDs and reject
   broken references or cycles.
2. Resolve exactly one selector:
   - `trace`: exact `run_id`, then the first supported failure signal;
   - `span`: exact `step_id`;
   - `finding`: exact `finding.id`, `finding_id`, or
     `plumbline.finding.id` attribute;
   - `outcome`: exact run outcome or span status.
3. Start with the selected span's parent-child subtree.
4. Add its complete parent ancestry, `caused_by` dependencies, hook target
   dependencies, and causally dependent failure-evidence spans. Do not add
   unrelated siblings merely because a retained ancestor is their parent.
5. Preserve source ordering as `sequence`; pseudonymize topology joins.
6. Project each span onto the fixed structural allowlist. Convert tool calls to
   inert descriptors and record every removed or replaced leaf.
7. Emit a local expected assertion plus the fixed no-capabilities safety marker,
   validate the result again, and write only preflighted explicit outputs.

This is deliberately separate from P07 ownership: P07 owns ancestry and gap
detection. P10 consumes trace topology but owns only reduction, sanitization,
fixture generation, and local replay-test scaffolding.

## Contracts by example

Generation request:

```json
{
  "schema_version": "SpanToTestGenerationRequestV1",
  "selector": { "kind": "span", "value": "failing-tool" }
}
```

The generated fixture contains pseudonymous joins and an inert descriptor:

```json
{
  "schema_version": "SanitizedReplayFixtureV1",
  "spans": [
    {
      "span_id": "span_cc9229ca6138c2231260",
      "kind": "tool_call",
      "status": "error",
      "attributes": {
        "tool": {
          "operation": "execute_tool",
          "name": "Shell",
          "argument_keys": ["command"],
          "result_kind": "shell",
          "descriptor_mode": "inert"
        }
      }
    }
  ],
  "safety": {
    "artifact_mode": "inert_data_only",
    "contains_executable_payloads": false
  }
}
```

The complete contract also requires resource metadata, topology fields,
expected assertion, and the fixed prohibited-capability list. See the golden
fixture in `tests/fixtures/span_to_test/`.

## Five-minute local demo

The checked-in source is synthetic and intentionally contains sentinel secrets
to prove they do not survive generation.

```sh
uv run plumbline span-to-test \
  tests/fixtures/span_to_test/failing-trace.json \
  --span failing-tool \
  --output demo-fixture.json \
  --receipt-output demo-receipt.json \
  --pytest-output test_demo_fixture.py

uv run pytest -q test_demo_fixture.py
```

Repeated generation to fresh explicit paths produces byte-identical fixture and
receipt JSON. Reusing an existing path fails unless `--force` is explicit.

Other selector examples:

```sh
plumbline span-to-test trace.json --trace RUN_ID -o fixture.json
plumbline span-to-test trace.json --finding FINDING_ID -o fixture.json
plumbline span-to-test trace.json --outcome failed -o fixture.json
```

Without `--receipt-output`, the receipt is printed to stdout. The fixture still
requires `--output`; no default artifact path exists.

## OpenTelemetry standards snapshot

As of **2026-08-11**, the official GenAI conventions are in the separate
[`open-telemetry/semantic-conventions-genai`](https://github.com/open-telemetry/semantic-conventions-genai)
repository. This design was checked against main commit
[`8d3e4a0f3c34a46f6edb9c71e8666e02e6bf3958`](https://github.com/open-telemetry/semantic-conventions-genai/tree/8d3e4a0f3c34a46f6edb9c71e8666e02e6bf3958)
and core Semantic Conventions
[`v1.44.0`](https://opentelemetry.io/docs/specs/semconv/).

Standard requirements and status:

- GenAI conventions are **Development** status.
- The `execute_tool` internal span uses
  `gen_ai.operation.name=execute_tool` and requires `gen_ai.tool.name`.
- `gen_ai.tool.call.arguments` and `gen_ai.tool.call.result` are opt-in content.
- `error.type` is the conditionally required, low-cardinality error class for
  failed operations; free-form exception messages may contain sensitive data.

Design inferences:

- Because captured tool content is opt-in and unsafe for a portable regression
  artifact, P10 keeps only the argument-key shape and inert tool descriptor.
- Because `error.type` is structural and intended to be low-cardinality, P10
  preserves a bounded safe value (or a pseudonym), while removing exception text.
- Plumbline topology and the local expected assertion are product-owned overlay
  behavior, not OTel requirements.

Local fixture behavior:

- No OTLP export, SDK instrumentation, provider request, network connection,
  shell launch, filesystem mutation replay, or credential lookup occurs.
- The fixture's `service.name` is always `plumbline.synthetic-replay`; it is not
  copied from the source environment.
- Development-stage OTel conventions can change. The pinned commit above is the
  review baseline, not a promise of future compatibility.
