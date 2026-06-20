# Reference Cases

A **case** is a small JSON file that declares the intended decision path for one scenario. It is the
rubric the scorer grades a trace against. Cases are plain data — a directory of JSON files with no code.

---

## Shape

```json
{
  "case_id": "string (required)",
  "description": "string (optional, human label)",
  "reference_path": [
    {
      "tool": "string (required)",
      "args": { "key": "any value" }
    }
  ]
}
```

### Fields

| Field | Required | Description |
|---|---|---|
| `case_id` | yes | Stable identifier. Appears in the Scorecard output. |
| `description` | no | Human-readable label for the scenario. Ignored by the scorer. |
| `reference_path` | yes | Ordered list of expected actions. May be empty (see conventions below). |
| `reference_path[].tool` | yes | The expected tool or subagent dispatch name (see naming below). |
| `reference_path[].args` | no | Expected argument keys for this call (see param_name scoring). |

---

## Tool naming

### Direct tool calls

Use the tool's canonical name as it appears in `gen_ai.tool.name` in the trace:

```json
{ "tool": "Read" }
{ "tool": "Edit" }
{ "tool": "Bash" }
```

Names are matched exactly (case-sensitive).

### Subagent dispatches

An `agent` step in the trace represents a subagent dispatch. Reference it as `agent:<type>`, where
`<type>` is the value of the `agent.type` attribute in the step:

```json
{ "tool": "agent:code-reviewer" }
{ "tool": "agent:python-reviewer" }
```

This naming convention lets cases distinguish between different subagent types dispatched in the same
run.

---

## The `args` field

`args` is optional. When present, it enables the `param_name` scoring axis for that reference node.

**The scorer reads only the keys of `args`, never the values.** Values are placeholders — they exist
so the JSON is valid but they do not affect the score. This means:

- Scrubbed traces (where argument values are replaced with `[SCRUBBED]`) score identically to
  unscubbed traces.
- You can use any value (e.g., `"anything"`, `""`, `null`) as a stand-in.

A case node without `args` does not contribute to `param_name` scoring. When no node in the
`reference_path` declares `args`, the `param_name` axis is absent from the composite entirely.

---

## Conventions

**Order matters for ordering.** The `reference_path` list is the expected execution order. `ordering`
(edge F1) compares consecutive-pair edges, so `[Read, Edit, Bash]` and `[Bash, Read, Edit]` score
differently on the ordering axis even if their selection scores are identical.

**Repeated tools are supported.** A reference path of `["Bash", "Bash"]` expects Bash to be called
twice; one realized Bash call would score `recall = 0.5` on selection.

**Empty reference path.** An empty `reference_path` paired with an empty realized path is a perfect
score (1.0). An empty reference path with a non-empty realized path scores 0 on selection and
edit_similarity.

---

## Worked examples

### Example 1 — selection + ordering only

A case for a rate-limit patch: read the config, apply the edit, run the tests.

```json
{
  "case_id": "rate-limit-ideal",
  "description": "Ideal path: read config, edit middleware, verify with Bash",
  "reference_path": [
    { "tool": "Read" },
    { "tool": "Edit" },
    { "tool": "Bash" }
  ]
}
```

No `args` on any node, so `param_name` is `null` in the Scorecard and does not affect `overall`.

Score this case:

```sh
plumbline score run.plumbline.json cases/rate-limit-ideal.json
```

### Example 2 — with param_name scoring and a subagent dispatch

A stricter case that also checks argument keys and expects a reviewer subagent.

```json
{
  "case_id": "rate-limit-with-review",
  "description": "Read config, dispatch reviewer, patch middleware, run tests",
  "reference_path": [
    { "tool": "Read",             "args": { "file_path": "" } },
    { "tool": "agent:code-reviewer" },
    { "tool": "Edit",             "args": { "file_path": "", "old_string": "", "new_string": "" } },
    { "tool": "Bash",             "args": { "command": "" } }
  ]
}
```

The scorer checks:

- **selection** — did the agent call Read, a code-reviewer subagent, Edit, and Bash?
- **ordering** — in that order?
- **edit_similarity** — how close is the sequence overall?
- **param_name** — did the Read call carry `file_path`? Did Edit carry `file_path`, `old_string`, and
  `new_string`? Did Bash carry `command`? (Values are ignored.)

Score only the reviewer subagent's path instead of the main agent:

```sh
plumbline score run.plumbline.json cases/rate-limit-with-review.json --subagent agent_rev1
```

---

## Organizing a case corpus

There is no required directory structure. A simple flat layout works well:

```
cases/
  rate-limit-ideal.json
  rate-limit-with-review.json
  auth-happy-path.json
```

Run a sweep in shell:

```sh
for case in cases/*.json; do
  plumbline score run.plumbline.json "$case" -o "results/$(basename $case .json).scorecard.json"
done
```
