"""Record a Claude Code session transcript into a Plumbline trace.

Input: the main `<session>.jsonl` transcript. Subagent sidechains are discovered
at `<session>/subagents/agent-*.jsonl` and merged, tagged by their `agentId`.

The recorder captures the *observable execution layer* — llm turns, tool calls
(with results merged back by `tool_use_id`), subagent dispatch, hook/guardrail
verdicts, mode transitions and compaction boundaries. It does not synthesize
`decision` steps: a meta-decision is a judgement the Phase 2 scorer infers from
this observable path, not a fact the transcript records.

Real transcripts vary in shape; the recorder is defensive about it (see
`_as_dict` and the id fallbacks) so a single malformed event never aborts a run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from plumbline.scrub import scrub_obj

PLUMBLINE_VERSION = "0.1.0"
HARNESS_NAME = "claude-code"
_EPOCH = "1970-01-01T00:00:00Z"
_SUMMARY_MAX = 600  # cap the observed outcome summary so a trace stays bounded
# Final-turn stop_reason -> outcome status. Only an explicit end_turn is a claimed
# completion; everything else stays conservative ("unknown") rather than guessing.
_STOP_REASON_STATUS = {"end_turn": "completed"}

# Tool name -> a coarse tool.result.kind tag (typed-result detail lives in args).
_RESULT_KIND = {
    "Read": "read",
    "Write": "write",
    "Edit": "edit",
    "MultiEdit": "edit",
    "Bash": "bash",
    "Agent": "agent",
    "Glob": "glob",
    "Grep": "grep",
}

JsonObj = dict[str, Any]


def _as_dict(value: Any) -> JsonObj:  # noqa: ANN401 - CC fields are sometimes str, sometimes object
    """Coerce a field that CC emits inconsistently (str or object) to a dict."""
    return value if isinstance(value, dict) else {}


def load_jsonl(path: Path) -> list[JsonObj]:
    """Parse a JSONL transcript into a list of events (streamed line by line)."""
    events: list[JsonObj] = []
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            stripped = raw.strip()
            if stripped:
                events.append(json.loads(stripped))
    return events


def _subagent_files(main_path: Path) -> list[Path]:
    sub_dir = main_path.with_suffix("") / "subagents"
    return sorted(sub_dir.glob("*.jsonl")) if sub_dir.is_dir() else []


def record_session(main_path: Path | str, *, scrub: bool = True) -> JsonObj:  # noqa: C901, PLR0912 - linear event dispatcher; clearer flat than split
    """Normalize a Claude Code session (main + subagents) into a Plumbline trace.

    PII scrubbing is ON by default (see `plumbline.scrub`); pass ``scrub=False`` to
    keep raw values (e.g. for local-only inspection).
    """
    main_path = Path(main_path)
    events = load_jsonl(main_path)
    for sub_file in _subagent_files(main_path):
        events.extend(load_jsonl(sub_file))

    run = _build_run(events)
    steps: list[JsonObj] = []
    tool_steps: dict[str, JsonObj] = {}
    last_ts = run["started_at"]
    last_mode: str | None = None
    mode_seq = 0

    for idx, ev in enumerate(events):
        ts = ev.get("timestamp") or last_ts
        if ev.get("timestamp"):
            last_ts = ev["timestamp"]
        sub_id = ev.get("agentId") if ev.get("isSidechain") else None
        etype = ev.get("type")

        if etype == "assistant":
            steps.append(_llm_step(ev, sub_id, ts, idx))
            for j, block in enumerate(_content(ev)):
                if block.get("type") == "tool_use":
                    block_id = block.get("id") or f"ev{idx}_tu{j}"
                    step = _tool_step(ev, block, sub_id, ts, block_id)
                    steps.append(step)
                    if block.get("id"):
                        tool_steps[block["id"]] = step
        elif etype == "user":
            for block in _content(ev):
                if block.get("type") == "tool_result":
                    _merge_result(tool_steps, block, ev)
        elif etype == "system":
            if step := _system_step(ev, sub_id, ts, idx):
                steps.append(step)
        elif etype in ("permission-mode", "mode"):
            mode_seq += 1
            step, last_mode = _mode_step(ev, etype, last_mode, mode_seq, ts)
            steps.append(step)

    # Stable sort: equal timestamps keep insertion (≈ event) order.
    steps.sort(key=lambda s: s["started_at"])
    trace: JsonObj = {"plumbline_version": PLUMBLINE_VERSION, "run": run, "steps": steps}
    return scrub_obj(trace) if scrub else trace


# --- builders ---------------------------------------------------------------


def _content(ev: JsonObj) -> list[JsonObj]:
    content = _as_dict(ev.get("message")).get("content")
    return content if isinstance(content, list) else []


def _attribution(ev: JsonObj) -> JsonObj | None:
    present = {
        key: ev[src]
        for key, src in (
            ("skill", "attributionSkill"),
            ("mcp_server", "attributionMcpServer"),
            ("mcp_tool", "attributionMcpTool"),
            ("agent", "attributionAgent"),
        )
        if ev.get(src) is not None
    }
    return present or None


def _base_step(step_id: str, kind: str, ev: JsonObj, sub_id: str | None, ts: str) -> JsonObj:
    step: JsonObj = {
        "step_id": step_id,
        "parent_step_id": ev.get("parentUuid"),
        "subagent_id": sub_id,
        "kind": kind,
        "started_at": ts,
    }
    if attribution := _attribution(ev):
        step["attribution"] = attribution
    return step


def _llm_step(ev: JsonObj, sub_id: str | None, ts: str, idx: int) -> JsonObj:
    msg = _as_dict(ev.get("message"))
    usage = _as_dict(msg.get("usage"))
    attrs: JsonObj = {}
    if msg.get("model"):
        attrs["gen_ai.request.model"] = msg["model"]
    if "input_tokens" in usage:
        attrs["gen_ai.usage.input_tokens"] = usage["input_tokens"]
    if "output_tokens" in usage:
        attrs["gen_ai.usage.output_tokens"] = usage["output_tokens"]
    if msg.get("stop_reason"):
        attrs["gen_ai.response.finish_reasons"] = [msg["stop_reason"]]
    if any(b.get("type") == "thinking" for b in _content(ev)):
        attrs["agent.reasoning"] = True
    step = _base_step(ev.get("uuid") or f"ev{idx}", "llm", ev, sub_id, ts)
    step["ended_at"] = ev.get("timestamp")
    step["status"] = "ok"
    step["attributes"] = attrs
    return step


def _tool_step(ev: JsonObj, block: JsonObj, sub_id: str | None, ts: str, block_id: str) -> JsonObj:
    name = block.get("name", "")
    inp = block.get("input") or {}
    if name == "Agent":
        attrs = {"agent.type": inp.get("subagent_type", "unknown")}
        if inp.get("name"):
            attrs["agent.name"] = inp["name"]
        if inp.get("model"):
            attrs["agent.model"] = inp["model"]
        kind = "agent"
    else:
        attrs = {
            "gen_ai.tool.name": name,
            "gen_ai.tool.call.id": block_id,
            "tool.arguments": inp,
            "tool.result.kind": _RESULT_KIND.get(name, "other"),
        }
        kind = "tool_call"
    step = _base_step(block_id, kind, ev, sub_id, ts)
    step["status"] = "ok"
    step["attributes"] = attrs
    return step


def _merge_result(tool_steps: dict[str, JsonObj], block: JsonObj, ev: JsonObj) -> None:
    step = tool_steps.get(block.get("tool_use_id", ""))
    if step is None:
        return
    result = _as_dict(ev.get("toolUseResult"))
    if result.get("interrupted"):
        step["status"] = "interrupted"
    elif block.get("is_error"):
        step["status"] = "error"
    else:
        step["status"] = "ok"
    step["ended_at"] = ev.get("timestamp")
    if step["kind"] == "agent" and result.get("agentId"):
        step["attributes"]["agent.spawns_subagent_id"] = result["agentId"]
        if result.get("resolvedModel"):
            step["attributes"]["agent.model"] = result["resolvedModel"]
        if result.get("totalTokens") is not None:
            step["attributes"]["agent.total_tokens"] = result["totalTokens"]


def _system_step(ev: JsonObj, sub_id: str | None, ts: str, idx: int) -> JsonObj | None:
    subtype = ev.get("subtype") or ""
    if "compact" in subtype or ev.get("isCompactSummary"):
        meta = _as_dict(ev.get("compactMetadata"))
        attrs: JsonObj = {"harness.compaction.reason": meta.get("trigger", "unknown")}
        if meta.get("preTokens") is not None:
            attrs["harness.compaction.tokens_before"] = meta["preTokens"]
        if meta.get("postTokens") is not None:
            attrs["harness.compaction.tokens_after"] = meta["postTokens"]
        step = _base_step(ev.get("uuid") or f"sys{idx}", "compaction", ev, sub_id, ts)
        step["attributes"] = attrs
        return step

    if ev.get("hookInfos") or ev.get("preventedContinuation") is not None or ev.get("hookErrors"):
        infos = ev.get("hookInfos") or []
        info = _as_dict(infos[0]) if infos else {}
        prevented = bool(ev.get("preventedContinuation"))
        verdict = "deny" if prevented or ev.get("hookErrors") else "allow"
        attrs = {
            "harness.hook.name": info.get("name", "unknown"),
            "harness.hook.verdict": verdict,
            "harness.hook.event": info.get("event", subtype or "unknown"),
            "harness.hook.prevented_continuation": prevented,
        }
        if ev.get("toolUseID"):
            attrs["harness.hook.target_step_id"] = ev["toolUseID"]
        step = _base_step(ev.get("uuid") or f"sys{idx}", "hook", ev, sub_id, ts)
        step["caused_by"] = ev.get("toolUseID")
        step["attributes"] = attrs
        return step
    return None


def _mode_step(
    ev: JsonObj, etype: str, last_mode: str | None, seq: int, ts: str
) -> tuple[JsonObj, str]:
    if etype == "permission-mode":
        to, kind_attr = ev.get("permissionMode", "unknown"), "permission_mode"
    else:
        to, kind_attr = ev.get("mode", "unknown"), "mode"
    attrs: JsonObj = {"harness.mode.kind": kind_attr, "harness.mode.to": to}
    if last_mode is not None:
        attrs["harness.mode.from"] = last_mode
    step: JsonObj = {
        "step_id": f"mode_{seq}",
        "parent_step_id": None,
        "subagent_id": None,
        "kind": "mode_change",
        "started_at": ts,
        "attributes": attrs,
    }
    return step, to


def _content_text(ev: JsonObj) -> str | None:
    """The text of a turn, whether ``message.content`` is a plain string (common for
    the first user prompt) or a list of blocks. Joins all text blocks; ``None`` if
    there is no text.
    """
    content = _as_dict(ev.get("message")).get("content")
    if isinstance(content, str):
        return content or None
    parts = [
        block["text"]
        for block in (content if isinstance(content, list) else [])
        if isinstance(block, dict) and block.get("type") == "text" and block.get("text")
    ]
    return "\n".join(parts) or None


def _build_run(events: list[JsonObj]) -> JsonObj:
    run_id = "unknown"
    version = entrypoint = cwd = git_branch = model = plan_text = None
    out_summary = out_stop = None
    started = ended = None
    for ev in events:
        if run_id == "unknown" and ev.get("sessionId"):
            run_id = ev["sessionId"]
        version = version or ev.get("version")
        entrypoint = entrypoint or ev.get("entrypoint")
        cwd = cwd or ev.get("cwd")
        git_branch = git_branch or ev.get("gitBranch")
        if ts := ev.get("timestamp"):
            started = started or ts
            ended = ts
        if ev.get("type") == "assistant" and not ev.get("isSidechain"):
            msg = _as_dict(ev.get("message"))
            model = model or msg.get("model")
            out_summary = _content_text(ev) or out_summary  # keep the last turn's text
            out_stop = msg.get("stop_reason") or out_stop
        if plan_text is None and ev.get("type") == "user" and not ev.get("isSidechain"):
            plan_text = _content_text(ev)

    run: JsonObj = {
        "run_id": run_id,
        "harness": {"name": HARNESS_NAME, "version": version, "entrypoint": entrypoint},
        "started_at": started or _EPOCH,
        "ended_at": ended,
        "model": model,
    }
    if cwd or git_branch:
        run["workspace"] = {"cwd": cwd, "git_branch": git_branch}
    if plan_text:
        run["plan"] = {"source": "user_prompt", "statement": plan_text}
    if out_stop is not None or out_summary is not None:
        run["outcome"] = {
            "status": _STOP_REASON_STATUS.get(out_stop or "", "unknown"),
            "summary": out_summary[:_SUMMARY_MAX] if out_summary else None,
        }
    return run
