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

import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from plumbline.recorders.background import mark_pending, resolve_from_output
from plumbline.recorders.test_outcome import parse_test_outcome
from plumbline.recorders.transcript_input import UnsupportedTranscriptFormatError, load_jsonl
from plumbline.scrub import scrub_obj, scrub_text

PLUMBLINE_VERSION = "0.1.0"
HARNESS_NAME = "claude-code"
_EPOCH = "1970-01-01T00:00:00Z"
_SUMMARY_MAX = 600  # cap the observed outcome summary so a trace stays bounded
# Final-turn stop_reason -> outcome status. Only an explicit end_turn is a claimed
# completion; everything else stays conservative ("unknown") rather than guessing.
_STOP_REASON_STATUS = {"end_turn": "completed"}

# Phase 4d text-signal decision detection. Opt-in (the lowest-precision tier): scans
# assistant prose for refuse/escalate language at record time and emits decision steps
# carrying only the kind + a short scrubbed rationale snippet, never the full prose.
_RATIONALE_MAX = 120
# First-person, declining-an-action forms only. Bare "won't" / "refuse" / "declined"
# appear constantly in narrative prose, so require "I [refuse/decline] to ..." or
# "I won't/will not <action verb>" to keep precision up on the lowest-precision tier.
_REFUSE_RE = re.compile(
    r"(?i)\bi(?:'m| am)? (?:(?:refuse|decline|declining) to"
    r"|(?:won['’]t|will not) (?:be )?(?:do|run|delete|drop|remove|push|execute"  # noqa: RUF001
    r"|touch|modify|make|create|change|proceed|implement|add))"
)
_ESCALATE_RE = re.compile(
    r"(?i)(?:could you clarify|can you clarify|please clarify|did you mean"
    r"|need (?:your )?confirmation|wait(?:ing)? for (?:your )?(?:approval|confirmation)"
    r"|before (?:i|you) proceed)"
)

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
IndexedStep = tuple[int, JsonObj]


def _as_dict(value: Any) -> JsonObj:  # noqa: ANN401 - CC fields are sometimes str, sometimes object
    """Coerce a field that CC emits inconsistently (str or object) to a dict."""
    return value if isinstance(value, dict) else {}


def _subagent_files(main_path: Path) -> list[Path]:
    sub_dir = main_path.with_suffix("") / "subagents"
    return sorted(sub_dir.glob("*.jsonl")) if sub_dir.is_dir() else []


def record_session(
    main_path: Path | str, *, scrub: bool = True, infer_text_decisions: bool = False
) -> JsonObj:
    """Normalize a Claude Code session (main + subagents) into a Plumbline trace.

    PII scrubbing is ON by default (see `plumbline.scrub`); pass ``scrub=False`` to
    keep raw values (e.g. for local-only inspection).

    ``infer_text_decisions`` (opt-in, Phase 4d) scans assistant prose for refuse/escalate
    language and emits `decision` steps carrying only the kind + a short scrubbed
    rationale; it never stores the full prose. Lowest-precision tier, off by default.
    """
    main_path = Path(main_path)
    events = load_jsonl(main_path)
    for sub_file in _subagent_files(main_path):
        try:
            events.extend(load_jsonl(sub_file))
        except UnsupportedTranscriptFormatError:
            sys.stderr.write("warning: skipped unsupported sidechain\n")

    run = _build_run(events)
    steps, record_steps = _record_steps(events, run["started_at"])
    if infer_text_decisions:
        steps.extend(_text_decision_steps(events, record_steps))
    _resolve_parents(steps, events, record_steps)
    # Stable sort: equal timestamps keep insertion (≈ event) order.
    steps.sort(key=lambda s: s["started_at"])
    trace: JsonObj = {"plumbline_version": PLUMBLINE_VERSION, "run": run, "steps": steps}
    return scrub_obj(trace) if scrub else trace


def _record_steps(  # noqa: C901, PLR0912 - dispatch transcript record types
    events: list[JsonObj], last_ts: str
) -> tuple[list[JsonObj], dict[int, str]]:
    steps: list[JsonObj] = []
    tool_steps: dict[str, IndexedStep] = {}
    pending: list[IndexedStep] = []
    invocations: dict[tuple[str | None, str], JsonObj] = {}
    record_steps: dict[int, str] = {}
    skipped: Counter[str] = Counter()
    blocking_hooks = _blocking_hooks(events)
    matched_hooks: set[str] = set()
    last_mode: str | None = None
    mode_seq = 0

    for idx, ev in enumerate(events):
        ts = last_ts = ev.get("timestamp") or last_ts
        sub_id = ev.get("agentId") if ev.get("isSidechain") else None
        etype = ev.get("type")

        if etype == "assistant":
            invocation = _record_invocation(ev, _llm_step(ev, sub_id, ts, idx), steps, invocations)
            record_steps[idx] = invocation["step_id"]
            for j, block in enumerate(_content(ev)):
                if block.get("type") == "tool_use":
                    block_id = block.get("id") or f"ev{idx}_tu{j}"
                    step = _tool_step(ev, block, sub_id, ts, block_id)
                    step["parent_step_id"] = invocation["step_id"]
                    steps.append(step)
                    if block.get("id"):
                        tool_steps[block["id"]] = (idx, step)
        elif etype == "user":
            for block in _content(ev):
                if block.get("type") == "tool_result":
                    _merge_result(tool_steps, block, ev, pending)
        elif etype == "system":
            if ev.get("subtype") == "api_error":
                skipped["system.api_error"] += 1
            elif step := _system_step(ev, sub_id, ts, idx, blocking_hooks):
                steps.append(step)
                if (
                    ev.get("subtype") == "stop_hook_summary"
                    and ev.get("toolUseID") in blocking_hooks
                ):
                    matched_hooks.add(ev["toolUseID"])
                if ev.get("uuid"):
                    record_steps[idx] = step["step_id"]
            else:
                skipped["system"] += 1
        elif etype in ("permission-mode", "mode"):
            mode_seq += 1
            step, last_mode = _mode_step(ev, etype, last_mode, mode_seq, ts)
            steps.append(step)
            if ev.get("uuid"):
                record_steps[idx] = step["step_id"]
        else:
            label = (
                scrub_text(etype)
                if isinstance(etype, str) and re.fullmatch(r"[\w.-]{1,80}", etype)
                else "unknown"
            )
            skipped[label] += 1

    skipped["attachment"] -= len(matched_hooks)
    for label, count in sorted((+skipped).items()):
        prefix = "warning: " if label in {"system.api_error", "data-flow-incomplete"} else ""
        sys.stderr.write(f"{prefix}skipped {label}: {count}\n")

    return steps, record_steps


# --- builders ---------------------------------------------------------------


def _record_invocation(
    ev: JsonObj,
    llm: JsonObj,
    steps: list[JsonObj],
    invocations: dict[tuple[str | None, str], JsonObj],
) -> JsonObj:
    message_id = ev["message"].get("id")
    if isinstance(message_id, str) and message_id:
        key = (llm["subagent_id"], message_id)
        if existing := invocations.get(key):
            existing["attributes"].update(llm["attributes"])
            if llm["ended_at"]:
                existing["ended_at"] = llm["ended_at"]
            if llm["status"] == "error":
                existing["status"] = "error"
            return existing
        invocations[key] = llm
    steps.append(llm)
    return llm


def _resolve_parents(
    steps: list[JsonObj], events: list[JsonObj], record_steps: dict[int, str]
) -> None:
    parents = {ev["uuid"]: ev.get("parentUuid") for ev in events if ev.get("uuid")}
    resolved: dict[str, str | None] = {
        events[idx].get("uuid") or step_id: step_id for idx, step_id in record_steps.items()
    }
    for step in steps:
        parent = step.get("parent_step_id")
        visited: set[str] = set()
        while parent and parent not in resolved and parent not in visited:
            visited.add(parent)
            parent = parents.get(parent)
        ancestor = resolved.get(parent)
        for uid in visited:
            resolved[uid] = ancestor
        step["parent_step_id"] = ancestor if ancestor != step["step_id"] else None


def _blocking_hooks(events: list[JsonObj]) -> dict[str, JsonObj]:
    hooks = {}
    for ev in events:
        if ev.get("type") == "attachment":
            attachment = _as_dict(ev.get("attachment"))
            task_id = attachment.get("toolUseID") or ev.get("toolUseID")
            if attachment.get("type") == "hook_blocking_error" and task_id:
                hooks[task_id] = attachment
    return hooks


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
    step["status"] = "error" if ev.get("isApiErrorMessage") is True else "ok"
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
    step["status"] = None
    step["attributes"] = attrs
    return step


def _merge_result(
    tool_steps: dict[str, IndexedStep], block: JsonObj, ev: JsonObj, pending: list[IndexedStep]
) -> None:
    call = tool_steps.get(block.get("tool_use_id", ""))
    if call is None:
        return
    step = call[1]
    result = _as_dict(ev.get("toolUseResult"))
    if result.get("interrupted"):
        step["status"] = "interrupted"
    elif block.get("is_error"):
        step["status"] = "error"
    else:
        step["status"] = "ok"
    step["ended_at"] = ev.get("timestamp")
    if step["attributes"].get("gen_ai.tool.name") in {"Bash", "TaskOutput"}:
        # Event output can acknowledge a background task, but several results can share it.
        _merge_output(
            call,
            result,
            _result_output(result, block, ev.get("toolUseResult")),
            _output_text(block.get("content")),
            pending,
        )
    if step["kind"] == "agent" and result.get("agentId"):
        step["attributes"]["agent.spawns_subagent_id"] = result["agentId"]
        if result.get("resolvedModel"):
            step["attributes"]["agent.model"] = result["resolvedModel"]
        if result.get("totalTokens") is not None:
            step["attributes"]["agent.total_tokens"] = result["totalTokens"]


def _merge_output(
    call: IndexedStep,
    result: JsonObj,
    output: str,
    block_output: str,
    pending: list[IndexedStep],
) -> None:
    call_index, step = call
    if step["attributes"]["gen_ai.tool.name"] == "TaskOutput":
        for index, (launch_index, launch) in enumerate(pending):
            if launch_index < call_index and resolve_from_output(
                launch, step, result, output, block_output
            ):
                pending.pop(index)
                break
        return
    if mark_pending(step, result, output):
        pending.append(call)
        return
    command = _as_dict(step["attributes"]["tool.arguments"]).get("command")
    if outcome := parse_test_outcome(block_output, command=command):
        step["attributes"]["test.outcome"] = outcome


def _result_output(result: JsonObj, block: JsonObj, raw_result: object) -> str:
    streams = [result.get("stdout"), result.get("stderr")]
    if output := "\n".join(s for s in streams if isinstance(s, str) and s):
        return output
    content = (
        _as_dict(result.get("task")).get("output")
        or _as_dict(result.get("file")).get("content")
        or result.get("output")
        or block.get("content")
        or raw_result
    )
    return _output_text(content)


def _output_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            text for item in content if isinstance(text := _as_dict(item).get("text"), str)
        )
    return ""


def _system_step(
    ev: JsonObj, sub_id: str | None, ts: str, idx: int, blocking_hooks: dict[str, JsonObj]
) -> JsonObj | None:
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

    blocking = (
        blocking_hooks.get(ev.get("toolUseID", ""), {}) if subtype == "stop_hook_summary" else {}
    )
    if (
        blocking
        or ev.get("hookInfos")
        or ev.get("preventedContinuation") is not None
        or ev.get("hookErrors")
    ):
        infos = ev.get("hookInfos") or []
        info = (
            _as_dict(infos[0])
            if infos and (len(infos) == 1 or subtype != "stop_hook_summary")
            else {}
        )
        prevented = bool(ev.get("preventedContinuation"))
        verdict = "deny" if prevented or ev.get("hookErrors") or blocking else "allow"
        attrs = {
            "harness.hook.name": blocking.get("hookName")
            or info.get("hookName")
            or info.get("name", "unknown"),
            "harness.hook.verdict": verdict,
            "harness.hook.event": blocking.get("hookEvent")
            or info.get("hookEvent")
            or info.get("event", subtype or "unknown"),
            "harness.hook.prevented_continuation": prevented,
        }
        if subtype == "stop_hook_summary":
            entries = [
                {
                    k: v
                    for k, v in _as_dict(entry).items()
                    if k in {"name", "event", "hookName", "hookEvent", "outcome"}
                    and isinstance(v, str)
                }
                for entry in infos
            ]
            attrs["harness.hook.entries"] = entries
            if any(entry.get("outcome") == "blocked" for entry in entries):
                attrs["harness.hook.verdict"] = "deny"
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
        "parent_step_id": ev.get("parentUuid"),
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


def _classify_text_decision(text: str) -> tuple[str, str] | None:
    """Detect a refuse/escalate meta-decision in assistant prose -> (kind, snippet).

    Returns a short snippet around the match (never the full prose); ``None`` if no
    high-signal phrase is present. Refuse takes priority over escalate.
    """
    for kind, pattern in (("refuse", _REFUSE_RE), ("escalate", _ESCALATE_RE)):
        match = pattern.search(text)
        if match:
            start = max(0, match.start() - 30)
            snippet = text[start : match.end() + 70].strip()
            return kind, snippet[:_RATIONALE_MAX]
    return None


def _text_decision_steps(events: list[JsonObj], record_steps: dict[int, str]) -> list[JsonObj]:
    """Phase 4d: decision steps inferred from assistant prose (opt-in, best-effort)."""
    steps: list[JsonObj] = []
    for idx, ev in enumerate(events):
        if ev.get("type") != "assistant":
            continue
        text = _content_text(ev)
        if not text:
            continue
        found = _classify_text_decision(text)
        if found is None:
            continue
        kind, snippet = found
        invocation_id = record_steps[idx]
        steps.append(
            {
                "step_id": f"textdec_{idx}",
                "parent_step_id": ev.get("parentUuid"),
                "subagent_id": ev.get("agentId") if ev.get("isSidechain") else None,
                "kind": "decision",
                "started_at": ev.get("timestamp") or _EPOCH,
                "caused_by": invocation_id,
                "attributes": {
                    "agent.decision.kind": kind,
                    "agent.decision.rationale": f"(inferred from agent text) {snippet}",
                    "agent.decision.inferred": True,
                    "agent.decision.source": "text_signal",
                    "agent.decision.evidence": [invocation_id],
                },
            }
        )
    return steps


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
