#!/usr/bin/env python3
"""Read-only reconciliation of Pier job output against saved harness records.

No provider requests, proxy billing, or corrections are performed. A nonzero
exit means at least one reported observation disagrees with native evidence.
Supports Codex, Pi, and OpenCode V2 (including embedded session trees).
"""

import argparse
import hashlib
import json
import math
from pathlib import Path


def jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").split("\n")
        if line.strip().startswith("{")
    ]


def audit(job_dir):
    errors, trials = [], []
    stats = json.loads((job_dir / "result.json").read_text())["stats"]
    for path in sorted(job_dir.glob("*/result.json")):
        result = json.loads(path.read_text())
        context = result.get("agent_result") or {}
        agent_dir = path.parent / "agent"
        trajectory = json.loads((agent_dir / "trajectory.json").read_text())
        metrics = trajectory.get("final_metrics") or {}
        name = result["agent_info"]["name"]
        label = path.parent.name

        def check(condition, detail):
            if not condition:
                errors.append(label + ": " + detail)

        check(result.get("canonical") is True, "root trial is not canonical")
        observation = stats.get("canonical_trials", {}).get(label, {})
        check(
            observation.get("id") == result["id"], "job selects a different attempt ID"
        )
        check(
            observation.get("attempt") == result.get("attempt"),
            "job selects a different attempt number",
        )

        for field, metric in (
            ("n_input_tokens", "total_prompt_tokens"),
            ("n_cache_tokens", "total_cached_tokens"),
            ("n_output_tokens", "total_completion_tokens"),
            ("cost_usd", "total_cost_usd"),
        ):
            check(
                context.get(field) == metrics.get(metric),
                field + " differs from trajectory",
            )

        def walk(node):
            yield node
            for child in node.get("subagent_trajectories") or []:
                yield from walk(child)

        nodes = list(walk(trajectory))
        calls = [
            step
            for node in nodes
            for step in node["steps"]
            if step.get("source") == "agent"
        ]
        prompts = [
            step["metrics"]["prompt_tokens"]
            for step in calls
            if (step.get("metrics") or {}).get("prompt_tokens") is not None
        ]
        check(
            context.get("peak_context_tokens") == (max(prompts) if prompts else None),
            "peak context is not maximum observed prompt input",
        )
        markers = [
            step
            for node in nodes
            for step in node["steps"]
            if (step.get("extra") or {}).get("compaction")
        ]
        if name == "codex":
            check(
                len(markers) == context.get("summarization_count"),
                "compaction markers disagree with count",
            )
        native = {}
        sources = []
        if name == "opencode-v2":
            sources = [agent_dir / "opencode-v2" / "opencode-v2-sessions.jsonl"]
            records = jsonl(sources[0])
            sessions = {
                record["session"]["id"]: record["session"] for record in records
            }
            native = {
                "n_input_tokens": 0,
                "n_cache_tokens": 0,
                "n_output_tokens": 0,
                "cost_usd": 0,
            }
            for session in sessions.values():
                tokens, cache = session["tokens"], session["tokens"]["cache"]
                native["n_input_tokens"] += (
                    tokens["input"] + cache["read"] + cache["write"]
                )
                native["n_cache_tokens"] += cache["read"]
                native["n_output_tokens"] += tokens["output"] + tokens["reasoning"]
                native["cost_usd"] += session["cost"]
            reported = context.get("session_reported_usage") or {}
            for key, value in native.items():
                check(
                    math.isclose(
                        value, reported.get(key, -1), rel_tol=1e-9, abs_tol=1e-12
                    ),
                    key + " differs from saved session totals",
                )
        elif name == "pi":
            sources = [agent_dir / "pi.txt"]
            pending, records = None, []
            for event in jsonl(sources[0]):
                if (
                    event.get("type") == "message_end"
                    and event.get("message", {}).get("role") == "assistant"
                ):
                    if pending:
                        records.append(pending)
                    pending = event["message"]
                elif event.get("type") == "turn_end":
                    if pending and (
                        pending.get("responseId"),
                        pending.get("timestamp"),
                    ) != (
                        event["message"].get("responseId"),
                        event["message"].get("timestamp"),
                    ):
                        records.append(pending)
                    pending = None
                    records.append(event["message"])
            if pending:
                records.append(pending)
            check(
                not any(
                    event.get("type") == "compaction_end"
                    or any(
                        isinstance(item.get("usage"), dict)
                        for item in event.get("toolResults", [])
                    )
                    for event in jsonl(sources[0])
                ),
                "audit supports Pi main-call usage only; compaction/tool usage needs reconciliation",
            )
            native = {
                "n_input_tokens": 0,
                "n_cache_tokens": 0,
                "n_output_tokens": 0,
                "cost_usd": 0,
            }
            for message in records:
                usage = message["usage"]
                native["n_input_tokens"] += (
                    usage["input"] + usage["cacheRead"] + usage["cacheWrite"]
                )
                native["n_cache_tokens"] += usage["cacheRead"]
                native["n_output_tokens"] += usage["output"]
                native["cost_usd"] += usage["cost"]["total"]
        elif name == "codex":
            sources = list((agent_dir / "sessions").rglob("*.jsonl"))
            if len(nodes) == 1 and len(sources) == 1:
                snapshots = [
                    event["payload"]["info"]["total_token_usage"]
                    for event in jsonl(sources[0])
                    if event.get("type") == "event_msg"
                    and event.get("payload", {}).get("type") == "token_count"
                    and event["payload"].get("info")
                ]
                usage = snapshots[-1]
                native = {
                    "n_input_tokens": usage["input_tokens"],
                    "n_cache_tokens": usage["cached_input_tokens"],
                    "n_output_tokens": usage["output_tokens"],
                }
                # Independently reconcile standard provider cache-aware pricing.
                config = result["config"]["agent"]
                pricing_name = config.get("kwargs", {}).get("pricing_model_name")
                if pricing_name:
                    import litellm

                    rates = litellm.model_cost[pricing_name]
                    p, c, o = (
                        native[key]
                        for key in (
                            "n_input_tokens",
                            "n_cache_tokens",
                            "n_output_tokens",
                        )
                    )
                    native["cost_usd"] = (
                        (p - c) * rates["input_cost_per_token"]
                        + c * rates["cache_read_input_token_cost"]
                        + o * rates["output_cost_per_token"]
                    )
            else:
                check(
                    False,
                    "audit needs Codex child-local cumulative-baseline reconciliation",
                )
        else:
            check(False, "unsupported native harness: " + name)
        for key, value in native.items():
            if context.get(key) is not None:
                check(
                    math.isclose(value, context[key], rel_tol=1e-9, abs_tol=1e-12),
                    key + " differs from native usage",
                )
        trials.append(
            {
                "trial": label,
                "agent": name,
                "attempt": result.get("attempt"),
                "usage": {
                    key: context.get(key)
                    for key in (
                        "n_input_tokens",
                        "n_cache_tokens",
                        "n_output_tokens",
                        "cost_usd",
                        "peak_context_tokens",
                        "summarization_count",
                        "cost_source",
                    )
                },
                "native": native,
                "sources_sha256": {
                    str(p.relative_to(job_dir)): hashlib.sha256(
                        p.read_bytes()
                    ).hexdigest()
                    for p in sources
                },
            }
        )
    if not trials:
        errors.append("job: no supported trial observations found")
    if stats.get("n_completed_trials") != len(trials):
        errors.append("job: completed count differs from canonical files")
    for key in ("n_input_tokens", "n_cache_tokens", "n_output_tokens", "cost_usd"):
        values = [trial["usage"][key] for trial in trials]
        expected = sum(values) if all(value is not None for value in values) else None
        actual = stats.get(key)
        if (
            expected is None
            and actual is not None
            or expected is not None
            and (
                actual is None
                or not math.isclose(expected, actual, rel_tol=1e-9, abs_tol=1e-12)
            )
        ):
            errors.append("job: " + key + " differs from canonical trial observations")
    return {"ok": not errors, "trials": trials, "errors": errors}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_dir", type=Path)
    args = parser.parse_args()
    report = audit(args.job_dir)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["ok"] else 1)
