import json
from pathlib import Path

from pier.agents.installed.claude_code import ClaudeCode, _is_compaction_boundary
from pier.models.agent.context import AgentContext


def test_top_level_compact_boundary_is_detected():
    # Real Claude Code session artifacts emit the marker as a top-level subtype.
    event = {
        "type": "system",
        "subtype": "compact_boundary",
        "compactMetadata": {"trigger": "auto"},
    }
    assert _is_compaction_boundary(event) is True


def test_non_compaction_system_events_are_ignored():
    assert _is_compaction_boundary({"type": "system", "subtype": "init"}) is False
    assert _is_compaction_boundary({"type": "system"}) is False
    # Wrong event type, even with a matching subtype, is not a boundary.
    assert (
        _is_compaction_boundary({"type": "user", "subtype": "compact_boundary"})
        is False
    )


def _assistant_turn(msg_id: str, usage: dict) -> str:
    return json.dumps(
        {
            "type": "assistant",
            "timestamp": "2025-01-01T00:00:00.000Z",
            "message": {
                "id": msg_id,
                "role": "assistant",
                "model": "claude-sonnet-4-5",
                "content": [{"type": "text", "text": "hello"}],
                "usage": usage,
            },
        }
    )


def test_final_metrics_include_subagent_usage_without_adding_subagent_steps(
    tmp_path: Path,
):
    """Claude Code writes Task (subagent) turns to a separate `subagents/` tree.

    Those turns never appear in the primary transcript, so the per-step sums
    under-report a delegated run. The terminal `result` event reports totals for
    the whole session tree, and must be used instead of (never added to) the
    per-step sums.
    """
    logs_dir = tmp_path / "logs"
    project_dir = logs_dir / "sessions" / "projects" / "-app"
    project_dir.mkdir(parents=True)

    # Primary session: 10 prompt / 2 completion tokens.
    (project_dir / "session-1.jsonl").write_text(
        _assistant_turn("msg_primary", {"input_tokens": 10, "output_tokens": 2}) + "\n",
        encoding="utf-8",
    )
    # Delegated subagent transcript: 100 prompt / 20 completion tokens.
    subagents_dir = project_dir / "session-1" / "subagents"
    subagents_dir.mkdir(parents=True)
    (subagents_dir / "agent-child.jsonl").write_text(
        _assistant_turn("msg_child", {"input_tokens": 100, "output_tokens": 20}) + "\n",
        encoding="utf-8",
    )
    # Claude Code's terminal result reports the tree-wide aggregate.
    (logs_dir / "claude-code.txt").write_text(
        json.dumps(
            {
                "type": "result",
                "total_cost_usd": 0.5,
                "usage": {
                    "input_tokens": 110,
                    "output_tokens": 22,
                    "cache_read_input_tokens": 7,
                    "cache_creation_input_tokens": 3,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    agent = ClaudeCode(logs_dir=logs_dir, model_name="anthropic/claude-sonnet-4-5")
    context = AgentContext()
    agent.populate_context_post_run(context)

    # Subagent usage is counted: 110 + 7 cache read + 3 cache creation.
    assert context.n_input_tokens == 120
    assert context.n_output_tokens == 22
    assert context.n_cache_tokens == 7
    assert context.cost_usd == 0.5
    # ...but the subagent turn did not become a primary trajectory step, and the
    # primary session's own 10/2 was not added on top of the aggregate.
    trajectory = json.loads((logs_dir / "trajectory.json").read_text(encoding="utf-8"))
    assert trajectory["final_metrics"]["total_steps"] == 1
