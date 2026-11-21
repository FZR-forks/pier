import json
from pathlib import Path

import pytest

from pier.agents.installed.pi import Pi
from pier.models.agent.context import AgentContext


def test_cleanup_recovery_preserves_reported_usage_as_incomplete(tmp_path):
    message = {
        "role": "assistant",
        "model": "claude-haiku-5-5",
        "timestamp": 1,
        "responseId": "r1",
        "stopReason": "stop",
        "content": [],
        "usage": {
            "input": 10,
            "cacheRead": 100,
            "cacheWrite": 5,
            "output": 20,
            "cost": {"total": 0.01},
        },
    }
    events = [
        {"type": "message_end", "message": message},
        {"type": "turn_end", "message": message, "toolResults": []},
        {
            "type": "entry_appended",
            "entry": {
                "type": "usage",
                "id": "warm1",
                "kind": "cache_warm",
                "usage": {
                    "input": 1,
                    "cacheRead": 200,
                    "cacheWrite": 0,
                    "output": 1,
                    "cost": {"total": 0.001},
                },
            },
        },
    ]
    stdout = tmp_path / "pi.txt"
    stdout.write_text("\n".join(json.dumps(e) for e in events))
    original = stdout.read_bytes()
    agent = Pi(logs_dir=tmp_path, model_name="anthropic/claude-haiku-5-5")
    context = AgentContext()
    agent.populate_local_context_after_cleanup_failure(context)
    assert (
        context.n_input_tokens
        is context.n_cache_tokens
        is context.n_output_tokens
        is None
    )
    assert context.cost_usd is None
    reported = context.session_reported_usage
    assert reported is not None and reported.complete is False
    assert reported.n_input_tokens == 316
    assert reported.n_cache_tokens == 300
    assert reported.n_output_tokens == 21
    assert stdout.read_bytes() == original
    trajectory = json.loads((tmp_path / "trajectory.json").read_text())
    assert trajectory["final_metrics"].get("total_prompt_tokens") is None
    assert trajectory["final_metrics"]["extra"]["cleanup_failed"] is True


@pytest.mark.parametrize("write_fails", [False, True])
def test_known_usage_survives_interrupted_response_and_trajectory_write_failure(
    tmp_path, monkeypatch, write_fails
):
    message = {
        "role": "assistant",
        "model": "claude-haiku-5-5",
        "timestamp": 1,
        "responseId": "r1",
        "stopReason": "toolUse",
        "content": [],
        "usage": {
            "input": 10,
            "cacheRead": 100,
            "cacheWrite": 5,
            "output": 20,
            "cost": {"total": 0.01},
        },
    }
    interrupted = {
        "role": "assistant",
        "model": "claude-haiku-5-5",
        "timestamp": 2,
        "responseId": "r2",
        "stopReason": "aborted",
        "content": [],
    }
    events = [
        {"type": "message_end", "message": message},
        {"type": "turn_end", "message": message, "toolResults": []},
        {"type": "message_end", "message": interrupted},
    ]
    (tmp_path / "pi.txt").write_text("\n".join(json.dumps(e) for e in events))
    if write_fails:
        original_write = Path.write_text

        def write(path, *args, **kwargs):
            if path.name == "trajectory.json":
                raise PermissionError("read-only trajectory")
            return original_write(path, *args, **kwargs)

        monkeypatch.setattr(Path, "write_text", write)
    agent = Pi(logs_dir=tmp_path, model_name="anthropic/claude-haiku-5-5")
    # Normal conversion still refuses exact totals for a response missing usage.
    assert (
        agent._convert_events_to_trajectory(events).final_metrics.total_prompt_tokens
        is None
    )
    context = AgentContext()
    agent.populate_local_context_after_cleanup_failure(context)
    assert context.session_reported_usage.complete is False
    assert context.session_reported_usage.n_input_tokens == 115
    assert context.session_reported_usage.n_cache_tokens == 100
    assert context.session_reported_usage.n_output_tokens == 20
    assert context.n_input_tokens is context.n_output_tokens is None


def test_no_reported_usage_is_unknown_not_zero(tmp_path):
    event = {
        "type": "message_end",
        "message": {"role": "assistant", "content": [], "stopReason": "aborted"},
    }
    (tmp_path / "pi.txt").write_text(json.dumps(event))
    context = AgentContext()
    Pi(logs_dir=tmp_path).populate_local_context_after_cleanup_failure(context)
    assert context.session_reported_usage.n_input_tokens is None
    assert context.session_reported_usage.n_output_tokens is None
