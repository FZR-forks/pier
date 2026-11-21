import json

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
