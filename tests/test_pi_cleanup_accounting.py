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
    normal = AgentContext()
    agent.populate_context_post_run(normal)
    assert normal.n_input_tokens is normal.n_output_tokens is None
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


def test_partial_warming_summary_preserves_present_input_components(tmp_path):
    events = [
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "content": [],
                "stopReason": "stop",
                "usage": {"input": 1, "output": 2, "cacheRead": 0, "cacheWrite": 0},
            },
        },
        {
            "type": "entry_appended",
            "entry": {
                "type": "usage",
                "id": "w1",
                "kind": "cache_warm",
                "usage": {"input": 7, "cacheRead": 300, "output": 2},
            },
        },
    ]
    (tmp_path / "pi.txt").write_text("\n".join(json.dumps(e) for e in events))
    context = AgentContext()
    Pi(logs_dir=tmp_path).populate_context_post_run(context)
    saved = json.loads((tmp_path / "trajectory.json").read_text())
    assert (
        saved["final_metrics"]["extra"]["usage_entries"]["cache_warm"]["prompt_tokens"]
        == 307
    )
    assert context.n_input_tokens is None


def test_invalid_warming_entry_does_not_erase_completed_exact_totals(tmp_path):
    message = {
        "role": "assistant",
        "content": [],
        "stopReason": "stop",
        "usage": {"input": 10, "cacheRead": 100, "cacheWrite": 5, "output": 20},
    }
    events = [
        {"type": "turn_end", "message": message, "toolResults": []},
        {
            "type": "entry_appended",
            "entry": {"type": "usage", "id": "bad", "usage": None},
        },
    ]
    (tmp_path / "pi.txt").write_text("\n".join(json.dumps(e) for e in events))
    exact, recovered = AgentContext(), AgentContext()
    agent = Pi(logs_dir=tmp_path)
    agent.populate_context_post_run(exact)
    agent.populate_local_context_after_cleanup_failure(recovered)
    assert (
        exact.n_input_tokens == recovered.session_reported_usage.n_input_tokens == 115
    )


def test_partial_recovery_retains_known_cost_as_observed_not_exact(tmp_path):
    events = [
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "content": [],
                "stopReason": "toolUse",
                "usage": {
                    "input": 10,
                    "cacheRead": 100,
                    "cacheWrite": 5,
                    "output": 20,
                    "cost": {"total": 0.01},
                },
            },
        },
        {
            "type": "message_end",
            "message": {"role": "assistant", "content": [], "stopReason": "aborted"},
        },
    ]
    (tmp_path / "pi.txt").write_text("\n".join(json.dumps(e) for e in events))
    agent = Pi(logs_dir=tmp_path)
    context = AgentContext()
    agent.populate_local_context_after_cleanup_failure(context)
    assert context.cost_usd is None
    assert context.session_reported_usage.cost_usd == 0.01
    assert context.session_reported_usage.complete is False


def test_exact_and_recovered_usage_share_tool_compaction_and_warming_sources(tmp_path):
    def usage(input_tokens, cached, output):
        return {
            "input": input_tokens,
            "cacheRead": cached,
            "cacheWrite": 0,
            "output": output,
            "cost": {"total": 0.01},
        }

    message = {
        "role": "assistant",
        "content": [
            {"type": "toolCall", "id": "t1", "name": "llm_tool", "arguments": {}}
        ],
        "stopReason": "stop",
        "usage": usage(2, 3, 5),
        "responseId": "r1",
        "timestamp": 1,
    }
    tool = {
        "role": "toolResult",
        "content": [],
        "toolCallId": "t1",
        "usage": usage(7, 11, 13),
    }
    warming = {
        "type": "entry_appended",
        "entry": {
            "type": "usage",
            "id": "w1",
            "kind": "cache_warm",
            "usage": usage(17, 19, 23),
        },
    }
    events = [
        {"type": "turn_end", "message": message, "toolResults": [tool]},
        {
            "type": "compaction_end",
            "result": {"summary": "summary", "usage": usage(29, 31, 37)},
        },
        warming,
        warming,
    ]
    (tmp_path / "pi.txt").write_text("\n".join(json.dumps(e) for e in events))
    agent = Pi(logs_dir=tmp_path)
    exact, recovered = AgentContext(), AgentContext()
    agent.populate_context_post_run(exact)
    agent.populate_local_context_after_cleanup_failure(recovered)
    observed = recovered.session_reported_usage
    assert [exact.n_input_tokens, exact.n_cache_tokens, exact.n_output_tokens] == [
        119,
        64,
        78,
    ]
    assert [
        observed.n_input_tokens,
        observed.n_cache_tokens,
        observed.n_output_tokens,
    ] == [119, 64, 78]
    assert recovered.n_input_tokens is None


@pytest.mark.parametrize("free", [False, True])
def test_default_zero_cost_is_not_misreported_as_free_recovery(
    tmp_path, monkeypatch, free
):
    import litellm

    monkeypatch.setattr(litellm, "model_cost", {})
    message = {
        "role": "assistant",
        "content": [],
        "stopReason": "stop",
        "usage": {
            "input": 1,
            "output": 2,
            "cacheRead": 0,
            "cacheWrite": 0,
            "cost": {"total": 0},
        },
    }
    (tmp_path / "pi.txt").write_text(
        json.dumps({"type": "message_end", "message": message})
    )
    config = (
        {
            "providers": {
                "openai": {
                    "modelOverrides": {
                        "unknown": {
                            "cost": {
                                "input": 0,
                                "output": 0,
                                "cacheRead": 0,
                                "cacheWrite": 0,
                            }
                        }
                    }
                }
            }
        }
        if free
        else None
    )
    agent = Pi(logs_dir=tmp_path, model_name="openai/unknown", pi_config=config)
    exact, recovered = AgentContext(), AgentContext()
    agent.populate_context_post_run(exact)
    agent.populate_local_context_after_cleanup_failure(recovered)
    assert exact.cost_usd == (0 if free else None)
    assert recovered.session_reported_usage.cost_usd == exact.cost_usd


def test_partial_usage_retains_known_components_without_claiming_zero_missing_cache(
    tmp_path,
):
    message = {
        "role": "assistant",
        "content": [],
        "stopReason": "aborted",
        "usage": {"input": 7, "cacheRead": 11, "output": 3},
    }
    (tmp_path / "pi.txt").write_text(
        json.dumps({"type": "message_end", "message": message})
    )
    agent = Pi(logs_dir=tmp_path)
    exact, recovered = AgentContext(), AgentContext()
    agent.populate_context_post_run(exact)
    agent.populate_local_context_after_cleanup_failure(recovered)
    reported = recovered.session_reported_usage
    assert reported.n_input_tokens == 18
    assert reported.n_output_tokens == 3
    assert reported.complete is False
    assert exact.n_input_tokens is recovered.n_input_tokens is None
