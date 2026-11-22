import json
from pathlib import Path

import pytest

from pier.agents.installed.pi import Pi
from pier.models.agent.context import AgentContext


def usage(input_tokens=10, cached=100, output=20, *, cache_write=5, cost=0.01):
    return {
        "input": input_tokens,
        "cacheRead": cached,
        "cacheWrite": cache_write,
        "output": output,
        "cost": {"total": cost},
    }


def assistant(tokens, *, stop="stop", content=None):
    message = {
        "role": "assistant",
        "model": "claude-haiku-5-5",
        "content": content or [],
        "stopReason": stop,
    }
    if tokens is not None:
        message["usage"] = tokens
    return {"type": "message_end", "message": message}


def capture(tmp_path, events, **kwargs):
    (tmp_path / "pi.txt").write_text("\n".join(json.dumps(e) for e in events))
    return Pi(logs_dir=tmp_path, **kwargs)


def test_exact_and_recovered_usage_include_all_sources_without_double_counting(
    tmp_path,
):
    call = assistant(
        usage(2, 3, 5, cache_write=0),
        content=[{"type": "toolCall", "id": "t1", "name": "llm_tool", "arguments": {}}],
    )
    tool = {
        "role": "toolResult",
        "content": [],
        "toolCallId": "t1",
        "usage": usage(7, 11, 13, cache_write=0),
    }
    warming = {
        "type": "entry_appended",
        "entry": {
            "type": "usage",
            "id": "w1",
            "kind": "cache_warm",
            "usage": usage(17, 19, 23, cache_write=0),
        },
    }
    agent = capture(
        tmp_path,
        [
            call,
            {"type": "turn_end", "message": call["message"], "toolResults": [tool]},
            {
                "type": "compaction_end",
                "result": {
                    "summary": "summary",
                    "usage": usage(29, 31, 37, cache_write=0),
                },
            },
            warming,
            warming,
        ],
    )
    original = (tmp_path / "pi.txt").read_bytes()
    exact, recovered = AgentContext(), AgentContext()
    agent.populate_context_post_run(exact)
    agent.populate_local_context_after_cleanup_failure(recovered)
    reported = recovered.session_reported_usage
    assert [exact.n_input_tokens, exact.n_cache_tokens, exact.n_output_tokens] == [
        119,
        64,
        78,
    ]
    assert [
        reported.n_input_tokens,
        reported.n_cache_tokens,
        reported.n_output_tokens,
    ] == [119, 64, 78]
    assert reported.complete is False
    assert [
        recovered.n_input_tokens,
        recovered.n_cache_tokens,
        recovered.n_output_tokens,
        recovered.cost_usd,
    ] == [None] * 4
    assert (tmp_path / "pi.txt").read_bytes() == original
    trajectory = json.loads((tmp_path / "trajectory.json").read_text())
    assert trajectory["final_metrics"].get("total_prompt_tokens") is None
    assert trajectory["final_metrics"]["extra"]["cleanup_failed"] is True


@pytest.mark.parametrize("write_fails", [False, True])
def test_interrupted_response_keeps_observed_tokens_and_cost_even_if_output_fails(
    tmp_path, monkeypatch, write_fails
):
    agent = capture(
        tmp_path, [assistant(usage(), stop="toolUse"), assistant(None, stop="aborted")]
    )
    if write_fails:
        original_write = Path.write_text

        def write(path, *args, **kwargs):
            if path.name == "trajectory.json":
                raise PermissionError("read-only trajectory")
            return original_write(path, *args, **kwargs)

        monkeypatch.setattr(Path, "write_text", write)
    exact, recovered = AgentContext(), AgentContext()
    agent.populate_context_post_run(exact)
    agent.populate_local_context_after_cleanup_failure(recovered)
    assert exact.n_input_tokens is exact.n_output_tokens is None
    reported = recovered.session_reported_usage
    assert [
        reported.n_input_tokens,
        reported.n_cache_tokens,
        reported.n_output_tokens,
    ] == [115, 100, 20]
    assert reported.cost_usd == 0.01
    assert reported.complete is False
    assert (
        recovered.n_input_tokens
        is recovered.n_output_tokens
        is recovered.cost_usd
        is None
    )


def test_no_reported_usage_is_unknown_not_zero(tmp_path):
    agent = capture(tmp_path, [assistant(None, stop="aborted")])
    context = AgentContext()
    agent.populate_local_context_after_cleanup_failure(context)
    assert context.session_reported_usage.n_input_tokens is None
    assert context.session_reported_usage.n_output_tokens is None


@pytest.mark.parametrize(
    "warm_usage,expected_prompt,summary_prompt",
    [
        (None, 115, None),
        ({"input": 7, "cacheRead": 300, "output": 2}, None, 307),
    ],
)
def test_warming_compatibility_with_missing_payload_or_component(
    tmp_path, warm_usage, expected_prompt, summary_prompt
):
    agent = capture(
        tmp_path,
        [
            assistant(usage()),
            {
                "type": "entry_appended",
                "entry": {
                    "type": "usage",
                    "id": "warm",
                    "kind": "cache_warm",
                    "usage": warm_usage,
                },
            },
        ],
    )
    exact = AgentContext()
    agent.populate_context_post_run(exact)
    assert exact.n_input_tokens == expected_prompt
    extra = json.loads((tmp_path / "trajectory.json").read_text())["final_metrics"][
        "extra"
    ]
    if summary_prompt is not None:
        assert extra["usage_entries"]["cache_warm"]["prompt_tokens"] == summary_prompt
    recovered = AgentContext()
    agent.populate_local_context_after_cleanup_failure(recovered)
    assert recovered.session_reported_usage.n_input_tokens == 115 + (
        summary_prompt or 0
    )


@pytest.mark.parametrize("free", [False, True])
def test_unpriced_zero_is_unknown_but_explicit_free_is_zero(
    tmp_path, monkeypatch, free
):
    import litellm

    monkeypatch.setattr(litellm, "model_cost", {})
    config = (
        {
            "providers": {
                "openai": {
                    "modelOverrides": {
                        "unknown": {
                            "cost": dict.fromkeys(
                                ["input", "output", "cacheRead", "cacheWrite"], 0
                            )
                        }
                    }
                }
            }
        }
        if free
        else None
    )
    agent = capture(
        tmp_path,
        [assistant(usage(1, 0, 2, cache_write=0, cost=0))],
        model_name="openai/unknown",
        pi_config=config,
    )
    exact, recovered = AgentContext(), AgentContext()
    agent.populate_context_post_run(exact)
    agent.populate_local_context_after_cleanup_failure(recovered)
    assert exact.cost_usd == (0 if free else None)
    assert recovered.session_reported_usage.cost_usd == exact.cost_usd


def test_partial_usage_keeps_known_components_without_claiming_exactness(tmp_path):
    agent = capture(
        tmp_path,
        [assistant({"input": 7, "cacheRead": 11, "output": 3}, stop="aborted")],
    )
    exact, recovered = AgentContext(), AgentContext()
    agent.populate_context_post_run(exact)
    agent.populate_local_context_after_cleanup_failure(recovered)
    reported = recovered.session_reported_usage
    assert [reported.n_input_tokens, reported.n_output_tokens] == [18, 3]
    assert reported.complete is False
    assert exact.n_input_tokens is recovered.n_input_tokens is None
