import json
from pathlib import Path

import pytest

from pier.agents.installed.claude_code import ClaudeCode


def _write_result(logs_dir: Path, model_usage: dict) -> None:
    result = {
        "type": "result",
        "total_cost_usd": 0.18,
        "modelUsage": model_usage,
    }
    (logs_dir / "claude-code.txt").write_text(json.dumps(result) + "\n")


@pytest.mark.parametrize(
    ("model_usage", "expected"),
    [
        ({"claude-opus-5-5": {"costUSD": 0.18, "costBasis": "list"}}, 0.18),
        ({"claude-opus-5-5": {"costUSD": 0.18, "costBasis": "managed"}}, 0.18),
        # Builds that predate the field: treat as list pricing.
        ({"claude-opus-5-5": {"costUSD": 0.18}}, 0.18),
        # A guess at the default model's rate is not a measured cost.
        ({"deepseek-v4p1-flash[1m]": {"costUSD": 0.18, "costBasis": "unknown"}}, None),
        (
            {
                "claude-haiku-5-5": {"costUSD": 0.01, "costBasis": "list"},
                "gateway-model": {"costUSD": 0.17, "costBasis": "unknown"},
            },
            None,
        ),
    ],
)
def test_total_cost_respects_cost_basis(tmp_path: Path, model_usage, expected):
    _write_result(tmp_path, model_usage)
    agent = ClaudeCode(logs_dir=tmp_path, model_name="deepseek-v4p1-flash[1m]")
    assert agent._parse_total_cost_from_stream_json() == expected


def test_unknown_basis_in_any_result_taints_the_total(tmp_path: Path):
    lines = [
        {
            "type": "result",
            "total_cost_usd": 0.05,
            "modelUsage": {"m": {"costUSD": 0.05, "costBasis": "unknown"}},
        },
        {
            "type": "result",
            "total_cost_usd": 0.07,
            "modelUsage": {"m": {"costUSD": 0.07, "costBasis": "managed"}},
        },
    ]
    (tmp_path / "claude-code.txt").write_text(
        "".join(json.dumps(line) + "\n" for line in lines)
    )
    agent = ClaudeCode(logs_dir=tmp_path, model_name="m")
    assert agent._parse_total_cost_from_stream_json() is None


def _session(tmp_path: Path, *, compacted: bool) -> Path:
    session_dir = tmp_path / "projects" / "-app" / "s"
    session_dir.mkdir(parents=True)
    events = [
        {
            "type": "user",
            "timestamp": "2026-01-01T00:00:00Z",
            "sessionId": "s",
            "message": {"role": "user", "content": "go"},
        },
        {
            "type": "assistant",
            "timestamp": "2026-01-01T00:00:01Z",
            "sessionId": "s",
            "message": {
                "id": "m1",
                "model": "m",
                "role": "assistant",
                "content": [{"type": "text", "text": "done"}],
                "usage": {
                    "input_tokens": 100,
                    "cache_read_input_tokens": 50,
                    "output_tokens": 10,
                },
            },
        },
    ]
    if compacted:
        events.insert(
            1,
            {
                "type": "system",
                "subtype": "compact_boundary",
                "compactMetadata": {"trigger": "auto"},
            },
        )
    (session_dir / "s.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events)
    )
    return session_dir


def _model_usage(input_tokens: int, output_tokens: int, cache_read: int) -> dict:
    return {
        "m": {
            "inputTokens": input_tokens,
            "outputTokens": output_tokens,
            "cacheReadInputTokens": cache_read,
            "cacheCreationInputTokens": 0,
            "costBasis": "list",
        }
    }


def test_session_totals_add_compaction_summary_calls(tmp_path: Path):
    """The summary call of a compaction is billed but absent from the transcript."""
    session_dir = _session(tmp_path, compacted=True)
    _write_result(tmp_path, _model_usage(130, 25, 70))
    agent = ClaudeCode(logs_dir=tmp_path, model_name="m")

    final = agent._convert_events_to_trajectory(session_dir).final_metrics

    assert final.total_prompt_tokens == 200
    assert final.total_completion_tokens == 25
    assert final.total_cached_tokens == 70
    assert final.extra["unattributed_usage"] == {
        "prompt_tokens": 50,
        "completion_tokens": 15,
        "cached_tokens": 20,
        "cache_write_tokens": 0,
    }
    # The cache breakdown follows the adopted totals.
    assert final.extra["total_cache_read_input_tokens"] == 70
    assert final.extra["total_cache_creation_input_tokens"] == 0


def test_inconsistent_session_totals_are_exposed_not_adopted(tmp_path: Path):
    session_dir = _session(tmp_path, compacted=False)
    # More cache but less uncached input than the steps: not just missing calls.
    _write_result(tmp_path, _model_usage(10, 10, 160))
    agent = ClaudeCode(logs_dir=tmp_path, model_name="m")

    final = agent._convert_events_to_trajectory(session_dir).final_metrics

    assert (final.total_prompt_tokens, final.total_cached_tokens) == (150, 50)
    assert final.extra["native_usage_mismatch"]["cached_tokens"] == 160
    assert "unattributed_usage" not in final.extra


def test_compaction_without_session_totals_is_flagged(tmp_path: Path):
    session_dir = _session(tmp_path, compacted=True)
    agent = ClaudeCode(logs_dir=tmp_path, model_name="m")

    final = agent._convert_events_to_trajectory(session_dir).final_metrics

    assert final.total_prompt_tokens == 150
    assert final.extra["compaction_usage_missing"] is True


def test_latest_cumulative_cost_is_used(tmp_path: Path):
    lines = [
        {
            "type": "result",
            "total_cost_usd": 0.05,
            "modelUsage": {"m": {"costUSD": 0.05, "costBasis": "list"}},
        },
        {
            "type": "result",
            "total_cost_usd": 0.07,
            "modelUsage": {"m": {"costUSD": 0.07, "costBasis": "list"}},
        },
    ]
    (tmp_path / "claude-code.txt").write_text(
        "".join(json.dumps(line) + "\n" for line in lines)
    )
    agent = ClaudeCode(logs_dir=tmp_path, model_name="m")
    assert agent._parse_total_cost_from_stream_json() == 0.07
