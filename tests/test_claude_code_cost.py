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
