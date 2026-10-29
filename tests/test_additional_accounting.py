from pier.agents.installed.codex import Codex
from pier.agents.installed.gemini_cli import GeminiCli
from pier.agents.installed.mini_swe_agent import _build_step_metrics
from pier.agents.installed.opencode import OpenCode


def test_gemini_tool_prompt_is_not_billed_as_output(tmp_path):
    agent = GeminiCli(logs_dir=tmp_path, model_name="test")
    trajectory = agent._convert_gemini_to_atif(
        {
            "sessionId": "test",
            "messages": [
                {
                    "id": "one",
                    "type": "gemini",
                    "content": "answer",
                    "tokens": {
                        "input": 100,
                        "output": 10,
                        "thoughts": 20,
                        "tool": 50,
                        "cached": 80,
                    },
                }
            ],
        }
    )
    assert trajectory.final_metrics.total_prompt_tokens == 150
    assert trajectory.final_metrics.total_completion_tokens == 30
    assert trajectory.steps[0].metrics.extra["tool_tokens"] == 50


def test_gemini_zero_cache_price_survives_fallback(tmp_path, monkeypatch):
    import litellm

    monkeypatch.setitem(
        litellm.model_cost,
        "free-cache",
        {
            "input_cost_per_token": 1,
            "output_cost_per_token": 2,
            "cache_read_input_token_cost": 0,
        },
    )
    agent = GeminiCli(logs_dir=tmp_path, model_name="free-cache")
    assert agent._compute_cost_from_pricing(100, 10, 90) == 30


def test_gemini_missing_usage_is_not_reported_as_free_zero_tokens(tmp_path):
    agent = GeminiCli(logs_dir=tmp_path, model_name="test")
    trajectory = agent._convert_gemini_to_atif(
        {
            "sessionId": "test",
            "messages": [{"type": "gemini", "content": "partial"}],
        }
    )
    assert trajectory.final_metrics.total_prompt_tokens is None
    assert trajectory.final_metrics.total_cost_usd is None


def test_codex_pricing_alias_does_not_reprice_other_subagent_models(
    tmp_path, monkeypatch
):
    import litellm

    monkeypatch.setattr(
        litellm,
        "model_cost",
        {
            "provider/billing-model": {
                "input_cost_per_token": 1,
                "output_cost_per_token": 2,
            },
            "child-model": {"input_cost_per_token": 3, "output_cost_per_token": 4},
        },
    )
    agent = Codex(
        logs_dir=tmp_path,
        model_name="custom/model",
        command_model_name="model",
        pricing_model_name="provider/billing-model",
    )
    assert agent._compute_cost_from_pricing(100, 10, 0, model_name="model") == 120
    assert agent._compute_cost_from_pricing(100, 10, 0, model_name="child-model") == 340


def test_opencode_v1_cache_creation_is_in_prompt_and_zero_cost_is_data(tmp_path):
    agent = OpenCode(logs_dir=tmp_path, model_name="test/test")
    trajectory = agent._convert_events_to_trajectory(
        [
            {"type": "step_start"},
            {
                "type": "step_finish",
                "part": {
                    "cost": 0,
                    "tokens": {
                        "input": 10,
                        "output": 20,
                        "reasoning": 5,
                        "cache": {"read": 30, "write": 40},
                    },
                },
            },
        ]
    )
    assert trajectory.final_metrics.total_prompt_tokens == 80
    assert trajectory.final_metrics.total_cached_tokens == 30
    assert trajectory.final_metrics.total_cost_usd == 0


def test_mini_swe_never_fabricates_per_call_cost_by_output_fraction():
    metrics = _build_step_metrics(
        prompt_tokens=100,
        completion_tokens=20,
        cached_tokens=0,
        prompt_tokens_details={},
        completion_tokens_details={},
        total_cost_usd=99,
        total_completion_tokens=40,
    )
    assert metrics.cost_usd is None


def test_native_audit_fails_closed_on_empty_job(tmp_path):
    import importlib.util
    import json
    from pathlib import Path

    path = Path(__file__).parents[1] / "scripts" / "audit_native_usage.py"
    spec = importlib.util.spec_from_file_location("native_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / "result.json").write_text(
        json.dumps({"stats": {"n_completed_trials": 0}})
    )
    report = module.audit(tmp_path)
    assert report["ok"] is False
    assert "no supported trial" in report["errors"][0]
