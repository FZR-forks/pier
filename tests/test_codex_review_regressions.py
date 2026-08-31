import json
from pathlib import Path

from pier.trial.trial import _agent_step_count_from_trajectory_path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from pier.agents.installed.codex import Codex
from pier.models.agent.context import AgentContext

ROOT = "root-thread"
CHILD = "child-thread"
ROOT_MODEL = "root-model"
CHILD_MODEL = "child-model"


def _session_meta(
    thread_id: str,
    *,
    parent_thread_id: str | None = None,
    thread_source: str = "user",
) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": thread_id,
        "thread_source": thread_source,
        "source": "exec",
        "cli_version": "0.149.1",
    }
    if parent_thread_id is not None:
        payload["parent_thread_id"] = parent_thread_id
        payload["source"] = {
            "subagent": {"thread_spawn": {"parent_thread_id": parent_thread_id}}
        }
    return {"type": "session_meta", "payload": payload}


def _token_count(
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
    cache_write_input_tokens: int = 0,
) -> dict[str, object]:
    usage = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_input_tokens": cached_input_tokens,
        "cache_write_input_tokens": cache_write_input_tokens,
    }
    return {
        "type": "event_msg",
        "payload": {
            "type": "token_count",
            "info": {
                "last_token_usage": dict(usage),
                "total_token_usage": dict(usage),
            },
        },
    }


def _thread_events(
    thread_id: str,
    usage: tuple[int, int] | None,
    *,
    model: str,
    parent_thread_id: str | None = None,
) -> list[dict[str, object]]:
    events: list[dict[str, object]] = [
        _session_meta(
            thread_id,
            parent_thread_id=parent_thread_id,
            thread_source="subagent" if parent_thread_id else "user",
        ),
        {"type": "turn_context", "payload": {"model": model}},
        {
            "type": "event_msg",
            "payload": {"type": "turn_started", "turn_id": f"{thread_id}-turn"},
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": thread_id}],
            },
        },
    ]
    if usage is not None:
        events.append(_token_count(*usage))
    return events


def _write_rollout(
    logs_dir: Path,
    filename: str,
    events: list[dict[str, object]],
    date: tuple[str, str, str] = ("2026", "01", "01"),
) -> None:
    path = logs_dir / "sessions" / date[0] / date[1] / date[2] / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(event) for event in events) + "\n")


def _write_stdout(
    logs_dir: Path, thread_id: str, usage: dict[str, int] | None = None
) -> None:
    events = [{"type": "thread.started", "thread_id": thread_id}]
    if usage is not None:
        events.append({"type": "turn.completed", "usage": usage})
    (logs_dir / "codex.txt").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n"
    )


def _convert(logs_dir: Path):
    trajectory = Codex(
        logs_dir=logs_dir, model_name=ROOT_MODEL
    )._convert_events_to_trajectory(logs_dir / "sessions")
    assert trajectory is not None
    return trajectory


def test_populate_context_discovers_rollouts_across_midnight(tmp_path: Path) -> None:
    _write_rollout(
        tmp_path,
        "root.jsonl",
        _thread_events(ROOT, (10, 2), model=ROOT_MODEL),
        date=("2026", "08", "24"),
    )
    _write_rollout(
        tmp_path,
        "child.jsonl",
        _thread_events(
            CHILD,
            (7, 3),
            model=CHILD_MODEL,
            parent_thread_id=ROOT,
        ),
        date=("2026", "08", "25"),
    )
    _write_stdout(tmp_path, ROOT)

    context = AgentContext()
    Codex(logs_dir=tmp_path, model_name=ROOT_MODEL).populate_context_post_run(context)

    trajectory_path = tmp_path / "trajectory.json"
    assert trajectory_path.is_file()
    saved = json.loads(trajectory_path.read_text())
    assert saved["trajectory_id"] == ROOT
    assert saved["subagent_trajectories"][0]["trajectory_id"] == CHILD
    assert context.n_input_tokens == 17
    assert context.n_output_tokens == 5


@pytest.mark.asyncio
async def test_supplied_model_catalog_is_narrowed_before_install(
    tmp_path: Path,
) -> None:
    benchmark_model = "benchmark-model"
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "slug": benchmark_model,
                        "supported_reasoning_levels": [
                            {"effort": "low", "description": "Fast"},
                            {"effort": "high", "description": "Deep"},
                        ],
                        "default_reasoning_level": "high",
                        "context_window": 1_000_000,
                        "custom_field": "preserve-me",
                    },
                    {
                        "slug": "another-model",
                        "supported_reasoning_levels": [
                            {"effort": "medium", "description": "Balanced"},
                            {"effort": "xhigh", "description": "Deep"},
                        ],
                        "default_reasoning_level": "medium",
                        "context_window": 128_000,
                    },
                ]
            }
        )
    )
    agent = Codex(
        logs_dir=tmp_path,
        model_name=benchmark_model,
        restrict_model_catalog=True,
        model_catalog_file=str(catalog_path),
        reasoning_effort="low",
    )
    agent.exec_as_agent = AsyncMock()

    await agent._install_model_catalog(SimpleNamespace(), benchmark_model, {})

    calls = agent.exec_as_agent.await_args_list
    assert len(calls) == 1
    command = calls[0].kwargs["command"]
    assert "codex debug models" not in command
    payload = command.split("<<'CATALOG'\n", 1)[1].rsplit("\nCATALOG", 1)[0]
    installed = json.loads(payload)
    assert len(installed["models"]) == 1
    model = installed["models"][0]
    assert model["slug"] == benchmark_model
    assert model["supported_reasoning_levels"] == [
        {"effort": "low", "description": "Fast"}
    ]
    assert model["context_window"] == 1_000_000
    assert model["custom_field"] == "preserve-me"


def test_missing_child_metrics_mark_tree_incomplete(tmp_path: Path) -> None:
    _write_rollout(
        tmp_path,
        "root.jsonl",
        _thread_events(ROOT, (10, 2), model=ROOT_MODEL),
    )
    _write_rollout(
        tmp_path,
        "child.jsonl",
        _thread_events(
            CHILD,
            None,
            model=CHILD_MODEL,
            parent_thread_id=ROOT,
        ),
    )

    trajectory = _convert(tmp_path)
    assert trajectory.final_metrics is not None
    metrics = trajectory.final_metrics
    assert (metrics.extra or {})["tree_metrics_complete"] is False
    assert metrics.total_prompt_tokens is None
    assert metrics.total_completion_tokens is None
    assert metrics.total_cached_tokens is None


def test_unpriced_thread_makes_tree_cost_unknown_but_keeps_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import litellm

    fake_model = "not-a-real-model-for-codex-regression"
    priced_model = "priced-codex-regression-model"
    monkeypatch.delitem(litellm.model_cost, fake_model, raising=False)
    monkeypatch.setitem(
        litellm.model_cost,
        priced_model,
        {
            "input_cost_per_token": 1e-6,
            "output_cost_per_token": 2e-6,
            "cache_read_input_token_cost": 1e-6,
        },
    )
    _write_rollout(
        tmp_path,
        "root.jsonl",
        _thread_events(ROOT, (10, 2), model=fake_model),
    )
    _write_rollout(
        tmp_path,
        "child.jsonl",
        _thread_events(
            CHILD,
            (20, 3),
            model=priced_model,
            parent_thread_id=ROOT,
        ),
    )

    trajectory = _convert(tmp_path)
    assert trajectory.final_metrics is not None
    metrics = trajectory.final_metrics
    extra = metrics.extra or {}
    assert extra["tree_cost_complete"] is False
    assert metrics.total_cost_usd is None
    assert metrics.total_prompt_tokens == 30
    assert metrics.total_completion_tokens == 5


def test_root_usage_falls_back_to_stdout(tmp_path: Path) -> None:
    _write_rollout(
        tmp_path,
        "root.jsonl",
        _thread_events(ROOT, None, model=ROOT_MODEL),
    )
    _write_stdout(
        tmp_path,
        ROOT,
        {
            "input_tokens": 17,
            "output_tokens": 4,
            "cached_input_tokens": 3,
            "reasoning_output_tokens": 2,
        },
    )

    trajectory = _convert(tmp_path)
    assert trajectory.final_metrics is not None
    metrics = trajectory.final_metrics
    assert metrics.total_prompt_tokens == 17
    assert metrics.total_completion_tokens == 4
    assert metrics.total_cached_tokens == 3
    assert (metrics.extra or {})["reasoning_output_tokens"] == 2


def test_child_usage_does_not_fall_back_to_root_stdout(tmp_path: Path) -> None:
    _write_rollout(
        tmp_path,
        "root.jsonl",
        _thread_events(ROOT, (8, 2), model=ROOT_MODEL),
    )
    _write_rollout(
        tmp_path,
        "child.jsonl",
        _thread_events(
            CHILD,
            None,
            model=CHILD_MODEL,
            parent_thread_id=ROOT,
        ),
    )
    _write_stdout(
        tmp_path,
        ROOT,
        {"input_tokens": 900, "output_tokens": 900, "cached_input_tokens": 0},
    )

    trajectory = _convert(tmp_path)
    assert trajectory.subagent_trajectories is not None
    assert trajectory.subagent_trajectories[0].final_metrics is None
    assert trajectory.final_metrics is not None
    assert (trajectory.final_metrics.extra or {})["tree_metrics_complete"] is False
    assert trajectory.final_metrics.total_prompt_tokens is None
    assert trajectory.final_metrics.total_completion_tokens is None


def test_step_count_ignores_non_dict_trajectory_documents(tmp_path: Path) -> None:
    """A malformed trajectory.json must not crash trial finalization.

    The counter runs while the trial is being finalized, so an uncaught
    AttributeError here would take down a run that had otherwise succeeded.
    """
    for payload in ("[]", "null", "42", '"text"'):
        path = tmp_path / "trajectory.json"
        path.write_text(payload)
        assert _agent_step_count_from_trajectory_path(path) is None, payload


def test_cache_writes_are_preserved_without_inflating_prompt_tokens(
    tmp_path: Path,
) -> None:
    """Codex's `input_tokens` already includes cache writes.

    Recording `cache_write_input_tokens` must therefore only add observability:
    prompt/cached totals stay exactly as Codex reported them.
    """
    events = _thread_events(ROOT, None, model=ROOT_MODEL)
    events.append(
        _token_count(100, 20, cached_input_tokens=30, cache_write_input_tokens=40)
    )
    _write_rollout(tmp_path, "root.jsonl", events)

    trajectory = _convert(tmp_path)

    step_metrics = [step.metrics for step in trajectory.steps if step.metrics]
    assert step_metrics
    assert any(
        (metrics.extra or {}).get("cache_write_input_tokens") == 40
        for metrics in step_metrics
    )
    for metrics in step_metrics:
        assert metrics.prompt_tokens == 100
        assert metrics.cached_tokens == 30

    assert trajectory.final_metrics is not None
    metrics = trajectory.final_metrics
    assert metrics.total_prompt_tokens == 100
    assert metrics.total_completion_tokens == 20
    assert metrics.total_cached_tokens == 30
    assert (metrics.extra or {})["total_cache_write_input_tokens"] == 40


def test_tree_cache_writes_sum_across_subagents(tmp_path: Path) -> None:
    """A delegating run publishes the tree total, not just the root's share."""
    root_events = _thread_events(ROOT, None, model=ROOT_MODEL)
    root_events.append(_token_count(100, 10, cache_write_input_tokens=40))
    _write_rollout(tmp_path, "root.jsonl", root_events)

    child_events = _thread_events(CHILD, None, model=ROOT_MODEL, parent_thread_id=ROOT)
    child_events.append(_token_count(50, 5, cache_write_input_tokens=7))
    _write_rollout(tmp_path, "child.jsonl", child_events)

    trajectory = _convert(tmp_path)
    assert trajectory.final_metrics is not None
    extra = trajectory.final_metrics.extra or {}
    assert extra["total_cache_write_input_tokens"] == 47
    assert extra["self_only"]["total_cache_write_input_tokens"] == 40
    assert trajectory.final_metrics.total_prompt_tokens == 150


def test_cache_writes_are_priced_at_the_cache_creation_rate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cache writes bill at cache-creation price, not the plain input rate."""
    import litellm

    model = "cache-write-priced-codex-model"
    monkeypatch.setitem(
        litellm.model_cost,
        model,
        {
            "input_cost_per_token": 1e-6,
            "output_cost_per_token": 2e-6,
            "cache_read_input_token_cost": 1e-7,
            "cache_creation_input_token_cost": 5e-6,
        },
    )
    events = _thread_events(ROOT, None, model=model)
    events.append(
        _token_count(100, 10, cached_input_tokens=30, cache_write_input_tokens=40)
    )
    _write_rollout(tmp_path, "root.jsonl", events)

    trajectory = Codex(
        logs_dir=tmp_path, model_name=model
    )._convert_events_to_trajectory(tmp_path / "sessions")
    assert trajectory is not None
    assert trajectory.final_metrics is not None
    expected = 30 * 1e-6 + 30 * 1e-7 + 40 * 5e-6 + 10 * 2e-6
    assert trajectory.final_metrics.total_cost_usd == pytest.approx(expected)


def test_cost_is_unchanged_when_the_model_has_no_cache_write_price(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A model with no cache-creation price bills writes at the input rate.

    Splitting cache writes out of the fresh-input remainder must not change the
    total for such a model, so pricing tables that say nothing about cache
    creation keep their previous cost.
    """
    import litellm

    model = "no-cache-write-price-codex-model"
    monkeypatch.setitem(
        litellm.model_cost,
        model,
        {
            "input_cost_per_token": 1e-6,
            "output_cost_per_token": 2e-6,
            "cache_read_input_token_cost": 1e-7,
        },
    )
    events = _thread_events(ROOT, None, model=model)
    events.append(
        _token_count(100, 10, cached_input_tokens=30, cache_write_input_tokens=40)
    )
    _write_rollout(tmp_path, "root.jsonl", events)

    trajectory = Codex(
        logs_dir=tmp_path, model_name=model
    )._convert_events_to_trajectory(tmp_path / "sessions")
    assert trajectory is not None
    assert trajectory.final_metrics is not None
    expected = 70 * 1e-6 + 30 * 1e-7 + 10 * 2e-6
    assert trajectory.final_metrics.total_cost_usd == pytest.approx(expected)


def test_incomplete_tree_withholds_the_cache_write_total(tmp_path: Path) -> None:
    """An unmeasured child must not let the root's count pose as the tree total.

    The surrounding aggregate deliberately withholds prompt/completion/cached
    and cost totals for an incomplete tree; the cache-write total follows the
    same rule, and the root's own figure stays under ``extra.self_only``.
    """
    root_events = _thread_events(ROOT, None, model=ROOT_MODEL)
    root_events.append(_token_count(100, 10, cache_write_input_tokens=40))
    _write_rollout(tmp_path, "root.jsonl", root_events)
    _write_rollout(
        tmp_path,
        "child.jsonl",
        _thread_events(CHILD, None, model=CHILD_MODEL, parent_thread_id=ROOT),
    )

    trajectory = _convert(tmp_path)
    assert trajectory.final_metrics is not None
    extra = trajectory.final_metrics.extra or {}
    assert extra["tree_metrics_complete"] is False
    assert "total_cache_write_input_tokens" not in extra
    assert extra["self_only"]["total_cache_write_input_tokens"] == 40


def test_tree_total_is_withheld_when_a_thread_reports_no_cache_writes(
    tmp_path: Path,
) -> None:
    """A thread that never reports cache writes has an unknown, not zero, count.

    Older Codex builds omit the field, so summing around such a thread would
    silently understate the tree.
    """
    root_events = _thread_events(ROOT, None, model=ROOT_MODEL)
    root_events.append(_token_count(100, 10, cache_write_input_tokens=40))
    _write_rollout(tmp_path, "root.jsonl", root_events)

    child_events = _thread_events(CHILD, None, model=ROOT_MODEL, parent_thread_id=ROOT)
    child_usage = {"input_tokens": 50, "output_tokens": 5}
    child_events.append(
        {
            "type": "event_msg",
            "payload": {
                "type": "token_count",
                "info": {
                    "last_token_usage": dict(child_usage),
                    "total_token_usage": dict(child_usage),
                },
            },
        }
    )
    _write_rollout(tmp_path, "child.jsonl", child_events)

    trajectory = _convert(tmp_path)
    assert trajectory.final_metrics is not None
    extra = trajectory.final_metrics.extra or {}
    # The tree itself is measurable; only the cache-write total is unknown.
    assert extra["tree_metrics_complete"] is True
    assert trajectory.final_metrics.total_prompt_tokens == 150
    assert "total_cache_write_input_tokens" not in extra
    assert extra["self_only"]["total_cache_write_input_tokens"] == 40


def test_a_reported_zero_cache_write_count_is_preserved(tmp_path: Path) -> None:
    """Zero is a measurement; it must not be flattened away like a missing field."""
    events = _thread_events(ROOT, None, model=ROOT_MODEL)
    events.append(_token_count(100, 10, cache_write_input_tokens=0))
    _write_rollout(tmp_path, "root.jsonl", events)

    trajectory = _convert(tmp_path)
    assert trajectory.final_metrics is not None
    assert (trajectory.final_metrics.extra or {})["total_cache_write_input_tokens"] == 0
    step_extras = [
        step.metrics.extra or {} for step in trajectory.steps if step.metrics
    ]
    assert any(extra.get("cache_write_input_tokens") == 0 for extra in step_extras)


def test_unreported_cache_writes_stay_absent_from_metrics(tmp_path: Path) -> None:
    """A Codex build that omits the field must not report a fabricated zero."""
    events = _thread_events(ROOT, None, model=ROOT_MODEL)
    usage = {"input_tokens": 100, "output_tokens": 10}
    events.append(
        {
            "type": "event_msg",
            "payload": {
                "type": "token_count",
                "info": {
                    "last_token_usage": dict(usage),
                    "total_token_usage": dict(usage),
                },
            },
        }
    )
    _write_rollout(tmp_path, "root.jsonl", events)

    trajectory = _convert(tmp_path)
    assert trajectory.final_metrics is not None
    assert trajectory.final_metrics.total_prompt_tokens == 100
    assert "total_cache_write_input_tokens" not in (
        trajectory.final_metrics.extra or {}
    )
    for step in trajectory.steps:
        if step.metrics:
            assert "cache_write_input_tokens" not in (step.metrics.extra or {})
