"""Regression coverage for PR #18 review findings."""

import asyncio
import importlib.util
import json
import logging
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient

from pier.agents.installed.codex import Codex
from pier.agents.installed.claude_code import ClaudeCode
from pier.agents.installed.gemini_cli import GeminiCli
from pier.agents.installed.mini_swe_agent import convert_mini_swe_agent_to_atif
from pier.agents.installed.opencode import OpenCode
from pier.agents.installed.opencode_v2 import OpenCodeV2
from pier.agents.installed.pi import Pi
from pier.job import Job
from pier.metrics.mean import Mean
from pier.models.agent.context import AgentContext
from pier.models.job.config import JobConfig, RetryConfig
from pier.models.task.id import LocalTaskId
from pier.models.trial.config import AgentConfig, TaskConfig, TrialConfig
from pier.models.trial.paths import TrialPaths
from pier.models.trial.result import AgentInfo, ExceptionInfo, TrialResult
from pier.trial.hooks import TrialEvent, TrialHookEvent
from pier.trial.queue import TrialQueue
from pier.trial.trial import Trial
from pier.utils.rejected_request import rejected_requests_from_errors
from pier.utils.trajectory_metrics import populate_context_from_final_metrics
from pier.viewer.server import create_app


def native_audit():
    path = Path(__file__).parents[1] / "scripts" / "audit_native_usage.py"
    spec = importlib.util.spec_from_file_location("review_native_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def token_snapshot(usage, window=None):
    return {
        "type": "event_msg",
        "payload": {
            "type": "token_count",
            "info": {
                "total_token_usage": usage,
                "model_context_window": window,
            },
        },
    }


def test_native_audit_handles_missing_and_malformed_snapshots():
    audit = native_audit()
    assert audit.codex_usage_snapshot([]) is None
    assert (
        audit.codex_usage_snapshot(
            [
                {"type": "event_msg", "payload": []},
                {"type": "event_msg", "payload": {"type": "token_count", "info": []}},
                token_snapshot([]),
            ]
        )
        is None
    )


def test_native_audit_reports_missing_codex_snapshot_without_crashing(tmp_path):
    result = trial_result(tmp_path)
    result.agent_info.name = "codex"
    path = tmp_path / "trial"
    (path / "agent" / "sessions").mkdir(parents=True)
    (path / "result.json").write_text(result.model_dump_json())
    (path / "agent" / "trajectory.json").write_text(json.dumps({"steps": []}))
    (path / "agent" / "sessions" / "rollout.jsonl").write_text(
        json.dumps(
            {"type": "event_msg", "payload": {"type": "token_count", "info": []}}
        )
        + "\n"
    )
    (tmp_path / "result.json").write_text(
        json.dumps(
            {
                "stats": {
                    "n_completed_trials": 1,
                    "canonical_trials": {"trial": {"id": str(result.id), "attempt": 1}},
                }
            }
        )
    )
    report = native_audit().audit(tmp_path)
    assert report["ok"] is False
    assert any("no native cumulative usage" in error for error in report["errors"])


def test_native_audit_skips_only_the_exact_context_full_sentinel():
    audit = native_audit()
    normal = {"input_tokens": 10, "output_tokens": 2, "cached_input_tokens": 0}
    sentinel = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_input_tokens": 0,
        "total_tokens": 1000,
    }
    assert (
        audit.codex_usage_snapshot(
            [
                token_snapshot(normal),
                token_snapshot(sentinel, window=1000),
            ]
        )
        == normal
    )
    for zero_usage, window in [
        ({"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}, 1000),
        ({**sentinel, "output_tokens": 2}, 1000),
        (sentinel, 0),
        (sentinel, None),
    ]:
        assert (
            audit.codex_usage_snapshot([token_snapshot(zero_usage, window)])
            == zero_usage
        )


@pytest.mark.parametrize(
    "evidence",
    [
        {"type": "compacted", "payload": {}},
        {
            "type": "event_msg",
            "payload": {
                "type": "error",
                "message": "Max Input Tokens=1048576, Got=1054336",
            },
        },
    ],
)
def test_codex_evidence_only_child_cannot_be_a_zero_usage_complete_tree(
    tmp_path, evidence
):
    agent = Codex(logs_dir=tmp_path, model_name="test")
    base = [
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "task"}],
            },
        }
    ]
    child = agent._convert_session_file_to_trajectory(
        tmp_path / "child.jsonl", base + [evidence]
    )
    assert child.final_metrics.extra["metrics_complete"] is False
    root = child.model_copy(deep=True)
    root.trajectory_id = "root"
    root.final_metrics = root.final_metrics.model_copy(
        update={
            "total_prompt_tokens": 10,
            "total_completion_tokens": 2,
            "total_cached_tokens": 0,
            "total_cost_usd": 0.01,
            "extra": {},
        }
    )
    root.subagent_trajectories = [child]
    tree = agent._with_tree_metrics(root)
    assert tree.final_metrics.total_prompt_tokens is None
    assert tree.final_metrics.total_cost_usd is None
    assert tree.final_metrics.extra["tree_metrics_complete"] is False


@pytest.mark.parametrize(
    "rates,expected_cost,source",
    [
        (None, 0.00003, "pricing_estimate"),
        (
            {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
            0,
            "harness_reported",
        ),
    ],
)
def test_pi_custom_default_zero_is_not_explicit_free_pricing(
    tmp_path, monkeypatch, rates, expected_cost, source
):
    import litellm

    monkeypatch.setattr(
        litellm,
        "model_cost",
        {
            "private-slug": {
                "input_cost_per_token": 1e-6,
                "output_cost_per_token": 2e-6,
            }
        },
    )
    model = {"id": "private-slug"}
    if rates is not None:
        model["cost"] = rates
    agent = Pi(
        logs_dir=tmp_path,
        model_name="private/private-slug",
        pi_config={
            "providers": {"private": {"models": [model]}},
        },
    )
    message = {
        "role": "assistant",
        "model": "private-slug",
        "provider": "private",
        "content": [{"type": "text", "text": "answer"}],
        "usage": {
            "input": 10,
            "output": 10,
            "cacheRead": 0,
            "cacheWrite": 0,
            "cost": {"total": 0},
        },
    }
    trajectory = agent._convert_events_to_trajectory(
        [{"type": "turn_end", "message": message}]
    )
    assert trajectory.final_metrics.total_cost_usd == pytest.approx(expected_cost)
    assert trajectory.final_metrics.extra["cost_source"] == source
    assert trajectory.steps[0].metrics.cost_usd == (0 if rates is not None else None)


def test_pi_unknown_custom_default_zero_remains_unknown(tmp_path, monkeypatch):
    import litellm

    monkeypatch.setattr(litellm, "model_cost", {})
    agent = Pi(
        logs_dir=tmp_path,
        model_name="private/private-slug",
        pi_config={
            "providers": {"private": {"models": [{"id": "private-slug"}]}},
        },
    )
    trajectory = agent._convert_events_to_trajectory(
        [
            {
                "type": "turn_end",
                "message": {
                    "role": "assistant",
                    "model": "private-slug",
                    "provider": "private",
                    "content": [],
                    "usage": {
                        "input": 1,
                        "output": 1,
                        "cacheRead": 0,
                        "cacheWrite": 0,
                        "cost": {"total": 0},
                    },
                },
            }
        ]
    )
    assert trajectory.final_metrics.total_cost_usd is None
    assert trajectory.final_metrics.extra["cost_source"] == "unknown"


@pytest.mark.parametrize(
    "message,expected",
    [
        (
            "maximum context length is 100 tokens. Your messages resulted in 123 tokens",
            123,
        ),
        ("maximum context length is 100 tokens", None),
        ("your messages resulted in 123 tokens", None),
    ],
)
def test_rejected_request_combined_pattern_has_two_groups_not_three_patterns(
    message, expected
):
    observations = rejected_requests_from_errors(
        {"type": "error", "message": message}, source="provider"
    )
    assert observations == (
        []
        if expected is None
        else [
            {
                "input_tokens": 123,
                "limit_tokens": 100,
                "source": "provider",
                "measurement": "provider_or_gateway_reported",
            }
        ]
    )


@pytest.mark.asyncio
async def test_cancellation_in_real_trial_start_hook_persists_before_queue_end(
    tmp_path, monkeypatch
):
    trial = Trial.__new__(Trial)
    config = TrialConfig(
        task=TaskConfig(path=tmp_path), trial_name="early", trials_dir=tmp_path
    )
    trial.config = config
    trial._trial_paths = TrialPaths(tmp_path / "early")
    trial._task = SimpleNamespace(name="task", checksum="checksum", has_steps=False)
    trial._agent = SimpleNamespace(
        to_agent_info=lambda: AgentInfo(name="test", version="1")
    )
    trial._environment = SimpleNamespace(
        env_paths=SimpleNamespace(agent_dir=Path("/logs/agent"))
    )
    trial._logger = logging.getLogger(__name__)
    trial._hooks = {event: [] for event in TrialEvent}
    trial._setup_environment = AsyncMock()
    trial._maybe_download_logs = AsyncMock()
    trial._maybe_populate_agent_context = Mock()
    trial._collect_artifacts = AsyncMock()
    trial._stop_agent_environment = AsyncMock()
    trial._close_logger_handler = Mock()

    async def create(cls, cfg):
        return trial

    async def cancel(event):
        raise asyncio.CancelledError()

    observed = []

    async def ended(event):
        saved = TrialResult.model_validate_json(
            trial._trial_paths.result_path.read_text()
        )
        assert saved.exception_info.exception_type == "CancelledError"
        assert saved.finished_at is not None
        observed.append(saved.id)

    monkeypatch.setattr(Trial, "create", classmethod(create))
    queue = TrialQueue(1)
    queue.add_hook(TrialEvent.START, cancel)
    queue.add_hook(TrialEvent.END, ended)
    with pytest.raises(asyncio.CancelledError):
        await queue._execute_trial_with_retries(config)
    assert observed == [trial.result.id]
    trial._setup_environment.assert_not_awaited()


def trial_result(tmp_path, name="trial", usage=None):
    return TrialResult(
        task_name="task",
        trial_name=name,
        trial_uri=(tmp_path / name).as_uri(),
        task_id=LocalTaskId(path=tmp_path),
        task_checksum="checksum",
        config=TrialConfig(
            task=TaskConfig(path=tmp_path), trial_name=name, trials_dir=tmp_path
        ),
        agent_info=AgentInfo(name="test", version="1"),
        agent_result=usage,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_at", ["backoff", "retry_hook"])
async def test_cancelled_retry_does_not_increment_job_or_resume_count(
    tmp_path, monkeypatch, cancel_at
):
    job = Job(
        JobConfig(
            job_name="job",
            jobs_dir=tmp_path,
            quiet=True,
            agents=[AgentConfig(name="nop")],
            retry=RetryConfig(max_retries=1),
        ),
        _task_configs=[TaskConfig(path=tmp_path)],
        _metrics=defaultdict(lambda: [Mean()]),
    )

    class FakeTrial:
        def __init__(self, cfg):
            self.config = cfg
            self.trial_dir = cfg.trials_dir / cfg.trial_name
            self.hooks = defaultdict(list)

        def add_hook(self, event, hook):
            self.hooks[event].append(hook)

        async def run(self, **kwargs):
            self.trial_dir.mkdir()
            result = trial_result(tmp_path, name=self.config.trial_name)
            result.config = self.config
            result.attempt = kwargs["attempt"]
            result.exception_info = ExceptionInfo.from_exception(
                ConnectionError("transport")
            )
            for hook in self.hooks[TrialEvent.START]:
                await hook(
                    TrialHookEvent(
                        event=TrialEvent.START,
                        trial_id=result.trial_name,
                        task_name=result.task_name,
                        config=self.config,
                        result=result,
                    )
                )
            (self.trial_dir / "config.json").write_text(self.config.model_dump_json())
            return result

    async def create(cls, cfg):
        return FakeTrial(cfg)

    async def cancelled(*args):
        raise asyncio.CancelledError()

    monkeypatch.setattr(Trial, "create", classmethod(create))
    if cancel_at == "backoff":
        monkeypatch.setattr("pier.trial.queue.asyncio.sleep", cancelled)
    else:
        monkeypatch.setattr("pier.trial.queue.asyncio.sleep", AsyncMock())
        job._trial_queue.add_hook(TrialEvent.RETRY, cancelled)
    with pytest.raises(asyncio.CancelledError):
        await job.run()
    saved = json.loads((job.job_dir / "result.json").read_text())["stats"]
    assert saved["n_retries"] == 0
    assert saved["n_completed_trials"] == 1
    resumed = Job(
        job.config,
        _task_configs=[TaskConfig(path=tmp_path)],
        _metrics=defaultdict(lambda: [Mean()]),
    )
    try:
        assert resumed._n_retries == 0
    finally:
        resumed._close_logger_handlers()


@pytest.mark.parametrize("complete", [False, True])
def test_viewer_peak_and_step_averages_require_all_trials(tmp_path, complete):
    for name, context in [
        ("known", AgentContext(peak_context_tokens=100, n_agent_steps=5)),
        (
            "missing",
            AgentContext(peak_context_tokens=200, n_agent_steps=7)
            if complete
            else AgentContext(),
        ),
    ]:
        result = trial_result(tmp_path, name=name, usage=context)
        path = tmp_path / "job" / name / "result.json"
        path.parent.mkdir(parents=True)
        path.write_text(result.model_dump_json())
    client = TestClient(create_app(tmp_path))
    response = client.get("/api/jobs/job/tasks")
    assert response.status_code == 200, response.text
    item = response.json()["items"][0]
    assert item["n_trials"] == 2
    assert item["avg_peak_context_tokens"] == (150 if complete else None)
    assert item["avg_agent_steps"] == (6 if complete else None)
    response = client.get("/api/jobs/job/heatmap")
    assert response.status_code == 200, response.text
    for row in response.json()["cells"].values():
        for cell in row.values():
            assert cell["avg_peak_context_tokens"] == (150 if complete else None)
            assert cell["avg_agent_steps"] == (6 if complete else None)


@pytest.mark.parametrize("partial_dump", [False, True])
def test_native_audit_reports_partial_opencode_categories_without_crashing(
    tmp_path, partial_dump
):
    result = trial_result(tmp_path)
    result.agent_info.name = "opencode-v2"
    path = tmp_path / "trial"
    dump_dir = path / "agent" / "opencode-v2"
    dump_dir.mkdir(parents=True)
    # One complete session, one without an output/cache-write/cost record.
    sessions = [
        {
            "id": "complete",
            "tokens": {
                "input": 10,
                "output": 2,
                "reasoning": 1,
                "cache": {"read": 5, "write": 0},
            },
            "cost": 0,
        },
        {"id": "partial", "tokens": {"input": 20, "cache": {"read": 0}}},
    ]
    from pier.models.agent.usage import ReportedUsage

    observed = ReportedUsage.aggregate(
        [
            OpenCodeV2._session_reported_usage(session, complete=False)
            for session in sessions
        ],
        source="opencode.session.tokens/cost",
    )
    result.agent_result = AgentContext(session_reported_usage=observed)
    (path / "result.json").write_text(result.model_dump_json())
    (path / "agent" / "trajectory.json").write_text(
        json.dumps(
            {
                "steps": [],
                "final_metrics": {},
            }
        )
    )
    name = (
        "opencode-v2-sessions.partial.jsonl"
        if partial_dump
        else "opencode-v2-sessions.jsonl"
    )
    (dump_dir / name).write_text(
        "".join(json.dumps({"session": session}) + "\n" for session in sessions)
    )
    (tmp_path / "result.json").write_text(
        json.dumps(
            {
                "stats": {
                    "n_completed_trials": 1,
                    "canonical_trials": {"trial": {"id": str(result.id), "attempt": 1}},
                }
            }
        )
    )
    report = native_audit().audit(tmp_path)
    assert report["ok"] is False
    assert any("missing categories" in error for error in report["errors"])
    observation = report["trials"][0]
    assert observation["native"]["n_input_tokens"] is None
    assert observation["native"]["n_output_tokens"] is None
    assert observation["native"]["n_cache_tokens"] == 5
    assert observation["native_best_known"]["n_input_tokens"] == 15
    assert observation["native_missing_observations"]["n_input_tokens"] == 1


def assert_cost_observation(metrics, value, source):
    assert metrics.total_cost_usd == value
    assert metrics.extra["cost_source"] == source
    context = AgentContext()
    populate_context_from_final_metrics(context, metrics)
    assert context.cost_usd == value
    assert context.cost_source == source


@pytest.mark.parametrize(
    "stream,cost,source",
    [
        (None, None, "unknown"),
        ("not-json\n", None, "unknown"),
        ('{"type":"result"}\n', None, "unknown"),
        ('{"type":"result","total_cost_usd":0}\n', 0, "harness_reported"),
        ('{"type":"result","total_cost_usd":0.25}\n', 0.25, "harness_reported"),
    ],
)
def test_claude_cost_provenance_requires_stream_cost(tmp_path, stream, cost, source):
    (tmp_path / "session.jsonl").write_text(
        json.dumps(
            {
                "type": "assistant",
                "sessionId": "session",
                "message": {
                    "id": "answer",
                    "role": "assistant",
                    "model": "test",
                    "content": [{"type": "text", "text": "answer"}],
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            }
        )
        + "\n"
    )
    if stream is not None:
        (tmp_path / "claude-code.txt").write_text(stream)
    trajectory = ClaudeCode(
        logs_dir=tmp_path, model_name="test"
    )._convert_events_to_trajectory(tmp_path)
    assert_cost_observation(trajectory.final_metrics, cost, source)


@pytest.mark.parametrize(
    "has_usage,price", [(False, None), (True, None), (True, 0), (True, 0.25)]
)
def test_codex_cost_provenance_requires_computed_estimate(
    tmp_path, monkeypatch, has_usage, price
):
    agent = Codex(logs_dir=tmp_path, model_name="test")
    monkeypatch.setattr(agent, "_compute_cost_from_pricing", lambda **kwargs: price)
    events = [
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "task"}],
            },
        }
    ]
    if has_usage:
        events.append(
            token_snapshot(
                {"input_tokens": 1, "output_tokens": 1, "cached_input_tokens": 0}
            )
        )
    else:
        events.append({"type": "compacted", "payload": {}})
    trajectory = agent._convert_session_file_to_trajectory(
        tmp_path / "codex.jsonl", events
    )
    assert_cost_observation(
        trajectory.final_metrics,
        price,
        "pricing_estimate" if price is not None else "unknown",
    )


@pytest.mark.parametrize(
    "has_usage,price", [(False, None), (True, None), (True, 0), (True, 0.25)]
)
def test_gemini_cost_provenance_requires_computed_estimate(
    tmp_path, monkeypatch, has_usage, price
):
    agent = GeminiCli(logs_dir=tmp_path, model_name="test")
    monkeypatch.setattr(agent, "_compute_cost_from_pricing", lambda *args: price)
    message = {"type": "gemini", "content": "answer"}
    if has_usage:
        message["tokens"] = {
            "input": 1,
            "output": 1,
            "thoughts": 0,
            "cached": 0,
            "tool": 0,
        }
    trajectory = agent._convert_gemini_to_atif(
        {"sessionId": "session", "messages": [message]}
    )
    assert_cost_observation(
        trajectory.final_metrics,
        price,
        "pricing_estimate" if price is not None else "unknown",
    )


@pytest.mark.parametrize("cost", [None, 0, 0.25])
def test_mini_swe_cost_provenance_requires_reported_cost(cost):
    info = {"model_stats": {} if cost is None else {"instance_cost": cost}}
    trajectory = convert_mini_swe_agent_to_atif(
        {
            "info": info,
            "messages": [
                {
                    "role": "assistant",
                    "content": "answer",
                    "extra": {
                        "response": {
                            "usage": {
                                "prompt_tokens": 1,
                                "completion_tokens": 1,
                            }
                        }
                    },
                }
            ],
        },
        "session",
    )
    assert_cost_observation(
        trajectory.final_metrics,
        cost,
        "harness_reported" if cost is not None else "unknown",
    )


@pytest.mark.parametrize("cost", [None, 0, 0.25])
def test_opencode_cost_provenance_requires_complete_step_costs(tmp_path, cost):
    part = {"tokens": {"input": 1, "output": 1, "cache": {"read": 0, "write": 0}}}
    if cost is not None:
        part["cost"] = cost
    trajectory = OpenCode(
        logs_dir=tmp_path, model_name="test/test"
    )._convert_events_to_trajectory(
        [
            {"type": "step_start"},
            {"type": "step_finish", "part": part},
        ]
    )
    assert_cost_observation(
        trajectory.final_metrics,
        cost,
        "harness_reported" if cost is not None else "unknown",
    )


@pytest.mark.parametrize("state", ["missing", "partial", "complete", "zero", "active"])
def test_opencode_v2_provenance_tracks_withheld_vs_session_cost(tmp_path, state):
    tokens = {
        "input": 10,
        "output": 1,
        "reasoning": 0,
        "cache": {"read": 0, "write": 0},
    }
    cost = 0 if state == "zero" else 0.25
    session = {"id": "session", "tokens": tokens, "cost": cost}
    message = {
        "id": "answer",
        "type": "assistant",
        "tokens": tokens,
        "cost": cost,
        "time": {"created": 1, "completed": 2},
        "content": [],
    }
    if state == "missing":
        message.pop("cost")
        session.pop("cost")
    if state == "partial":
        message.pop("tokens")
    agent = OpenCodeV2(logs_dir=tmp_path, model_name="test/test")
    trajectory = agent._convert_session_to_trajectory(
        {
            "session": session,
            "messages": [message],
            "active": True if state == "active" else None,
        },
        {},
    )
    complete = state in {"complete", "zero"}
    assert_cost_observation(
        trajectory.final_metrics,
        cost if complete else None,
        "harness_reported" if complete else "unknown",
    )
    context = AgentContext()
    agent._populate_context_if_usage_known(context, trajectory.final_metrics)
    assert context.cost_source == ("harness_reported" if complete else "unknown")
    assert context.session_reported_usage.cost_source == (
        "unknown" if state == "missing" else "harness_reported"
    )
    if complete:
        marked = agent._mark_collection_incomplete(trajectory)
        assert_cost_observation(marked.final_metrics, None, "unknown")
        assert (
            marked.final_metrics.extra["session_reported_usage"]["cost_source"]
            == "harness_reported"
        )


@pytest.mark.parametrize("name", ["codex", "opencode-v2"])
def test_tree_cost_provenance_is_unknown_if_child_cost_is_missing(tmp_path, name):
    from pier.models.trajectories import Agent, FinalMetrics, Step, Trajectory

    root = Trajectory(
        schema_version="ATIF-v1.7",
        session_id="root",
        trajectory_id="root",
        agent=Agent(name=name, version="test"),
        steps=[Step(step_id=1, source="user", message="task")],
        final_metrics=FinalMetrics(
            total_prompt_tokens=10,
            total_completion_tokens=1,
            total_cached_tokens=0,
            total_cost_usd=0.25,
            extra={"cost_source": "harness_reported"},
        ),
    )
    child = root.model_copy(deep=True)
    child.session_id = child.trajectory_id = "child"
    child.final_metrics.total_cost_usd = None
    child.final_metrics.extra["cost_source"] = "unknown"
    root.subagent_trajectories = [child]
    agent = (
        Codex(logs_dir=tmp_path, model_name="test")
        if name == "codex"
        else OpenCodeV2(logs_dir=tmp_path, model_name="test/test")
    )
    tree = (
        agent._with_tree_metrics(root)
        if name == "codex"
        else agent._with_tree_metrics(root, 1)
    )
    assert_cost_observation(tree.final_metrics, None, "unknown")


def test_job_withheld_cost_source_differs_from_best_known_cost_source(tmp_path):
    from pier.models.job.result import JobStats
    from pier.models.agent.usage import ReportedUsage

    known = trial_result(
        tmp_path, "known", AgentContext(cost_usd=0.25, cost_source="harness_reported")
    )
    missing = trial_result(tmp_path, "missing", AgentContext(cost_source="unknown"))
    stats = JobStats.from_trial_results([known, missing])
    assert stats.cost_usd is None
    assert stats.cost_source == "unknown"
    assert stats.canonical_trials["missing"]["cost_source"] == "unknown"
    best = ReportedUsage.aggregate(
        [known.reported_usage(), missing.reported_usage()], source="test"
    )
    assert best.cost_usd == 0.25
    assert best.cost_source == "mixed"
    unknown = ReportedUsage.aggregate([missing.reported_usage()], source="test")
    assert unknown.cost_usd is None
    assert unknown.cost_source == "unknown"
