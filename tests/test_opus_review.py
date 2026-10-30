"""Independent Opus review regressions for generic Pier behavior."""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from rich.console import Console

from pier.agents.installed.opencode_v2 import OpenCodeV2
from pier.cli import jobs
from pier.models.agent.context import AgentContext
from pier.models.agent.usage import ReportedUsage
from pier.models.job.config import RetryConfig
from pier.models.job.result import JobStats
from pier.models.task.id import LocalTaskId
from pier.models.trial.config import TrialConfig, TaskConfig
from pier.models.trial.result import (
    TrialResult,
    AgentInfo,
    ExceptionInfo,
    StepResult,
    TimingInfo,
)
from pier.models.verifier.result import VerifierResult
from pier.trial.queue import TrialQueue
from pier.trial.trial import Trial


def result(tmp_path, *, error=None, graded=False):
    config = TrialConfig(
        task=TaskConfig(path=tmp_path), trials_dir=tmp_path, trial_name="trial"
    )
    return TrialResult(
        task_name="task",
        trial_name="trial",
        trial_uri=(tmp_path / "trial").as_uri(),
        task_id=LocalTaskId(path=tmp_path),
        task_checksum="test",
        config=config,
        agent_info=AgentInfo(name="test", version="1"),
        agent_result=AgentContext(
            n_input_tokens=10, n_cache_tokens=0, n_output_tokens=1, cost_usd=0.01
        ),
        exception_info=ExceptionInfo(
            exception_type=error,
            exception_message=error,
            exception_traceback="",
            occurred_at=datetime.now(timezone.utc),
        )
        if error
        else None,
        verifier_result=VerifierResult(rewards={"reward": 0}) if graded else None,
    )


@pytest.mark.parametrize(
    "second",
    [
        ReportedUsage(source="native", complete=True, n_input_tokens=1, cost_usd=0.01),
        ReportedUsage(
            source="native", lower_bound=True, n_input_tokens=1, cost_usd=0.01
        ),
        ReportedUsage(source="native", lower_bound=True),
    ],
)
def test_contradictory_usage_never_becomes_lower_bound_by_aggregation(tmp_path, second):
    first = ReportedUsage(
        source="native", contradictory=True, n_input_tokens=100, cost_usd=1
    )
    trial = result(tmp_path)
    trial.agent_result = AgentContext(
        session_reported_usage=first, usage_discrepancies={"cost": "conflict"}
    )
    combined = ReportedUsage.aggregate([trial.reported_usage(), second], source="job")
    assert combined.contradictory is True
    assert combined.complete is combined.lower_bound is False
    nested = ReportedUsage.aggregate([combined], source="nested")
    assert nested.contradictory is True
    assert nested.lower_bound is False


def test_unqualified_partial_usage_does_not_claim_a_proven_lower_bound():
    unknown = ReportedUsage(source="native", complete=False, n_input_tokens=100)
    bounded = ReportedUsage(source="native", lower_bound=True, n_input_tokens=1)
    assert (
        ReportedUsage.aggregate([unknown, bounded], source="job").lower_bound is False
    )
    complete = ReportedUsage(source="native", complete=True, n_input_tokens=1)
    assert (
        ReportedUsage.aggregate([bounded, complete], source="job").lower_bound is True
    )


def test_contradictory_native_session_survives_trial_and_cli_summary(
    tmp_path, monkeypatch
):
    session = {
        "id": "session",
        "tokens": {
            "input": 10,
            "output": 1,
            "reasoning": 0,
            "cache": {"read": 0, "write": 0},
        },
        "cost": 0.1,
    }
    message = {
        "id": "answer",
        "type": "assistant",
        "time": {"created": 1, "completed": 2},
        "tokens": {**session["tokens"], "input": 20},
        "cost": 0.1,
        "content": [],
    }
    agent = OpenCodeV2(logs_dir=tmp_path, model_name="test/test")
    trajectory = agent._convert_session_to_trajectory(
        {"session": session, "messages": [message]}, {}
    )
    context = AgentContext()
    agent._populate_context_if_usage_known(context, trajectory.final_metrics)
    assert context.session_reported_usage.contradictory is True
    trial = result(tmp_path)
    trial.agent_result = context
    stats = JobStats.from_trial_results([trial])
    stats.session_reported_usage = trial.reported_usage()
    console = Console(record=True, force_terminal=False)
    monkeypatch.setattr(jobs, "console", console)
    jobs.print_job_results_tables(SimpleNamespace(stats=stats, trial_results=[trial]))
    text = console.export_text()
    assert "lower bounds are" not in text
    assert "contradictory" in text
    assert stats.session_reported_usage.contradictory is True


@pytest.mark.asyncio
@pytest.mark.parametrize("graded", [False, True])
async def test_explicit_nonzero_retry_policy_is_honored_and_recorded(
    tmp_path, monkeypatch, graded
):
    first = result(tmp_path, error="NonZeroAgentExitCodeError", graded=graded)
    second = result(tmp_path)
    calls = []

    class FakeTrial:
        trial_dir = tmp_path / "trial"

        async def run(self, **kwargs):
            self.trial_dir.mkdir()
            calls.append(kwargs)
            return first if len(calls) == 1 else second

    async def create(cls, cfg):
        return FakeTrial()

    monkeypatch.setattr(Trial, "create", classmethod(create))
    queue = TrialQueue(
        1,
        RetryConfig(
            max_retries=1,
            include_exceptions={"NonZeroAgentExitCodeError"},
            min_wait_sec=0,
        ),
    )
    kept = await queue._execute_trial_with_retries(first.config)
    assert len(calls) == 2
    assert kept.attempt == 2
    assert kept.retry_attempts[0].trial_id == first.id
    assert kept.retry_attempts[0].reason == "explicit_retry_policy"
    assert kept.canonical is True


def test_explicit_retry_exclusion_wins_and_cancellation_is_never_retried():
    assert not TrialQueue(
        1,
        RetryConfig(
            include_exceptions={"NonZeroAgentExitCodeError"},
            exclude_exceptions={"NonZeroAgentExitCodeError"},
        ),
    )._should_retry_exception("NonZeroAgentExitCodeError")
    assert not TrialQueue(
        1, RetryConfig(include_exceptions={"CancelledError"}, exclude_exceptions=set())
    )._should_retry_exception("CancelledError")


@pytest.mark.asyncio
async def test_unreadable_archive_is_preserved_with_unknown_attempt_identity(
    tmp_path, monkeypatch
):
    path = tmp_path / ".retry-attempts" / "trial" / "attempt-2" / "result.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"truncated":')
    kept = result(tmp_path)

    class FakeTrial:
        trial_dir = tmp_path / "trial"

        async def run(self, **kwargs):
            self.trial_dir.mkdir()
            return kept

    async def create(cls, cfg):
        return FakeTrial()

    monkeypatch.setattr(Trial, "create", classmethod(create))
    kept = await TrialQueue(1)._execute_trial_with_retries(kept.config)
    assert kept.attempt == 3
    assert kept.retry_attempts[0].trial_id is None
    assert kept.retry_attempts[0].reason == "unreadable_archived_attempt"
    assert kept.retry_history_errors == [str(path.relative_to(tmp_path))]
    assert path.read_text() == '{"truncated":'


def test_job_reuses_unreadable_archives_surviving_config_identity(tmp_path):
    from collections import defaultdict
    from pier.job import Job
    from pier.models.job.config import JobConfig

    config = JobConfig(job_name="job", jobs_dir=tmp_path)
    task_configs = [TaskConfig(path=tmp_path / "task")]
    original = Job(config, _task_configs=task_configs, _metrics=defaultdict(list))
    try:
        trial_config = original._trial_configs[0]
        (original.job_dir / "config.json").write_text(config.model_dump_json())
        archive = (
            original.job_dir / ".retry-attempts" / trial_config.trial_name / "attempt-1"
        )
        archive.mkdir(parents=True)
        (archive / "result.json").write_text('{"truncated":')
        (archive / "config.json").write_text(trial_config.model_dump_json())
    finally:
        original._close_logger_handlers()
    resumed = Job(config, _task_configs=task_configs, _metrics=defaultdict(list))
    try:
        assert resumed._remaining_trial_configs[0].trial_name == trial_config.trial_name
    finally:
        resumed._close_logger_handlers()


@pytest.mark.parametrize("steps", [False, True])
def test_finalized_trial_before_agent_start_is_known_zero_not_missing(tmp_path, steps):
    trial = result(tmp_path, error="EnvironmentStartTimeoutError")
    trial.agent_result = None
    trial.started_at = trial.finished_at = datetime.now(timezone.utc)
    if steps:
        trial.step_results = [
            StepResult(step_name="setup", exception_info=trial.exception_info)
        ]
    assert trial.compute_token_cost_totals() == (0, 0, 0, 0)
    assert trial.reported_usage().complete is True
    assert trial.reported_usage().source == "trial.agent_not_started"
    stats = JobStats.from_trial_results([trial])
    assert stats.cost_usd == stats.n_input_tokens == 0
    assert stats.n_trials_missing_usage == 0
    # An execution that started but never reported usage is still unknown.
    trial.agent_execution = TimingInfo(started_at=trial.started_at)
    assert trial.compute_token_cost_totals() == (None, None, None, None)


def test_native_audit_uses_independent_opencode_mapping(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).parents[1] / "scripts" / "audit_native_usage.py"
    spec = importlib.util.spec_from_file_location("opus_native_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def wrong_mapping(*args, **kwargs):
        raise AssertionError("native audit must not call the adapter under test")

    monkeypatch.setattr(OpenCodeV2, "_session_reported_usage", wrong_mapping)
    assert module.opencode_session_usage(
        {
            "tokens": {
                "input": 10,
                "output": 2,
                "reasoning": 3,
                "cache": {"read": 4, "write": 5},
            },
            "cost": 0,
        }
    ) == {
        "n_input_tokens": 19,
        "n_cache_tokens": 4,
        "n_output_tokens": 5,
        "cost_usd": 0,
    }


@pytest.mark.parametrize("source", ["pricing_estimate", "harness_reported", "unknown"])
def test_never_started_zero_cost_is_neutral_in_job_provenance(tmp_path, source):
    ran = result(tmp_path)
    ran.agent_result.cost_source = source
    never_started = result(tmp_path, error="EnvironmentStartTimeoutError")
    never_started.trial_name = "not-started"
    never_started.agent_result = None
    never_started.started_at = never_started.finished_at = datetime.now(timezone.utc)
    stats = JobStats.from_trial_results([ran, never_started])
    assert stats.cost_source == source
    aggregate = ReportedUsage.aggregate(
        [ran.reported_usage(), never_started.reported_usage()], source="job"
    )
    assert aggregate.cost_source == source
    assert aggregate.cost_usd == 0.01
    all_zero = JobStats.from_trial_results([never_started])
    assert all_zero.cost_source == "no_agent_execution"
    assert (
        ReportedUsage.aggregate(
            [never_started.reported_usage()], source="job"
        ).cost_source
        == "no_agent_execution"
    )
