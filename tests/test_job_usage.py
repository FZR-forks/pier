from collections import defaultdict
from datetime import datetime

from pier.job import Job
from pier.models.agent.context import AgentContext
from pier.models.job.config import JobConfig
from pier.models.job.result import JobStats
from pier.models.trial.result import StepResult
from test_trial_queue_retry_archives import trial_result
import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from pier.metrics.mean import Mean
from pier.models.task.config import TaskConfig as TaskSettings
from pier.models.trial.config import AgentConfig, TaskConfig
from pier.models.trial.paths import TrialPaths
from pier.models.trial.result import TimingInfo, AgentInfo, TrialResult
from pier.models.verifier.result import VerifierResult
from pier.trial.trial import Trial


def test_job_totals_are_exact_per_category_not_partial(tmp_path):
    known = trial_result(tmp_path)
    partial = trial_result(tmp_path)
    partial.agent_result.n_cache_tokens = None
    partial.agent_result.cost_usd = None
    stats = JobStats.from_trial_results([known, partial])
    assert stats.n_input_tokens == 20
    assert stats.n_output_tokens == 2
    assert stats.n_cache_tokens is stats.cost_usd is None
    not_started = trial_result(tmp_path)
    not_started.agent_result = None
    stats = JobStats.from_trial_results([known, not_started])
    assert stats.n_input_tokens == 10 and stats.cost_usd == 0.01


def test_multi_step_missing_category_is_not_summed_around(tmp_path):
    result = trial_result(tmp_path)
    result.agent_result = None
    result.step_results = [
        StepResult(
            step_name="one", agent_result=AgentContext(n_input_tokens=10, cost_usd=0)
        ),
        StepResult(step_name="two", agent_result=AgentContext(n_input_tokens=20)),
    ]
    assert result.compute_token_cost_totals() == (30, None, None, None)


def test_retry_overhead_is_separate_and_unreadable_attempt_is_unknown(tmp_path):
    job = Job(
        JobConfig(job_name="job", jobs_dir=tmp_path),
        _task_configs=[],
        _metrics=defaultdict(list),
    )
    try:
        canonical = trial_result(tmp_path)
        job._previous_trial_results = {"trial": canonical}
        job._job_result = type(
            "State", (), {"stats": JobStats.from_trial_results([canonical])}
        )()
        archive = job.job_dir / ".retry-attempts" / "trial" / "attempt-1"
        archive.mkdir(parents=True)
        (archive / "result.json").write_text(
            trial_result(tmp_path, "ConnectionError").model_dump_json()
        )
        overhead = job._retry_overhead()
        assert overhead.n_attempts == 1 and overhead.cost_usd == 0.01
        assert job._job_result.stats.cost_usd == 0.01
        archived = trial_result(tmp_path)
        (archive / "result.json").write_text(archived.model_dump_json())
        job._previous_trial_results = {"trial": archived}
        assert job._retry_overhead() is None  # still current while backoff waits
        job._previous_trial_results = {"trial": canonical}
        bad = archive.parent / "attempt-2"
        bad.mkdir()
        (bad / "result.json").write_text("truncated")
        overhead = job._retry_overhead()
        assert overhead.n_attempts == 2
        assert overhead.n_input_tokens is overhead.cost_usd is None
    finally:
        job._close_logger_handlers()


def test_live_job_summary_uses_the_same_canonical_usage(tmp_path):
    from datetime import datetime
    from pier.models.job.result import JobResult

    job = Job(
        JobConfig(job_name="job", jobs_dir=tmp_path),
        _task_configs=[],
        _metrics=defaultdict(list),
    )
    try:
        known = trial_result(tmp_path)
        partial = trial_result(tmp_path)
        partial.trial_name = "partial"
        partial.agent_result.cost_usd = None
        job._previous_trial_results = {
            item.trial_name: item for item in [known, partial]
        }
        job._job_result = JobResult(
            id=job.id,
            started_at=datetime.now(),
            n_total_trials=2,
            stats=JobStats.from_trial_results([known, partial]),
        )
        job._write_job_result(exclude_trial_results=True)
        saved = JobResult.model_validate_json((job.job_dir / "result.json").read_text())
        assert saved.stats.n_input_tokens == 20
        assert saved.stats.n_cache_tokens == 0
        assert saved.stats.cost_usd is None
        # On replacement, an existing result no longer contributes to totals.
        job._previous_trial_results.pop("partial")
        job._write_job_result(exclude_trial_results=True)
        assert (
            JobResult.model_validate_json(
                (job.job_dir / "result.json").read_text()
            ).stats.cost_usd
            == 0.01
        )
    finally:
        job._close_logger_handlers()


def lifecycle_trial(
    config, *, retryable_failure=False, cleanup_failure=False, reward=1
):
    """Use real Trial.run/finalization and queue/job hooks, no sandbox."""
    trial = Trial.__new__(Trial)
    trial.config = config
    trial._trial_paths = TrialPaths(config.trials_dir / config.trial_name)
    trial._trial_paths.mkdir()
    trial._task = SimpleNamespace(
        name="task",
        checksum="test",
        has_steps=False,
        config=TaskSettings(),
        instruction="task",
    )
    trial._agent = SimpleNamespace(
        to_agent_info=lambda: AgentInfo(name="test", version="1")
    )
    trial._environment = SimpleNamespace(
        run_healthcheck=AsyncMock(),
        default_user=None,
        stop=AsyncMock(
            side_effect=ConnectionError("cleanup") if cleanup_failure else None
        ),
        env_paths=SimpleNamespace(agent_dir=trial._trial_paths.agent_dir),
    )
    trial._is_agent_environment_stopped = False
    trial._logger = logging.getLogger(__name__)
    trial._log_handler = None
    from pier.trial.hooks import TrialEvent

    trial._hooks = {event: [] for event in TrialEvent}
    for name in (
        "_setup_environment",
        "_setup_agent",
        "_maybe_download_logs",
        "_run_pre_artifacts_script",
        "_run_collect_hooks",
        "_maybe_upload_agent_logs",
        "_collect_artifacts",
    ):
        setattr(trial, name, AsyncMock())

    async def execute():
        trial.result.agent_execution = TimingInfo(started_at=datetime.now())
        trial.result.agent_result = AgentContext(
            n_input_tokens=10, n_cache_tokens=0, n_output_tokens=1, cost_usd=0.01
        )
        trial.result.agent_execution.finished_at = datetime.now()
        if retryable_failure:
            raise ConnectionError("transport")

    async def verify():
        trial.result.verifier = TimingInfo(
            started_at=datetime.now(), finished_at=datetime.now()
        )
        trial.result.verifier_result = VerifierResult(rewards={"reward": reward})

    trial._execute_agent = execute
    trial._run_verification = verify
    return trial


def lifecycle_job(tmp_path):
    from pier.models.job.config import RetryConfig

    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "instruction.md").write_text("task")
    return Job(
        JobConfig(
            job_name="job",
            jobs_dir=tmp_path,
            quiet=True,
            agents=[AgentConfig(name="nop")],
            retry=RetryConfig(max_retries=1, min_wait_sec=0),
        ),
        _task_configs=[TaskConfig(path=task_dir)],
        _metrics=defaultdict(lambda: [Mean()]),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit", [False, True])
async def test_completed_reward_cannot_be_replaced_after_cleanup_failure(
    tmp_path, monkeypatch, explicit
):
    job = lifecycle_job(tmp_path)
    if explicit:
        job._trial_queue._retry_config.include_exceptions = {"ConnectionError"}
    attempts = []

    async def create(cls, config):
        trial = lifecycle_trial(
            config, cleanup_failure=True, reward=0 if not attempts else 1
        )
        attempts.append(trial)
        return trial

    monkeypatch.setattr(Trial, "create", classmethod(create))
    result = await job.run()
    assert len(attempts) == 1
    kept = result.trial_results[0]
    assert kept.verifier_result.rewards == {"reward": 0}
    assert (
        kept.exception_info.exception_type == "ConnectionError"
    )  # cleanup still observable
    assert result.stats.n_retries == 0
    assert result.stats.retry_overhead is None
    assert result.stats.cost_usd == kept.agent_result.cost_usd == 0.01
    disk = TrialResult.model_validate_json(
        (job.job_dir / kept.trial_name / "result.json").read_text()
    )
    assert disk.id == kept.id and disk.verifier_result.rewards == {"reward": 0}
    assert next(iter(result.stats.evals.values())).metrics == [{"mean": 0}]


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_at", ["backoff", "replacement_create"])
async def test_retry_cancellation_and_resume_preserve_canonical_identity(
    tmp_path, monkeypatch, cancel_at
):
    from pier.models.job.result import JobResult

    job = lifecycle_job(tmp_path)
    task_configs = job._task_configs
    attempts = []

    async def create(cls, config):
        if attempts and cancel_at == "replacement_create":
            # Constructors can create scratch before being interrupted.
            (config.trials_dir / config.trial_name).mkdir()
            raise asyncio.CancelledError()
        trial = lifecycle_trial(config, retryable_failure=True)
        attempts.append(trial)
        return trial

    monkeypatch.setattr(Trial, "create", classmethod(create))
    if cancel_at == "backoff":

        async def cancel(delay):
            raise asyncio.CancelledError()

        monkeypatch.setattr("pier.trial.queue.asyncio.sleep", cancel)
    with pytest.raises(asyncio.CancelledError):
        await job.run()
    kept = attempts[0].result
    path = job.job_dir / kept.trial_name / "result.json"
    disk = TrialResult.model_validate_json(path.read_text())
    assert disk.id == kept.id and disk.attempt == 1
    assert not list((job.job_dir / ".retry-attempts").glob("*/*/result.json"))
    summary = JobResult.model_validate_json((job.job_dir / "result.json").read_text())
    assert summary.stats.n_completed_trials == 1
    assert summary.stats.n_retries == 0 and summary.stats.retry_overhead is None
    assert summary.stats.cost_usd == disk.agent_result.cost_usd
    resumed = Job(
        job.config, _task_configs=task_configs, _metrics=defaultdict(lambda: [Mean()])
    )
    try:
        assert not resumed._remaining_trial_configs
        assert resumed._existing_trial_results[0].id == kept.id
        assert resumed._trial_configs[0].trial_name == kept.trial_name
    finally:
        resumed._close_logger_handlers()


@pytest.mark.asyncio
async def test_replacement_start_cancellation_persists_current_attempt(
    tmp_path, monkeypatch
):
    from pier.models.job.result import JobResult
    from pier.trial.hooks import TrialEvent

    job = lifecycle_job(tmp_path)
    attempts = []

    async def create(cls, config):
        trial = lifecycle_trial(config, retryable_failure=not attempts)
        attempts.append(trial)
        return trial

    async def cancel_second_start(event):
        if event.result.attempt == 2:
            raise asyncio.CancelledError()

    monkeypatch.setattr(Trial, "create", classmethod(create))
    job.add_hook(TrialEvent.START, cancel_second_start)
    with pytest.raises(asyncio.CancelledError):
        await job.run()
    current = attempts[1].result
    disk = TrialResult.model_validate_json(
        (job.job_dir / current.trial_name / "result.json").read_text()
    )
    assert disk.id == current.id and disk.attempt == 2
    assert disk.exception_info.exception_type == "CancelledError"
    summary = JobResult.model_validate_json((job.job_dir / "result.json").read_text())
    assert summary.stats.n_completed_trials == 1
    assert summary.stats.n_retries == 1
    assert summary.stats.retry_overhead.n_attempts == 1
    assert summary.stats.retry_overhead.cost_usd == 0.01
    resumed = Job(
        job.config,
        _task_configs=job._task_configs,
        _metrics=defaultdict(lambda: [Mean()]),
    )
    try:
        assert not resumed._remaining_trial_configs
        assert resumed._existing_trial_results[0].id == current.id
    finally:
        resumed._close_logger_handlers()


@pytest.mark.asyncio
async def test_completed_retry_summary_and_overhead_reconcile_with_files(
    tmp_path, monkeypatch
):
    job = lifecycle_job(tmp_path)
    attempts = []

    async def create(cls, config):
        trial = lifecycle_trial(config, retryable_failure=not attempts)
        attempts.append(trial)
        return trial

    monkeypatch.setattr(Trial, "create", classmethod(create))
    result = await job.run()
    kept = result.trial_results[0]
    assert kept.attempt == 2 and kept.verifier_result.rewards == {"reward": 1}
    archive = (
        job.job_dir / ".retry-attempts" / kept.trial_name / "attempt-1" / "result.json"
    )
    previous = TrialResult.model_validate_json(archive.read_text())
    assert previous.id == attempts[0].result.id != kept.id
    assert result.stats.cost_usd == kept.agent_result.cost_usd == 0.01
    assert (
        result.stats.retry_overhead.cost_usd == previous.agent_result.cost_usd == 0.01
    )
    assert result.stats.retry_overhead.n_attempts == result.stats.n_retries == 1
