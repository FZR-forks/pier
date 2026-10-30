from collections import defaultdict

from pier.job import Job
from pier.models.agent.context import AgentContext
from pier.models.job.config import JobConfig
from pier.models.job.result import JobStats
from pier.models.trial.result import StepResult
from test_trial_queue_retry_archives import trial_result


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
