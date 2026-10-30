from types import SimpleNamespace
from datetime import datetime

import pytest

from pier.models.job.config import RetryConfig
from pier.trial.queue import TrialQueue
from pier.trial.trial import Trial
from pier.models.agent.context import AgentContext
from pier.models.task.id import LocalTaskId
from pier.models.trial.config import TrialConfig, TaskConfig
from pier.models.trial.result import AgentInfo, TrialResult, ExceptionInfo


def trial_result(tmp_path, error=None):
    config = TrialConfig(
        task=TaskConfig(path=tmp_path), trial_name="trial", trials_dir=tmp_path
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
            occurred_at=datetime.now(),
        )
        if error
        else None,
    )


def test_archive_retry_attempt_preserves_contents_and_avoids_collisions(tmp_path):
    job_dir = tmp_path / "job"
    trial_dir = job_dir / "task__abc123"
    trial_dir.mkdir(parents=True)
    (trial_dir / "trial.log").write_text("first failure", encoding="utf-8")

    first_archive = TrialQueue._archive_retry_attempt(trial_dir, attempt=0)

    assert first_archive == (job_dir / ".retry-attempts" / "task__abc123" / "attempt-1")
    assert (first_archive / "trial.log").read_text(encoding="utf-8") == "first failure"
    assert not trial_dir.exists()

    trial_dir.mkdir()
    (trial_dir / "trial.log").write_text("second failure", encoding="utf-8")

    second_archive = TrialQueue._archive_retry_attempt(trial_dir, attempt=0)

    assert second_archive == (
        job_dir / ".retry-attempts" / "task__abc123" / "attempt-1-2"
    )
    assert (second_archive / "trial.log").read_text(
        encoding="utf-8"
    ) == "second failure"
    assert (first_archive / "trial.log").read_text(encoding="utf-8") == "first failure"


@pytest.mark.asyncio
async def test_retry_keeps_observation_if_archiving_fails(monkeypatch, tmp_path):
    failed_trial_dir = tmp_path / "job" / "task__abc123"
    failed_trial_dir.mkdir(parents=True)
    (failed_trial_dir / "trial.log").write_text("failed", encoding="utf-8")

    failed_result = SimpleNamespace(
        exception_info=SimpleNamespace(exception_type="ConnectionError"),
        attempt=1,
    )
    successful_result = SimpleNamespace(exception_info=None)

    class FakeTrial:
        def __init__(self, trial_dir, result):
            self.trial_dir = trial_dir
            self._result = result

        async def run(self, **kwargs):
            return self._result

    trials = iter(
        [
            FakeTrial(failed_trial_dir, failed_result),
            FakeTrial(tmp_path / "job" / "task__abc123", successful_result),
        ]
    )

    async def fake_create(cls, trial_config):
        return next(trials)

    def fail_archive(trial_dir, attempt):
        raise OSError("archive failed")

    monkeypatch.setattr(Trial, "create", classmethod(fake_create))
    monkeypatch.setattr(
        TrialQueue, "_archive_retry_attempt", staticmethod(fail_archive)
    )

    queue = TrialQueue(
        n_concurrent=1,
        retry_config=RetryConfig(max_retries=1, min_wait_sec=0, max_wait_sec=0),
    )

    result = await queue._execute_trial_with_retries(
        SimpleNamespace(trial_name="task__abc123", trials_dir=tmp_path / "job")
    )

    assert result is failed_result
    assert failed_trial_dir.exists()


@pytest.mark.parametrize(
    "error,eligible",
    [
        ("ConnectionError", True),
        ("NonZeroAgentExitCodeError", False),
        ("RuntimeError", False),
        ("AgentTimeoutError", False),
        ("QuestionError", False),
        ("CancelledError", False),
    ],
)
def test_default_retries_only_explicit_infrastructure_types(error, eligible):
    assert TrialQueue(1)._should_retry_exception(error) is eligible


def test_explicit_retry_policy_empty_set_and_exclusions():
    assert not TrialQueue(
        1, RetryConfig(include_exceptions=set())
    )._should_retry_exception("ConnectionError")
    assert TrialQueue(
        1, RetryConfig(include_exceptions={"NonZeroAgentExitCodeError"})
    )._should_retry_exception("NonZeroAgentExitCodeError")
    assert not TrialQueue(
        1,
        RetryConfig(
            include_exceptions={"ConnectionError"},
            exclude_exceptions={"ConnectionError"},
        ),
    )._should_retry_exception("ConnectionError")


@pytest.mark.asyncio
async def test_retry_numbers_and_archived_evidence_identify_canonical_result(
    tmp_path, monkeypatch
):
    failed = trial_result(tmp_path, "ConnectionError")
    success = trial_result(tmp_path)
    attempts = []

    class FakeTrial:
        trial_dir = tmp_path / "trial"

        def add_hook(self, *args):
            pass

        async def run(self, *, attempt):
            self.trial_dir.mkdir()
            result = failed if not attempts else success
            result.attempt = attempt
            attempts.append(attempt)
            (self.trial_dir / "result.json").write_text(result.model_dump_json())
            return result

    async def create(cls, cfg):
        return FakeTrial()

    monkeypatch.setattr(Trial, "create", classmethod(create))
    result = await TrialQueue(
        1, RetryConfig(max_retries=1, min_wait_sec=0)
    )._execute_trial_with_retries(success.config)
    assert attempts == [1, 2]
    assert result.id == success.id
    archived = TrialResult.model_validate_json(
        (
            tmp_path / ".retry-attempts" / "trial" / "attempt-1" / "result.json"
        ).read_text()
    )
    kept = TrialResult.model_validate_json(
        (tmp_path / "trial" / "result.json").read_text()
    )
    assert archived.id == failed.id and archived.attempt == 1
    assert kept.id == success.id and kept.attempt == 2
