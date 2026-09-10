from types import SimpleNamespace

import pytest

from pier.models.job.config import RetryConfig
from pier.trial.queue import TrialQueue
from pier.trial.trial import Trial


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
async def test_retry_continues_if_archiving_fails(monkeypatch, tmp_path):
    failed_trial_dir = tmp_path / "job" / "task__abc123"
    failed_trial_dir.mkdir(parents=True)
    (failed_trial_dir / "trial.log").write_text("failed", encoding="utf-8")

    failed_result = SimpleNamespace(
        exception_info=SimpleNamespace(exception_type="RetryableError")
    )
    successful_result = SimpleNamespace(exception_info=None)

    class FakeTrial:
        def __init__(self, trial_dir, result):
            self.trial_dir = trial_dir
            self._result = result

        async def run(self):
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
        SimpleNamespace(trial_name="task__abc123")
    )

    assert result is successful_result
    assert not failed_trial_dir.exists()
