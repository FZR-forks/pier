from collections import defaultdict

from pier.job import Job
from pier.models.job.config import JobConfig


def test_resume_cleanup_preserves_critiques_metadata_dir(tmp_path):
    config = JobConfig(job_name="job", jobs_dir=tmp_path)
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "config.json").write_text(config.model_dump_json(), encoding="utf-8")

    critiques_dir = job_dir / ".critiques" / "olympus-qa-v3"
    critiques_dir.mkdir(parents=True)
    (critiques_dir / "result.json").write_text("{}", encoding="utf-8")

    incomplete_trial_dir = job_dir / "task__abc123"
    incomplete_trial_dir.mkdir()
    (incomplete_trial_dir / "config.json").write_text("{}", encoding="utf-8")

    job = Job(config, _task_configs=[], _metrics=defaultdict(list))
    job._close_logger_handlers()

    assert (job_dir / ".critiques").is_dir()
    assert critiques_dir.is_dir()
    assert not incomplete_trial_dir.exists()


def test_resume_cleanup_preserves_retry_attempt_archives(tmp_path):
    config = JobConfig(job_name="job", jobs_dir=tmp_path)
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "config.json").write_text(config.model_dump_json(), encoding="utf-8")

    archive_dir = job_dir / ".retry-attempts" / "task__abc123" / "attempt-1"
    archive_dir.mkdir(parents=True)
    (archive_dir / "config.json").write_text("{}", encoding="utf-8")
    (archive_dir / "result.json").write_text("{}", encoding="utf-8")
    (archive_dir / "trial.log").write_text("failed", encoding="utf-8")

    job = Job(config, _task_configs=[], _metrics=defaultdict(list))
    job._close_logger_handlers()

    assert archive_dir.is_dir()
    assert (archive_dir / "trial.log").read_text(encoding="utf-8") == "failed"
