import asyncio
import re
import shutil
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

from pier.models.job.config import RETRYABLE_INFRASTRUCTURE_EXCEPTIONS, RetryConfig
from pier.models.trial.config import TrialConfig
from pier.models.trial.result import RetryAttempt, TrialResult
from pier.trial.hooks import HookCallback, TrialEvent, TrialHookEvent
from pier.utils.logger import logger


class TrialQueue:
    """
    Handles orchestration of concurrent trials.

    Receives TrialConfigs, creates Trial objects internally, runs them
    with retry logic, and returns TrialResult tasks. Concurrency is
    bounded by an asyncio.Semaphore. Hooks are wired to each Trial
    instance — Trial handles all event invocations.
    """

    def __init__(
        self,
        n_concurrent: int,
        retry_config: RetryConfig | None = None,
        hooks: dict[TrialEvent, list[HookCallback]] | None = None,
    ):
        if hooks is None:
            hooks = {event: [] for event in TrialEvent}
        else:
            for event in TrialEvent:
                hooks.setdefault(event, [])

        self._n_concurrent = n_concurrent
        self._retry_config = retry_config if retry_config is not None else RetryConfig()
        self._hooks = hooks
        self._logger = logger.getChild(__name__)
        self._semaphore = asyncio.Semaphore(n_concurrent)

    def add_hook(self, event: TrialEvent, callback: HookCallback) -> "TrialQueue":
        """Register a callback for a trial lifecycle event and return the queue."""
        self._hooks[event].append(callback)
        return self

    def on_trial_started(self, callback: HookCallback) -> "TrialQueue":
        """Register a callback that runs when a queued trial starts."""
        return self.add_hook(TrialEvent.START, callback)

    def on_environment_started(self, callback: HookCallback) -> "TrialQueue":
        """Register a callback that runs when a trial environment starts."""
        return self.add_hook(TrialEvent.ENVIRONMENT_START, callback)

    def on_agent_started(self, callback: HookCallback) -> "TrialQueue":
        """Register a callback that runs when a trial agent starts."""
        return self.add_hook(TrialEvent.AGENT_START, callback)

    def on_verification_started(self, callback: HookCallback) -> "TrialQueue":
        """Register a callback that runs when trial verification starts."""
        return self.add_hook(TrialEvent.VERIFICATION_START, callback)

    def on_trial_ended(self, callback: HookCallback) -> "TrialQueue":
        """Register a callback that runs when a queued trial ends."""
        return self.add_hook(TrialEvent.END, callback)

    def on_trial_cancelled(self, callback: HookCallback) -> "TrialQueue":
        """Register a callback that runs when a queued trial is cancelled."""
        return self.add_hook(TrialEvent.CANCEL, callback)

    def _should_retry_exception(self, exception_type: str) -> bool:
        """Check if an exception should trigger a retry."""
        if exception_type == "CancelledError":
            return False
        if (
            self._retry_config.exclude_exceptions
            and exception_type in self._retry_config.exclude_exceptions
        ):
            self._logger.debug(
                f"Exception {exception_type} is in exclude_exceptions, not retrying"
            )
            return False

        allowed = self._retry_config.include_exceptions
        if allowed is None:
            allowed = RETRYABLE_INFRASTRUCTURE_EXCEPTIONS
        if exception_type not in allowed:
            self._logger.debug(
                f"Exception {exception_type} is not in include_exceptions, not retrying"
            )
            return False

        return True

    def _calculate_backoff_delay(self, attempt: int) -> float:
        """Calculate the backoff delay for a retry attempt."""
        delay = self._retry_config.min_wait_sec * (
            self._retry_config.wait_multiplier**attempt
        )
        return min(delay, self._retry_config.max_wait_sec)

    @staticmethod
    def _archive_retry_attempt(trial_dir: Path, attempt: int) -> Path:
        """Move a failed retryable attempt below a job-level metadata directory.

        The extra nesting level keeps the top-level ``.retry-attempts`` directory
        from matching Pier's trial-directory discovery heuristics. The leading dot
        is cosmetic rather than the safety mechanism. Existing archive names are
        preserved by adding a numeric suffix.
        """
        archive_root = trial_dir.parent / ".retry-attempts" / trial_dir.name
        archive_root.mkdir(parents=True, exist_ok=True)

        archive_dir = archive_root / f"attempt-{attempt + 1}"
        suffix = 2
        while archive_dir.exists():
            archive_dir = archive_root / f"attempt-{attempt + 1}-{suffix}"
            suffix += 1

        shutil.move(str(trial_dir), str(archive_dir))
        return archive_dir

    def _setup_hooks(self, trial) -> None:
        """Wire queue-level hooks to the trial."""
        for event, hooks in self._hooks.items():
            if event in {TrialEvent.END, TrialEvent.RETRY}:
                continue
            for hook in hooks:
                trial.add_hook(event, hook)

    async def _execute_trial_with_retries(
        self, trial_config: TrialConfig
    ) -> TrialResult:
        """Execute a trial with retry logic."""
        from pier.trial.trial import Trial

        history: list[RetryAttempt] = []
        history_errors: list[str] = []
        for path in sorted(
            (
                trial_config.trials_dir / ".retry-attempts" / trial_config.trial_name
            ).glob("*/result.json")
        ):
            try:
                previous = TrialResult.model_validate_json(path.read_text())
            except (ValueError, OSError):
                relative_path = path.relative_to(trial_config.trials_dir).as_posix()
                history_errors.append(relative_path)
                self._logger.warning(
                    "Unreadable archived attempt at %s; retaining unknown history", path
                )
                number = re.fullmatch(r"attempt-(\d+)(?:-\d+)?", path.parent.name)
                if number and int(number[1]) > 0:
                    history.append(
                        RetryAttempt(
                            attempt=int(number[1]),
                            trial_id=None,
                            result_path=relative_path,
                            exception_type="Unknown",
                            reason="unreadable_archived_attempt",
                        )
                    )
                continue
            history.append(
                RetryAttempt(
                    attempt=previous.attempt,
                    trial_id=previous.id,
                    result_path=path.relative_to(trial_config.trials_dir).as_posix(),
                    exception_type=previous.exception_info.exception_type
                    if previous.exception_info
                    else "ManualResume",
                    reason="previous_archived_attempt",
                )
            )
        first_attempt = max((item.attempt for item in history), default=0) + 1

        async def notify(event: TrialEvent, result: TrialResult) -> None:
            for hook in self._hooks[event]:
                await hook(
                    TrialHookEvent(
                        event=event,
                        trial_id=result.trial_name,
                        task_name=result.task_name,
                        config=trial_config,
                        result=result,
                    )
                )

        async def finish(result: TrialResult) -> TrialResult:
            await notify(TrialEvent.END, result)
            return result

        for attempt in range(self._retry_config.max_retries + 1):
            trial = await Trial.create(trial_config)
            self._setup_hooks(trial)
            try:
                result = await trial.run(
                    attempt=first_attempt + attempt,
                    retry_attempts=history,
                    retry_history_errors=history_errors,
                )
            except asyncio.CancelledError:
                # Trial finalization persisted the cancelled observation. It is
                # still canonical and must reach job accounting before re-raise.
                await notify(TrialEvent.END, trial.result)
                raise
            result.attempt = first_attempt + attempt
            result.retry_attempts = list(history)
            result.retry_history_errors = list(history_errors)
            result.canonical = True
            result_path = trial.trial_dir / "result.json"
            result_path.write_text(result.model_dump_json(indent=4))

            # A graded observation is already a benchmark attempt, even if
            # cleanup or a recovered harness stream left an exception behind.
            if result.exception_info is None:
                return await finish(result)
            if (
                result.verifier_result is not None
                and self._retry_config.include_exceptions is None
            ):
                return await finish(result)

            if not self._should_retry_exception(result.exception_info.exception_type):
                self._logger.debug(
                    "Not retrying trial because the exception is not in "
                    "include_exceptions or the maximum number of retries has been "
                    "reached"
                )
                return await finish(result)
            if attempt == self._retry_config.max_retries:
                self._logger.debug(
                    "Not retrying trial because the maximum number of retries has been "
                    "reached"
                )
                return await finish(result)

            try:
                archive_dir = self._archive_retry_attempt(
                    trial.trial_dir, result.attempt - 1
                )
            except OSError as exc:
                self._logger.warning(
                    "Could not archive failed attempt for trial %s: %s. "
                    "Keeping the original observation; not retrying.",
                    trial_config.trial_name,
                    exc,
                )
                return await finish(result)

            result.canonical = False
            result.trial_uri = archive_dir.resolve().as_uri()
            (archive_dir / "result.json").write_text(result.model_dump_json(indent=4))
            history.append(
                RetryAttempt(
                    attempt=result.attempt,
                    trial_id=result.id,
                    result_path=(archive_dir / "result.json")
                    .relative_to(trial.trial_dir.parent)
                    .as_posix(),
                    exception_type=result.exception_info.exception_type,
                    reason="explicit_retry_policy"
                    if self._retry_config.include_exceptions is not None
                    else "allowlisted_infrastructure_failure",
                )
            )
            delay = self._calculate_backoff_delay(attempt)

            self._logger.debug(
                "Trial %s failed with exception %s. Archived failed attempt at %s. "
                "Retrying in %.2f seconds...",
                trial_config.trial_name,
                result.exception_info.exception_type,
                archive_dir,
                delay,
            )

            try:
                await asyncio.sleep(delay)
                await notify(TrialEvent.RETRY, result)
            except asyncio.CancelledError:
                # No replacement benchmark attempt started. Restore the last
                # observation as canonical instead of leaving an archive only.
                shutil.move(str(archive_dir), str(trial.trial_dir))
                result.canonical = True
                result.trial_uri = trial.trial_dir.resolve().as_uri()
                result_path.write_text(result.model_dump_json(indent=4))
                await notify(TrialEvent.END, result)
                raise

        raise RuntimeError(
            f"Trial {trial_config.trial_name} produced no result. This should never "
            "happen."
        )

    async def _run_trial(self, trial_config: TrialConfig) -> TrialResult:
        """Execute a single trial, acquiring the semaphore for concurrency control."""
        async with self._semaphore:
            return await self._execute_trial_with_retries(trial_config)

    def submit(self, trial_config: TrialConfig) -> Coroutine[Any, Any, TrialResult]:
        """
        Return a coroutine that executes one trial.

        The caller decides how to schedule it (await, gather, TaskGroup).
        """
        return self._run_trial(trial_config)

    def submit_batch(
        self, configs: list[TrialConfig]
    ) -> list[Coroutine[Any, Any, TrialResult]]:
        """
        Return coroutines for multiple trials, ordered to match `configs`.
        """
        return [self.submit(config) for config in configs]
