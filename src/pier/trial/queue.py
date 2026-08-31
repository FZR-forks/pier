import asyncio
import shutil
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

from pier.models.job.config import RetryConfig
from pier.models.trial.config import TrialConfig
from pier.models.trial.result import TrialResult
from pier.trial.hooks import HookCallback, TrialEvent
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
        if (
            self._retry_config.exclude_exceptions
            and exception_type in self._retry_config.exclude_exceptions
        ):
            self._logger.debug(
                f"Exception {exception_type} is in exclude_exceptions, not retrying"
            )
            return False

        if (
            self._retry_config.include_exceptions
            and exception_type not in self._retry_config.include_exceptions
        ):
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
            for hook in hooks:
                trial.add_hook(event, hook)

    async def _execute_trial_with_retries(
        self, trial_config: TrialConfig
    ) -> TrialResult:
        """Execute a trial with retry logic."""
        from pier.trial.trial import Trial

        for attempt in range(self._retry_config.max_retries + 1):
            trial = await Trial.create(trial_config)
            self._setup_hooks(trial)
            result = await trial.run()

            if result.exception_info is None:
                return result

            if not self._should_retry_exception(result.exception_info.exception_type):
                self._logger.debug(
                    "Not retrying trial because the exception is not in "
                    "include_exceptions or the maximum number of retries has been "
                    "reached"
                )
                return result
            if attempt == self._retry_config.max_retries:
                self._logger.debug(
                    "Not retrying trial because the maximum number of retries has been "
                    "reached"
                )
                return result

            archive_dir: Path | None = None
            try:
                archive_dir = self._archive_retry_attempt(trial.trial_dir, attempt)
            except OSError as exc:
                self._logger.warning(
                    "Could not archive failed attempt for trial %s: %s. "
                    "Deleting it before retry.",
                    trial_config.trial_name,
                    exc,
                )
                shutil.rmtree(trial.trial_dir, ignore_errors=True)

            delay = self._calculate_backoff_delay(attempt)

            if archive_dir is not None:
                self._logger.debug(
                    "Trial %s failed with exception %s. Archived failed attempt at %s. "
                    "Retrying in %.2f seconds...",
                    trial_config.trial_name,
                    result.exception_info.exception_type,
                    archive_dir,
                    delay,
                )
            else:
                self._logger.debug(
                    "Trial %s failed with exception %s. Failed attempt could not be "
                    "archived. Retrying in %.2f seconds...",
                    trial_config.trial_name,
                    result.exception_info.exception_type,
                    delay,
                )

            await asyncio.sleep(delay)

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
