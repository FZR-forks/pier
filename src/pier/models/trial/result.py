import traceback
from datetime import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from pier.models.agent.context import AgentContext
from pier.models.agent.usage import ReportedUsage
from pier.models.task.id import GitTaskId, LocalTaskId, PackageTaskId
from pier.models.trial.config import TrialConfig
from pier.models.verifier.result import VerifierResult


class TimingInfo(BaseModel):
    """Timing information for a phase of trial execution."""

    started_at: datetime | None = None
    finished_at: datetime | None = None


class ExceptionInfo(BaseModel):
    """Information about an exception that occurred during trial execution."""

    exception_type: str
    exception_message: str
    exception_traceback: str
    occurred_at: datetime

    @classmethod
    def from_exception(cls, e: BaseException) -> "ExceptionInfo":
        return cls(
            exception_type=type(e).__name__,
            exception_message=str(e),
            exception_traceback=traceback.format_exc(),
            occurred_at=datetime.now(),
        )


class ModelInfo(BaseModel):
    """Information about a model that participated in a trial.

    ``provider`` is optional: when the user runs e.g. ``-m gpt-5.4`` with no
    ``<provider>/`` prefix, the CLI records the model name without a
    provider. Downstream writes to the ``model`` table omit the column so
    the DB default (``'unknown'``) takes over, keeping both sides honest
    about "not specified" vs "explicitly unknown".
    """

    name: str
    provider: str | None = None


class AgentInfo(BaseModel):
    """Information about an agent that participated in a trial."""

    name: str
    version: str
    model_info: ModelInfo | None = None


class StepResult(BaseModel):
    step_name: str
    agent_result: AgentContext | None = None
    verifier_result: VerifierResult | None = None
    exception_info: ExceptionInfo | None = None
    agent_execution: TimingInfo | None = None
    verifier: TimingInfo | None = None


class RetryAttempt(BaseModel):
    attempt: int
    trial_id: UUID
    result_path: str
    exception_type: str
    reason: str = "allowlisted_infrastructure_failure"


class TrialResult(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    task_name: str
    trial_name: str
    trial_uri: str
    task_id: LocalTaskId | GitTaskId | PackageTaskId
    source: str | None = None
    task_checksum: str
    config: TrialConfig
    agent_info: AgentInfo
    agent_result: AgentContext | None = None
    verifier_result: VerifierResult | None = None
    exception_info: ExceptionInfo | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    environment_setup: TimingInfo | None = None
    agent_setup: TimingInfo | None = None
    agent_execution: TimingInfo | None = None
    verifier: TimingInfo | None = None
    n_agent_steps: int | None = None
    step_results: list[StepResult] | None = None
    attempt: int = Field(default=1, ge=1)
    canonical: bool = True
    retry_attempts: list[RetryAttempt] = Field(default_factory=list)

    def agent_step_count(self) -> int | None:
        if self.n_agent_steps is not None:
            return self.n_agent_steps
        if (
            self.agent_result is not None
            and self.agent_result.n_agent_steps is not None
        ):
            return self.agent_result.n_agent_steps
        if not self.step_results:
            return None

        values = [
            step.agent_result.n_agent_steps if step.agent_result else None
            for step in self.step_results
        ]
        return sum(values) if all(value is not None for value in values) else None

    def compute_token_cost_totals(
        self,
    ) -> tuple[int | None, int | None, int | None, float | None]:
        """Sum (n_input_tokens, n_cache_tokens, n_output_tokens, cost_usd)."""
        if self.agent_result is not None:
            contexts = [self.agent_result]
        elif self.step_results:
            contexts = [
                step_result.agent_result or AgentContext()
                for step_result in self.step_results
            ]
        else:
            contexts = []

        if not contexts:
            return None, None, None, None

        def total(field: str):
            values = [getattr(context, field) for context in contexts]
            # Multi-step trials must not publish partial sums as full totals.
            return sum(values) if all(value is not None for value in values) else None

        return tuple(
            total(field)
            for field in (
                "n_input_tokens",
                "n_cache_tokens",
                "n_output_tokens",
                "cost_usd",
            )
        )

    def reported_usage(self) -> ReportedUsage:
        """Best-known usage, explicitly qualified and never used as exact totals."""
        contexts = (
            [self.agent_result]
            if self.agent_result is not None
            else [
                step.agent_result or AgentContext() for step in self.step_results or []
            ]
        )
        observations = []
        for context in contexts:
            if context.session_reported_usage is not None:
                observations.append(context.session_reported_usage)
            else:
                observations.append(
                    ReportedUsage(
                        source="trial.agent_result",
                        complete=all(
                            value is not None
                            for value in (
                                context.n_input_tokens,
                                context.n_cache_tokens,
                                context.n_output_tokens,
                                context.cost_usd,
                            )
                        ),
                        n_input_tokens=context.n_input_tokens,
                        n_cache_tokens=context.n_cache_tokens,
                        n_output_tokens=context.n_output_tokens,
                        cost_usd=context.cost_usd,
                        cost_source=context.cost_source,
                        n_cache_write_tokens=context.n_cache_write_tokens,
                        n_reasoning_tokens=context.n_reasoning_tokens,
                    )
                )
        return ReportedUsage.aggregate(observations, source="trial_observations")
