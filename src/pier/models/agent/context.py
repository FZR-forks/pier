from typing import Any

from pydantic import BaseModel, Field

from pier.models.agent.rollout_detail import RolloutDetail
from pier.models.agent.usage import ReportedUsage


class AgentContext(BaseModel):
    session_reported_usage: ReportedUsage | None = Field(
        default=None,
        description=(
            "Harness running totals. Inspect complete/lower_bound and any "
            "usage_discrepancies: an in-flight request or uncollected child may "
            "be absent. Missing categories remain null, not zero."
        ),
    )
    rejected_request_tokens: list[dict[str, Any]] | None = Field(
        default=None,
        description="Provider/gateway-reported rejected request sizes, not model usage.",
    )
    n_input_tokens: int | None = Field(
        default=None, description="The number of input tokens used including cache."
    )
    n_cache_tokens: int | None = Field(
        default=None, description="The number of cache tokens used."
    )
    n_output_tokens: int | None = Field(
        default=None, description="The number of output tokens used."
    )
    n_cache_write_tokens: int | None = Field(
        default=None,
        description="Cache-creation input tokens, included in n_input_tokens.",
    )
    n_reasoning_tokens: int | None = Field(
        default=None,
        description="Reasoning output tokens, included in n_output_tokens.",
    )
    cost_usd: float | None = Field(
        default=None, description="The cost in USD for the agent execution."
    )
    cost_source: str | None = Field(
        default=None,
        description="Cost provenance: harness_reported, pricing_estimate, mixed, or unknown.",
    )
    usage_discrepancies: dict[str, Any] | None = Field(
        default=None,
        description="Contradictions between harness session and per-call totals; exact totals are withheld.",
    )
    metrics_complete: bool | None = Field(
        default=None,
        description="False when the harness tree or usage collection is known incomplete.",
    )
    peak_context_tokens: int | None = Field(
        default=None,
        description="The largest input context observed for a single agent LLM call.",
    )
    summarization_count: int | None = Field(
        default=None,
        description="The number of detected context summarization or compaction events.",
    )
    n_agent_steps: int | None = Field(
        default=None,
        description="The number of agent-authored trajectory steps.",
    )
    rollout_details: list[RolloutDetail] | None = Field(
        default=None,
        description=(
            "Detailed information about each rollout trajectory including token IDs, "
            "loss masks, and logprobs. Each element represents one trajectory. For a "
            "linear chat history, there is only one rollout trajectory."
        ),
    )
    metadata: dict[str, Any] | None = Field(
        default=None, description="Additional metadata about the agent execution."
    )

    def is_empty(self) -> bool:
        return all(value is None for value in self.model_dump().values())
