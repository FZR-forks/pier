"""Usage preserved from a harness's saved running session totals."""

from pydantic import BaseModel, Field


class SessionUsage(BaseModel):
    source: str = "opencode.session.tokens/cost"
    # False means observed usage only: an in-flight call/child can be absent.
    complete: bool = False
    n_input_tokens: int | None = Field(default=None, ge=0)
    n_cache_tokens: int | None = Field(default=None, ge=0)
    n_output_tokens: int | None = Field(default=None, ge=0)
    cost_usd: float | None = Field(default=None, ge=0)
