"""Explicitly qualified usage observations, never implicit replacements for totals."""

from collections.abc import Iterable

from pydantic import BaseModel, Field


class ReportedUsage(BaseModel):
    source: str
    complete: bool = False
    n_input_tokens: int | None = Field(default=None, ge=0)
    n_cache_tokens: int | None = Field(default=None, ge=0)
    n_output_tokens: int | None = Field(default=None, ge=0)
    n_cache_write_tokens: int | None = Field(default=None, ge=0)
    n_reasoning_tokens: int | None = Field(default=None, ge=0)
    cost_usd: float | None = Field(default=None, ge=0)
    cost_source: str | None = None
    lower_bound: bool = False
    missing_observations: dict[str, int] = Field(default_factory=dict)
    session_ids: list[str] = Field(default_factory=list)

    @classmethod
    def aggregate(
        cls, observations: Iterable["ReportedUsage"], *, source: str
    ) -> "ReportedUsage":
        observations = list(observations)
        if not observations:
            return cls(source=source)
        values = {}
        missing = {}
        for field in (
            "n_input_tokens",
            "n_cache_tokens",
            "n_output_tokens",
            "n_cache_write_tokens",
            "n_reasoning_tokens",
            "cost_usd",
        ):
            reported = [getattr(item, field) for item in observations]
            values[field] = (
                sum(value for value in reported if value is not None)
                if any(value is not None for value in reported)
                else None
            )
            missing[field] = sum(
                item.missing_observations.get(field, int(getattr(item, field) is None))
                for item in observations
            )
        cost_sources = {item.cost_source or "unknown" for item in observations}
        return cls(
            source=source,
            complete=bool(observations) and all(item.complete for item in observations),
            lower_bound=any(item.lower_bound for item in observations)
            or any(
                missing[field]
                for field in (
                    "n_input_tokens",
                    "n_cache_tokens",
                    "n_output_tokens",
                    "cost_usd",
                )
            ),
            cost_source=next(iter(cost_sources)) if len(cost_sources) == 1 else "mixed",
            missing_observations=missing,
            session_ids=list(
                dict.fromkeys(sid for item in observations for sid in item.session_ids)
            ),
            **values,
        )
