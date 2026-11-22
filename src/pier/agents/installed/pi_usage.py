"""Pi's billed usage sources, shared by exact and incomplete accounting."""

import math
from typing import Any


def reported_cost(usage: Any) -> float | None:
    cost = usage.get("cost") if isinstance(usage, dict) else None
    value = cost.get("total") if isinstance(cost, dict) else None
    return (
        float(value)
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
        else None
    )


class PiUsage:
    """Exact totals require complete valid usage; observed totals sum known parts.

    Costs have the same two views. Zero still needs the adapter's explicit-free
    policy because Pi's absent pricing metadata also produces zero.
    """

    def __init__(self, records: list[dict[str, Any]], events: list[dict[str, Any]]):
        self.usages: list[Any] = []
        for record in records:
            if record["kind"] == "agent":
                self.usages.append(record["message"].get("usage"))
                self.usages.extend(
                    result["usage"]
                    for result in record["tool_results"]
                    if isinstance(result.get("usage"), dict)
                )
            else:
                self.usages.append(record["result"].get("usage"))
        warming = {}
        for event in events:
            entry = (
                event.get("entry") if event.get("type") == "entry_appended" else None
            )
            if (
                isinstance(entry, dict)
                and entry.get("type") == "usage"
                and isinstance(entry.get("usage"), dict)
            ):
                warming[str(entry.get("id") or len(warming))] = entry
        self.warming = list(warming.values())
        self.usages.extend(entry.get("usage") for entry in self.warming)

    @staticmethod
    def _count(usage: Any, *fields: str) -> int | None:
        if not isinstance(usage, dict) or not all(
            isinstance(usage.get(key), int)
            and not isinstance(usage[key], bool)
            and usage[key] >= 0
            for key in fields
        ):
            return None
        return sum(usage[key] for key in fields)

    def observed(self, *fields: str) -> int | None:
        counts = [self._count(item, field) for item in self.usages for field in fields]
        known = [value for value in counts if value is not None]
        return sum(known) if known else None

    @property
    def complete(self) -> bool:
        return bool(self.usages) and all(
            self._count(item, "input", "output", "cacheRead", "cacheWrite") is not None
            for item in self.usages
        )

    @property
    def cost(self) -> float | None:
        costs = [reported_cost(item) for item in self.usages]
        return (
            sum(costs) if costs and all(value is not None for value in costs) else None
        )

    @property
    def observed_cost(self) -> float | None:
        known = [
            value for item in self.usages if (value := reported_cost(item)) is not None
        ]
        return sum(known) if known else None

    def warming_summary(self) -> dict[str, dict[str, int]]:
        result = {}
        for entry in self.warming:
            summary = result.setdefault(
                str(entry.get("kind") or "unknown"),
                {"count": 0, "prompt_tokens": 0, "completion_tokens": 0},
            )
            summary["count"] += 1
            summary["prompt_tokens"] += sum(
                self._count(entry.get("usage"), field) or 0
                for field in ("input", "cacheRead", "cacheWrite")
            )
            summary["completion_tokens"] += (
                self._count(entry.get("usage"), "output") or 0
            )
        return result
