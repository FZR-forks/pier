"""Explicit rejected input sizes are diagnostics, not successful model usage."""

import re
from typing import Any


def rejected_requests_from_errors(value: Any, *, source: str) -> list[dict[str, Any]]:
    observations = []

    def visit(item: Any, error: bool = False) -> None:
        if isinstance(item, dict):
            kind = item.get("type")
            error = error or (
                isinstance(kind, str) and kind in {"error", "turn.failed"}
            )
            for key, child in item.items():
                visit(
                    child,
                    error
                    or key
                    in {"error", "errorMessage", "error_message", "responseBody"},
                )
        elif isinstance(item, list):
            for child in item:
                visit(child, error)
        elif isinstance(item, str) and error:
            for pattern in (
                r"Max Input Tokens\s*[=:]\s*([\d,]+)\s*,\s*Got\s*[=:]\s*([\d,]+)",
                r"maximum context length is\s*([\d,]+)\s*tokens.*?your messages resulted in\s*([\d,]+)\s*tokens",
            ):
                for match in re.finditer(pattern, item, re.IGNORECASE | re.DOTALL):
                    try:
                        limit = int(match[1].replace(",", ""))
                        size = int(match[2].replace(",", ""))
                    except ValueError:
                        # Diagnostics must not prevent conversion of valid
                        # usage when a malformed/oversized count is reported.
                        continue
                    observation = {
                        "input_tokens": size,
                        "limit_tokens": limit,
                        "source": source,
                        "measurement": "provider_or_gateway_reported",
                    }
                    if observation not in observations:
                        observations.append(observation)

    visit(value)
    return observations
