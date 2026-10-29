"""Extract explicitly reported rejected input sizes, never inferred model usage."""

import re
from typing import Any


def rejected_requests_from_errors(value: Any, *, source: str) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []

    def visit(item: Any, in_error: bool = False) -> None:
        if isinstance(item, dict):
            is_error = in_error or item.get("type") in {"error", "turn.failed"}
            for key, child in item.items():
                visit(
                    child,
                    is_error
                    or key
                    in {
                        "error",
                        "errorMessage",
                        "error_message",
                        "responseBody",
                    },
                )
        elif isinstance(item, list):
            for child in item:
                visit(child, in_error)
        elif isinstance(item, str) and in_error:
            # LiteLLM's preflight error states the transformed gateway input.
            patterns = (
                r"Max Input Tokens\s*[=:]\s*([\d,]+)\s*,\s*Got\s*[=:]\s*([\d,]+)",
                r"maximum context length is\s*([\d,]+)\s*tokens.*?"
                r"your messages resulted in\s*([\d,]+)\s*tokens",
            )
            for pattern in patterns:
                for match in re.finditer(pattern, item, re.IGNORECASE | re.DOTALL):
                    observation = {
                        "input_tokens": int(match[2].replace(",", "")),
                        "limit_tokens": int(match[1].replace(",", "")),
                        "source": source,
                        "measurement": "provider_or_gateway_reported",
                    }
                    if observation not in observations:
                        observations.append(observation)

    visit(value)
    return observations
