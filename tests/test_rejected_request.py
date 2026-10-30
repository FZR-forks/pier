import pytest

from pier.agents.installed.pi import Pi
from pier.utils.rejected_request import rejected_requests_from_errors


@pytest.mark.parametrize(
    "message",
    [
        "Max Input Tokens=1,048,576, Got=1,054,336",
        "maximum context length is 1,048,576 tokens. Your messages resulted in 1,054,336 tokens",
    ],
)
def test_rejected_size_is_explicit_error_evidence_only(message):
    assert rejected_requests_from_errors({"text": message}, source="test") == []
    assert rejected_requests_from_errors(
        {"type": "error", "message": message}, source="test"
    ) == [
        {
            "input_tokens": 1054336,
            "limit_tokens": 1048576,
            "source": "test",
            "measurement": "provider_or_gateway_reported",
        }
    ]


def test_pi_rejection_does_not_inflate_input_peak(tmp_path):
    event = {
        "type": "message_end",
        "message": {
            "role": "assistant",
            "content": [],
            "stopReason": "error",
            "errorMessage": "Max Input Tokens=1048576, Got=1054336",
        },
    }
    trajectory = Pi(
        logs_dir=tmp_path, model_name="test/test"
    )._convert_events_to_trajectory([event])
    assert trajectory.final_metrics.extra.get("peak_context_tokens") is None
    assert trajectory.final_metrics.total_prompt_tokens is None
    assert (
        trajectory.final_metrics.extra["rejected_request_tokens"][0]["input_tokens"]
        == 1054336
    )
