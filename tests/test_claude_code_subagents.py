import json
from pathlib import Path
from typing import Any, cast

import pytest

from pier.agents.installed.claude_code import ClaudeCode
from pier.environments.base import BaseEnvironment, ExecResult
from pier.models.agent.context import AgentContext


class FakeEnvironment:
    def __init__(self) -> None:
        self.exec_calls: list[dict[str, Any]] = []

    def agent_process_env(self, env: dict[str, str] | None) -> dict[str, str] | None:
        return env

    async def exec(self, **kwargs: Any) -> ExecResult:
        self.exec_calls.append(kwargs)
        return ExecResult(return_code=0, stdout="", stderr="")

    @property
    def envs(self) -> list[dict[str, str]]:
        return [call.get("env") or {} for call in self.exec_calls]


def _make_assistant_event(
    content: list[dict[str, Any]],
    *,
    session_id: str = "test-session",
    timestamp: str = "2026-01-01T00:00:00Z",
    model: str = "claude-opus-4-6",
    input_tokens: int = 100,
    output_tokens: int = 50,
    msg_id: str | None = None,
) -> dict[str, Any]:
    message: dict[str, Any] = {
        "model": model,
        "role": "assistant",
        "content": content,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    }
    if msg_id is not None:
        message["id"] = msg_id
    return {
        "type": "assistant",
        "timestamp": timestamp,
        "sessionId": session_id,
        "version": "2.1.50",
        "message": message,
    }


def _make_user_event(
    content: str | list[dict[str, Any]],
    *,
    session_id: str = "test-session",
    timestamp: str = "2026-01-01T00:00:01Z",
) -> dict[str, Any]:
    return {
        "type": "user",
        "timestamp": timestamp,
        "sessionId": session_id,
        "message": {"role": "user", "content": content},
    }


def _make_tool_result_event(
    *,
    tool_id: str,
    content: str,
    timestamp: str,
    session_id: str = "test-session",
) -> dict[str, Any]:
    return _make_user_event(
        [
            {
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": content,
            }
        ],
        session_id=session_id,
        timestamp=timestamp,
    )


def _sidechain(
    event: dict[str, Any], *, agent_id: str = "agent-1", uuid: str | None = None
) -> dict[str, Any]:
    marked = {**event, "isSidechain": True, "agentId": agent_id}
    if uuid is not None:
        marked["uuid"] = uuid
    return marked


def _write_session(logs_dir: Path, events: list[dict[str, Any]]) -> Path:
    session_dir = logs_dir / "projects" / "test-project" / "test-session"
    session_dir.mkdir(parents=True)
    (session_dir / "session.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n"
    )
    return session_dir


def _write_subagent_file(
    session_dir: Path, events: list[dict[str, Any]], agent_id: str = "agent-1"
) -> None:
    session_file = next(session_dir.glob("*.jsonl"))
    subagents_dir = session_dir / session_file.stem / "subagents"
    subagents_dir.mkdir(parents=True)
    (subagents_dir / f"{agent_id}.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n"
    )


class TestSubagentTranscripts:
    def test_subagent_files_become_sidechain_steps_with_usage(self, tmp_path: Path):
        agent = ClaudeCode(logs_dir=tmp_path, model_name="claude-opus-4-6")
        main_events = [
            _make_user_event(
                "Fix the bug.",
                timestamp="2026-01-01T00:00:00Z",
            ),
            _make_assistant_event(
                [
                    {
                        "type": "tool_use",
                        "id": "toolu_task",
                        "name": "Task",
                        "input": {"prompt": "explore the repo"},
                    }
                ],
                timestamp="2026-01-01T00:00:01Z",
                input_tokens=100,
                output_tokens=10,
                msg_id="msg_main_1",
            ),
            _make_tool_result_event(
                tool_id="toolu_task",
                content="subagent finished",
                timestamp="2026-01-01T00:00:04Z",
            ),
            _make_assistant_event(
                [{"type": "text", "text": "All done."}],
                timestamp="2026-01-01T00:00:05Z",
                input_tokens=200,
                output_tokens=20,
                msg_id="msg_main_2",
            ),
        ]
        subagent_events = [
            _sidechain(
                _make_user_event(
                    "explore the repo",
                    timestamp="2026-01-01T00:00:02Z",
                )
            ),
            _sidechain(
                _make_assistant_event(
                    [{"type": "text", "text": "Repo explored."}],
                    timestamp="2026-01-01T00:00:03Z",
                    model="claude-haiku-4-5",
                    input_tokens=50,
                    output_tokens=5,
                    msg_id="msg_sub_1",
                )
            ),
        ]
        session_dir = _write_session(tmp_path, main_events)
        _write_subagent_file(session_dir, subagent_events)

        trajectory = agent._convert_events_to_trajectory(session_dir)

        assert trajectory is not None
        sidechain_steps = [
            step for step in trajectory.steps if (step.extra or {}).get("is_sidechain")
        ]
        assert len(sidechain_steps) == 2
        subagent_step = next(step for step in sidechain_steps if step.source == "agent")
        assert subagent_step.model_name == "claude-haiku-4-5"
        assert subagent_step.metrics is not None
        assert subagent_step.metrics.prompt_tokens == 50
        assert subagent_step.metrics.completion_tokens == 5
        assert trajectory.agent.model_name == "claude-opus-4-6"
        assert trajectory.final_metrics is not None
        assert trajectory.final_metrics.total_prompt_tokens == 350
        assert trajectory.final_metrics.total_completion_tokens == 35
        assert next(
            step for step in trajectory.steps if step.source == "user"
        ).message == ("Fix the bug.")

        messages = [step.message for step in trajectory.steps if step.message]
        assert messages.index("Fix the bug.") < messages.index("Repo explored.")
        assert messages.index("Repo explored.") < messages.index("All done.")

    def test_subagent_uuid_is_deduplicated_across_files(self, tmp_path: Path):
        agent = ClaudeCode(logs_dir=tmp_path, model_name="claude-opus-4-6")
        duplicated = _sidechain(
            _make_assistant_event(
                [{"type": "text", "text": "Repo explored."}],
                timestamp="2026-01-01T00:00:02Z",
                model="claude-haiku-4-5",
                input_tokens=50,
                output_tokens=5,
                msg_id="msg_sub_1",
            ),
            uuid="uuid-sub-1",
        )
        session_dir = _write_session(
            tmp_path,
            [
                _make_user_event(
                    "Fix the bug.",
                    timestamp="2026-01-01T00:00:00Z",
                ),
                duplicated,
            ],
        )
        _write_subagent_file(session_dir, [duplicated])

        trajectory = agent._convert_events_to_trajectory(session_dir)

        assert trajectory is not None
        assert (
            len([step for step in trajectory.steps if step.message == "Repo explored."])
            == 1
        )
        assert trajectory.final_metrics is not None
        assert trajectory.final_metrics.total_prompt_tokens == 50
        assert trajectory.final_metrics.total_completion_tokens == 5

    def test_inline_sidechain_events_stay_chronological(self, tmp_path: Path):
        agent = ClaudeCode(logs_dir=tmp_path, model_name="claude-opus-4-6")
        events = [
            _make_user_event(
                "Fix the bug.",
                timestamp="2026-01-01T00:00:00Z",
            ),
            _sidechain(
                _make_user_event(
                    "explore the repo",
                    timestamp="2026-01-01T00:00:01Z",
                )
            ),
            _make_assistant_event(
                [{"type": "text", "text": "All done."}],
                timestamp="2026-01-01T00:00:02Z",
                msg_id="msg_main_1",
            ),
        ]
        session_dir = _write_session(tmp_path, events)

        trajectory = agent._convert_events_to_trajectory(session_dir)

        assert trajectory is not None
        assert [step.message for step in trajectory.steps] == [
            "Fix the bug.",
            "explore the repo",
            "All done.",
        ]
        assert (trajectory.steps[1].extra or {}).get("is_sidechain") is True

    def test_root_model_prefers_main_chain_over_earlier_sidechain(self, tmp_path: Path):
        agent = ClaudeCode(logs_dir=tmp_path, model_name=None)
        main_events = [
            _make_user_event(
                "Fix the bug.",
                timestamp="2026-01-01T00:00:00Z",
            ),
            _make_assistant_event(
                [{"type": "text", "text": "Done."}],
                timestamp="2026-01-01T00:00:03Z",
                model="claude-opus-4-6",
                msg_id="msg_main_1",
            ),
        ]
        subagent_events = [
            _sidechain(
                _make_assistant_event(
                    [{"type": "text", "text": "Repo explored."}],
                    timestamp="2026-01-01T00:00:01Z",
                    model="claude-haiku-4-5",
                    msg_id="msg_sub_1",
                )
            )
        ]
        session_dir = _write_session(tmp_path, main_events)
        _write_subagent_file(session_dir, subagent_events)

        trajectory = agent._convert_events_to_trajectory(session_dir)

        assert trajectory is not None
        assert trajectory.agent.model_name == "claude-opus-4-6"

    def test_production_session_layout_is_found_by_get_session_dir(
        self, tmp_path: Path
    ):
        agent = ClaudeCode(logs_dir=tmp_path, model_name="claude-opus-4-6")
        session_id = "0ae161f9-9d22-4708-8d5c-1f453f0eb05c"
        project_dir = tmp_path / "sessions" / "projects" / "-Users-anna-dev-app"
        subagents_dir = project_dir / session_id / "subagents"
        subagents_dir.mkdir(parents=True)
        main_events = [
            _make_user_event(
                "Fix the bug.",
                session_id=session_id,
                timestamp="2026-01-01T00:00:00Z",
            ),
            _make_assistant_event(
                [{"type": "text", "text": "All done."}],
                session_id=session_id,
                timestamp="2026-01-01T00:00:03Z",
                msg_id="msg_main_1",
            ),
        ]
        subagent_events = [
            _sidechain(
                _make_assistant_event(
                    [{"type": "text", "text": "Repo explored."}],
                    session_id=session_id,
                    timestamp="2026-01-01T00:00:01Z",
                    model="claude-haiku-4-5",
                    msg_id="msg_sub_1",
                ),
                agent_id="a4e70aa0cbe15c6ff",
            )
        ]
        (project_dir / f"{session_id}.jsonl").write_text(
            "\n".join(json.dumps(event) for event in main_events) + "\n"
        )
        (subagents_dir / "agent-a4e70aa0cbe15c6ff.jsonl").write_text(
            "\n".join(json.dumps(event) for event in subagent_events) + "\n"
        )

        session_dir = agent._get_session_dir()

        assert session_dir == project_dir
        assert session_dir is not None
        trajectory = agent._convert_events_to_trajectory(session_dir)
        assert trajectory is not None
        assert trajectory.session_id == session_id
        sidechain_steps = [
            step for step in trajectory.steps if (step.extra or {}).get("is_sidechain")
        ]
        assert [step.message for step in sidechain_steps] == ["Repo explored."]
        assert sidechain_steps[0].model_name == "claude-haiku-4-5"


class TestTrajectoryConversionRobustness:
    def test_user_text_content_block_is_unwrapped(self, tmp_path: Path):
        agent = ClaudeCode(logs_dir=tmp_path, model_name="claude-opus-4-6")
        skill_doc = (
            "Base directory for this skill: /logs/agent/sessions/skills/xlsx\n\n"
            "All Excel files must be deterministic."
        )
        events = [
            _make_user_event(
                [{"type": "text", "text": skill_doc}],
                timestamp="2026-01-01T00:00:00Z",
            ),
            _make_assistant_event(
                [{"type": "text", "text": "Got it."}],
                timestamp="2026-01-01T00:00:01Z",
            ),
        ]
        session_dir = _write_session(tmp_path, events)

        trajectory = agent._convert_events_to_trajectory(session_dir)

        assert trajectory is not None
        user_steps = [step for step in trajectory.steps if step.source == "user"]
        assert [step.message for step in user_steps] == [skill_doc]
        assert not user_steps[0].message.startswith('{"type":')

    def test_duplicate_session_uuid_tool_result_is_deduped(self, tmp_path: Path):
        agent = ClaudeCode(logs_dir=tmp_path, model_name="claude-opus-4-6")
        tool_use = _make_assistant_event(
            [
                {
                    "type": "tool_use",
                    "id": "toolu_duplicate",
                    "name": "Bash",
                    "input": {"command": "echo ok"},
                }
            ],
            timestamp="2026-01-01T00:00:01Z",
            msg_id="msg_tool",
        )
        tool_use["uuid"] = "assistant-tool-use"
        tool_result = _make_tool_result_event(
            tool_id="toolu_duplicate",
            content="ok",
            timestamp="2026-01-01T00:00:02Z",
        )
        tool_result["uuid"] = "duplicate-tool-result"
        duplicate_tool_result = {**tool_result}
        events = [
            _make_user_event("Run the command", timestamp="2026-01-01T00:00:00Z"),
            tool_use,
            tool_result,
            {
                "type": "system",
                "subtype": "compact_boundary",
                "uuid": "compact-boundary",
                "timestamp": "2026-01-01T00:00:03Z",
            },
            duplicate_tool_result,
            _make_assistant_event(
                [{"type": "text", "text": "Done."}],
                timestamp="2026-01-01T00:00:04Z",
            ),
        ]
        session_dir = _write_session(tmp_path, events)

        trajectory = agent._convert_events_to_trajectory(session_dir)

        assert trajectory is not None
        assert [step.step_id for step in trajectory.steps] == list(
            range(1, len(trajectory.steps) + 1)
        )
        tool_steps = [step for step in trajectory.steps if step.tool_calls]
        assert len(tool_steps) == 1
        assert tool_steps[0].tool_calls[0].function_name == "Bash"
        assert tool_steps[0].observation is not None
        assert tool_steps[0].observation.results[0].content == "ok"

    def test_duplicate_completed_tool_result_is_skipped(self, tmp_path: Path):
        agent = ClaudeCode(logs_dir=tmp_path, model_name="claude-opus-4-6")
        events = [
            _make_user_event("Run the command", timestamp="2026-01-01T00:00:00Z"),
            _make_assistant_event(
                [
                    {
                        "type": "tool_use",
                        "id": "toolu_completed",
                        "name": "Bash",
                        "input": {"command": "echo ok"},
                    }
                ],
                timestamp="2026-01-01T00:00:01Z",
                msg_id="msg_tool",
            ),
            _make_tool_result_event(
                tool_id="toolu_completed",
                content="ok",
                timestamp="2026-01-01T00:00:02Z",
            ),
            _make_tool_result_event(
                tool_id="toolu_completed",
                content="duplicate",
                timestamp="2026-01-01T00:00:03Z",
            ),
            _make_assistant_event(
                [{"type": "text", "text": "Done."}],
                timestamp="2026-01-01T00:00:04Z",
            ),
        ]
        session_dir = _write_session(tmp_path, events)

        trajectory = agent._convert_events_to_trajectory(session_dir)

        assert trajectory is not None
        assert len(trajectory.steps) == 3
        assert [step.step_id for step in trajectory.steps] == [1, 2, 3]
        tool_steps = [step for step in trajectory.steps if step.tool_calls]
        assert len(tool_steps) == 1
        assert tool_steps[0].observation is not None
        assert tool_steps[0].observation.results[0].content == "ok"
        assert trajectory.steps[-1].message == "Done."

    def test_orphan_tool_result_is_skipped_without_a_step_gap(self, tmp_path: Path):
        agent = ClaudeCode(logs_dir=tmp_path, model_name="claude-opus-4-6")
        events = [
            _make_user_event("Start", timestamp="2026-01-01T00:00:00Z"),
            _make_tool_result_event(
                tool_id="toolu_orphan",
                content="orphan output",
                timestamp="2026-01-01T00:00:01Z",
            ),
            _make_assistant_event(
                [{"type": "text", "text": "Still converted."}],
                timestamp="2026-01-01T00:00:02Z",
            ),
        ]
        session_dir = _write_session(tmp_path, events)

        trajectory = agent._convert_events_to_trajectory(session_dir)

        assert trajectory is not None
        assert [step.step_id for step in trajectory.steps] == [1, 2]
        assert not any(step.tool_calls for step in trajectory.steps)
        assert trajectory.steps[-1].message == "Still converted."


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model_name", "extra_env"),
    [
        ("anthropic/claude-opus-4-6", {}),
        (
            "anthropic/claude-opus-4-6",
            {"ANTHROPIC_BASE_URL": "https://proxy.example.com"},
        ),
        (
            "bedrock/global.anthropic.claude-sonnet-4-5-20250929-v1:0",
            {"CLAUDE_CODE_USE_BEDROCK": "1"},
        ),
    ],
    ids=["official-api", "custom-base-url", "bedrock"],
)
async def test_all_claude_model_channels_are_pinned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_name: str,
    extra_env: dict[str, str],
):
    for key in (
        "ANTHROPIC_BASE_URL",
        "CLAUDE_CODE_USE_BEDROCK",
        "AWS_BEARER_TOKEN_BEDROCK",
    ):
        monkeypatch.delenv(key, raising=False)

    agent = ClaudeCode(logs_dir=tmp_path, model_name=model_name, extra_env=extra_env)
    environment = FakeEnvironment()
    await agent.run("Fix the bug", cast(BaseEnvironment, environment), AgentContext())

    env = environment.envs[-1]
    assert env["ANTHROPIC_MODEL"]
    for key in (
        "CLAUDE_CODE_SUBAGENT_MODEL",
        "ANTHROPIC_DEFAULT_SONNET_MODEL",
        "ANTHROPIC_DEFAULT_OPUS_MODEL",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL",
        "ANTHROPIC_SMALL_FAST_MODEL",
    ):
        assert env[key] == env["ANTHROPIC_MODEL"]
