import json
from pathlib import Path
from typing import Any, cast

import pytest

from pier.agents.factory import AgentFactory
from pier.agents.installed.base import NonZeroAgentExitCodeError
from pier.agents.installed.pi import (
    _CURRENT_PI_PACKAGE,
    _LEGACY_PI_PACKAGE,
    Pi,
)
from pier.environments.base import BaseEnvironment, ExecResult
from pier.models.agent.context import AgentContext
from pier.models.agent.name import AgentName


class FakeEnvironment:
    session_id = "trial-session"

    def __init__(self, *, agent_install_spec: Any = None) -> None:
        self.agent_install_spec = agent_install_spec
        self.exec_calls: list[dict[str, Any]] = []

    def agent_process_env(self, env: dict[str, str] | None) -> dict[str, str] | None:
        return env

    async def exec(self, **kwargs: Any) -> ExecResult:
        self.exec_calls.append(kwargs)
        return ExecResult(return_code=0, stdout="", stderr="")

    @property
    def commands(self) -> list[str]:
        return [call["command"] for call in self.exec_calls]

    @property
    def envs(self) -> list[dict[str, str]]:
        return [call.get("env") or {} for call in self.exec_calls]


@pytest.fixture
def fake_environment() -> FakeEnvironment:
    return FakeEnvironment()


def write_stdout(path: Path, events: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(event) for event in events) + "\n")


def usage(
    *,
    input: int = 0,
    output: int = 0,
    cache_read: int = 0,
    cache_write: int = 0,
    reasoning: int = 0,
    cost: float = 0.0,
) -> dict:
    """A pi usage object, shaped as pi-ai emits it."""
    return {
        "input": input,
        "output": output,
        "cacheRead": cache_read,
        "cacheWrite": cache_write,
        "reasoning": reasoning,
        "totalTokens": input + output + cache_read + cache_write,
        "cost": {
            "input": 0.0,
            "output": 0.0,
            "cacheRead": 0.0,
            "cacheWrite": 0.0,
            "total": cost,
        },
    }


def assistant_message(
    *,
    content: list[dict] | None = None,
    stop_reason: str = "stop",
    timestamp: int = 1786379751430,
    response_id: str = "resp-1",
    model: str = "gpt-5.6-luna",
    **usage_kwargs,
) -> dict:
    return {
        "role": "assistant",
        "content": content
        if content is not None
        else [{"type": "text", "text": "done"}],
        "api": "openai-completions",
        "provider": "openai",
        "model": model,
        "responseId": response_id,
        "usage": usage(**usage_kwargs),
        "stopReason": stop_reason,
        "timestamp": timestamp,
    }


def tool_result(
    *,
    call_id: str = "call-1",
    name: str = "read",
    text: str = "hi",
    is_error: bool = False,
) -> dict:
    return {
        "role": "toolResult",
        "toolCallId": call_id,
        "toolName": name,
        "content": [{"type": "text", "text": text}],
        "isError": is_error,
        "timestamp": 1786379751500,
    }


@pytest.fixture
def agent(tmp_path: Path) -> Pi:
    return Pi(logs_dir=tmp_path, model_name="openai/gpt-5.6-luna")


class TestRegistration:
    def test_pi_is_registered(self, tmp_path: Path):
        created = AgentFactory.create_agent_from_name(
            AgentName.PI, logs_dir=tmp_path, model_name="openai/gpt-5.6-luna"
        )
        assert isinstance(created, Pi)
        assert Pi.name() == "pi"

    def test_supports_atif(self):
        assert Pi.SUPPORTS_ATIF is True


class TestInstall:
    def test_install_spec_uses_nvm_and_current_package(self, agent: Pi):
        spec = agent.install_spec()
        run = spec.steps[-1].run
        assert "nvm install 22" in run
        assert f"npm install -g --ignore-scripts {_CURRENT_PI_PACKAGE}@latest" in run
        assert spec.agent_name == "pi"

    def test_pinned_version_is_installed(self, tmp_path: Path):
        agent = Pi(
            logs_dir=tmp_path, model_name="openai/gpt-5.6-luna", version="0.84.1"
        )
        assert f"{_CURRENT_PI_PACKAGE}@0.84.1" in agent.install_spec().steps[-1].run

    @pytest.mark.parametrize("version", ["0.73.1", "0.1.0"])
    def test_legacy_package_for_pre_rename_versions(self, tmp_path: Path, version: str):
        agent = Pi(logs_dir=tmp_path, model_name="openai/gpt-5.6-luna", version=version)
        assert _LEGACY_PI_PACKAGE in agent.install_spec().steps[-1].run

    @pytest.mark.parametrize("version", ["0.74.0", "1.0.0"])
    def test_current_package_from_rename_version_onward(
        self, tmp_path: Path, version: str
    ):
        agent = Pi(logs_dir=tmp_path, model_name="openai/gpt-5.6-luna", version=version)
        assert _CURRENT_PI_PACKAGE in agent.install_spec().steps[-1].run

    def test_unparseable_version_falls_back_to_current_package(self, tmp_path: Path):
        agent = Pi(
            logs_dir=tmp_path, model_name="openai/gpt-5.6-luna", version="nightly"
        )
        assert _CURRENT_PI_PACKAGE in agent.install_spec().steps[-1].run

    def test_version_parsing_takes_last_line(self, agent: Pi):
        assert agent.parse_version("Downloading node...\n0.84.1\n") == "0.84.1"


class TestRunCommand:
    async def _run(self, agent: Pi, environment: FakeEnvironment) -> None:
        await agent.run(
            "Fix the bug", cast(BaseEnvironment, environment), AgentContext()
        )

    @pytest.mark.asyncio
    async def test_command_shape(self, agent: Pi, fake_environment):
        await self._run(agent, fake_environment)
        command = fake_environment.commands[-1]
        assert ". ~/.nvm/nvm.sh;" in command
        assert "pi --print --mode json" in command
        assert "--session-dir /logs/agent/pi/sessions" in command
        assert "--provider openai --model gpt-5.6-luna" in command
        assert "/logs/agent/pi.txt" in command

    @pytest.mark.asyncio
    async def test_instruction_is_piped_on_stdin(self, agent: Pi, fake_environment):
        # pi has no `--` separator, treats a leading `-` as an unknown option and a
        # leading `@` as a file reference, so the prompt must not be a positional.
        await agent.run(
            "--not-a-flag", cast(BaseEnvironment, fake_environment), AgentContext()
        )
        command = fake_environment.commands[-1]
        # bash's printf only parses options before the format operand, so a
        # dash-leading instruction is emitted literally rather than as a flag.
        assert "printf '%s' --not-a-flag | pi --print" in command
        assert " -- " not in command

    @pytest.mark.asyncio
    async def test_noisy_events_are_filtered_from_capture(
        self, agent: Pi, fake_environment
    ):
        await self._run(agent, fake_environment)
        command = fake_environment.commands[-1]
        assert "message_update" in command
        assert "entry_appended" in command
        assert "agent_end" in command

    @pytest.mark.asyncio
    async def test_offline_env_is_set_and_overridable(
        self, tmp_path: Path, fake_environment
    ):
        agent = Pi(logs_dir=tmp_path, model_name="openai/gpt-5.6-luna")
        await self._run(agent, fake_environment)
        assert fake_environment.envs[-1]["PI_OFFLINE"] == "1"
        assert fake_environment.envs[-1]["PI_SKIP_VERSION_CHECK"] == "1"

        overridden = Pi(
            logs_dir=tmp_path,
            model_name="openai/gpt-5.6-luna",
            extra_env={"PI_OFFLINE": "0"},
        )
        await self._run(overridden, fake_environment)
        assert fake_environment.envs[-1]["PI_OFFLINE"] == "0"

    @pytest.mark.asyncio
    async def test_provider_env_is_forwarded(self, tmp_path: Path, fake_environment):
        agent = Pi(
            logs_dir=tmp_path,
            model_name="anthropic/claude-opus-4-8",
            extra_env={
                "ANTHROPIC_API_KEY": "sk-ant",
                "ANTHROPIC_OAUTH_TOKEN": "oauth",
            },
        )
        await self._run(agent, fake_environment)
        env = fake_environment.envs[-1]
        assert env["ANTHROPIC_API_KEY"] == "sk-ant"
        assert env["ANTHROPIC_OAUTH_TOKEN"] == "oauth"

    @pytest.mark.asyncio
    async def test_ambient_keys_are_scoped_to_the_provider(
        self, tmp_path: Path, fake_environment, monkeypatch: pytest.MonkeyPatch
    ):
        """Only the requested provider's keys are read out of the ambient env.

        ``agent.env`` (``extra_env``) is deliberately forwarded wholesale by
        ``build_process_env`` as Pier's escape hatch, so scoping only applies to
        variables picked up from the process environment.
        """
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
        agent = Pi(logs_dir=tmp_path, model_name="anthropic/claude-opus-4-8")
        await self._run(agent, fake_environment)
        env = fake_environment.envs[-1]
        assert env["ANTHROPIC_API_KEY"] == "sk-ant"
        assert "OPENAI_API_KEY" not in env

    @pytest.mark.asyncio
    async def test_thinking_flag(self, tmp_path: Path, fake_environment):
        agent = Pi(
            logs_dir=tmp_path, model_name="openai/gpt-5.6-luna", thinking="medium"
        )
        await self._run(agent, fake_environment)
        assert "--thinking medium" in fake_environment.commands[-1]

    def test_thinking_accepts_max(self, tmp_path: Path):
        agent = Pi(logs_dir=tmp_path, model_name="openai/gpt-5.6-luna", thinking="max")
        assert "--thinking max" in agent.build_cli_flags()

    def test_thinking_rejects_unknown_level(self, tmp_path: Path):
        with pytest.raises(ValueError):
            Pi(logs_dir=tmp_path, model_name="openai/gpt-5.6-luna", thinking="ultra")

    @pytest.mark.asyncio
    async def test_model_name_must_include_provider(
        self, tmp_path: Path, fake_environment
    ):
        for model_name in (None, "gpt-5.6-luna"):
            agent = Pi(logs_dir=tmp_path, model_name=model_name)
            with pytest.raises(ValueError, match="provider/model_name"):
                await self._run(agent, fake_environment)

    @pytest.mark.asyncio
    async def test_skills_are_copied(self, tmp_path: Path, fake_environment):
        agent = Pi(
            logs_dir=tmp_path,
            model_name="openai/gpt-5.6-luna",
            skills_dir="/mnt/skills",
        )
        await self._run(agent, fake_environment)
        assert any(
            "$HOME/.agents/skills" in command for command in fake_environment.commands
        )


class TestModelsJson:
    def test_no_config_written_without_endpoint_or_key(self, agent: Pi):
        assert agent._build_register_models_command() is None

    def test_base_url_only_keeps_builtin_catalog(self, tmp_path: Path):
        """A baseUrl override must not declare a `models` entry.

        Declaring one for an id that matches a built-in replaces it and drops the
        shipped cost/capability metadata, so real OpenAI slugs stay untouched.
        """
        agent = Pi(
            logs_dir=tmp_path,
            model_name="openai/gpt-5.6-luna",
            extra_env={
                "OPENAI_BASE_URL": "https://proxy.example.com/v1",
                "OPENAI_API_KEY": "sk-test",
            },
        )
        config = agent._build_models_config()
        assert (
            config["providers"]["openai"]["baseUrl"] == "https://proxy.example.com/v1"
        )
        assert "models" not in config["providers"]["openai"]

    def test_api_key_is_referenced_by_env_name(self, tmp_path: Path):
        agent = Pi(
            logs_dir=tmp_path,
            model_name="openai/gpt-5.6-luna",
            extra_env={"OPENAI_API_KEY": "sk-secret"},
        )
        config = agent._build_models_config()
        assert config["providers"]["openai"]["apiKey"] == "$OPENAI_API_KEY"
        assert "sk-secret" not in json.dumps(config)

    def test_pi_config_deep_merges_and_adds_custom_slug(self, tmp_path: Path):
        agent = Pi(
            logs_dir=tmp_path,
            model_name="openai/my-proxy-slug",
            extra_env={"OPENAI_BASE_URL": "https://proxy.example.com/v1"},
            pi_config={
                "providers": {
                    "openai": {
                        "api": "openai-completions",
                        "models": [
                            {
                                "id": "my-proxy-slug",
                                "cost": {"input": 1.25, "output": 10.0},
                            }
                        ],
                    }
                }
            },
        )
        provider = agent._build_models_config()["providers"]["openai"]
        # Generated baseUrl survives the merge alongside the caller's keys.
        assert provider["baseUrl"] == "https://proxy.example.com/v1"
        assert provider["api"] == "openai-completions"
        assert provider["models"][0]["cost"]["input"] == 1.25

    def test_register_command_writes_models_json(self, tmp_path: Path):
        agent = Pi(
            logs_dir=tmp_path,
            model_name="openai/gpt-5.6-luna",
            extra_env={"OPENAI_BASE_URL": "https://proxy.example.com/v1"},
        )
        command = agent._build_register_models_command()
        assert command is not None
        assert "$HOME/.pi/agent/models.json" in command


class TestNetworkAllowlist:
    def _domains(self, agent: Pi) -> set[str]:
        return set(agent.network_allowlist().domains)

    def test_default_provider_domain(self, tmp_path: Path):
        agent = Pi(logs_dir=tmp_path, model_name="anthropic/claude-opus-4-8")
        assert "api.anthropic.com" in self._domains(agent)

    def test_custom_base_url_is_allowlisted(self, tmp_path: Path):
        agent = Pi(
            logs_dir=tmp_path,
            model_name="openai/gpt-5.6-luna",
            extra_env={"OPENAI_BASE_URL": "https://proxy.example.com/v1"},
        )
        assert {"api.openai.com", "proxy.example.com"} <= self._domains(agent)

    def test_base_url_from_pi_config_is_allowlisted(self, tmp_path: Path):
        agent = Pi(
            logs_dir=tmp_path,
            model_name="openai/my-proxy-slug",
            pi_config={
                "providers": {"openai": {"baseUrl": "https://gateway.example.com/v1"}}
            },
        )
        assert "gateway.example.com" in self._domains(agent)

    def test_no_allowlist_without_provider_prefix(self, tmp_path: Path):
        agent = Pi(logs_dir=tmp_path, model_name="gpt-5.6-luna")
        assert self._domains(agent) == set()


class TestTrajectoryConversion:
    def _convert(self, agent: Pi, events: list[dict]) -> dict:
        write_stdout(agent.logs_dir / "pi.txt", events)
        context = AgentContext()
        agent._instruction = "Fix the bug"
        agent.populate_context_post_run(context)
        return {
            "context": context,
            "trajectory": json.loads((agent.logs_dir / "trajectory.json").read_text()),
        }

    def _basic_events(self) -> list[dict]:
        message = assistant_message(
            content=[
                {"type": "thinking", "thinking": "checking the file"},
                {"type": "text", "text": "Reading it now."},
                {
                    "type": "toolCall",
                    "id": "call-1",
                    "name": "read",
                    "arguments": {"path": "note.txt"},
                },
            ],
            stop_reason="toolUse",
            input=100,
            output=50,
            cache_read=20,
            cache_write=10,
            reasoning=8,
            cost=0.005,
        )
        return [
            {"type": "session", "version": 3, "id": "session-uuid"},
            {"type": "message_end", "message": message},
            {
                "type": "turn_end",
                "message": message,
                "toolResults": [tool_result(text="hi")],
            },
        ]

    def test_missing_output_file_is_tolerated(self, agent: Pi):
        context = AgentContext()
        agent.populate_context_post_run(context)
        assert context.is_empty()

    def test_malformed_lines_are_skipped(self, agent: Pi):
        path = agent.logs_dir / "pi.txt"
        path.write_text(
            "not json\n"
            + json.dumps({"type": "session", "id": "s1"})
            + "\n\n"
            + json.dumps(
                {"type": "turn_end", "message": assistant_message(input=5, output=5)}
            )
            + "\n"
        )
        context = AgentContext()
        agent._instruction = "Fix the bug"
        agent.populate_context_post_run(context)
        trajectory = json.loads((agent.logs_dir / "trajectory.json").read_text())
        assert trajectory["session_id"] == "s1"

    def test_step_content_is_mapped(self, agent: Pi):
        trajectory = self._convert(agent, self._basic_events())["trajectory"]
        steps = trajectory["steps"]

        assert steps[0]["source"] == "user"
        assert steps[0]["message"] == "Fix the bug"

        step = steps[1]
        assert step["source"] == "agent"
        assert step["message"] == "Reading it now."
        # Thinking is kept out of `message` and carried separately.
        assert step["reasoning_content"] == "checking the file"
        assert step["tool_calls"][0]["function_name"] == "read"
        assert step["tool_calls"][0]["arguments"] == {"path": "note.txt"}
        assert step["observation"]["results"][0]["source_call_id"] == "call-1"
        assert step["observation"]["results"][0]["content"] == "hi"
        assert step["llm_call_count"] == 1
        assert step["extra"]["stop_reason"] == "toolUse"
        assert step["timestamp"].startswith("2026-")

    def test_turn_end_supersedes_matching_message_end(self, agent: Pi):
        """message_end then turn_end for one turn must yield a single step."""
        trajectory = self._convert(agent, self._basic_events())["trajectory"]
        agent_steps = [s for s in trajectory["steps"] if s["source"] == "agent"]
        assert len(agent_steps) == 1
        assert trajectory["final_metrics"]["total_steps"] == 2

    def test_trailing_message_end_without_turn_end_is_kept(self, agent: Pi):
        """A turn cut short by a timeout still contributes its step and usage."""
        events = self._basic_events()
        events.append(
            {
                "type": "message_end",
                "message": assistant_message(
                    response_id="resp-2", input=7, output=3, cost=0.001
                ),
            }
        )
        trajectory = self._convert(agent, events)["trajectory"]
        agent_steps = [s for s in trajectory["steps"] if s["source"] == "agent"]
        assert len(agent_steps) == 2
        assert trajectory["final_metrics"]["total_prompt_tokens"] == 137

    def test_token_and_cost_accounting(self, agent: Pi):
        result = self._convert(agent, self._basic_events())
        metrics = result["trajectory"]["steps"][1]["metrics"]
        # pi reports `input` net of cache, so prompt = input + cacheRead + cacheWrite.
        assert metrics["prompt_tokens"] == 130
        assert metrics["completion_tokens"] == 50
        assert metrics["cached_tokens"] == 20
        assert metrics["cost_usd"] == pytest.approx(0.005)
        assert metrics["extra"]["cache_write_tokens"] == 10
        assert metrics["extra"]["reasoning_tokens"] == 8

        context = result["context"]
        assert context.n_input_tokens == 130
        assert context.n_output_tokens == 50
        assert context.n_cache_tokens == 20
        assert context.cost_usd == pytest.approx(0.005)
        assert context.peak_context_tokens == 130

        final = result["trajectory"]["final_metrics"]
        assert final["extra"]["total_cache_write_tokens"] == 10
        assert final["extra"]["total_reasoning_tokens"] == 8

    def test_tool_reported_usage_is_included_in_totals(self, agent: Pi):
        """pi bills usage reported by tools, so the run totals must include it."""
        events = self._basic_events()
        events[-1]["toolResults"][0]["usage"] = usage(input=11, output=4, cost=0.002)
        final = self._convert(agent, events)["trajectory"]["final_metrics"]
        assert final["total_prompt_tokens"] == 141
        assert final["total_completion_tokens"] == 54
        assert final["total_cost_usd"] == pytest.approx(0.007)

    def test_tool_error_is_flagged(self, agent: Pi):
        events = self._basic_events()
        events[-1]["toolResults"] = [tool_result(text="boom", is_error=True)]
        trajectory = self._convert(agent, events)["trajectory"]
        result = trajectory["steps"][1]["observation"]["results"][0]
        assert result["extra"]["is_error"] is True

    def test_response_model_wins_over_requested_model(self, agent: Pi):
        events = self._basic_events()
        events[-1]["message"]["responseModel"] = "gpt-5.6-luna-2026-01"
        trajectory = self._convert(agent, events)["trajectory"]
        assert trajectory["steps"][1]["model_name"] == "gpt-5.6-luna-2026-01"

    def test_api_retry_count_is_recorded(self, agent: Pi):
        events = self._basic_events()
        events.insert(1, {"type": "auto_retry_start", "attempt": 1})
        events.insert(2, {"type": "auto_retry_start", "attempt": 2})
        final = self._convert(agent, events)["trajectory"]["final_metrics"]
        assert final["extra"]["api_retry_count"] == 2


class TestCompaction:
    def _events_with_compaction(self, *, aborted: bool = False) -> list[dict]:
        first = assistant_message(
            response_id="resp-1", input=100, output=10, cost=0.001
        )
        second = assistant_message(response_id="resp-2", input=40, output=5, cost=0.001)
        return [
            {"type": "session", "id": "s1"},
            {"type": "turn_end", "message": first, "toolResults": []},
            {"type": "compaction_start", "reason": "threshold"},
            {
                "type": "compaction_end",
                "reason": "threshold",
                "aborted": aborted,
                "willRetry": False,
                "result": {
                    "summary": "Earlier work summarised.",
                    "firstKeptEntryId": "abcd1234",
                    "tokensBefore": 120000,
                    "estimatedTokensAfter": 20000,
                    "usage": usage(input=1000, output=200, cost=0.004),
                },
            },
            {"type": "turn_end", "message": second, "toolResults": []},
        ]

    def _convert(self, agent: Pi, events: list[dict]) -> dict:
        write_stdout(agent.logs_dir / "pi.txt", events)
        context = AgentContext()
        agent._instruction = "Fix the bug"
        agent.populate_context_post_run(context)
        return {
            "context": context,
            "trajectory": json.loads((agent.logs_dir / "trajectory.json").read_text()),
        }

    def test_compaction_emits_system_step(self, agent: Pi):
        trajectory = self._convert(agent, self._events_with_compaction())["trajectory"]
        system_steps = [s for s in trajectory["steps"] if s["source"] == "system"]
        assert len(system_steps) == 1
        step = system_steps[0]
        assert step["message"] == "Earlier work summarised."
        assert step["extra"]["compaction"]["reason"] == "threshold"
        assert step["extra"]["compaction"]["tokens_before"] == 120000
        assert step["extra"]["compaction"]["estimated_tokens_after"] == 20000

    def test_compaction_is_counted_and_flagged(self, agent: Pi):
        result = self._convert(agent, self._events_with_compaction())
        final = result["trajectory"]["final_metrics"]
        assert final["extra"]["summarization_count"] == 1
        assert final["extra"]["compacted"] is True
        assert final["extra"]["compaction_reasons"] == ["threshold"]
        assert result["context"].summarization_count == 1

    def test_summarization_usage_is_billed_into_totals(self, agent: Pi):
        """pi's own totals include summary generation, so Pier's must too."""
        final = self._convert(agent, self._events_with_compaction())["trajectory"][
            "final_metrics"
        ]
        assert final["total_prompt_tokens"] == 1140
        assert final["total_completion_tokens"] == 215
        assert final["total_cost_usd"] == pytest.approx(0.006)

    def test_aborted_compaction_is_not_counted(self, agent: Pi):
        trajectory = self._convert(agent, self._events_with_compaction(aborted=True))[
            "trajectory"
        ]
        assert "summarization_count" not in (trajectory["final_metrics"]["extra"] or {})
        assert all(step["source"] != "system" for step in trajectory["steps"])

    def test_compaction_steps_do_not_count_as_agent_steps(self, agent: Pi):
        """`n_agent_steps` is derived from source=="agent" steps only."""
        trajectory = self._convert(agent, self._events_with_compaction())["trajectory"]
        agent_steps = [s for s in trajectory["steps"] if s["source"] == "agent"]
        assert len(agent_steps) == 2


class TestErrorSurfacing:
    def _write(self, agent: Pi, events: list[dict]) -> None:
        write_stdout(agent.logs_dir / "pi.txt", events)
        agent._instruction = "Fix the bug"

    def test_final_error_turn_raises(self, agent: Pi):
        self._write(
            agent,
            [
                {"type": "session", "id": "s1"},
                {
                    "type": "turn_end",
                    "message": assistant_message(stop_reason="error", input=5, output=0)
                    | {"errorMessage": "Connection error."},
                    "toolResults": [],
                },
            ],
        )
        with pytest.raises(NonZeroAgentExitCodeError, match="Connection error."):
            agent.populate_context_post_run(AgentContext())

    def test_recovered_intermediate_error_does_not_raise(self, agent: Pi):
        """pi retries provider errors itself; only the final turn decides failure."""
        self._write(
            agent,
            [
                {"type": "session", "id": "s1"},
                {
                    "type": "turn_end",
                    "message": assistant_message(
                        response_id="resp-1", stop_reason="error"
                    )
                    | {"errorMessage": "transient"},
                    "toolResults": [],
                },
                {"type": "auto_retry_start", "attempt": 1},
                {
                    "type": "turn_end",
                    "message": assistant_message(
                        response_id="resp-2", stop_reason="stop", input=10, output=5
                    ),
                    "toolResults": [],
                },
            ],
        )
        agent.populate_context_post_run(AgentContext())
        trajectory = json.loads((agent.logs_dir / "trajectory.json").read_text())
        assert trajectory["final_metrics"]["extra"]["stop_reason"] == "stop"

    def test_trajectory_is_written_before_raising(self, agent: Pi):
        """A failed run must still leave its trajectory behind for analysis."""
        self._write(
            agent,
            [
                {"type": "session", "id": "s1"},
                {
                    "type": "turn_end",
                    "message": assistant_message(
                        stop_reason="error", input=9, output=1
                    ),
                    "toolResults": [],
                },
            ],
        )
        with pytest.raises(NonZeroAgentExitCodeError):
            agent.populate_context_post_run(AgentContext())
        assert (agent.logs_dir / "trajectory.json").exists()


class TestCostFallback:
    def test_litellm_fallback_prices_known_model(self, tmp_path: Path):
        agent = Pi(logs_dir=tmp_path, model_name="anthropic/claude-haiku-4-5")
        cost = agent._compute_cost_from_pricing(1000, 500, 0)
        assert cost is not None and cost > 0

    def test_unknown_model_yields_no_cost(self, tmp_path: Path):
        agent = Pi(logs_dir=tmp_path, model_name="openai/definitely-not-a-real-slug")
        assert agent._compute_cost_from_pricing(1000, 500, 0) is None

    def test_pi_reported_cost_is_preferred(self, tmp_path: Path):
        agent = Pi(logs_dir=tmp_path, model_name="anthropic/claude-haiku-4-5")
        write_stdout(
            agent.logs_dir / "pi.txt",
            [
                {"type": "session", "id": "s1"},
                {
                    "type": "turn_end",
                    "message": assistant_message(input=1000, output=500, cost=0.25),
                    "toolResults": [],
                },
            ],
        )
        agent._instruction = "Fix the bug"
        context = AgentContext()
        agent.populate_context_post_run(context)
        assert context.cost_usd == pytest.approx(0.25)

    def test_zero_reported_cost_falls_back_to_litellm(self, tmp_path: Path):
        agent = Pi(logs_dir=tmp_path, model_name="anthropic/claude-haiku-4-5")
        write_stdout(
            agent.logs_dir / "pi.txt",
            [
                {"type": "session", "id": "s1"},
                {
                    "type": "turn_end",
                    "message": assistant_message(input=1000, output=500, cost=0.0),
                    "toolResults": [],
                },
            ],
        )
        agent._instruction = "Fix the bug"
        context = AgentContext()
        agent.populate_context_post_run(context)
        assert context.cost_usd is not None and context.cost_usd > 0
