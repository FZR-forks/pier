"""Focused tests for the OpenCode V2 (opencode-v2) Pier agent."""

import copy
import json
import logging
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest

from pier.agents.factory import AgentFactory
from pier.agents.installed.base import NonZeroAgentExitCodeError
from pier.agents.installed import opencode_v2_runner as runner_module
from pier.agents.installed.opencode_v2 import OpenCodeV2
from pier.environments.base import ExecResult
from pier.models.agent.context import AgentContext
from pier.models.agent.name import AgentName
from pier.models.task.config import MCPServerConfig

FIXTURES = Path(__file__).parent / "fixtures" / "opencode_v2"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeEnvironment:
    session_id = "trial-session"
    persistent_env: dict[str, str] = {}

    def __init__(self) -> None:
        self.exec_calls: list[dict[str, Any]] = []
        self.downloads: list[tuple[str, Path]] = []
        self.uploads: list[tuple[str, str, str]] = []

    def agent_process_env(self, env: dict[str, str] | None) -> dict[str, str] | None:
        return env

    async def exec(self, **kwargs: Any) -> ExecResult:
        self.exec_calls.append(kwargs)
        return ExecResult(return_code=0, stdout="", stderr="")

    async def download_file(self, remote: str, local: Path) -> None:
        self.downloads.append((remote, local))
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text("")

    async def upload_file(self, source: Path | str, target: str) -> None:
        source_path = Path(source)
        self.uploads.append((str(source_path), target, source_path.read_text()))


def make_agent(logs_dir: Path, **kwargs: Any) -> OpenCodeV2:
    kwargs.setdefault("model_name", "litellm/kimi-k3")
    kwargs.setdefault("variant", "max")
    kwargs.setdefault("version", "2.0.3")
    return OpenCodeV2(logs_dir=logs_dir, **kwargs)


def load_fixture(name: str = "root_child_inspections.json") -> list[dict[str, Any]]:
    data = json.loads((FIXTURES / name).read_text())
    return list(data.values())


def write_inspections(
    logs_dir: Path,
    inspections: list[dict[str, Any]],
) -> None:
    target = logs_dir / "opencode-v2" / "opencode-v2-sessions.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(json.dumps(item) + "\n" for item in inspections))
    session_ids = [
        str(item.get("session", {}).get("id"))
        for item in inspections
        if item.get("session", {}).get("id")
    ]
    roots = [
        str(item["session"]["id"])
        for item in inspections
        if item.get("session", {}).get("id")
        and not item.get("session", {}).get("parentID")
    ]
    (target.parent / "runner-result.json").write_text(
        json.dumps(
            {
                "collection_complete": True,
                "root_id": roots[0] if len(roots) == 1 else None,
                "discovered_session_ids": session_ids,
            }
        )
    )


# ---------------------------------------------------------------------------
# Registration and naming
# ---------------------------------------------------------------------------


def test_opencode_v2_is_registered():
    agent = AgentFactory.create_agent_from_name(
        AgentName.OPENCODE_V2,
        logs_dir=Path("/tmp/opencode-v2-test-logs"),
        model_name="litellm/kimi-k3",
        version="2.0.3",
    )
    assert agent.name() == "opencode-v2"
    assert AgentName("opencode-v2") is AgentName.OPENCODE_V2


def test_v1_opencode_registration_unchanged():
    agent = AgentFactory.create_agent_from_name(
        AgentName.OPENCODE,
        logs_dir=Path("/tmp/opencode-v2-test-logs"),
        model_name="litellm/kimi-k3",
    )
    assert agent.name() == "opencode"


def test_opencode_v2_is_in_installed_agents_for_ca_tests():
    from pier.agents.installed.base import BaseInstalledAgent

    assert issubclass(OpenCodeV2, BaseInstalledAgent)
    assert OpenCodeV2 in AgentFactory._AGENTS


# ---------------------------------------------------------------------------
# Config: providers/agents/model conflict
# ---------------------------------------------------------------------------


def test_config_carries_model_and_variant(tmp_path: Path):
    agent = make_agent(tmp_path, restrict_model=True)
    config = agent._build_runtime_config(include_mcp=False)
    assert config["providers"]["litellm"]["models"] == {"kimi-k3": {}}
    assert config["agents"]["build"]["model"] == "litellm/kimi-k3#max"
    assert config["agents"]["general"]["model"] == "litellm/kimi-k3#max"


def test_restrict_model_pins_builtin_agents(tmp_path: Path):
    agent = make_agent(tmp_path, restrict_model=True)
    config = agent._build_runtime_config(include_mcp=False)
    pinned = {name: agent_cfg["model"] for name, agent_cfg in config["agents"].items()}
    assert pinned == {
        "build": "litellm/kimi-k3#max",
        "plan": "litellm/kimi-k3#max",
        "general": "litellm/kimi-k3#max",
        "explore": "litellm/kimi-k3#max",
        "title": "litellm/kimi-k3#max",
        "summary": "litellm/kimi-k3#max",
        "compaction": "litellm/kimi-k3#max",
    }
    assert config["agents"]["title"]["disabled"] is True
    assert config["agents"]["summary"]["disabled"] is True


def test_restrict_model_overwrites_caller_model_contamination(tmp_path: Path):
    agent = make_agent(
        tmp_path,
        restrict_model=True,
        opencode_v2_config={
            "agents": {
                "build": {
                    "model": "openai/gpt-5.6-luna",
                    "permission": {"read": "allow"},
                }
            },
        },
    )
    config = agent._build_runtime_config(include_mcp=False)
    assert config["model"] == "litellm/kimi-k3#max"
    assert config["agents"]["build"]["model"] == "litellm/kimi-k3#max"
    assert config["agents"]["build"]["permission"] == {"read": "allow"}


def test_restrict_model_rejects_extra_provider_model(tmp_path: Path):
    agent = make_agent(
        tmp_path,
        restrict_model=True,
        opencode_v2_config={
            "providers": {
                "litellm": {"models": {"other-model": {}}},
            },
        },
    )
    with pytest.raises(ValueError, match="other-model"):
        agent._build_runtime_config(include_mcp=False)


def test_restrict_model_rejects_wrong_top_level_model(tmp_path: Path):
    agent = make_agent(tmp_path, restrict_model=True)
    config = agent._build_runtime_config(include_mcp=False)
    config["model"] = "openai/gpt-5.6-luna"
    with pytest.raises(ValueError, match="could resolve models other than"):
        agent._assert_single_model(config)


# ---------------------------------------------------------------------------
# Caller immutability
# ---------------------------------------------------------------------------


def test_caller_opencode_v2_config_is_never_mutated(tmp_path: Path):
    caller_config = {
        "provider": {
            "litellm": {
                "models": {"kimi-k3": {"limit": {"context": 1048576, "output": 131072}}}
            }
        },
    }
    frozen = copy.deepcopy(caller_config)
    agent = make_agent(tmp_path, opencode_v2_config=caller_config)
    for _ in range(2):
        agent._build_runtime_config(include_mcp=True)
        assert caller_config == frozen


def test_runtime_config_is_fresh_across_calls(tmp_path: Path):
    agent = make_agent(tmp_path)
    first = agent._build_runtime_config(include_mcp=False)
    second = agent._build_runtime_config(include_mcp=False)
    first["providers"]["litellm"]["models"]["kimi-k3"]["injected"] = True
    assert "injected" not in second["providers"]["litellm"]["models"]["kimi-k3"]


# ---------------------------------------------------------------------------
# Model body override / transport correctness
# ---------------------------------------------------------------------------


def test_output_limit_metadata_does_not_synthesize_transport_body(tmp_path: Path):
    agent = make_agent(
        tmp_path,
        opencode_v2_config={
            "providers": {
                "litellm": {
                    "models": {
                        "kimi-k3": {"limit": {"context": 1048576, "output": 131072}}
                    }
                }
            },
        },
    )
    config = agent._build_runtime_config(include_mcp=False)
    model = config["providers"]["litellm"]["models"]["kimi-k3"]
    assert model["limit"]["output"] == 131072
    assert "body" not in model


def test_model_level_responses_package_preserves_explicit_body(tmp_path: Path):
    agent = make_agent(
        tmp_path,
        opencode_v2_config={
            "providers": {
                "litellm": {
                    "package": "aisdk:@ai-sdk/openai-compatible",
                    "models": {
                        "kimi-k3": {
                            "package": (
                                "@opencode-ai/ai/providers/openai-compatible/responses"
                            ),
                            "limit": {"output": 131072},
                            "body": {"max_output_tokens": 8192},
                        }
                    },
                }
            },
        },
    )
    config = agent._build_runtime_config(include_mcp=False)
    body = config["providers"]["litellm"]["models"]["kimi-k3"]["body"]
    assert body == {"max_output_tokens": 8192}


def test_model_level_responses_package_rejects_chat_output_field(tmp_path: Path):
    agent = make_agent(
        tmp_path,
        opencode_v2_config={
            "providers": {
                "litellm": {
                    "package": "aisdk:@ai-sdk/openai-compatible",
                    "models": {
                        "kimi-k3": {
                            "package": (
                                "@opencode-ai/ai/providers/openai-compatible/responses"
                            ),
                            "limit": {"output": 131072},
                            "body": {"max_tokens": 8192},
                        }
                    },
                }
            }
        },
    )
    with pytest.raises(ValueError, match="must be max_output_tokens"):
        agent._build_runtime_config(include_mcp=False)


def test_explicit_body_override_is_preserved_when_restrict_model(tmp_path: Path):
    agent = make_agent(
        tmp_path,
        restrict_model=True,
        opencode_v2_config={
            "providers": {
                "litellm": {
                    "models": {
                        "kimi-k3": {
                            "limit": {"output": 64000},
                            "body": {"max_tokens": 64000},
                        }
                    },
                }
            },
        },
    )
    config = agent._build_runtime_config(include_mcp=False)
    body = config["providers"]["litellm"]["models"]["kimi-k3"]["body"]
    assert body == {"max_tokens": 64000}
    assert (
        config["providers"]["litellm"]["models"]["kimi-k3"]["limit"]["output"] == 64000
    )


def test_test_only_smaller_body_cap_is_allowed(tmp_path: Path):
    agent = make_agent(
        tmp_path,
        opencode_v2_config={
            "providers": {
                "litellm": {
                    "models": {
                        "kimi-k3": {
                            "limit": {"output": 54321},
                            "body": {"max_tokens": 8192},
                        }
                    }
                }
            }
        },
    )
    config = agent._build_runtime_config(include_mcp=False)
    model = config["providers"]["litellm"]["models"]["kimi-k3"]
    assert model["limit"]["output"] == 54321
    assert model["body"] == {"max_tokens": 8192}


def test_no_output_limit_means_no_body_override(tmp_path: Path):
    agent = make_agent(tmp_path)
    config = agent._build_runtime_config(include_mcp=False)
    assert "options" not in config["providers"]["litellm"]


def test_v1_provider_options_syntax_is_rejected(tmp_path: Path):
    agent = make_agent(
        tmp_path,
        opencode_v2_config={
            "providers": {
                "litellm": {
                    "options": {"body": {"max_completion_tokens": 999}},
                    "models": {"kimi-k3": {"limit": {"output": 131072}}},
                }
            },
        },
    )
    with pytest.raises(ValueError, match="V1 compatibility syntax"):
        agent._build_runtime_config(include_mcp=False)


# ---------------------------------------------------------------------------
# Network allowlist
# ---------------------------------------------------------------------------


def test_allowlist_defaults_to_provider_domain(tmp_path: Path):
    agent = make_agent(tmp_path, model_name="anthropic/claude-opus-5")
    assert "api.anthropic.com" in agent.network_allowlist().domains


def test_allowlist_picks_up_base_url(tmp_path: Path):
    agent = make_agent(
        tmp_path,
        model_name="openai/kimi-k3",
        variant="max",
        extra_env={"OPENAI_BASE_URL": "https://gateway.example.com/v1"},
    )
    assert "gateway.example.com" in agent.network_allowlist().domains


def test_allowlist_ignores_openai_base_url_for_other_provider(tmp_path: Path):
    agent = make_agent(
        tmp_path,
        extra_env={"OPENAI_BASE_URL": "http://unrelated.example.com/v1"},
    )
    assert "unrelated.example.com" not in agent.network_allowlist().domains


def test_allowlist_picks_up_config_urls(tmp_path: Path):
    agent = make_agent(
        tmp_path,
        opencode_v2_config={
            "providers": {
                "litellm": {"settings": {"baseURL": "https://gw.example.com/v1"}}
            },
        },
    )
    assert "gw.example.com" in agent.network_allowlist().domains


def test_allowlist_includes_enabled_agent_provider_urls(tmp_path: Path):
    agent = make_agent(
        tmp_path,
        opencode_v2_config={
            "providers": {
                "openai": {
                    "settings": {"baseURL": "https://child-gateway.example.com/v1"}
                }
            },
            "agents": {"general": {"model": "openai/child-model#low"}},
        },
    )

    domains = agent.network_allowlist().domains
    assert "child-gateway.example.com" in domains


def test_allowlist_includes_top_level_inherited_agent_provider(tmp_path: Path):
    agent = make_agent(
        tmp_path,
        opencode_v2_config={
            "model": "openai/inherited-model#low",
            "providers": {
                "openai": {
                    "settings": {"baseURL": "https://inherited-gateway.example.com/v1"}
                }
            },
            # An enabled agent with no model inherits config.model.
            "agents": {"general": {"permission": {"read": "allow"}}},
        },
    )

    assert "inherited-gateway.example.com" in agent.network_allowlist().domains


def test_allowlist_includes_enabled_agent_default_provider_and_openai_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://child-openai.example.com/v1")
    agent = make_agent(
        tmp_path,
        opencode_v2_config={"agents": {"general": {"model": "openai/child-model#low"}}},
    )

    domains = agent.network_allowlist().domains
    assert "api.openai.com" in domains
    assert "child-openai.example.com" in domains


def test_allowlist_ignores_disabled_agent_provider(tmp_path: Path):
    agent = make_agent(
        tmp_path,
        opencode_v2_config={
            "providers": {
                "anthropic": {
                    "settings": {"baseURL": "https://disabled.example.com/v1"}
                }
            },
            "agents": {
                "explore": {
                    "model": "anthropic/child-model",
                    "disabled": True,
                }
            },
        },
    )

    assert "disabled.example.com" not in agent.network_allowlist().domains


def test_restricted_allowlist_does_not_admit_contaminating_agent_provider(
    tmp_path: Path,
):
    agent = make_agent(
        tmp_path,
        restrict_model=True,
        opencode_v2_config={
            "providers": {
                "openai": {
                    "settings": {"baseURL": "https://contaminating.example.com/v1"}
                }
            },
            "agents": {"general": {"model": "openai/other-model#high"}},
        },
    )

    assert "contaminating.example.com" not in agent.network_allowlist().domains


def test_allowlist_resolves_config_env_template(tmp_path: Path):
    agent = make_agent(
        tmp_path,
        extra_env={
            "LITELLM_OPENAI_BASE_URL": "https://templated-gateway.example.com/v1",
            "LITELLM_API_KEY": "sk.secret.value",
        },
        opencode_v2_config={
            "providers": {
                "litellm": {
                    "settings": {
                        "baseURL": "{env:LITELLM_OPENAI_BASE_URL}",
                        "apiKey": "{env:LITELLM_API_KEY}",
                    }
                }
            }
        },
    )
    assert "templated-gateway.example.com" in agent.network_allowlist().domains
    assert "sk.secret.value" not in agent.network_allowlist().domains


def test_allowlist_includes_remote_mcp_host(tmp_path: Path):
    agent = make_agent(
        tmp_path,
        mcp_servers=[
            MCPServerConfig(
                name="docs",
                transport="streamable-http",
                url="https://mcp.example.com/api",
            )
        ],
    )
    assert "mcp.example.com" in agent.network_allowlist().domains


def test_allowlist_rejects_unresolved_url_template(tmp_path: Path):
    agent = make_agent(
        tmp_path,
        opencode_v2_config={
            "providers": {
                "litellm": {"settings": {"baseURL": "{env:TASK_ONLY_GATEWAY_URL}"}}
            }
        },
    )
    with pytest.raises(ValueError, match="provide it through agent.env"):
        agent.network_allowlist()


def test_allowlist_rejects_partially_templated_url(tmp_path: Path):
    agent = make_agent(
        tmp_path,
        opencode_v2_config={
            "providers": {
                "litellm": {"settings": {"baseURL": "https://{env:GATEWAY_HOST}/v1"}}
            }
        },
    )

    with pytest.raises(ValueError, match="only partially templated"):
        agent.network_allowlist()


def test_provider_url_requires_https_except_loopback(tmp_path: Path):
    remote = make_agent(
        tmp_path,
        opencode_v2_config={
            "providers": {
                "litellm": {"settings": {"baseURL": "http://gateway.example.com/v1"}}
            }
        },
    )
    with pytest.raises(ValueError, match="must use HTTPS"):
        remote.network_allowlist()

    malformed = make_agent(
        tmp_path,
        opencode_v2_config={
            "providers": {"litellm": {"settings": {"baseURL": "https://"}}}
        },
    )
    with pytest.raises(ValueError, match="must include a hostname"):
        malformed.network_allowlist()

    loopback = make_agent(
        tmp_path,
        opencode_v2_config={
            "providers": {
                "litellm": {"settings": {"baseURL": "http://127.0.0.1:8080/v1"}}
            }
        },
    )
    assert "127.0.0.1" in loopback.network_allowlist().domains


# ---------------------------------------------------------------------------
# Environment restrictions and AGENTS.md
# ---------------------------------------------------------------------------


def test_runner_env_is_restricted(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENCODE_CONFIG_PROJECT_DISABLE", "")
    agent = make_agent(tmp_path)
    env = agent.build_process_env(
        {"OPENCODE_CONFIG_PROJECT_DISABLE": "1", "OPENCODE_DISABLE_MODELS_FETCH": "1"}
    )
    assert env["OPENCODE_CONFIG_PROJECT_DISABLE"] == "1"
    assert env["OPENCODE_DISABLE_MODELS_FETCH"] == "1"


def test_run_writes_config_to_private_home(tmp_path: Path):
    environment = FakeEnvironment()
    agent = make_agent(
        tmp_path,
        opencode_v2_config={
            "providers": {"litellm": {"settings": {"apiKey": "direct-secret"}}}
        },
    )

    import asyncio

    asyncio.run(agent.run("do the thing", environment, AgentContext()))

    config_uploads = [
        upload for upload in environment.uploads if upload[1].endswith("opencode.json")
    ]
    assert config_uploads, "config file was not uploaded"
    assert config_uploads[0][2].find('"apiKey": "direct-secret"') >= 0
    assert all(
        "direct-secret" not in call["command"] for call in environment.exec_calls
    )
    command = next(
        call["command"]
        for call in environment.exec_calls
        if "mkdir -p" in call["command"]
    )
    # Private per-trial HOME, config path injected via OPENCODE_CONFIG.
    assert "/tmp/opencode-v2-home-" in command
    assert config_uploads[0][1].endswith("config/opencode/opencode.json")
    assert "mkdir -p" in command


def test_run_never_passes_server_password_through_logged_exec_env(tmp_path: Path):
    environment = FakeEnvironment()
    agent = make_agent(tmp_path)

    import asyncio

    asyncio.run(agent.run("do the thing", environment, AgentContext()))

    assert environment.exec_calls
    for call in environment.exec_calls:
        assert "OPENCODE_PASSWORD" not in (call.get("env") or {})
        assert "OPENCODE_SERVER_PASSWORD" not in (call.get("env") or {})


def test_run_forwards_ambient_config_template_values(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CUSTOM_PROVIDER_HEADER", "ambient-value")
    environment = FakeEnvironment()
    agent = make_agent(
        tmp_path,
        opencode_v2_config={
            "providers": {
                "litellm": {
                    "settings": {
                        "headers": {"X-Custom": "{env:CUSTOM_PROVIDER_HEADER}"}
                    }
                }
            }
        },
    )

    import asyncio

    asyncio.run(agent.run("do the thing", environment, AgentContext()))

    assert environment.exec_calls
    assert all(
        call["env"]["CUSTOM_PROVIDER_HEADER"] == "ambient-value"
        for call in environment.exec_calls
    )


def test_run_forwards_and_redacts_config_template_in_list(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    secret = "mcp-secret-that-must-be-redacted"
    monkeypatch.setenv("MCP_SECRET", secret)
    environment = FakeEnvironment()
    agent = make_agent(
        tmp_path,
        opencode_v2_config={
            "mcp": {
                "fixture": {
                    "type": "local",
                    "command": ["fixture-mcp", "--key", "{env:MCP_SECRET}"],
                }
            }
        },
    )
    caplog.set_level(logging.DEBUG)

    import asyncio

    asyncio.run(agent.run("do the thing", environment, AgentContext()))

    assert all(call["env"]["MCP_SECRET"] == secret for call in environment.exec_calls)
    env_records = [record for record in caplog.records if hasattr(record, "env")]
    assert env_records
    assert all(secret not in repr(record.__dict__) for record in env_records)
    assert any(record.env.get("MCP_SECRET") == "<redacted>" for record in env_records)


def test_run_rejects_missing_config_template_in_list(tmp_path: Path):
    environment = FakeEnvironment()
    agent = make_agent(
        tmp_path,
        opencode_v2_config={
            "mcp": {
                "fixture": {
                    "type": "local",
                    "command": ["fixture-mcp", "{env:MISSING_MCP_SECRET}"],
                }
            }
        },
    )

    import asyncio

    with pytest.raises(ValueError, match="MISSING_MCP_SECRET is not set"):
        asyncio.run(agent.run("do the thing", environment, AgentContext()))


def test_config_template_values_are_redacted_only_from_debug_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    secret = "credential-that-must-not-enter-debug-metadata"
    monkeypatch.setenv("CUSTOM_PROVIDER_HEADER", secret)
    environment = FakeEnvironment()
    agent = make_agent(
        tmp_path,
        opencode_v2_config={
            "providers": {
                "litellm": {
                    "settings": {
                        "headers": {"X-Custom": "{env:CUSTOM_PROVIDER_HEADER}"}
                    }
                }
            }
        },
    )
    caplog.set_level(logging.DEBUG)

    import asyncio

    asyncio.run(agent.run("do the thing", environment, AgentContext()))

    assert environment.exec_calls
    assert all(
        call["env"]["CUSTOM_PROVIDER_HEADER"] == secret
        for call in environment.exec_calls
    )
    env_records = [record for record in caplog.records if hasattr(record, "env")]
    assert env_records
    assert all(secret not in repr(record.__dict__) for record in env_records)
    assert any(
        record.env.get("CUSTOM_PROVIDER_HEADER") == "<redacted>"
        for record in env_records
    )


@pytest.mark.parametrize(
    ("model_name", "env_name"),
    [
        ("google/gemini-test", "GOOGLE_API_KEY"),
        ("google/gemini-test", "GOOGLE_GENAI_USE_VERTEXAI"),
        ("llama/llama-test", "LLAMA_API_KEY"),
        ("amazon-bedrock/claude-test", "AWS_SESSION_TOKEN"),
    ],
)
def test_run_forwards_v1_provider_environment_parity(
    tmp_path: Path, monkeypatch, model_name: str, env_name: str
):
    monkeypatch.setenv(env_name, "configured-value")
    environment = FakeEnvironment()
    agent = make_agent(tmp_path, model_name=model_name)

    import asyncio

    asyncio.run(agent.run("do the thing", environment, AgentContext()))

    assert environment.exec_calls[-1]["env"][env_name] == "configured-value"


def test_server_password_env_is_reserved(tmp_path: Path):
    with pytest.raises(ValueError, match="runner-owned"):
        make_agent(tmp_path, extra_env={"OPENCODE_PASSWORD": "must-not-be-logged"})


def test_run_preserves_project_agents_md_discovery(tmp_path: Path):
    """OPENCODE_CONFIG_PROJECT_DISABLE=1 must not disable AGENTS.md discovery.

    The env var only suppresses project opencode.json discovery, which the
    adapter already prevents by pointing OPENCODE_CONFIG at the generated
    file. So the flag stays set and AGENTS.md keeps working.
    """
    environment = FakeEnvironment()
    agent = make_agent(tmp_path)

    import asyncio

    asyncio.run(agent.run("do the thing", environment, AgentContext()))

    run_calls = [
        call
        for call in environment.exec_calls
        if "opencode_v2_runner.py" in call["command"]
        and "--logs-dir" in call["command"]
    ]
    assert run_calls
    runner_command = run_calls[0]["command"]
    assert str(agent._RUNNER_PATH) in runner_command
    assert "--restrict-model" not in runner_command
    assert "2>&1 </dev/null | stdbuf -oL tee" in runner_command
    assert "> >(stdbuf" not in runner_command


def test_run_copies_skills_into_isolated_config_dir(tmp_path: Path):
    environment = FakeEnvironment()
    agent = make_agent(tmp_path, skills_dir="/mnt/task-skills")

    import asyncio

    asyncio.run(agent.run("do the thing", environment, AgentContext()))

    setup_call = next(
        call
        for call in environment.exec_calls
        if 'mkdir -p "$OPENCODE_CONFIG_DIR/skills"' in call["command"]
    )
    setup_command = setup_call["command"]
    assert "/mnt/task-skills/*" in setup_command
    assert "~/.config/opencode/skills" not in setup_command
    assert setup_call["env"]["OPENCODE_CONFIG_DIR"].endswith("/config/opencode")


def test_run_passes_model_restriction_to_runner(tmp_path: Path):
    environment = FakeEnvironment()
    agent = make_agent(tmp_path, restrict_model=True)

    import asyncio

    asyncio.run(agent.run("do the thing", environment, AgentContext()))

    runner_command = next(
        call["command"]
        for call in environment.exec_calls
        if "opencode_v2_runner.py" in call["command"]
        and "--logs-dir" in call["command"]
    )
    assert "--restrict-model" in runner_command


# ---------------------------------------------------------------------------
# Conversion: fixture totals
# ---------------------------------------------------------------------------


def test_fixture_tree_totals(tmp_path: Path):
    agent = make_agent(tmp_path)
    write_inspections(tmp_path, load_fixture())
    context = AgentContext()

    agent.populate_context_post_run(context)

    trajectory = json.loads((tmp_path / "trajectory.json").read_text())

    # Root completion includes output plus reasoning: 10+5 + 20+15 = 50.
    # Child completion is 40+30 = 70; tree completion is 120.
    metrics = trajectory["final_metrics"]
    assert metrics["total_prompt_tokens"] == 223
    assert metrics["total_completion_tokens"] == 120
    assert metrics["total_cached_tokens"] == 30
    assert metrics["extra"]["total_cache_write_input_tokens"] == 33
    assert metrics["extra"]["total_reasoning_tokens"] == 50
    assert metrics["extra"]["subagent_count"] == 1
    assert metrics["extra"]["tree_metrics_complete"] is True
    assert metrics["extra"]["tree_cost_complete"] is True

    self_only = metrics["extra"]["self_only"]
    assert self_only["total_prompt_tokens"] == 121
    assert self_only["total_completion_tokens"] == 50
    assert self_only["total_cached_tokens"] == 10
    assert self_only["total_cache_write_input_tokens"] == 11

    assert context.n_input_tokens == 223
    assert context.n_output_tokens == 120
    assert context.n_cache_tokens == 30

    children = trajectory["subagent_trajectories"]
    assert len(children) == 1
    assert children[0]["trajectory_id"] == "ses_child0000000000000000000001"
    assert children[0]["extra"]["parent_session_id"] == (
        "ses_root0000000000000000000001"
    )
    assert children[0]["extra"]["is_subagent"] is True


def test_spec_reference_totals(tmp_path: Path):
    """Canonical root/child normalized usage from the implementation brief."""
    inspections = [
        {
            "session": {
                "id": "ses_root0000000000000000000001",
                "cost": 2.0,
                "tokens": {
                    "input": 100,
                    "output": 30,
                    "reasoning": 20,
                    "cache": {"read": 40, "write": 60},
                },
            },
            "messages": [
                {
                    "type": "assistant",
                    "id": "msg_a1",
                    "agent": "build",
                    "model": {"id": "kimi-k3", "providerID": "litellm"},
                    "time": {"created": 1, "completed": 2},
                    "finish": "stop",
                    "cost": 2.0,
                    "tokens": {
                        "input": 100,
                        "output": 40,
                        "reasoning": 60,
                        "cache": {"read": 30, "write": 20},
                    },
                    "content": [{"type": "text", "text": "done"}],
                }
            ],
            "active": None,
            "terminal": None,
        },
        {
            "session": {
                "id": "ses_child0000000000000000000001",
                "parentID": "ses_root0000000000000000000001",
                "cost": 0.5,
                "tokens": {
                    "input": 10,
                    "output": 3,
                    "reasoning": 2,
                    "cache": {"read": 4, "write": 6},
                },
            },
            "messages": [
                {
                    "type": "assistant",
                    "id": "msg_ca1",
                    "agent": "explore",
                    "model": {"id": "kimi-k3", "providerID": "litellm"},
                    "time": {"created": 1, "completed": 2},
                    "finish": "stop",
                    "cost": 0.5,
                    "tokens": {
                        "input": 10,
                        "output": 4,
                        "reasoning": 6,
                        "cache": {"read": 3, "write": 2},
                    },
                    "content": [{"type": "text", "text": "explored"}],
                }
            ],
            "active": None,
            "terminal": None,
        },
    ]
    agent = make_agent(tmp_path)
    write_inspections(tmp_path, inspections)
    context = AgentContext()

    agent.populate_context_post_run(context)

    metrics = json.loads((tmp_path / "trajectory.json").read_text())["final_metrics"]
    root = metrics["extra"]["self_only"]
    assert root["total_prompt_tokens"] == 150
    assert root["total_completion_tokens"] == 100
    assert root["total_cached_tokens"] == 30
    assert root["total_cache_write_input_tokens"] == 20
    assert metrics["total_prompt_tokens"] == 165
    assert metrics["total_completion_tokens"] == 110
    assert metrics["total_cached_tokens"] == 33
    assert metrics["extra"]["total_cache_write_input_tokens"] == 22
    assert metrics["extra"]["total_reasoning_tokens"] == 66
    assert metrics["extra"]["subagent_count"] == 1
    assert context.n_input_tokens == metrics["total_prompt_tokens"]


def test_nested_tree_totals_and_parentage(tmp_path: Path):
    """A grandchild contributes once and remains nested below its real parent."""
    inspections = [
        {
            "session": {"id": "ses_root0000000000000000000001"},
            "messages": [
                {
                    "type": "assistant",
                    "id": "msg_root",
                    "agent": "build",
                    "model": {"id": "kimi-k3", "providerID": "litellm"},
                    "time": {"created": 1, "completed": 2},
                    "finish": "stop",
                    "cost": 1.0,
                    "tokens": {
                        "input": 100,
                        "output": 40,
                        "reasoning": 60,
                        "cache": {"read": 30, "write": 20},
                    },
                    "content": [{"type": "text", "text": "root"}],
                }
            ],
            "active": None,
            "terminal": None,
        },
        {
            "session": {
                "id": "ses_child0000000000000000000001",
                "parentID": "ses_root0000000000000000000001",
            },
            "messages": [
                {
                    "type": "assistant",
                    "id": "msg_child",
                    "agent": "general",
                    "model": {"id": "kimi-k3", "providerID": "litellm"},
                    "time": {"created": 3, "completed": 4},
                    "finish": "stop",
                    "cost": 0.5,
                    "tokens": {
                        "input": 10,
                        "output": 4,
                        "reasoning": 6,
                        "cache": {"read": 3, "write": 2},
                    },
                    "content": [{"type": "text", "text": "child"}],
                }
            ],
            "active": None,
            "terminal": None,
        },
        {
            "session": {
                "id": "ses_grandchild000000000000001",
                "parentID": "ses_child0000000000000000000001",
            },
            "messages": [
                {
                    "type": "assistant",
                    "id": "msg_grandchild",
                    "agent": "explore",
                    "model": {"id": "kimi-k3", "providerID": "litellm"},
                    "time": {"created": 5, "completed": 6},
                    "finish": "stop",
                    "cost": 0.1,
                    "tokens": {
                        "input": 1,
                        "output": 4,
                        "reasoning": 5,
                        "cache": {"read": 2, "write": 3},
                    },
                    "content": [{"type": "text", "text": "grandchild"}],
                }
            ],
            "active": None,
            "terminal": None,
        },
    ]
    agent = make_agent(tmp_path)
    write_inspections(tmp_path, inspections)
    agent.populate_context_post_run(AgentContext())

    trajectory = json.loads((tmp_path / "trajectory.json").read_text())
    metrics = trajectory["final_metrics"]
    assert metrics["total_prompt_tokens"] == 171
    assert metrics["total_completion_tokens"] == 119
    assert metrics["total_cached_tokens"] == 35
    assert metrics["extra"]["total_cache_write_input_tokens"] == 25
    assert metrics["extra"]["total_reasoning_tokens"] == 71
    assert metrics["extra"]["subagent_count"] == 2

    child = trajectory["subagent_trajectories"][0]
    assert child["trajectory_id"] == "ses_child0000000000000000000001"
    assert len(child["subagent_trajectories"]) == 1
    assert child["subagent_trajectories"][0]["trajectory_id"] == (
        "ses_grandchild000000000000001"
    )


def test_prompt_completion_cache_normalization(tmp_path: Path):
    """prompt = input + cache.read + cache.write; completion = output + reasoning;
    cached = cache.read; extras carry reasoning and cache write."""
    inspection = {
        "session": {"id": "ses_solo0000000000000000000001"},
        "messages": [
            {
                "type": "assistant",
                "id": "msg_x1",
                "agent": "build",
                "model": {"id": "m", "providerID": "p"},
                "time": {"created": 1, "completed": 2},
                "finish": "stop",
                "cost": 0.1,
                "tokens": {
                    "input": 5,
                    "output": 7,
                    "reasoning": 3,
                    "cache": {"read": 4, "write": 6},
                },
                "content": [{"type": "text", "text": "ok"}],
            }
        ],
        "active": None,
        "terminal": None,
    }
    agent = make_agent(tmp_path)
    write_inspections(tmp_path, [inspection])
    agent.populate_context_post_run(AgentContext())

    trajectory = json.loads((tmp_path / "trajectory.json").read_text())
    step = trajectory["steps"][0]
    assert step["metrics"]["prompt_tokens"] == 5 + 4 + 6
    assert step["metrics"]["completion_tokens"] == 7 + 3
    assert step["metrics"]["cached_tokens"] == 4
    assert step["metrics"]["extra"]["reasoning_tokens"] == 3
    assert step["metrics"]["extra"]["cache_write_tokens"] == 6

    final = trajectory["final_metrics"]
    assert final["total_prompt_tokens"] == 15
    assert final["total_completion_tokens"] == 7 + 3
    assert final["total_cached_tokens"] == 4
    assert final["extra"]["total_cache_write_input_tokens"] == 6
    assert final["extra"]["total_reasoning_tokens"] == 3


def test_zero_usage_is_preserved_not_missing(tmp_path: Path):
    """A recorded zero is data; an absent usage record is missing."""
    inspection = {
        "session": {"id": "ses_zero0000000000000000000001"},
        "messages": [
            {
                "type": "assistant",
                "id": "msg_z1",
                "agent": "build",
                "model": {"id": "m", "providerID": "p"},
                "time": {"created": 1, "completed": 2},
                "finish": "stop",
                "cost": 0.0,
                "tokens": {
                    "input": 0,
                    "output": 0,
                    "reasoning": 0,
                    "cache": {"read": 0, "write": 0},
                },
                "content": [{"type": "text", "text": "ok"}],
            }
        ],
        "active": None,
        "terminal": None,
    }
    agent = make_agent(tmp_path)
    write_inspections(tmp_path, [inspection])
    agent.populate_context_post_run(AgentContext())

    trajectory = json.loads((tmp_path / "trajectory.json").read_text())
    final = trajectory["final_metrics"]
    extra = final.get("extra") or {}
    assert extra.get("metrics_complete") is None
    assert extra.get("unfinished") is None
    assert final["total_prompt_tokens"] == 0
    assert final["total_completion_tokens"] == 0
    assert final["total_cached_tokens"] == 0
    assert final["total_cost_usd"] == 0.0


def test_zero_usage_is_preserved_in_zero_subtree(tmp_path: Path):
    inspections = [
        {
            "session": {"id": "ses_zero_root000000000000000001"},
            "messages": [
                {
                    "type": "assistant",
                    "id": "msg_zero_root",
                    "model": {"id": "m", "providerID": "p", "variant": "max"},
                    "time": {"created": 1, "completed": 2},
                    "finish": "stop",
                    "tokens": {
                        "input": 0,
                        "output": 0,
                        "reasoning": 0,
                        "cache": {"read": 0, "write": 0},
                    },
                    "content": [{"type": "text", "text": "root"}],
                }
            ],
            "active": None,
            "terminal": None,
        },
        {
            "session": {
                "id": "ses_zero_child00000000000000001",
                "parentID": "ses_zero_root000000000000000001",
            },
            "messages": [
                {
                    "type": "assistant",
                    "id": "msg_zero_child",
                    "model": {"id": "m", "providerID": "p", "variant": "max"},
                    "time": {"created": 1, "completed": 2},
                    "finish": "stop",
                    "tokens": {
                        "input": 0,
                        "output": 0,
                        "reasoning": 0,
                        "cache": {"read": 0, "write": 0},
                    },
                    "content": [{"type": "text", "text": "child"}],
                }
            ],
            "active": None,
            "terminal": None,
        },
    ]
    agent = make_agent(tmp_path)
    write_inspections(tmp_path, inspections)
    agent.populate_context_post_run(AgentContext())

    metrics = json.loads((tmp_path / "trajectory.json").read_text())["final_metrics"]
    assert metrics["total_prompt_tokens"] == 0
    assert metrics["total_completion_tokens"] == 0
    assert metrics["total_cached_tokens"] == 0


def test_missing_model_provenance_is_unknown(tmp_path: Path):
    inspection = {
        "session": {"id": "ses_nomodel000000000000000000001"},
        "messages": [
            {
                "type": "assistant",
                "id": "msg_nomodel",
                "time": {"created": 1, "completed": 2},
                "finish": "stop",
                "tokens": {
                    "input": 1,
                    "output": 1,
                    "reasoning": 0,
                    "cache": {"read": 0, "write": 0},
                },
                "content": [{"type": "text", "text": "unknown"}],
            }
        ],
        "active": None,
        "terminal": None,
    }
    agent = make_agent(tmp_path, restrict_model=True)
    write_inspections(tmp_path, [inspection])
    context = AgentContext()
    agent.populate_context_post_run(context)

    trajectory = json.loads((tmp_path / "trajectory.json").read_text())
    assert trajectory["steps"][0].get("model_name") is None
    assert trajectory["final_metrics"]["extra"]["model_provenance_complete"] is False
    assert trajectory["final_metrics"]["total_prompt_tokens"] == 1
    assert context.n_input_tokens == 1


def test_missing_variant_provenance_is_unknown_not_contamination(tmp_path: Path):
    inspection = {
        "session": {"id": "ses_novariant00000000000000001"},
        "messages": [
            {
                "type": "assistant",
                "id": "msg_novariant",
                "model": {"id": "kimi-k3", "providerID": "litellm"},
                "time": {"created": 1, "completed": 2},
                "finish": "stop",
                "tokens": {
                    "input": 1,
                    "output": 1,
                    "reasoning": 0,
                    "cache": {"read": 0, "write": 0},
                },
                "content": [{"type": "text", "text": "unknown variant"}],
            }
        ],
    }
    agent = make_agent(tmp_path, restrict_model=True)
    write_inspections(tmp_path, [inspection])
    context = AgentContext()

    agent.populate_context_post_run(context)

    metrics = json.loads((tmp_path / "trajectory.json").read_text())["final_metrics"]
    assert metrics["extra"]["model_provenance_complete"] is False
    assert metrics["extra"]["model_mismatches"] == ["missing_variant:msg_novariant"]
    assert metrics["total_prompt_tokens"] == 1


def test_malformed_message_record_is_preserved_as_incomplete_gap(tmp_path: Path):
    inspection = {
        "session": {"id": "ses_badmessage00000000000000001"},
        "messages": ["malformed"],
    }
    agent = make_agent(tmp_path, restrict_model=True)
    write_inspections(tmp_path, [inspection])

    agent.populate_context_post_run(AgentContext())

    trajectory = json.loads((tmp_path / "trajectory.json").read_text())
    assert trajectory["steps"][0]["extra"]["collection_gap"] is True
    assert trajectory["final_metrics"]["extra"]["metrics_complete"] is False
    assert trajectory["final_metrics"]["extra"]["unfinished"] is True
    assert "total_prompt_tokens" not in trajectory["final_metrics"]


def test_malformed_message_mixed_with_valid_usage_withholds_totals(tmp_path: Path):
    inspection = {
        "session": {"id": "ses_mixedmessage0000000000000001"},
        "messages": [
            {
                "type": "assistant",
                "id": "msg_valid",
                "model": {
                    "id": "kimi-k3",
                    "providerID": "litellm",
                    "variant": "max",
                },
                "time": {"created": 1, "completed": 2},
                "finish": "stop",
                "tokens": {
                    "input": 7,
                    "output": 3,
                    "reasoning": 1,
                    "cache": {"read": 0, "write": 0},
                },
                "content": [{"type": "text", "text": "partial evidence"}],
            },
            "malformed",
        ],
    }
    agent = make_agent(tmp_path, restrict_model=True)
    write_inspections(tmp_path, [inspection])
    context = AgentContext()

    agent.populate_context_post_run(context)

    trajectory = json.loads((tmp_path / "trajectory.json").read_text())
    assert trajectory["steps"][0]["message"] == "partial evidence"
    assert trajectory["final_metrics"]["extra"]["metrics_complete"] is False
    assert trajectory["final_metrics"]["extra"]["unfinished"] is True
    assert "total_prompt_tokens" not in trajectory["final_metrics"]
    assert context.n_input_tokens is None


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"time": []}, id="assistant-time"),
        pytest.param(
            {
                "content": [
                    {
                        "type": "tool",
                        "id": "tool_bad_state",
                        "name": "read",
                        "state": [],
                    }
                ]
            },
            id="assistant-tool-state",
        ),
    ],
)
def test_malformed_nested_assistant_fields_preserve_incomplete_session(
    tmp_path: Path, overrides: dict[str, Any]
):
    common = {
        "type": "assistant",
        "model": {
            "id": "kimi-k3",
            "providerID": "litellm",
            "variant": "max",
        },
        "time": {"created": 1, "completed": 2},
        "finish": "stop",
        "tokens": {
            "input": 7,
            "output": 3,
            "reasoning": 1,
            "cache": {"read": 0, "write": 0},
        },
        "content": [{"type": "text", "text": "partial evidence"}],
    }
    malformed = {**common, "id": "msg_malformed_nested", **overrides}
    valid = {**common, "id": "msg_valid_nested"}
    inspection = {
        "session": {"id": "ses_malformed_nested00000000001"},
        "messages": [valid, malformed],
    }
    agent = make_agent(tmp_path, restrict_model=True)
    write_inspections(tmp_path, [inspection])
    context = AgentContext()

    agent.populate_context_post_run(context)

    trajectory = json.loads((tmp_path / "trajectory.json").read_text())
    assert trajectory["steps"][0]["message"] == "partial evidence"
    assert trajectory["final_metrics"]["extra"]["metrics_complete"] is False
    assert trajectory["final_metrics"]["extra"]["unfinished"] is True
    assert "total_prompt_tokens" not in trajectory["final_metrics"]
    assert context.n_input_tokens is None


def test_root_candidate_write_failure_is_recorded(tmp_path: Path, monkeypatch):
    errors: list[str] = []

    def fail_write(_path: Path, _contents: str) -> None:
        raise OSError("disk unavailable")

    monkeypatch.setattr(Path, "write_text", fail_write)

    runner_module._persist_root_candidates(
        tmp_path / "opencode-v2-root-candidates.json",
        {"candidate_session_ids": ["ses_root"]},
        errors,
    )

    assert errors == ["root candidate persistence: OSError: disk unavailable"]


def test_missing_runner_manifest_withholds_aggregate(tmp_path: Path):
    inspection = {
        "session": {"id": "ses_nomani000000000000000000001"},
        "messages": [
            {
                "type": "assistant",
                "id": "msg_nomani",
                "model": {"id": "kimi-k3", "providerID": "litellm", "variant": "max"},
                "time": {"created": 1, "completed": 2},
                "finish": "stop",
                "tokens": {
                    "input": 1,
                    "output": 1,
                    "reasoning": 0,
                    "cache": {"read": 0, "write": 0},
                },
                "content": [{"type": "text", "text": "partial"}],
            }
        ],
        "active": None,
        "terminal": None,
    }
    agent = make_agent(tmp_path)
    write_inspections(tmp_path, [inspection])
    (tmp_path / "opencode-v2" / "runner-result.json").unlink()
    agent.populate_context_post_run(AgentContext())

    metrics = json.loads((tmp_path / "trajectory.json").read_text())["final_metrics"]
    assert metrics.get("total_prompt_tokens") is None
    assert metrics["extra"]["collection_incomplete"] is True


def test_incomplete_runner_manifest_withholds_aggregate(tmp_path: Path):
    inspection = {
        "session": {"id": "ses_incomplete_manifest"},
        "messages": [
            {
                "type": "assistant",
                "id": "msg_incomplete_manifest",
                "model": {"id": "kimi-k3", "providerID": "litellm"},
                "time": {"created": 1, "completed": 2},
                "finish": "stop",
                "tokens": {"input": 1, "output": 1, "reasoning": 0},
                "content": [{"type": "text", "text": "partial"}],
            }
        ],
        "active": None,
        "terminal": None,
    }
    agent = make_agent(tmp_path)
    write_inspections(tmp_path, [inspection])
    (tmp_path / "opencode-v2" / "runner-result.json").write_text("{}")

    agent.populate_context_post_run(AgentContext())

    metrics = json.loads((tmp_path / "trajectory.json").read_text())["final_metrics"]
    assert metrics.get("total_prompt_tokens") is None
    assert metrics["extra"]["collection_incomplete"] is True


def test_empty_collection_writes_loud_incomplete_trajectory(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    runner_dir = tmp_path / "opencode-v2"
    runner_dir.mkdir()
    (runner_dir / "runner-result.json").write_text(
        json.dumps(
            {
                "collection_complete": False,
                "collection_errors": ["root session unresolved"],
            }
        )
    )
    context = AgentContext()
    caplog.set_level(logging.ERROR)

    make_agent(tmp_path).populate_context_post_run(context)

    trajectory = json.loads((tmp_path / "trajectory.json").read_text())
    assert trajectory["steps"][0]["extra"] == {
        "collection_gap": True,
        "collection_errors": ["root session unresolved"],
    }
    metrics = trajectory["final_metrics"]
    assert metrics["extra"]["metrics_complete"] is False
    assert metrics["extra"]["collection_incomplete"] is True
    assert "total_prompt_tokens" not in metrics
    assert context.n_input_tokens is None
    assert "root session unresolved" in caplog.text


def test_missing_usage_withholds_totals(tmp_path: Path):
    inspection = {
        "session": {"id": "ses_miss0000000000000000000001"},
        "messages": [
            {
                "type": "assistant",
                "id": "msg_m1",
                "agent": "build",
                "model": {"id": "m", "providerID": "p"},
                "time": {"created": 1, "completed": 2},
                "finish": "stop",
                "content": [{"type": "text", "text": "ok"}],
            }
        ],
        "active": None,
        "terminal": None,
    }
    agent = make_agent(tmp_path)
    write_inspections(tmp_path, [inspection])
    context = AgentContext()
    agent.populate_context_post_run(context)

    trajectory = json.loads((tmp_path / "trajectory.json").read_text())
    final = trajectory["final_metrics"]
    assert final.get("total_prompt_tokens") is None
    assert final.get("total_completion_tokens") is None
    assert final.get("total_cached_tokens") is None
    assert final.get("total_cost_usd") is None
    assert context.n_input_tokens is None
    assert context.cost_usd is None


def test_missing_usage_category_is_unknown_not_zero(tmp_path: Path):
    inspection = {
        "session": {"id": "ses_partial0000000000000000001"},
        "messages": [
            {
                "type": "assistant",
                "id": "msg_partial",
                "agent": "build",
                "model": {
                    "id": "kimi-k3",
                    "providerID": "litellm",
                    "variant": "max",
                },
                "time": {"created": 1, "completed": 2},
                "finish": "stop",
                # reasoning is deliberately absent; it must not become zero.
                "tokens": {
                    "input": 7,
                    "output": 3,
                    "cache": {"read": 2, "write": 1},
                },
                "cost": 0.01,
                "content": [{"type": "text", "text": "partial usage"}],
            }
        ],
        "active": None,
        "terminal": None,
    }
    agent = make_agent(tmp_path)
    write_inspections(tmp_path, [inspection])
    agent.populate_context_post_run(AgentContext())

    trajectory = json.loads((tmp_path / "trajectory.json").read_text())
    final = trajectory["final_metrics"]
    assert final.get("total_prompt_tokens") is None
    assert final.get("total_completion_tokens") is None
    assert final.get("total_cached_tokens") is None
    assert final["extra"]["metrics_complete"] is False
    step = trajectory["steps"][0]
    assert step.get("metrics") is None
    assert step["extra"]["usage_partial"] == {
        "input": 7,
        "output": 3,
        "reasoning": None,
        "cache_read": 2,
        "cache_write": 1,
    }


def test_missing_cost_withholds_only_cost(tmp_path: Path):
    inspection = {
        "session": {"id": "ses_nocost00000000000000000001"},
        "messages": [
            {
                "type": "assistant",
                "id": "msg_nocost",
                "agent": "build",
                "model": {
                    "id": "kimi-k3",
                    "providerID": "litellm",
                    "variant": "max",
                },
                "time": {"created": 1, "completed": 2},
                "finish": "stop",
                "tokens": {
                    "input": 7,
                    "output": 3,
                    "reasoning": 4,
                    "cache": {"read": 2, "write": 1},
                },
                "content": [{"type": "text", "text": "known tokens"}],
            }
        ],
        "active": None,
        "terminal": None,
    }
    agent = make_agent(tmp_path)
    write_inspections(tmp_path, [inspection])
    agent.populate_context_post_run(AgentContext())

    final = json.loads((tmp_path / "trajectory.json").read_text())["final_metrics"]
    assert final["total_prompt_tokens"] == 10
    assert final["total_completion_tokens"] == 7
    assert final["total_cached_tokens"] == 2
    assert final.get("total_cost_usd") is None
    assert final["extra"]["cost_complete"] is False


def test_error_finish_counts_persisted_paid_attempt_but_keeps_step(tmp_path: Path):
    inspection = {
        "session": {"id": "ses_err00000000000000000000001"},
        "messages": [
            {
                "type": "assistant",
                "id": "msg_e1",
                "agent": "build",
                "model": {"id": "m", "providerID": "p"},
                "time": {"created": 1, "completed": 2},
                "finish": "error",
                "error": {"type": "provider_error", "message": "boom"},
                "tokens": {
                    "input": 9,
                    "output": 1,
                    "reasoning": 0,
                    "cache": {"read": 0, "write": 0},
                },
                "content": [],
            }
        ],
        "active": None,
        "terminal": None,
    }
    agent = make_agent(tmp_path)
    write_inspections(tmp_path, [inspection])
    agent.populate_context_post_run(AgentContext())

    trajectory = json.loads((tmp_path / "trajectory.json").read_text())
    assert trajectory["steps"][0]["extra"]["finish"] == "error"
    final = trajectory["final_metrics"]
    assert final["total_prompt_tokens"] == 9
    assert final["total_completion_tokens"] == 1
    assert final["extra"]["failed_attempts"] == 1


def test_length_finish_keeps_usage(tmp_path: Path):
    """A truncated reply that completed still carries authoritative usage."""
    inspection = {
        "session": {"id": "ses_len00000000000000000000001"},
        "messages": [
            {
                "type": "assistant",
                "id": "msg_l1",
                "agent": "build",
                "model": {"id": "m", "providerID": "p"},
                "time": {"created": 1, "completed": 2},
                "finish": "length",
                "tokens": {
                    "input": 9,
                    "output": 100,
                    "reasoning": 0,
                    "cache": {"read": 0, "write": 0},
                },
                "content": [{"type": "text", "text": "partial outpu"}],
            }
        ],
        "active": None,
        "terminal": None,
    }
    agent = make_agent(tmp_path)
    write_inspections(tmp_path, [inspection])
    agent.populate_context_post_run(AgentContext())

    trajectory = json.loads((tmp_path / "trajectory.json").read_text())
    final = trajectory["final_metrics"]
    assert final["total_prompt_tokens"] == 9
    assert final["total_completion_tokens"] == 100
    assert final["extra"].get("metrics_complete") is None


def test_unfinished_message_is_preserved(tmp_path: Path):
    inspection = {
        "session": {"id": "ses_unf00000000000000000000001"},
        "messages": [
            {
                "type": "assistant",
                "id": "msg_u9",
                "agent": "build",
                "model": {"id": "m", "providerID": "p"},
                "time": {"created": 1},
                "finish": "unknown",
                "tokens": {
                    "input": 9,
                    "output": 5,
                    "reasoning": 0,
                    "cache": {"read": 0, "write": 0},
                },
                "content": [{"type": "text", "text": "half a thought"}],
            }
        ],
        "active": {"type": "running"},
        "terminal": None,
    }
    agent = make_agent(tmp_path)
    write_inspections(tmp_path, [inspection])
    agent.populate_context_post_run(AgentContext())

    trajectory = json.loads((tmp_path / "trajectory.json").read_text())
    final = trajectory["final_metrics"]
    assert final["extra"]["interrupted"] is True
    assert final["extra"]["unfinished"] is True
    assert final.get("total_prompt_tokens") is None


def test_still_running_session_preserved(tmp_path: Path):
    inspection = {
        "session": {
            "id": "ses_run00000000000000000000001",
            "outcome": "succeeded",
        },
        "messages": [
            {
                "type": "assistant",
                "id": "msg_r1",
                "agent": "build",
                "model": {"id": "m", "providerID": "p"},
                "time": {"created": 1, "completed": 2},
                "finish": "stop",
                "tokens": {
                    "input": 5,
                    "output": 5,
                    "reasoning": 0,
                    "cache": {"read": 0, "write": 0},
                },
                "content": [{"type": "text", "text": "ok"}],
            }
        ],
        "active": {"type": "running"},
        "terminal": None,
    }
    agent = make_agent(tmp_path)
    write_inspections(tmp_path, [inspection])
    agent.populate_context_post_run(AgentContext())

    trajectory = json.loads((tmp_path / "trajectory.json").read_text())
    assert trajectory["final_metrics"]["extra"]["interrupted"] is True


def test_content_filter_finish_preserved(tmp_path: Path):
    inspection = {
        "session": {"id": "ses_cf000000000000000000000001"},
        "messages": [
            {
                "type": "assistant",
                "id": "msg_cf1",
                "agent": "build",
                "model": {"id": "m", "providerID": "p"},
                "time": {"created": 1, "completed": 2},
                "finish": "content-filter",
                "tokens": {
                    "input": 5,
                    "output": 0,
                    "reasoning": 0,
                    "cache": {"read": 0, "write": 0},
                },
                "content": [],
            }
        ],
        "active": None,
        "terminal": None,
    }
    agent = make_agent(tmp_path)
    write_inspections(tmp_path, [inspection])
    agent.populate_context_post_run(AgentContext())

    trajectory = json.loads((tmp_path / "trajectory.json").read_text())
    step = trajectory["steps"][0]
    assert step["extra"]["finish"] == "content-filter"
    # A refused reply still bills its prompt-side usage; the run is complete.
    final = trajectory["final_metrics"]
    assert final["total_prompt_tokens"] == 5
    assert (final.get("extra") or {}).get("metrics_complete") is None


# ---------------------------------------------------------------------------
# Compaction
# ---------------------------------------------------------------------------


def test_compaction_counted_once(tmp_path: Path):
    inspection = {
        "session": {"id": "ses_cmp00000000000000000000001"},
        "messages": [
            {
                "type": "user",
                "id": "msg_cu1",
                "text": "go",
                "time": {"created": 1},
            },
            {
                "type": "compaction",
                "id": "msg_comp1",
                "status": "completed",
                "reason": "auto",
                "summary": "earlier context",
                "time": {"created": 2},
                "cost": 0.2,
                "tokens": {
                    "input": 50,
                    "output": 20,
                    "reasoning": 0,
                    "cache": {"read": 0, "write": 0},
                },
            },
            {
                "type": "assistant",
                "id": "msg_a1",
                "agent": "build",
                "model": {"id": "m", "providerID": "p"},
                "time": {"created": 3, "completed": 4},
                "finish": "stop",
                "cost": 0.1,
                "tokens": {
                    "input": 10,
                    "output": 5,
                    "reasoning": 0,
                    "cache": {"read": 0, "write": 0},
                },
                "content": [{"type": "text", "text": "ok"}],
            },
        ],
        "active": None,
        "terminal": None,
    }
    agent = make_agent(tmp_path)
    write_inspections(tmp_path, [inspection])
    agent.populate_context_post_run(AgentContext())

    trajectory = json.loads((tmp_path / "trajectory.json").read_text())
    final = trajectory["final_metrics"]
    # Compaction usage (50 in + 20 out) and the assistant step (10 + 5) both
    # bill to the run; summarization counted once.
    assert final["total_prompt_tokens"] == 50 + 10
    assert final["total_completion_tokens"] == 20 + 5
    assert final["extra"]["summarization_count"] == 1
    assert final["extra"]["compacted"] is True
    compaction_steps = [
        step for step in trajectory["steps"] if step["source"] == "system"
    ]
    assert len(compaction_steps) == 1
    assert compaction_steps[0]["extra"]["compaction"]["usage"] == {
        "prompt_tokens": 50,
        "completion_tokens": 20,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
        "reasoning_tokens": 0,
        "cost_usd": 0.2,
    }


def test_compaction_wrapper_reference_is_not_counted_again(tmp_path: Path):
    """A linked wrapper is deduped by record ID, never by equal token totals."""
    inspection = {
        "session": {"id": "ses_wrap0000000000000000000001"},
        "messages": [
            {
                "type": "assistant",
                "id": "msg_paid1",
                "model": {"id": "kimi-k3", "providerID": "litellm", "variant": "max"},
                "time": {"created": 1, "completed": 2},
                "finish": "stop",
                "tokens": {
                    "input": 10,
                    "output": 4,
                    "reasoning": 2,
                    "cache": {"read": 1, "write": 1},
                },
                "content": [{"type": "text", "text": "ok"}],
            },
            {
                "type": "compaction",
                "id": "msg_wrapper",
                "assistantMessageID": "msg_paid1",
                "status": "completed",
                "reason": "auto",
                "summary": "wrapper",
                "tokens": {
                    "input": 10,
                    "output": 4,
                    "reasoning": 2,
                    "cache": {"read": 1, "write": 1},
                },
            },
        ],
        "active": None,
        "terminal": None,
    }
    agent = make_agent(tmp_path)
    write_inspections(tmp_path, [inspection])
    agent.populate_context_post_run(AgentContext())

    trajectory = json.loads((tmp_path / "trajectory.json").read_text())
    assert trajectory["final_metrics"]["total_prompt_tokens"] == 12
    assert trajectory["final_metrics"]["total_completion_tokens"] == 6
    assert trajectory["final_metrics"]["extra"]["compacted"] is True
    compaction_steps = [
        step for step in trajectory["steps"] if step["source"] == "system"
    ]
    assert len(compaction_steps) == 1
    assert (
        compaction_steps[0]["extra"]["compaction"]["usage_duplicate_of"] == "msg_paid1"
    )


def test_failed_compaction_marks_incomplete(tmp_path: Path):
    inspection = {
        "session": {"id": "ses_cmpfail00000000000000000001"},
        "messages": [
            {
                "type": "compaction",
                "id": "msg_comp1",
                "status": "failed",
                "reason": "auto",
                "error": {"type": "provider_error", "message": "context"},
                "time": {"created": 2},
                "tokens": {
                    "input": 5,
                    "output": 0,
                    "reasoning": 0,
                    "cache": {"read": 0, "write": 0},
                },
            }
        ],
        "active": None,
        "terminal": None,
    }
    agent = make_agent(tmp_path)
    write_inspections(tmp_path, [inspection])
    agent.populate_context_post_run(AgentContext())

    trajectory = json.loads((tmp_path / "trajectory.json").read_text())
    final = trajectory["final_metrics"]
    assert final["total_prompt_tokens"] == 5
    assert final["total_completion_tokens"] == 0
    assert final["extra"]["failed_attempts"] == 1
    assert final["extra"]["compacted"] is True


# ---------------------------------------------------------------------------
# Model contamination
# ---------------------------------------------------------------------------


def test_model_contamination_is_recorded_not_silently_used(tmp_path: Path):
    inspection = {
        "session": {"id": "ses_contam000000000000000000001"},
        "messages": [
            {
                "type": "assistant",
                "id": "msg_c1",
                "agent": "build",
                "model": {"id": "other-model", "providerID": "litellm"},
                "time": {"created": 1, "completed": 2},
                "finish": "stop",
                "cost": 0.5,
                "tokens": {
                    "input": 5,
                    "output": 5,
                    "reasoning": 0,
                    "cache": {"read": 0, "write": 0},
                },
                "content": [{"type": "text", "text": "from another model"}],
            }
        ],
        "active": None,
        "terminal": None,
    }
    agent = make_agent(tmp_path)
    write_inspections(tmp_path, [inspection])
    agent.populate_context_post_run(AgentContext())

    trajectory = json.loads((tmp_path / "trajectory.json").read_text())
    # Unrestricted mode preserves each agent's actual configured model.
    assert trajectory["steps"][0]["model_name"] == "litellm/other-model"
    assert (
        trajectory["final_metrics"].get("extra", {}).get("model_provenance_complete")
        is not False
    )

    strict_dir = tmp_path / "strict"
    strict_agent = make_agent(
        strict_dir,
        model_name="litellm/kimi-k3",
        restrict_model=True,
    )
    write_inspections(strict_dir, [inspection])
    context = AgentContext()
    with pytest.raises(NonZeroAgentExitCodeError, match="msg_c1:litellm/other-model"):
        strict_agent.populate_context_post_run(context)
    strict_trajectory = json.loads((strict_dir / "trajectory.json").read_text())
    assert strict_trajectory["final_metrics"]["total_prompt_tokens"] == 5
    assert (
        strict_trajectory["final_metrics"]["extra"]["model_provenance_complete"]
        is False
    )
    assert context.n_input_tokens == 5


def test_restricted_model_check_uses_last_duplicate_record(tmp_path: Path):
    messages = [
        {
            "type": "assistant",
            "id": "msg_replayed",
            "model": {"id": "other-model", "providerID": "litellm"},
            "time": {"created": 1, "completed": 2},
            "finish": "stop",
            "tokens": {"input": 1, "output": 1, "reasoning": 0},
            "content": [{"type": "text", "text": "old"}],
        },
        {
            "type": "assistant",
            "id": "msg_replayed",
            "model": {"id": "kimi-k3", "providerID": "litellm"},
            "time": {"created": 1, "completed": 2},
            "finish": "stop",
            "tokens": {"input": 1, "output": 1, "reasoning": 0},
            "content": [{"type": "text", "text": "final"}],
        },
    ]
    agent = make_agent(tmp_path, model_name="litellm/kimi-k3", restrict_model=True)
    write_inspections(
        tmp_path,
        [{"session": {"id": "ses_replayed"}, "messages": messages}],
    )

    agent.populate_context_post_run(AgentContext())


def test_restricted_child_model_mismatch_hard_fails_whole_tree(tmp_path: Path):
    inspections = load_fixture()
    child = next(
        item for item in inspections if item["session"].get("parentID") is not None
    )
    assistant = next(
        message for message in child["messages"] if message["type"] == "assistant"
    )
    assistant["model"] = {"providerID": "litellm", "id": "other-model"}
    agent = make_agent(
        tmp_path,
        model_name="litellm/kimi-k3",
        restrict_model=True,
    )
    write_inspections(tmp_path, inspections)
    with pytest.raises(NonZeroAgentExitCodeError, match="other-model"):
        agent.populate_context_post_run(AgentContext())

    trajectory = json.loads((tmp_path / "trajectory.json").read_text())
    assert (
        trajectory["final_metrics"]["extra"]["tree_model_provenance_complete"] is False
    )
    assert trajectory["final_metrics"]["total_prompt_tokens"] == 223


def test_restricted_mismatch_hard_fails_when_root_is_ambiguous(tmp_path: Path):
    inspection = {
        "messages": [
            {
                "type": "assistant",
                "id": "msg_wrong_root",
                "model": {"providerID": "litellm", "id": "other-model"},
                "finish": "stop",
                "tokens": {
                    "input": 1,
                    "output": 1,
                    "reasoning": 0,
                    "cache": {"read": 0, "write": 0},
                },
                "content": [{"type": "text", "text": "wrong model"}],
            }
        ],
        "active": None,
        "terminal": None,
    }
    inspections = [
        inspection | {"session": {"id": "ses_root_a"}},
        copy.deepcopy(inspection) | {"session": {"id": "ses_root_b"}},
    ]
    agent = make_agent(
        tmp_path,
        model_name="litellm/kimi-k3",
        restrict_model=True,
    )
    write_inspections(tmp_path, inspections)
    with pytest.raises(NonZeroAgentExitCodeError, match="msg_wrong_root"):
        agent.populate_context_post_run(AgentContext())
    assert not (tmp_path / "trajectory.json").exists()
    assert (tmp_path / "opencode-v2" / "opencode-v2-sessions.jsonl").exists()


def test_restrict_model_pins_contaminating_agent_config(tmp_path: Path):
    agent = make_agent(
        tmp_path,
        restrict_model=True,
        opencode_v2_config={
            "agents": {"general": {"model": "openai/gpt-5.6-luna"}},
        },
    )
    config = agent._build_runtime_config(include_mcp=False)
    assert config["agents"]["general"]["model"] == "litellm/kimi-k3#max"


# ---------------------------------------------------------------------------
# Resumed child idempotence / duplicate records
# ---------------------------------------------------------------------------


def test_repeated_records_are_idempotent(tmp_path: Path):
    """Snapshots replace by (sessionID, type, id) ledger key."""
    inspection = load_fixture()[0]
    # Re-report the same session as one merged snapshot: same session, same
    # message ids, exactly the runner's stable-snapshot contract.
    merged = copy.deepcopy(inspection)
    merged["messages"] = merged["messages"] + [copy.deepcopy(inspection["messages"][0])]
    agent = make_agent(tmp_path)
    write_inspections(tmp_path, [merged])
    context = AgentContext()

    agent.populate_context_post_run(context)

    trajectory = json.loads((tmp_path / "trajectory.json").read_text())
    # Re-observed records replace rather than double count.
    metrics = trajectory["final_metrics"]
    assert metrics["total_prompt_tokens"] == 121
    assert metrics["total_completion_tokens"] == 50
    assert len(trajectory["steps"]) == 3


def test_resumed_child_keeps_single_record_per_message(tmp_path: Path):
    inspections = load_fixture()
    child = copy.deepcopy(inspections[1])
    # The resume re-reports the same child with the same message ids: the
    # snapshot replaces by ledger key, so totals do not double.
    child["messages"] = child["messages"] + [copy.deepcopy(child["messages"][-1])]
    agent = make_agent(tmp_path)
    write_inspections(tmp_path, [inspections[0], child])

    context = AgentContext()
    agent.populate_context_post_run(context)

    trajectory = json.loads((tmp_path / "trajectory.json").read_text())
    child_trajectory = trajectory["subagent_trajectories"][0]
    assert child_trajectory["final_metrics"]["total_completion_tokens"] == 40 + 30
    assert child_trajectory["final_metrics"]["total_prompt_tokens"] == 102


# ---------------------------------------------------------------------------
# Root ambiguity
# ---------------------------------------------------------------------------


def test_ambiguous_root_is_refused(tmp_path: Path):
    inspections = load_fixture()
    # Strip the child's parentID so both look like roots.
    inspections[1]["session"].pop("parentID")
    agent = make_agent(tmp_path)
    write_inspections(tmp_path, inspections)
    context = AgentContext()

    agent.populate_context_post_run(context)

    assert not (tmp_path / "trajectory.json").exists()
    assert context.is_empty()


def test_empty_root_preserves_child_but_withholds_tree_totals(tmp_path: Path):
    inspections = load_fixture()
    # The parent's record carries no messages (it never got collected), so
    # only the child produces a trajectory.
    inspections[0]["messages"] = []
    agent = make_agent(tmp_path)
    write_inspections(tmp_path, inspections)
    context = AgentContext()

    agent.populate_context_post_run(context)

    trajectory = json.loads((tmp_path / "trajectory.json").read_text())
    assert trajectory["trajectory_id"] == "ses_root0000000000000000000001"
    assert trajectory["subagent_trajectories"][0]["trajectory_id"] == (
        "ses_child0000000000000000000001"
    )
    assert trajectory["final_metrics"].get("total_prompt_tokens") is None
    assert trajectory["final_metrics"]["extra"]["tree_metrics_complete"] is False


def test_incomplete_child_does_not_leave_root_metrics_complete_true(tmp_path: Path):
    inspections = load_fixture()
    inspections[1]["messages"] = []
    agent = make_agent(tmp_path)
    write_inspections(tmp_path, inspections)

    agent.populate_context_post_run(AgentContext())

    extra = json.loads((tmp_path / "trajectory.json").read_text())["final_metrics"][
        "extra"
    ]
    assert extra["tree_metrics_complete"] is False
    assert "metrics_complete" not in extra


def test_malformed_session_line_does_not_create_unknown_root(tmp_path: Path):
    inspections = load_fixture()
    write_inspections(tmp_path, inspections)
    sessions_path = tmp_path / "opencode-v2" / "opencode-v2-sessions.jsonl"
    with sessions_path.open("a") as stream:
        stream.write('{"session":')

    context = AgentContext()
    make_agent(tmp_path).populate_context_post_run(context)

    trajectory = json.loads((tmp_path / "trajectory.json").read_text())
    assert trajectory["trajectory_id"] == "ses_root0000000000000000000001"
    assert context.n_input_tokens == 223


# ---------------------------------------------------------------------------
# Runner behavior: multipage, nested, background terminal, failure paths
# ---------------------------------------------------------------------------


class FakeHTTP:
    def __init__(self, sessions_pages, messages_pages, active=None, terminal=None):
        self.sessions_pages = sessions_pages
        self.messages_pages = messages_pages
        self.active = active or {}
        self.terminal = terminal or {}
        self.calls: list[str] = []


def test_server_start_preserves_readiness_error_when_cleanup_also_fails(monkeypatch):
    class FakeProcess:
        stderr: list[str] = []

    monkeypatch.setattr(
        runner_module.subprocess, "Popen", lambda *args, **kwargs: FakeProcess()
    )

    def fail_readiness(process):
        raise RuntimeError("readiness evidence")

    monkeypatch.setattr(runner_module, "wait_for_server", fail_readiness)
    server = runner_module.OpenCodeV2Server(
        binary="opencode", cwd="/tmp", password="pw", env={}
    )

    def fail_cleanup():
        raise RuntimeError("cleanup evidence")

    monkeypatch.setattr(server, "stop", fail_cleanup)

    with pytest.raises(RuntimeError, match="readiness evidence") as raised:
        server.start()

    assert any("cleanup evidence" in note for note in raised.value.__notes__)


@pytest.mark.parametrize(
    ("statuses", "paths", "succeeds"),
    [
        ([200], ["api/status"], True),
        ([404], ["api/status"], False),
        ([401], ["api/status"], False),
        ([503], ["api/status"], False),
    ],
)
def test_server_readiness_uses_status_endpoint(monkeypatch, statuses, paths, succeeds):
    class FakeProcess:
        stderr: list[str] = []

    monkeypatch.setattr(
        runner_module.subprocess, "Popen", lambda *a, **k: FakeProcess()
    )
    monkeypatch.setattr(
        runner_module, "wait_for_server", lambda process: "http://127.0.0.1:1"
    )
    calls = []
    replies = iter(statuses)

    def get(url, password, timeout):
        calls.append(url.removeprefix("http://127.0.0.1:1/"))
        return next(replies), "{}"

    monkeypatch.setattr(runner_module, "http_get", get)
    server = runner_module.OpenCodeV2Server("opencode", "/tmp", "pw", {})
    stopped = []
    monkeypatch.setattr(server, "stop", lambda: stopped.append(True))
    if succeeds:
        assert server.start() == "http://127.0.0.1:1"
        assert not stopped
    else:
        with pytest.raises(RuntimeError, match="readiness check"):
            server.start()
        assert stopped == [True]
    assert calls == paths


@pytest.mark.parametrize(
    ("statuses", "succeeds"),
    [
        ([204], True),
        ([200], True),
        ([404], False),
        ([401], False),
        ([503], False),
    ],
)
def test_session_wait_uses_experimental_endpoint(monkeypatch, statuses, succeeds):
    calls = []
    replies = iter(statuses)

    def post(url, password, payload, timeout):
        calls.append(url)
        assert payload == {}
        return next(replies)

    monkeypatch.setattr(runner_module, "http_post", post)
    server = runner_module.OpenCodeV2Server("opencode", "/tmp", "pw", {})
    server.url = "http://127.0.0.1:1"
    assert server.wait_session("ses_test") is succeeds
    expected = ["http://127.0.0.1:1/api/experimental/session/ses_test/wait"]
    assert calls == expected


def test_collect_sessions_follows_cursor_pages(tmp_path: Path, monkeypatch):
    pages = [
        ([{"id": "ses_1"}, {"id": "ses_2"}], "cursor-1"),
        ([{"id": "ses_3"}], "cursor-2"),
        ([{"id": "ses_4"}], None),
    ]
    monkeypatch.setattr(
        runner_module.OpenCodeV2Server,
        "page_sessions",
        lambda self, parent_id=None, cursor=None: pages.pop(0),
    )
    server = runner_module.OpenCodeV2Server(
        binary="opencode", cwd="/tmp", password="pw", env={}
    )
    result = server.collect_sessions()
    assert [s["id"] for s in result] == ["ses_1", "ses_2", "ses_3", "ses_4"]


def test_collect_sessions_rejects_repeated_cursor(monkeypatch):
    pages = [([], "cursor-1"), ([], "cursor-1")]
    monkeypatch.setattr(
        runner_module.OpenCodeV2Server,
        "page_sessions",
        lambda self, parent_id=None, cursor=None: pages.pop(0),
    )
    server = runner_module.OpenCodeV2Server(
        binary="opencode", cwd="/tmp", password="pw", env={}
    )
    with pytest.raises(RuntimeError, match="session pagination cursor repeated"):
        server.collect_sessions()


def test_collect_sessions_scoped_to_parent(tmp_path: Path, monkeypatch):
    seen: list[str | None] = []

    def fake_page_sessions(self, parent_id=None, cursor=None):
        seen.append(parent_id)
        return [], None

    monkeypatch.setattr(
        runner_module.OpenCodeV2Server, "page_sessions", fake_page_sessions
    )
    server = runner_module.OpenCodeV2Server(
        binary="opencode", cwd="/tmp", password="pw", env={}
    )
    server.collect_sessions(parent_id="ses_parent")
    assert seen == ["ses_parent"]


def test_collect_descendants_deduplicates_repeated_page_records(monkeypatch):
    pages = {
        "ses_root": [
            {"id": "ses_child", "parentID": "ses_root"},
            {"id": "ses_child", "parentID": "ses_root"},
        ],
        "ses_child": [],
    }
    server = runner_module.OpenCodeV2Server(
        binary="opencode", cwd="/tmp", password="pw", env={}
    )
    monkeypatch.setattr(
        server,
        "collect_sessions",
        lambda parent_id=None: (
            pages.get(parent_id, []) if parent_id is not None else [{"id": "ses_root"}]
        ),
    )
    result = server.collect_descendants("ses_root")
    assert [item["id"] for item in result] == ["ses_root", "ses_child"]


def test_collect_messages_follows_cursor_pages(tmp_path: Path, monkeypatch):
    pages = [
        ([{"id": "msg_1"}], "m-cursor-1"),
        ([{"id": "msg_2"}], "m-cursor-2"),
        ([{"id": "msg_3"}], None),
    ]
    monkeypatch.setattr(
        runner_module.OpenCodeV2Server,
        "page_messages",
        lambda self, session_id, cursor=None: pages.pop(0),
    )
    server = runner_module.OpenCodeV2Server(
        binary="opencode", cwd="/tmp", password="pw", env={}
    )
    result = server.collect_messages("ses_1")
    assert [m["id"] for m in result] == ["msg_1", "msg_2", "msg_3"]


def test_collect_messages_rejects_repeated_cursor(monkeypatch):
    pages = [([], "cursor-1"), ([], "cursor-1")]
    monkeypatch.setattr(
        runner_module.OpenCodeV2Server,
        "page_messages",
        lambda self, session_id, cursor=None: pages.pop(0),
    )
    server = runner_module.OpenCodeV2Server(
        binary="opencode", cwd="/tmp", password="pw", env={}
    )
    with pytest.raises(RuntimeError, match="message pagination cursor repeated"):
        server.collect_messages("ses_1")


def test_page_messages_rejects_malformed_records(monkeypatch):
    monkeypatch.setattr(
        runner_module,
        "http_get_json",
        lambda *args, **kwargs: (200, {"data": ["not-a-message"]}),
    )
    server = runner_module.OpenCodeV2Server(
        binary="opencode", cwd="/tmp", password="pw", env={}
    )
    server.url = "http://127.0.0.1:1"

    with pytest.raises(RuntimeError, match="returned malformed records"):
        server.page_messages("ses_1")


def test_raw_page_evidence_is_deduplicated_and_bounded(monkeypatch):
    monkeypatch.setattr(runner_module, "RAW_PAGE_LIMIT", 2)
    server = runner_module.OpenCodeV2Server(
        binary="opencode", cwd="/tmp", password="pw", env={}
    )
    first = {"endpoint": "/api/session", "status": 200, "response": {"n": 1}}
    server._record_raw_page(first)
    server._record_raw_page(first)
    server._record_raw_page(
        {"endpoint": "/api/session", "status": 200, "response": {"n": 2}}
    )
    server._record_raw_page(
        {"endpoint": "/api/session", "status": 200, "response": {"n": 3}}
    )

    assert [item["response"]["n"] for item in server.raw_pages] == [2, 3]
    assert server.raw_pages_deduplicated == 1
    assert server.raw_pages_dropped == 1


def test_oversize_raw_page_is_replaced_by_bounded_digest_marker(monkeypatch):
    monkeypatch.setattr(runner_module, "RAW_PAGE_BYTES_LIMIT", 1024)
    server = runner_module.OpenCodeV2Server(
        binary="opencode", cwd="/tmp", password="pw", env={}
    )
    server._record_raw_page(
        {
            "endpoint": "/api/session/ses_1/message",
            "status": 200,
            "response": {"data": "x" * 4096},
        }
    )

    assert server._raw_page_bytes <= runner_module.RAW_PAGE_BYTES_LIMIT
    assert server.raw_pages[0]["oversize_page"]["response_omitted"] is True
    assert server.raw_pages[0]["oversize_page"]["encoded_bytes"] > 4096
    assert server.raw_pages_oversize == 1


def test_message_pages_use_order_only_before_cursor(monkeypatch):
    seen_queries: list[str] = []

    def fake_page_messages(self, session_id, cursor=None):
        return [], None

    def fake_http_get_json(url, password, timeout=60.0):
        seen_queries.append(url)
        next_cursor = "cursor with /?" if len(seen_queries) == 1 else None
        return 200, {"data": [], "cursor": {"next": next_cursor}}

    monkeypatch.setattr(runner_module, "http_get_json", fake_http_get_json)
    server = runner_module.OpenCodeV2Server(
        binary="opencode", cwd="/tmp", password="pw", env={}
    )
    server.url = "http://127.0.0.1:1"
    server.collect_messages("ses_1")
    assert len(seen_queries) == 2
    assert "order=asc" in seen_queries[0]
    assert "order=" not in seen_queries[1]
    assert "cursor=cursor+with+%2F%3F" in seen_queries[1]


def test_interrupt_all_interrupts_discovered_sessions():
    interrupted: list[str] = []

    class Server(runner_module.OpenCodeV2Server):
        def interrupt_session(self, session_id: str) -> None:
            interrupted.append(session_id)

    server = Server(binary="opencode", cwd="/tmp", password="pw", env={})
    server.interrupt_all({"ses_b", "ses_a"})
    assert interrupted == ["ses_a", "ses_b"]


def test_runner_cleans_process_group_on_stop():
    process = subprocess.Popen(
        ["sleep", "30"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    server = runner_module.OpenCodeV2Server(
        binary="opencode", cwd="/tmp", password="pw", env={}
    )
    server.process = process
    server.stop()
    assert process.poll() is not None


def test_runner_kills_server_when_stdin_close_is_not_enough():
    process = subprocess.Popen(
        ["bash", "-c", "read; sleep 30"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    server = runner_module.OpenCodeV2Server(
        binary="opencode", cwd="/tmp", password="pw", env={}
    )
    server.process = process
    try:
        # stdin close alone does not terminate this fake server.
        server.process.stdin.close()
        with pytest.raises(subprocess.TimeoutExpired):
            server.process.wait(timeout=0.5)
        server.stop()
        assert process.poll() is not None
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def test_runner_kills_owned_descendant_after_server_leader_exits(tmp_path: Path):
    child_pid_file = tmp_path / "child.pid"
    process = subprocess.Popen(
        [
            "bash",
            "-c",
            f"sleep 30 & echo $! > {child_pid_file}; exit 0",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    process.wait(timeout=5)
    child_pid = int(child_pid_file.read_text())
    server = runner_module.OpenCodeV2Server(
        binary="opencode", cwd="/tmp", password="pw", env={}
    )
    server.process = process
    server.stop()
    deadline = time.monotonic() + 3
    while Path(f"/proc/{child_pid}").exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not Path(f"/proc/{child_pid}").exists()


def test_runner_reports_process_group_that_survives_sigkill(monkeypatch):
    class FakeProcess:
        pid = 424242
        stdin = None

        @staticmethod
        def wait(timeout=None):
            return 0

    ticks = iter(range(0, 100, 4))
    monkeypatch.setattr(runner_module.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(runner_module.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(runner_module.os, "killpg", lambda _pid, _signal: None)
    server = runner_module.OpenCodeV2Server(
        binary="opencode", cwd="/tmp", password="pw", env={}
    )
    server.process = FakeProcess()

    with pytest.raises(RuntimeError, match="survived SIGKILL"):
        server.stop()


def test_collect_tree_timeout_interrupts_discovered_tree(monkeypatch):
    interrupted: list[str] = []

    class Server:
        collection_deadline = None

        def collect_descendants(self, root_id):
            return [
                {"id": root_id},
                {"id": "ses_child", "parentID": root_id},
            ]

        def interrupt_all(self, session_ids):
            interrupted.extend(sorted(session_ids))

        def inspect_session(self, session):
            return {
                "session": session,
                "messages": [],
                "active": None,
                "inbox": [],
            }

    errors: list[str] = []
    inspections, settled = runner_module._collect_tree(
        Server(),
        "ses_root",
        deadline=time.monotonic(),
        errors=errors,
    )
    assert settled is False
    assert interrupted == ["ses_child", "ses_root"]
    assert {item["session"]["id"] for item in inspections} == {
        "ses_root",
        "ses_child",
    }
    assert errors


def test_collect_tree_uses_bounded_backoff_before_incomplete_result(monkeypatch):
    now = 0.0
    sleeps: list[float] = []

    def monotonic():
        return now

    def sleep(seconds):
        nonlocal now
        sleeps.append(seconds)
        now += seconds

    class Server:
        collection_deadline = None

        def collect_descendants(self, root_id):
            return [{"id": root_id}]

        def wait_session(self, session_id):
            return True

        def running_session_ids(self):
            return set()

        def inbox_items(self, session_id):
            return []

        def inspect_session(self, session, active_ids=None):
            return {
                "session": session,
                "messages": [],
                "active": None,
                "inbox": [],
            }

        def interrupt_all(self, session_ids):
            return None

    monkeypatch.setattr(runner_module.time, "monotonic", monotonic)
    monkeypatch.setattr(runner_module.time, "sleep", sleep)
    errors: list[str] = []

    _, settled = runner_module._collect_tree(
        Server(), "ses_root", deadline=10.0, errors=errors
    )

    assert settled is False
    assert sleeps == [1.0, 2.0, 4.0, 3.0]
    assert runner_module.DEFAULT_SETTLE_SECONDS <= 120
    assert errors == ["sessions lack terminal idle outcome: ses_root"]


def test_session_metadata_outcome_is_a_terminal_state_without_idle_message():
    inspections = [
        {
            "session": {"id": "ses_root", "outcome": "succeeded"},
            "messages": [],
        }
    ]
    assert runner_module._sessions_without_terminal_outcome(inspections) == []


def test_collection_is_incomplete_when_owned_server_died():
    assert runner_module._collection_complete(
        settled=True, errors=[], root_id="ses_root", server_exit_code=None
    )
    assert not runner_module._collection_complete(
        settled=True, errors=[], root_id="ses_root", server_exit_code=1
    )


def test_collection_errors_do_not_fail_successful_agent_execution():
    runner_module._raise_runner_failure(None, None)

    with pytest.raises(SystemExit, match="CLI exited"):
        runner_module._raise_runner_failure(None, "OpenCode CLI exited with status 1")


def test_root_id_uses_retained_cli_candidates_after_event_tail_eviction():
    sessions = [
        {"id": "ses_root"},
        {"id": "ses_child", "parentID": "ses_root"},
    ]
    events = [{"sessionID": "ses_child"}]

    assert runner_module._root_id(events, sessions, set()) is None
    assert (
        runner_module._root_id(
            events, sessions, set(), candidate_ids={"ses_root", "ses_child"}
        )
        == "ses_root"
    )
    assert (
        runner_module._root_id(
            [],
            [{"id": "ses_old"}, {"id": "ses_new"}],
            {"ses_old"},
            candidate_ids={"ses_old"},
        )
        is None
    )


def test_runner_preserves_primary_error_on_cancellation(tmp_path: Path):
    """A failed runner run must keep its artifacts for post-run diagnosis."""

    class FailingEnvironment(FakeEnvironment):
        async def exec(self, **kwargs):
            self.exec_calls.append(kwargs)
            return ExecResult(
                return_code=1,
                stdout="",
                stderr="runner exploded",
            )

    environment = FailingEnvironment()
    agent = make_agent(tmp_path)

    import asyncio

    with pytest.raises(Exception):
        asyncio.run(agent.run("doomed", environment, AgentContext()))

    # Artifact collection still ran (best effort) even though the run failed.
    assert environment.downloads


def test_inspect_session_collects_background_terminal(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        runner_module.OpenCodeV2Server,
        "collect_messages",
        lambda self, session_id: [{"id": "msg_1"}],
    )
    monkeypatch.setattr(
        runner_module.OpenCodeV2Server,
        "running_session_ids",
        lambda self: {"ses_bg"},
    )
    monkeypatch.setattr(
        runner_module.OpenCodeV2Server,
        "inbox_items",
        lambda self, session_id: [],
    )
    server = runner_module.OpenCodeV2Server(
        binary="opencode", cwd="/tmp", password="pw", env={}
    )
    server.url = "http://127.0.0.1:1"

    inspection = runner_module.inspect_session(server, {"id": "ses_bg"})
    assert inspection["active"] == {"type": "running"}
    assert "terminal" not in inspection
    assert inspection["messages"] == [{"id": "msg_1"}]


def test_inspect_session_uses_snapshot_active_ids(monkeypatch):
    server = runner_module.OpenCodeV2Server(
        binary="opencode", cwd="/tmp", password="pw", env={}
    )
    monkeypatch.setattr(server, "collect_messages", lambda _session_id: [])
    monkeypatch.setattr(server, "inbox_items", lambda _session_id: [])
    monkeypatch.setattr(
        server,
        "running_session_ids",
        lambda: (_ for _ in ()).throw(AssertionError("active set fetched twice")),
    )

    inspection = runner_module.inspect_session(
        server, {"id": "ses_snapshot"}, active_ids={"ses_snapshot"}
    )

    assert inspection["active"] == {"type": "running"}


def test_session_active_state_distinguishes_running(monkeypatch):
    responses = {
        "http://127.0.0.1:1/api/session/active": (
            200,
            {"data": {"ses_bg": {"type": "running"}}},
        ),
    }

    def fake_http_get_json(url, password, timeout=10.0):
        return responses.get(url, (200, {"data": {}}))

    monkeypatch.setattr(runner_module, "http_get_json", fake_http_get_json)
    server = runner_module.OpenCodeV2Server(
        binary="opencode", cwd="/tmp", password="pw", env={}
    )
    server.url = "http://127.0.0.1:1"
    assert server.running_session_ids() == {"ses_bg"}


def test_runner_records_server_stderr_and_events(tmp_path: Path):
    """Server stderr and CLI stdout events are collected for diagnosis."""
    events_path = tmp_path / "opencode-v2-cli-events.jsonl"
    events_path.write_text('{"type": "step_start"}\nnot json\n')
    records = OpenCodeV2._read_jsonl(events_path)
    assert len(records) == 2
    assert records[1]["type"] == "cli-stdout"


def test_private_state_excludes_credentials_and_symlinks(tmp_path: Path):
    state = tmp_path / "state"
    state.mkdir()
    (state / "session.db").write_text("safe")
    (state / "auth.json").write_text("secret")
    secret = tmp_path / "outside-secret"
    secret.write_text("secret")
    (state / "linked-secret").symlink_to(secret)

    copied = runner_module._preserve_private_state(
        {"XDG_STATE_HOME": str(state)}, tmp_path / "logs"
    )

    assert any(path.endswith("session.db") for path in copied)
    assert not any(path.endswith("auth.json") for path in copied)
    assert not any(path.endswith("linked-secret") for path in copied)


def test_cli_capture_line_is_bounded():
    line = runner_module._bounded_cli_line(
        "x" * (runner_module.CLI_CAPTURE_MAX_LINE_BYTES * 2)
    )
    assert len(line.encode()) <= runner_module.CLI_CAPTURE_MAX_LINE_BYTES


def test_preflight_evidence_redacts_env_urls_and_header_credentials():
    value = {
        "settings": {
            "baseURL": "https://user:password@gateway.example/v1?key=secret",
            "apiKey": "provider-secret",
        },
        "headers": {
            "X-API-Key": "header-secret",
            "Authorization": "Bearer secret",
            "Content-Type": "application/json",
        },
    }
    redacted = runner_module._redact(value)
    assert redacted == {
        "settings": {
            "baseURL": "<redacted-url>",
            "apiKey": "<redacted>",
        },
        "headers": {
            "X-API-Key": "<redacted>",
            "Authorization": "<redacted>",
            "Content-Type": "application/json",
        },
    }


def test_preflight_allows_configured_subagent_model_when_unrestricted(
    tmp_path: Path, monkeypatch
):
    config_path = tmp_path / "opencode.json"
    config_path.write_text(
        json.dumps(
            {
                "model": "litellm/root#low",
                "providers": {
                    "litellm": {
                        "models": {
                            "root": {"body": {"max_tokens": 8192}},
                            "child": {},
                        }
                    }
                },
                "agents": {
                    "build": {"model": "litellm/root#low"},
                    "general": {"model": "litellm/child#high"},
                },
            }
        )
    )
    server = runner_module.OpenCodeV2Server(
        binary="opencode", cwd="/tmp", password="pw", env={}
    )

    def locations(_server, endpoint):
        if endpoint == "model":
            return [
                {
                    "providerID": "litellm",
                    "id": "root",
                    "variants": [{"id": "low"}],
                    "body": {"max_tokens": 8192},
                }
            ]
        return [
            {
                "id": "build",
                "model": {"providerID": "litellm", "id": "root", "variant": "low"},
            },
            {
                "id": "general",
                "model": {
                    "providerID": "litellm",
                    "id": "child",
                    "variant": "high",
                },
            },
        ]

    monkeypatch.setattr(runner_module, "_location_data", locations)
    result = runner_module.preflight_runtime(
        server,
        model_spec="litellm/root#low",
        config_file=str(config_path),
        restrict_model=False,
        timeout=0.1,
    )
    assert {item["id"] for item in result["resolved_agents"]} == {
        "build",
        "general",
    }


def test_preflight_rejects_subagent_model_when_restricted(tmp_path: Path, monkeypatch):
    config_path = tmp_path / "opencode.json"
    config_path.write_text(
        json.dumps(
            {
                "providers": {"litellm": {"models": {"root": {}}}},
                "agents": {"general": {"model": "litellm/child#high"}},
            }
        )
    )
    server = runner_module.OpenCodeV2Server(
        binary="opencode", cwd="/tmp", password="pw", env={}
    )

    def locations(_server, endpoint):
        if endpoint == "model":
            return [
                {
                    "providerID": "litellm",
                    "id": "root",
                    "variants": [{"id": "low"}],
                }
            ]
        return [
            {
                "id": "general",
                "model": {
                    "providerID": "litellm",
                    "id": "child",
                    "variant": "high",
                },
            }
        ]

    monkeypatch.setattr(runner_module, "_location_data", locations)
    with pytest.raises(RuntimeError, match="expected 'litellm/root#low'"):
        runner_module.preflight_runtime(
            server,
            model_spec="litellm/root#low",
            config_file=str(config_path),
            restrict_model=True,
            timeout=0.01,
        )


def test_preflight_rejects_inherited_contradictory_model_limits(
    tmp_path: Path, monkeypatch
):
    config_path = tmp_path / "opencode.json"
    config_path.write_text(
        json.dumps(
            {
                "providers": {
                    "litellm": {
                        "models": {
                            "gpt-5.6-luna": {
                                "limit": {"context": 272000, "output": 128000}
                            }
                        }
                    }
                }
            }
        )
    )
    server = runner_module.OpenCodeV2Server(
        binary="opencode", cwd="/tmp", password="pw", env={}
    )

    def locations(_server, endpoint):
        if endpoint == "model":
            return [
                {
                    "providerID": "litellm",
                    "id": "gpt-5.6-luna",
                    "variants": [{"id": "low"}],
                    "limit": {
                        "context": 272000,
                        "input": 922000,
                        "output": 128000,
                    },
                }
            ]
        return []

    monkeypatch.setattr(runner_module, "_location_data", locations)
    with pytest.raises(RuntimeError, match="limits are contradictory"):
        runner_module.preflight_runtime(
            server,
            model_spec="litellm/gpt-5.6-luna#low",
            config_file=str(config_path),
            restrict_model=True,
            timeout=0.01,
        )


# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------


def test_install_spec_uses_configured_version_checksum_and_target_architecture(
    tmp_path: Path,
):
    agent = make_agent(
        tmp_path,
        opencode_v2_checksums={
            "linux-x64": (
                "4b8c2cad67297c715adff18a569c8808b22fe23c7197fd1775bc11cbfa04022d"
            ),
            "linux-arm64": (
                "bc35547e678c68aaec1b2aa1623d1d77ec2585db6204574724826e40f20a7693"
            ),
        },
    )
    spec = agent.install_spec()
    assert spec.agent_name == "opencode-v2"
    assert spec.version == "2.0.3"
    assert spec.steps[0].run == "apt-get update && apt-get install -y curl python3"
    joined = "\n".join(step.run for step in spec.steps)
    assert 'machine="$(uname -m)"' in joined
    assert "target=linux-x64" in joined
    assert "target=linux-arm64" in joined
    assert "@opencode%2fcli-${target}" in joined
    assert "cli-${target}-${version}.tgz" in joined
    assert "sha256sum -c" in joined
    assert "4b8c2cad67297c715adff18a569c8808b22fe23c7197fd1775bc11cbfa04022d" in joined
    assert "bc35547e678c68aaec1b2aa1623d1d77ec2585db6204574724826e40f20a7693" in joined
    assert "OpenCode version mismatch" in joined
    assert spec.steps[-1].user == "root"
    assert "install -d -m 755 /installed-agent" in spec.steps[-1].run
    subprocess.run(
        ["bash", "-n"],
        input=spec.steps[-1].run,
        text=True,
        check=True,
    )


def test_install_spec_respects_explicit_version(tmp_path: Path):
    agent = make_agent(tmp_path, version="2.0.2")
    joined = "\n".join(step.run for step in agent.install_spec().steps)
    assert "2.0.2" in joined


def test_install_spec_without_version_resolves_latest_for_generic_use(tmp_path: Path):
    agent = OpenCodeV2(logs_dir=tmp_path, model_name="litellm/kimi-k3")
    spec = agent.install_spec()
    assert spec.version is None
    assert '["dist-tags"]["latest"]' in spec.steps[-1].run
    subprocess.run(
        ["bash", "-n"],
        input=spec.steps[-1].run,
        text=True,
        check=True,
    )


def test_unpinned_install_disables_cross_trial_layer_cache_reuse(tmp_path: Path):
    first = OpenCodeV2(logs_dir=tmp_path / "first", model_name="litellm/kimi-k3")
    second = OpenCodeV2(logs_dir=tmp_path / "second", model_name="litellm/kimi-k3")

    assert first.install_spec().fingerprint() == first.install_spec().fingerprint()
    assert first.install_spec().fingerprint() != second.install_spec().fingerprint()
    assert first.install_spec().cache_key != second.install_spec().cache_key
    assert first.install_spec().steps[-1].run == second.install_spec().steps[-1].run


def test_pinned_install_keeps_stable_cross_trial_cache_identity(tmp_path: Path):
    first = make_agent(tmp_path / "first")
    second = make_agent(tmp_path / "second")

    assert first.install_spec().fingerprint() == second.install_spec().fingerprint()
    assert first.install_spec().steps[-1].run == second.install_spec().steps[-1].run


def test_install_spec_rejects_unsafe_version(tmp_path: Path):
    agent = make_agent(tmp_path, version="2.0.3; touch /tmp/injected")
    with pytest.raises(ValueError, match="unsupported characters"):
        agent.install_spec()


def test_version_command_uses_remote_binary(tmp_path: Path):
    agent = make_agent(tmp_path)
    assert str(agent._REMOTE_BINARY.as_posix()) in agent.get_version_command()
