from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from pier.environments.docker.docker import DockerEnvironment


def configure_client(environment):
    environment._env_vars = SimpleNamespace(
        to_env_dict=lambda **_: {"DOCKER_HOST": "host-a"}
    )
    environment._compose_task_env = {
        "DOCKER_HOST": "host-b",
        "DOCKER_CONTEXT": "task-context",
    }
    environment._persistent_env = {"DOCKER_HOST": "host-c"}
    environment._windows_container_name = None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state,expected",
    [
        ({"Running": False, "Pid": 0, "Status": "exited"}, True),
        ({"Running": True, "Pid": 42, "Status": "running"}, False),
        ({"Running": False, "Pid": 42, "Status": "restarting"}, False),
        ({"Running": False, "Pid": 0, "Status": "restarting"}, False),
        ({"Running": True, "Pid": 42, "Status": "paused"}, False),
    ],
)
async def test_docker_stop_confirmation_checks_runtime_state(
    monkeypatch, state, expected
):
    import json

    processes = [
        SimpleNamespace(
            returncode=0, communicate=AsyncMock(return_value=(b"abcdef123456\n", b""))
        ),
        SimpleNamespace(
            returncode=0,
            communicate=AsyncMock(
                return_value=(json.dumps([{"State": state}]).encode(), b"")
            ),
        ),
    ]
    create = AsyncMock(side_effect=processes)
    monkeypatch.setattr("asyncio.create_subprocess_exec", create)
    environment = DockerEnvironment.__new__(DockerEnvironment)
    environment.session_id = "Test__AbC"
    configure_client(environment)
    assert await environment.agent_sandbox_is_stopped() is expected
    assert "label=com.docker.compose.project=test__abc" in create.call_args_list[0].args
    for call in create.call_args_list:
        assert call.kwargs["env"]["DOCKER_HOST"] == "host-c"
        assert call.kwargs["env"]["DOCKER_CONTEXT"] == "task-context"


@pytest.mark.asyncio
async def test_empty_project_is_stopped_but_query_failure_is_not_success(monkeypatch):
    environment = DockerEnvironment.__new__(DockerEnvironment)
    environment.session_id = "test"
    configure_client(environment)
    process = SimpleNamespace(
        returncode=0, communicate=AsyncMock(return_value=(b"", b""))
    )
    monkeypatch.setattr(
        "asyncio.create_subprocess_exec", AsyncMock(return_value=process)
    )
    assert await environment.agent_sandbox_is_stopped() is True
    process.returncode = 1
    with pytest.raises(RuntimeError, match="state query failed"):
        await environment.agent_sandbox_is_stopped()
