"""Opt-in isolated Docker regression: no provider calls, no external network."""

import asyncio
import os
import shlex
import uuid
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from pier.environments.base import ExecResult
from pier.environments.docker.docker import DockerEnvironment
from pier.trial.processes import AGENT_RUN_ENV, AgentProcessCleanupError, AgentProcesses


class IsolatedDocker:
    """Only the named disposable project may be stopped by this test."""

    def __init__(self, name):
        self.name = self.session_id = name

    async def exec(self, command, **kwargs):
        args = ["docker", "exec"]
        if kwargs.get("user") is not None:
            args += ["--user", str(kwargs["user"])]
        for key, value in (kwargs.get("env") or {}).items():
            args += ["--env", f"{key}={value}"]
        process = await asyncio.create_subprocess_exec(
            *args,
            self.name,
            "bash",
            "-c",
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=kwargs.get("timeout_sec", 10)
            )
        except BaseException:
            if process.returncode is None:
                process.kill()
            await process.wait()
            raise
        return ExecResult(
            return_code=process.returncode,
            stdout=stdout.decode(),
            stderr=stderr.decode(),
        )

    async def stop(self, delete=False):
        process = await asyncio.create_subprocess_exec(
            "docker",
            "stop",
            "--time",
            "1",
            self.name,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await process.communicate()
        assert process.returncode == 0, stderr.decode()

    async def agent_sandbox_is_stopped(self):
        return await DockerEnvironment.agent_sandbox_is_stopped(self)

    def _docker_client_env(self):
        return os.environ.copy()

    def agent_process_env(self, env):
        return env


@pytest.mark.asyncio
@pytest.mark.skipif(os.getenv("PIER_TEST_DOCKER") != "1", reason="opt-in Docker test")
@pytest.mark.parametrize("transient", [False, True])
async def test_same_uid_nondumpable_process_never_silently_passes_cleanup(transient):
    name = f"pier-pi-cleanup-regression-{uuid.uuid4().hex[:8]}"
    create = await asyncio.create_subprocess_exec(
        "docker",
        "run",
        "--pull=never",
        "--network=none",
        "--detach",
        "--name",
        name,
        "--label",
        f"com.docker.compose.project={name}",
        "python:3.13-slim",
        "sleep",
        "infinity",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await create.communicate()
    assert create.returncode == 0, stderr.decode()
    env = IsolatedDocker(name)
    try:
        processes = await AgentProcesses.start(env)
        code = (
            "import ctypes,pathlib,time; "
            "libc=ctypes.CDLL(None); libc.prctl(4,0); "
            "pathlib.Path('/tmp/ready').touch(); "
            + ("time.sleep(.15); libc.prctl(4,1); " if transient else "")
            + "time.sleep(5); pathlib.Path('/tmp/late-write').touch()"
        )
        start = await env.exec(
            f"python3 -c {shlex.quote(code)} >/dev/null 2>&1 &",
            env={AGENT_RUN_ENV: processes.token},
        )
        assert start.return_code == 0
        assert (
            await env.exec("while [ ! -e /tmp/ready ]; do sleep .01; done")
        ).return_code == 0
        if transient:
            # Give the regression a deterministic bound for slow CI hosts;
            # permissions reopen before the final permitted scan.
            processes._UNREADABLE_RETRIES = 10
            await processes.wait_stopped()
            assert processes.stopped
            assert not await env.agent_sandbox_is_stopped()
            assert (await env.exec("test ! -e /tmp/late-write")).return_code == 0
        else:
            with pytest.raises(AgentProcessCleanupError) as caught:
                await processes.wait_stopped()
            assert caught.value.sandbox_stopped is True
            assert await env.agent_sandbox_is_stopped()
    finally:
        remove = await asyncio.create_subprocess_exec(
            "docker",
            "rm",
            "--force",
            name,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await remove.wait()


@pytest.mark.asyncio
@pytest.mark.skipif(os.getenv("PIER_TEST_DOCKER") != "1", reason="opt-in Docker test")
async def test_compose_fallback_shutdown_is_confirmed_on_the_same_project(tmp_path):
    name = f"pier-pi-cleanup-compose-{uuid.uuid4().hex[:8]}"
    compose = tmp_path / "compose.yaml"
    compose.write_text(
        "services:\n  main:\n    image: python:3.13-slim\n    pull_policy: never\n"
        '    network_mode: none\n    command: ["sleep", "infinity"]\n'
    )

    class ComposeEnvironment(DockerEnvironment):
        @property
        def _docker_compose_paths(self):
            return [compose]

    environment = ComposeEnvironment.__new__(ComposeEnvironment)
    environment.session_id = environment.environment_name = name
    environment.environment_dir = tmp_path
    environment._env_vars = SimpleNamespace(
        to_env_dict=lambda **kwargs: os.environ.copy()
    )
    environment._compose_task_env = environment._persistent_env = {}
    environment._windows_container_name = None
    environment._keep_containers = False
    environment.prepare_logs_for_host = AsyncMock()
    environment._cleanup_resources_compose_file = lambda: None
    environment.logger = logging.getLogger(__name__)
    try:
        await environment._run_docker_compose_command(["up", "--detach", "--wait"])
        assert not await environment.agent_sandbox_is_stopped()
        await environment.stop(delete=False)
        assert await environment.agent_sandbox_is_stopped()
    finally:
        await environment._run_docker_compose_command(["down", "--remove-orphans"])
