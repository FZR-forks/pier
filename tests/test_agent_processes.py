"""The Docker tests are opt-in: PIER_TEST_DOCKER=1 uv run pytest tests/test_agent_processes.py."""

import asyncio
import os
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from pier.agents.installed.pi import Pi
from pier.environments.base import ExecResult
from pier.models.agent.context import AgentContext
from pier.trial.execution import AgentTimeoutError, TrialExecution
from pier.trial.processes import AgentProcessCleanupError, AgentProcesses


@pytest.mark.asyncio
async def test_cleanup_failure_is_not_an_agent_timeout():
    environment = SimpleNamespace(
        exec=AsyncMock(side_effect=TimeoutError("exec unavailable")),
        stop=AsyncMock(),
    )
    with pytest.raises(AgentProcessCleanupError):
        await AgentProcesses(environment, {"1:123"}).stop()
    environment.stop.assert_awaited_once_with(delete=False)


@pytest.mark.asyncio
async def test_cleanup_finishes_before_cancellation_returns():
    entered, release = asyncio.Event(), asyncio.Event()
    environment = SimpleNamespace(exec=AsyncMock())

    async def stop(*_args, **_kwargs):
        entered.set()
        await release.wait()
        return ExecResult(return_code=0)

    environment.exec.side_effect = stop
    processes = AgentProcesses(environment, {"1:123"})
    task = asyncio.create_task(processes.__aexit__())
    await entered.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert processes.stopped


@pytest.mark.asyncio
@pytest.mark.skipif(os.getenv("PIER_TEST_DOCKER") != "1", reason="opt-in Docker test")
async def test_tools_stop_before_collectors_then_final_sweep():
    name = f"pier-process-test-{uuid.uuid4().hex[:10]}"
    start = await asyncio.create_subprocess_exec(
        "docker",
        "run",
        "--rm",
        "-d",
        "--name",
        name,
        "python:3.13-slim",
        "sleep",
        "infinity",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await start.communicate()
    assert start.returncode == 0
    environment = DockerShell(name)
    try:
        processes = await AgentProcesses.start(environment)
        collector = await environment.exec("sleep 300 >/dev/null 2>&1 & echo $!")
        pid = int(collector.stdout.strip())
        await environment.exec(
            "setsid bash -c 'sleep 5; touch /late-write' >/dev/null 2>&1 &"
        )
        await processes.wait_stopped((pid,))
        assert not processes.stopped
        assert (await environment.exec(f"kill -0 {pid}")).return_code == 0
        await processes.wait_stopped()
        assert processes.stopped
        await asyncio.sleep(5)
        assert (await environment.exec("test ! -f /late-write")).return_code == 0
        state = await environment.exec(f"cat /proc/{pid}/stat 2>/dev/null || true")
        assert not state.stdout or state.stdout.split(") ")[1].startswith("Z ")
    finally:
        remove = await asyncio.create_subprocess_exec(
            "docker", "rm", "-f", name, stdout=asyncio.subprocess.DEVNULL
        )
        await remove.wait()


class DockerShell:
    """Cancel only the docker client, reproducing the real timeout failure."""

    def __init__(self, name):
        self.name = name

    def agent_process_env(self, env):
        return env

    async def exec(self, command, **_kwargs):
        process = await asyncio.create_subprocess_exec(
            "docker",
            "exec",
            self.name,
            "bash",
            "-c",
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await process.communicate()
        except asyncio.CancelledError:
            process.kill()
            await process.wait()
            raise
        return ExecResult(
            return_code=process.returncode,
            stdout=stdout.decode(),
            stderr=stderr.decode(),
        )


@pytest.mark.asyncio
@pytest.mark.skipif(os.getenv("PIER_TEST_DOCKER") != "1", reason="opt-in Docker test")
@pytest.mark.parametrize("outcome", ["timeout", "cancel", "success", "error"])
async def test_sandbox_sweep_stops_detached_tools_and_preserves_services(
    tmp_path, outcome
):
    name = f"pier-process-test-{uuid.uuid4().hex[:10]}"
    start = await asyncio.create_subprocess_exec(
        "docker",
        "run",
        "--rm",
        "-d",
        "--name",
        name,
        "python:3.13-slim",
        "sleep",
        "infinity",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await start.communicate()
    assert start.returncode == 0, stderr.decode()
    environment = DockerShell(name)
    try:
        service = await environment.exec("sleep 300 >/dev/null 2>&1 & echo $!")
        agent = Pi(logs_dir=tmp_path, model_name="openai/test")

        async def run(instruction, environment, context):
            # Ignore TERM, detach into a new session, and let the parent exit.
            await environment.exec(
                'setsid bash -c \'trap "" TERM; touch /started; '
                "sleep 5; touch /late-write' >/dev/null 2>&1 &"
            )
            if outcome in {"timeout", "cancel"}:
                await agent.exec_as_agent(environment, "sleep 300")
            if outcome == "error":
                raise ValueError("agent failed")

        agent.run = run
        execution = TrialExecution(
            None, agent, environment, 0.5 if outcome == "timeout" else 30, 1, 1
        )
        task = asyncio.create_task(
            execution.run_agent(instruction="", context=AgentContext())
        )
        if outcome == "cancel":
            for _ in range(50):
                if (await environment.exec("test -f /started")).return_code == 0:
                    break
                await asyncio.sleep(0.02)
            task.cancel()
        error = {
            "timeout": AgentTimeoutError,
            "cancel": asyncio.CancelledError,
            "error": ValueError,
        }.get(outcome)
        if error:
            with pytest.raises(error):
                await task
        else:
            await task
        assert (
            await environment.exec(f"kill -0 {service.stdout.strip()}")
        ).return_code == 0
        await asyncio.sleep(5)
        assert (await environment.exec("test ! -f /late-write")).return_code == 0
    finally:
        remove = await asyncio.create_subprocess_exec(
            "docker",
            "rm",
            "-f",
            name,
            stdout=asyncio.subprocess.DEVNULL,
        )
        await remove.wait()
