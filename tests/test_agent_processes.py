"""The Docker tests are opt-in: PIER_TEST_DOCKER=1 uv run pytest tests/test_agent_processes.py."""

import asyncio
import os
import shlex
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from pier.agents.installed.pi import Pi
from pier.environments.base import ExecResult
from pier.models.agent.context import AgentContext
from pier.trial.execution import AgentTimeoutError, TrialExecution
from pier.trial.processes import AGENT_RUN_ENV, AgentProcessCleanupError, AgentProcesses


@pytest.mark.asyncio
async def test_cleanup_failure_is_not_an_agent_timeout():
    environment = SimpleNamespace(
        exec=AsyncMock(side_effect=TimeoutError("exec unavailable")),
        stop=AsyncMock(),
    )
    with pytest.raises(AgentProcessCleanupError):
        await AgentProcesses(environment, "test-run").stop()
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
    processes = AgentProcesses(environment, "test-run")
    task = asyncio.create_task(processes.__aexit__())
    await entered.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert processes.stopped


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter_cleanup", [False, True])
@pytest.mark.parametrize("fails", [False, True])
async def test_finished_agent_is_not_timed_out_during_cleanup(
    tmp_path, adapter_cleanup, fails
):
    entered, release = asyncio.Event(), asyncio.Event()

    async def cleanup(command, **kwargs):
        entered.set()
        await release.wait()
        return ExecResult(return_code=0)

    environment = SimpleNamespace(exec=cleanup)
    agent = Pi(logs_dir=tmp_path, model_name="openai/test")

    async def run(*args, **kwargs):
        try:
            await asyncio.sleep(0.03)
            if fails:
                raise ValueError("agent failure")
        finally:
            if adapter_cleanup:
                await agent.stop_processes()  # Codex/Gemini/OpenCode path

    agent.run = run
    execution = TrialExecution(None, agent, environment, 0.05, 1, 1)
    task = asyncio.create_task(
        execution.run_agent(instruction="", context=AgentContext())
    )
    await asyncio.wait_for(entered.wait(), 1)
    await asyncio.sleep(0.1)  # execution deadline passes while cleanup is held
    assert not task.done()
    release.set()
    if fails:
        with pytest.raises(ValueError, match="agent failure"):
            await task
    else:
        await task


@pytest.mark.asyncio
async def test_codex_copies_logs_after_cleanup_cancellation(tmp_path):
    from pier.agents.installed.codex import Codex

    agent = Codex(
        logs_dir=tmp_path,
        model_name="openai/test",
        extra_env={"OPENAI_API_KEY": "test-key"},
    )
    agent.exec_as_agent = AsyncMock(return_value=ExecResult(return_code=0))
    agent.stop_processes = AsyncMock(side_effect=asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await agent.run("task", SimpleNamespace(), AgentContext())
    commands = [call.kwargs["command"] for call in agent.exec_as_agent.await_args_list]
    assert any('rm -f "$CODEX_HOME/auth.json"' in cmd for cmd in commands)
    assert any('cp -R "$CODEX_HOME/sessions"' in cmd for cmd in commands)


@pytest.mark.asyncio
async def test_codex_removes_credentials_but_skips_copy_if_shutdown_fails(tmp_path):
    from pier.agents.installed.codex import Codex

    agent = Codex(
        logs_dir=tmp_path,
        model_name="openai/test",
        extra_env={"OPENAI_API_KEY": "test-key"},
    )
    agent.exec_as_agent = AsyncMock(return_value=ExecResult(return_code=0))

    async def fail():
        commands = [
            call.kwargs["command"] for call in agent.exec_as_agent.await_args_list
        ]
        assert any('rm -f "$CODEX_HOME/auth.json"' in cmd for cmd in commands)
        agent._process_cleanup_failed = True
        raise AgentProcessCleanupError("failed")

    agent.stop_processes = fail
    with pytest.raises(AgentProcessCleanupError):
        await agent.run("task", SimpleNamespace(), AgentContext())
    commands = [call.kwargs["command"] for call in agent.exec_as_agent.await_args_list]
    assert not any('cp -R "$CODEX_HOME/sessions"' in cmd for cmd in commands)


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
        env = {AGENT_RUN_ENV: processes.token}
        collector = await environment.exec(
            "sleep 300 >/dev/null 2>&1 & echo $!", env=env
        )
        pid = int(collector.stdout.strip())
        await environment.exec(
            "setsid bash -c 'sleep 5; touch /late-write' >/dev/null 2>&1 &", env=env
        )
        await processes.wait_stopped((pid,))
        assert not processes.stopped
        assert (
            await environment.exec(
                f"grep -qE '^State:[[:space:]]+[^Z]' /proc/{pid}/status"
            )
        ).return_code == 0
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

    def __init__(self, name, default_user=None):
        self.name = name
        self.default_user = default_user

    def agent_process_env(self, env):
        return env

    async def exec(self, command, **_kwargs):
        options = []
        for key, value in (_kwargs.get("env") or {}).items():
            options.extend(["-e", f"{key}={value}"])
        if user := _kwargs.get("user") or self.default_user:
            options.extend(["-u", user])
        process = await asyncio.create_subprocess_exec(
            "docker",
            "exec",
            *options,
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
@pytest.mark.parametrize("user", [None, "tester"])
async def test_sandbox_sweep_stops_detached_tools_and_preserves_services(
    tmp_path, outcome, user
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
        if user:
            assert (
                await environment.exec("useradd -m tester; chmod 777 /")
            ).return_code == 0
            environment.default_user = user
        # A pre-existing service creates its worker only after the agent starts.
        supervisor = (
            "import pathlib,subprocess,time; "
            "p=pathlib.Path('/spawn-worker'); "
            "exec('while not p.exists(): time.sleep(.01)'); "
            "worker=subprocess.Popen(['sleep','300']); "
            "pathlib.Path('/worker-pid').write_text(str(worker.pid)); time.sleep(300)"
        )
        service = await environment.exec(
            f"python3 -c {shlex.quote(supervisor)} >/dev/null 2>&1 & echo $!"
        )
        agent = Pi(logs_dir=tmp_path, model_name="openai/test")

        async def run(instruction, environment, context):
            await agent.exec_as_agent(environment, "touch /spawn-worker")
            for _ in range(100):
                if (await environment.exec("test -f /worker-pid")).return_code == 0:
                    break
                await asyncio.sleep(0.01)
            assert (await environment.exec("test -f /worker-pid")).return_code == 0
            # Ignore TERM, detach into a new session, and let the parent exit.
            await agent.exec_as_agent(
                environment,
                'setsid bash -c \'trap "" TERM; touch /started; '
                "sleep 5; touch /late-write' >/dev/null 2>&1 &",
            )
            for _ in range(100):
                if (await environment.exec("test -f /started")).return_code == 0:
                    break
                await asyncio.sleep(0.01)
            assert (await environment.exec("test -f /started")).return_code == 0
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
            await environment.exec(
                f"grep -qE '^State:[[:space:]]+[^Z]' /proc/{service.stdout.strip()}/status"
            )
        ).return_code == 0
        worker = (await environment.exec("cat /worker-pid")).stdout.strip()
        assert (
            await environment.exec(
                f"grep -qE '^State:[[:space:]]+[^Z]' /proc/{worker}/status"
            )
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
