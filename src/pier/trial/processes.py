"""Bound an installed agent's processes to its run, not the host exec client."""

import asyncio
import shlex

from pier.environments.base import BaseEnvironment


class AgentProcessCleanupError(RuntimeError):
    """The sandbox is unsafe to verify; do not treat this as an agent timeout."""


# Use /proc and shell builtins: installed agents need neither Python nor ps.
_PROCESS_LIST = """
for path in /proc/[0-9]*; do
    IFS= read -r stat < "$path/stat" 2>/dev/null || continue
    fields=${stat##*) }
    set -- $fields
    [ "$1" = Z ] && continue
    shift 19
    printf '%s:%s\n' "${path##*/}" "$1"
done
"""


class AgentProcesses:
    """Preserve pre-existing task services; sweep new processes, even setsid tools.

    Each trial owns its sandbox. A process-group kill alone cannot reach tools
    that detached or were reparented after their harness exited.
    """

    def __init__(self, environment: BaseEnvironment, baseline: set[str]):
        self.environment = environment
        self.baseline = baseline
        self.stopped = False

    @classmethod
    async def start(cls, environment: BaseEnvironment) -> "AgentProcesses":
        result = await environment.exec(_PROCESS_LIST, user="root", timeout_sec=10)
        if result.return_code != 0 or not result.stdout:
            raise RuntimeError("Could not snapshot sandbox processes before agent run")
        return cls(environment, set(result.stdout.split()))

    async def stop(self) -> None:
        if self.stopped:
            return
        # Protect the cleanup exec and its ancestors too. They were not present
        # in the snapshot, but killing them would cut off the sweep itself.
        command = f"""
baseline={shlex.quote(" " + " ".join(sorted(self.baseline)) + " ")}
keep=' '
pid=$$
while [ "$pid" -gt 0 ]; do
    keep="$keep$pid "
    IFS= read -r stat < "/proc/$pid/stat" || exit 1
    fields=${{stat##*) }}
    set -- $fields
    pid=$2
done
for signal in TERM KILL KILL CHECK; do
    remaining=0
    for path in /proc/[0-9]*; do
        pid=${{path##*/}}
        case "$keep" in *" $pid "*) continue ;; esac
        IFS= read -r stat < "$path/stat" 2>/dev/null || continue
        fields=${{stat##*) }}
        set -- $fields
        [ "$1" = Z ] && continue
        shift 19
        case "$baseline" in *" $pid:$1 "*) continue ;; esac
        remaining=1
        if [ "$signal" != CHECK ]; then
            kill -"$signal" "$pid" 2>/dev/null || true
        fi
    done
    [ "$remaining" = 0 ] && exit 0
    [ "$signal" = CHECK ] && exit 1
    sleep 1
done
"""
        try:
            result = await self.environment.exec(command, user="root", timeout_sec=10)
            if result.return_code != 0:
                raise RuntimeError("Agent processes could not be confirmed stopped")
        except Exception:
            # Do not allow verification on a workspace that is still changing.
            try:
                await self.environment.stop(delete=False)
            finally:
                raise AgentProcessCleanupError("Could not stop sandbox agent processes")
        self.stopped = True

    async def wait_stopped(self) -> None:
        """Do not leave cleanup running in the background on cancellation."""
        cleanup = asyncio.create_task(self.stop())
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            await cleanup
            raise

    async def __aenter__(self) -> "AgentProcesses":
        return self

    async def __aexit__(self, *_exc) -> None:
        await self.wait_stopped()
