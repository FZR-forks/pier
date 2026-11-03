"""Bound an installed agent's processes to its run, not the host exec client."""

import asyncio
import contextlib
import shlex
import uuid

from pier.environments.base import BaseEnvironment


class AgentProcessCleanupError(RuntimeError):
    """The sandbox is unsafe to verify; do not treat this as an agent timeout."""


AGENT_RUN_ENV = "PIER_AGENT_RUN"


class AgentProcesses:
    """Stop this run's tagged processes, including detached/reparented tools.

    A tag inherited from the harness distinguishes agent tools from workers
    spawned by task services during the run. Age and ancestry alone cannot.
    """

    _STOP_TIMEOUT_SEC = 60

    def __init__(self, environment: BaseEnvironment, token: str):
        self.environment = environment
        self.token = token
        self.stopped = False

    @classmethod
    async def start(cls, environment: BaseEnvironment) -> "AgentProcesses":
        return cls(environment, uuid.uuid4().hex)

    def owns_pid(self, expression: str) -> str:
        """Shell guard for adapter PID files: never signal an untagged reuse."""
        tag = shlex.quote(AGENT_RUN_ENV + "=" + self.token)
        return (
            f"{{ owned=false; while IFS= read -r -d '' entry; do "
            f'if [ "$entry" = {tag} ]; then owned=true; break; fi; '
            f'done < "/proc/{expression}/environ"; [ "$owned" = true ]; }} 2>/dev/null'
        )

    async def stop(self, keep_pids: tuple[int, ...] = ()) -> None:
        if self.stopped:
            return
        # Use /proc and shell builtins, without requiring Python or ps. The
        # cleanup exec is untagged. Preserve collectors and their exec shells
        # temporarily when OpenCode needs them for final session collection.
        command = f"""
tag={shlex.quote(AGENT_RUN_ENV + "=" + self.token)}
my_uid=$(id -u) || exit 1
keep=' '
for pid in $$ {" ".join(map(str, keep_pids))}; do
    while [ "$pid" -gt 0 ]; do
        keep="$keep$pid "
        IFS= read -r stat < "/proc/$pid/stat" 2>/dev/null || break
        fields=${{stat##*) }}
        set -- $fields
        pid=$2
    done
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
        owned=false
        readable=false
        {{
            readable=true
            while IFS= read -r -d '' entry; do
                if [ "$entry" = "$tag" ]; then owned=true; break; fi
            done
            true
        }} < "$path/environ" 2>/dev/null || readable=false
        if [ "$readable" = false ]; then
            [ -d "$path" ] || continue
            uid=''
            while read -r key value rest; do
                [ "$key" = Uid: ] && {{ uid=$value; break; }}
            done < "$path/status"
            [ -n "$uid" ] || exit 1
            # Retry other owners as their own UID. An unreadable environment
            # even as its owner is unknown, not evidence that it is unowned.
            [ "$uid" != "$my_uid" ] || exit 1
            printf '%s\n' "$uid"
            continue
        fi
        [ "$owned" = true ] || continue
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
            # Container root may lack CAP_SYS_PTRACE. Retry unreadable owners
            # as their UID, including tools that switched away from the agent.
            users = ["root"]
            for user in users:
                result = await self.environment.exec(command, user=user, timeout_sec=10)
                if result.return_code != 0:
                    raise RuntimeError("Agent processes could not be confirmed stopped")
                for uid in (result.stdout or "").split():
                    if not uid.isdigit():
                        raise RuntimeError("Invalid process owner in cleanup output")
                    if uid not in users and uid != "0":
                        users.append(uid)
        except Exception:
            # Do not allow verification on a workspace that is still changing.
            try:
                await asyncio.wait_for(
                    self.environment.stop(delete=False), timeout=self._STOP_TIMEOUT_SEC
                )
            finally:
                raise AgentProcessCleanupError("Could not stop sandbox agent processes")
        self.stopped = not keep_pids

    async def wait_stopped(self, keep_pids: tuple[int, ...] = ()) -> None:
        """Do not leave cleanup running in the background on cancellation."""
        cleanup = asyncio.create_task(self.stop(keep_pids))
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            while not cleanup.done():
                with contextlib.suppress(asyncio.CancelledError):
                    await asyncio.shield(cleanup)
            cleanup.result()
            raise

    async def __aenter__(self) -> "AgentProcesses":
        return self

    async def __aexit__(self, *_exc) -> None:
        await self.wait_stopped()
