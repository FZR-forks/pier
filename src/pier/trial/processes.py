"""Bound an installed agent's processes to its run, not the host exec client."""

import asyncio
import contextlib
import shlex
import uuid
from typing import TYPE_CHECKING

from pier.environments.base import BaseEnvironment

if TYPE_CHECKING:
    from pier.models.agent.context import AgentContext


class AgentProcessCleanupError(RuntimeError):
    """The sandbox is unsafe to verify; do not treat this as an agent timeout."""

    def __init__(self, message: str, *, sandbox_stopped: bool = False):
        super().__init__(message)
        self.sandbox_stopped = sandbox_stopped
        self.primary_exception: BaseException | None = None
        self.cancellation: asyncio.CancelledError | None = None
        self.agent_context: AgentContext | None = None

    def preserve_primary(self, primary: BaseException | None) -> None:
        while isinstance(primary, AgentProcessCleanupError):
            if primary.cancellation is not None:
                self.cancellation = primary.cancellation
            primary = primary.primary_exception
        if primary is not None:
            self.primary_exception = primary

    def classify_deadline(self, expired: bool) -> None:
        if expired and (
            self.primary_exception is None
            or isinstance(self.primary_exception, asyncio.CancelledError)
        ):
            self.primary_exception = TimeoutError("Agent execution deadline expired")


AGENT_RUN_ENV = "PIER_AGENT_RUN"


class AgentProcesses:
    """Stop this run's tagged processes, including detached/reparented tools.

    A tag inherited from the harness distinguishes agent tools from workers
    spawned by task services during the run. Age and ancestry alone cannot.
    """

    _STOP_TIMEOUT_SEC = 60
    _SCAN_TIMEOUT_SEC = 10
    _UNREADABLE_RETRIES = 2

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
        # Task images need not provide Python or ps. OpenCode collectors and
        # their exec shells must survive until final session collection.
        command = f"""
tag={shlex.quote(AGENT_RUN_ENV + "=" + self.token)}
my_uid=$(id -u) || exit 1
keep=' '
for pid in $$ {" ".join(map(str, keep_pids))}; do
    while [ "$pid" -gt 0 ]; do
        keep="$keep$pid "
        IFS= read -r stat 2>/dev/null < "/proc/$pid/stat" || break
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
        IFS= read -r stat 2>/dev/null < "$path/stat" || continue
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
        }} 2>/dev/null < "$path/environ" || readable=false
        if [ "$readable" = false ]; then
            [ -d "$path" ] || continue
            # Zombies retain /proc entries but cannot write the workspace.
            IFS= read -r stat 2>/dev/null < "$path/stat" || continue
            fields=${{stat##*) }}
            set -- $fields
            [ "$1" = Z ] && continue
            uid=''
            while read -r key value rest; do
                [ "$key" = Uid: ] && {{ uid=$value; break; }}
            done 2>/dev/null < "$path/status"
            [ -n "$uid" ] || exit 1
            # Unreadability is unknown ownership, not evidence of no owner.
            if [ "$uid" = "$my_uid" ]; then
                # No command lines or environment values belong in diagnostics.
                echo "Unreadable process environment: pid=$pid uid=$uid state=$1" >&2
                exit 75
            fi
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
                for attempt in range(self._UNREADABLE_RETRIES + 1):
                    result = await self.environment.exec(
                        command, user=user, timeout_sec=self._SCAN_TIMEOUT_SEC
                    )
                    if result.return_code != 75 or attempt == self._UNREADABLE_RETRIES:
                        break
                    await asyncio.sleep(0.05)
                if result.return_code != 0:
                    raise RuntimeError(
                        "Agent processes could not be confirmed stopped: "
                        f"{result.stderr or result.stdout}"
                    )
                for uid in (result.stdout or "").split():
                    if not uid.isdigit():
                        raise RuntimeError(
                            f"Invalid process owner in cleanup output: {uid}"
                        )
                    if uid not in users and uid != "0":
                        users.append(uid)
        except Exception as cause:
            # stop() is best-effort on some backends; runtime confirmation is
            # required before already-local logs can be considered immutable.
            sandbox_stopped = await self._stop_sandbox()
            raise AgentProcessCleanupError(
                "Could not stop sandbox agent processes",
                sandbox_stopped=sandbox_stopped,
            ) from cause
        self.stopped = not keep_pids

    async def _stop_sandbox(self) -> bool:
        try:
            await asyncio.wait_for(
                self.environment.stop(delete=False), timeout=self._STOP_TIMEOUT_SEC
            )
            return (
                await asyncio.wait_for(
                    self.environment.agent_sandbox_is_stopped(), timeout=10
                )
                is True
            )
        except Exception:
            return False

    async def wait_stopped(self, keep_pids: tuple[int, ...] = ()) -> None:
        """Do not leave cleanup running in the background on cancellation."""
        cleanup = asyncio.create_task(self.stop(keep_pids))
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError as cancelled:
            try:
                while not cleanup.done():
                    with contextlib.suppress(asyncio.CancelledError):
                        await asyncio.shield(cleanup)
                cleanup.result()
            except AgentProcessCleanupError as exc:
                exc.cancellation = cancelled
                if exc.primary_exception is None:
                    exc.preserve_primary(cancelled)
                raise
            raise

    async def __aenter__(self) -> "AgentProcesses":
        return self

    async def __aexit__(self, _type=None, primary=None, _tb=None) -> None:
        if isinstance(primary, AgentProcessCleanupError):
            # Adapter cleanup already performed the fallback and captured its
            # runtime proof. A second sweep cannot improve that evidence.
            return
        try:
            await self.wait_stopped()
        except AgentProcessCleanupError as exc:
            exc.preserve_primary(primary)
            raise
