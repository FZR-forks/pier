"""Runner for the OpenCode V2 benchmark trial.

Copied into the trial environment and executed by the Pier ``opencode-v2``
agent. It owns one OpenCode V2 server per run so every session the run creates
is accountable to it, and it records the evidence Pier needs to build the
trajectory tree afterwards:

- ``opencode serve --stdio`` prints the ready server as one JSON line
  (``{"url": ...}`` on stdout) and exits when stdin closes, so the runner keeps
  stdin open for the whole run.
- The server answers Basic auth as ``opencode`` plus a fresh password generated
  inside this runner and shared only with its server and CLI subprocesses.
- The CLI connects with ``run --server <url>`` instead of booting its own
  background service, so every session belongs to *this* server and the runner
  can page it, interrupt it, and watch it die.

Only the standard library is used.
"""

import argparse
import base64
import hashlib
import json
import os
import secrets
import select
import shutil
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from pathlib import Path
from typing import Any, Iterable

READY_TIMEOUT_SECONDS = 30
AUTH_USERNAME = "opencode"
SESSION_PAGE_SIZE = 200
MESSAGE_PAGE_SIZE = 200
INSPECT_ATTEMPTS = 3
DEFAULT_SETTLE_SECONDS = 600.0
SESSION_WAIT_REQUEST_SECONDS = 30.0
SETTLE_INTERVAL_SECONDS = 1.0
SETTLE_MAX_INTERVAL_SECONDS = 5.0
RAW_PAGE_LIMIT = 512
RAW_PAGE_BYTES_LIMIT = 16 * 1024 * 1024
CLI_CAPTURE_MAX_LINES = 256
CLI_CAPTURE_MAX_LINE_BYTES = 16 * 1024
MAX_ROOT_CANDIDATES = 1024

# Live (write-through) observability artifacts. These mirror state the runner
# already holds; nothing here issues a request to the OpenCode server.
STATUS_FILENAME = "opencode-v2-status.json"
INCIDENTS_FILENAME = "opencode-v2-incidents.jsonl"
CLI_STREAM_FILENAME = "opencode-v2-cli-stream.jsonl"
CLI_STDERR_FILENAME = "opencode-v2-cli-stderr.log"
SERVER_STDERR_FILENAME = "opencode-v2-server-stderr.log"
PARTIAL_SESSIONS_FILENAME = "opencode-v2-sessions.partial.jsonl"

LIVE_LOG_MAX_BYTES = 64 * 1024 * 1024
STATUS_MIN_INTERVAL_SECONDS = 1.0
STATUS_HEARTBEAT_SECONDS = 5.0


# -- write-through observability ---------------------------------------------
#
# Pi, Codex and Claude Code stream agent output straight to a file while the
# agent runs, so an interrupted trial still shows what the agent was doing.
# This runner consumes the CLI and server pipes itself, so without the helpers
# below its evidence would only reach disk during final collection -- exactly
# the path a killed or timed-out run never reaches.
#
# Two rules keep this benchmark-neutral. Everything recorded is state the
# runner already observes, so the OpenCode process is never polled or
# perturbed. And every write is best effort: the threads that feed these logs
# are the same threads that keep the benchmarked CLI's pipes drained, so a
# failing log must degrade to silence rather than raise.


def _iso(timestamp: float) -> str:
    return (
        time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(timestamp))
        + f".{int((timestamp % 1) * 1000):03d}Z"
    )


class LiveLog:
    """An append-only, byte-capped, never-raising text log."""

    def __init__(self, path: Path, max_bytes: int = LIVE_LOG_MAX_BYTES):
        self.path = path
        self.max_bytes = max_bytes
        self.written = 0
        self.truncated = False
        self.disabled_reason: str | None = None
        self._handle: Any = None
        self._lock = threading.Lock()

    def _open(self) -> Any:
        if self._handle is None and self.disabled_reason is None:
            try:
                self._handle = self.path.open("a", encoding="utf-8", errors="replace")
            except OSError as error:
                self.disabled_reason = f"{type(error).__name__}: {error}"
        return self._handle

    def write(self, text: str) -> None:
        """Persist ``text`` immediately, or give up permanently on failure."""
        if not text:
            return
        with self._lock:
            if self.truncated or self.disabled_reason is not None:
                return
            handle = self._open()
            if handle is None:
                return
            size = len(text.encode("utf-8", "replace"))
            if self.written + size > self.max_bytes:
                text = (
                    f"\n[pier] live log truncated after {self.written} bytes "
                    f"(cap {self.max_bytes})\n"
                )
                self.truncated = True
            try:
                handle.write(text)
                handle.flush()
            except (OSError, ValueError) as error:
                self.disabled_reason = f"{type(error).__name__}: {error}"
                return
            if not self.truncated:
                self.written += size

    def close(self) -> None:
        with self._lock:
            handle, self._handle = self._handle, None
        if handle is not None:
            try:
                handle.close()
            except OSError:
                pass

    def state(self) -> dict[str, Any]:
        return {
            "path": self.path.name,
            "bytes": self.written,
            "truncated": self.truncated,
            "disabled": self.disabled_reason,
        }


class LiveRecorder:
    """A durable, continuously-updated view of what the runner is doing.

    The runner's own stdout is already teed to a file by Pier, so stage and
    incident lines are echoed there too. That gives a human-readable timeline
    even when the logs directory itself cannot be written.
    """

    def __init__(
        self,
        logs_dir: Path,
        *,
        echo: bool = True,
        heartbeat_seconds: float = STATUS_HEARTBEAT_SECONDS,
    ):
        self.logs_dir = logs_dir
        self.echo = echo
        self._lock = threading.RLock()
        self._start = time.monotonic()
        self._last_status_write = 0.0
        self._incident_count = 0
        self.cli_stream = LiveLog(logs_dir / CLI_STREAM_FILENAME)
        self.cli_stderr = LiveLog(logs_dir / CLI_STDERR_FILENAME)
        self.server_stderr = LiveLog(logs_dir / SERVER_STDERR_FILENAME)
        self.incidents = LiveLog(logs_dir / INCIDENTS_FILENAME)
        self._status: dict[str, Any] = {
            "stage": "starting",
            "stage_history": [],
            "started_at": _iso(time.time()),
            "runner_pid": os.getpid(),
            "server_started": False,
            "server_url_known": False,
            "server_exit_code": None,
            "cli_started": False,
            "cli_pid": None,
            "cli_returncode": None,
            "root_id": None,
            "root_candidates": [],
            "cli_stdout_lines": 0,
            "cli_stderr_lines": 0,
            "server_stderr_lines": 0,
            "last_cli_activity_at": None,
            "last_server_activity_at": None,
            "settle_iterations": 0,
            "sessions_snapshot_count": None,
            "sessions_snapshot_at": None,
            "incident_count": 0,
            "last_incident": None,
        }
        self._last_cli_activity: float | None = None
        self._last_server_activity: float | None = None
        self.stage("starting")
        # A hung agent stops calling _activity, so without an independent
        # heartbeat the status file would freeze with its last-known
        # "seconds_since_cli_activity" of ~0 and imply the agent was busy at
        # the moment it died. The heartbeat keeps idleness observable.
        self._stop_heartbeat = threading.Event()
        self._heartbeat_seconds = heartbeat_seconds
        self._heartbeat: threading.Thread | None = None
        if heartbeat_seconds > 0:
            self._heartbeat = threading.Thread(
                target=self._heartbeat_loop,
                name="opencode-v2-status-heartbeat",
                daemon=True,
            )
            self._heartbeat.start()

    def _heartbeat_loop(self) -> None:
        while not self._stop_heartbeat.wait(self._heartbeat_seconds):
            with self._lock:
                self._write_status()

    # -- emission ----------------------------------------------------------

    def _echo(self, text: str) -> None:
        if not self.echo:
            return
        try:
            print(text, flush=True)
        except (OSError, ValueError):
            self.echo = False

    def stage(self, name: str, **fields: Any) -> None:
        """Record that the runner reached a new stage of execution."""
        with self._lock:
            elapsed = round(time.monotonic() - self._start, 3)
            self._status["stage"] = name
            history = self._status["stage_history"]
            history.append({"stage": name, "at": _iso(time.time()), "elapsed": elapsed})
            del history[:-64]
            self._status.update(fields)
            self._write_status()
        detail = " ".join(f"{key}={value!r}" for key, value in sorted(fields.items()))
        self._echo(f"[opencode-v2] stage={name} elapsed={elapsed}s {detail}".rstrip())

    def update(self, **fields: Any) -> None:
        with self._lock:
            self._status.update(fields)
            self._write_status()

    def note(self, kind: str, message: str, **fields: Any) -> None:
        """Persist an error or notable observation at the moment it happens."""
        with self._lock:
            self._incident_count += 1
            record = {
                "at": _iso(time.time()),
                "elapsed": round(time.monotonic() - self._start, 3),
                "stage": self._status.get("stage"),
                "kind": kind,
                "message": message,
            }
            record.update(fields)
            self._status["incident_count"] = self._incident_count
            self._status["last_incident"] = record
            # default=str mirrors _write_status: this runs on the CLI drain
            # thread, and a TypeError here would stop draining and stall the
            # benchmarked CLI on a full pipe.
            self.incidents.write(
                json.dumps(record, ensure_ascii=False, default=str) + "\n"
            )
            self._write_status()
        self._echo(f"[opencode-v2] {kind}: {message}")

    # -- streamed activity -------------------------------------------------

    def cli_stdout_line(self, line: str) -> None:
        self.cli_stream.write(line if line.endswith("\n") else line + "\n")
        self._activity("cli_stdout_lines", source="cli")

    def cli_stderr_line(self, line: str) -> None:
        self.cli_stderr.write(line if line.endswith("\n") else line + "\n")
        self._activity("cli_stderr_lines", source="cli")

    def server_stderr_line(self, line: str) -> None:
        self.server_stderr.write(line if line.endswith("\n") else line + "\n")
        self._activity("server_stderr_lines", source="server")

    def observe_event(self, event: dict[str, Any]) -> None:
        """Note what the CLI is doing right now, without forcing a write.

        The throttled counter update that follows each line publishes this,
        so describing activity costs nothing beyond the parse the drain
        thread already performs for root-session capture.
        """
        part = event.get("part")
        part = part if isinstance(part, dict) else {}
        state = part.get("state")
        state = state if isinstance(state, dict) else {}
        described = {
            "type": event.get("type"),
            "tool": part.get("tool"),
            "status": state.get("status"),
            "sessionID": event.get("sessionID"),
            "timestamp": event.get("timestamp"),
        }
        with self._lock:
            self._status["last_cli_event"] = {
                key: value for key, value in described.items() if value is not None
            }
            if part.get("tool"):
                self._status["last_tool"] = {
                    "tool": part.get("tool"),
                    "status": state.get("status"),
                    "at": _iso(time.time()),
                }

    def _activity(self, counter: str, *, source: str) -> None:
        """Count a line and refresh status, throttled so a chatty CLI cannot
        turn every line of output into a status rewrite.

        CLI and server activity are timed separately on purpose. The server
        logs on its own schedule, so folding it into the CLI's clock would
        let a chatty server make a hung agent look busy -- destroying the
        signal that distinguishes "still working" from "stopped".
        """
        now = time.monotonic()
        with self._lock:
            self._status[counter] = int(self._status.get(counter) or 0) + 1
            if source == "cli":
                self._last_cli_activity = now
                self._status["last_cli_activity_at"] = _iso(time.time())
            else:
                self._last_server_activity = now
                self._status["last_server_activity_at"] = _iso(time.time())
            if now - self._last_status_write >= STATUS_MIN_INTERVAL_SECONDS:
                self._write_status()

    # -- snapshots ---------------------------------------------------------

    def write_partial_sessions(self, inspections: list[dict]) -> None:
        """Preserve the newest full inspection snapshot from the settle loop.

        Collection can legitimately run for minutes; without this, a kill
        anywhere inside it discards every session record already fetched.
        """
        path = self.logs_dir / PARTIAL_SESSIONS_FILENAME
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            temporary.write_text(
                "".join(
                    json.dumps(item, ensure_ascii=False) + "\n" for item in inspections
                )
            )
            os.replace(temporary, path)
        except (OSError, TypeError, ValueError) as error:
            self.note("partial-snapshot-failed", f"{type(error).__name__}: {error}")
            return
        with self._lock:
            self._status["sessions_snapshot_count"] = len(inspections)
            self._status["sessions_snapshot_at"] = _iso(time.time())
            self._write_status()

    # -- status file -------------------------------------------------------

    def _write_status(self) -> None:
        """Replace the status file atomically. Caller holds the lock."""
        now = time.monotonic()
        self._last_status_write = now
        payload = dict(self._status)
        payload["updated_at"] = _iso(time.time())
        payload["elapsed_seconds"] = round(now - self._start, 3)
        payload["seconds_since_cli_activity"] = (
            round(now - self._last_cli_activity, 3)
            if self._last_cli_activity is not None
            else None
        )
        payload["seconds_since_server_activity"] = (
            round(now - self._last_server_activity, 3)
            if self._last_server_activity is not None
            else None
        )
        payload["live_logs"] = {
            "cli_stream": self.cli_stream.state(),
            "cli_stderr": self.cli_stderr.state(),
            "server_stderr": self.server_stderr.state(),
            "incidents": self.incidents.state(),
        }
        path = self.logs_dir / STATUS_FILENAME
        temporary = path.with_suffix(".json.tmp")
        try:
            temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n")
            os.replace(temporary, path)
        except (OSError, TypeError, ValueError):
            # Status is a convenience; the incident and stream logs remain.
            pass

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._status)

    def close(self) -> None:
        self._stop_heartbeat.set()
        if self._heartbeat is not None:
            self._heartbeat.join(timeout=1)
        with self._lock:
            self._write_status()
        for log in (
            self.cli_stream,
            self.cli_stderr,
            self.server_stderr,
            self.incidents,
        ):
            log.close()


class RecordedErrors(list):
    """A ``collection_errors`` list that persists each entry as it is added.

    Subclassing the list keeps every existing ``append``/``extend`` call site
    working unchanged, and makes it impossible to add a collection error that
    is only visible once ``runner-result.json`` is written.
    """

    def __init__(self, recorder: LiveRecorder):
        super().__init__()
        self._recorder = recorder

    def append(self, item: Any) -> None:
        super().append(item)
        self._recorder.note("collection-error", str(item))

    def extend(self, items: Iterable[Any]) -> None:
        for item in items:
            self.append(item)


def auth_header(password: str) -> str:
    token = base64.b64encode(f"{AUTH_USERNAME}:{password}".encode()).decode()
    return f"Basic {token}"


def http_get(url: str, password: str, timeout: float = 60.0) -> tuple[int, str]:
    request = urllib.request.Request(
        url, headers={"Authorization": auth_header(password)}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8", "replace")


def http_get_json(url: str, password: str, timeout: float = 60.0) -> tuple[int, Any]:
    status, body = http_get(url, password, timeout=timeout)
    try:
        return status, json.loads(body)
    except json.JSONDecodeError:
        return status, None


def http_post(url: str, password: str, payload: Any, timeout: float = 10.0) -> int:
    request = urllib.request.Request(
        url,
        method="POST",
        headers={
            "Authorization": auth_header(password),
            "Content-Type": "application/json",
        },
        data=json.dumps(payload).encode(),
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response.read()
            return response.status
    except urllib.error.HTTPError as error:
        error.read()
        return error.code
    except (urllib.error.URLError, OSError):
        return 0


def wait_for_server(process, timeout_seconds: float = READY_TIMEOUT_SECONDS) -> str:
    """Read the readiness JSON line from ``opencode serve --stdio`` stdout."""
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process.stdout is None:
            break
        ready, _, _ = select.select(
            [process.stdout], [], [], max(0.0, deadline - time.monotonic())
        )
        if not ready:
            break
        line = process.stdout.readline()
        if line:
            try:
                payload = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            url = payload.get("url") if isinstance(payload, dict) else None
            if isinstance(url, str) and url:
                parsed = urllib.parse.urlsplit(url)
                if (
                    parsed.scheme not in {"http", "https"}
                    or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
                    or parsed.port is None
                    or not 1 <= parsed.port <= 65535
                ):
                    raise RuntimeError(
                        f"OpenCode readiness URL is not loopback: {url!r}"
                    )
                return url
            continue
        if process.poll() is not None:
            break
        time.sleep(0.05)
    raise RuntimeError(
        "opencode serve --stdio did not report a ready server within "
        f"{timeout_seconds:.0f}s (exit={process.poll()})"
    )


class OpenCodeV2Server:
    """One ``opencode serve --stdio`` process plus its authenticated client."""

    def __init__(
        self,
        binary: str,
        cwd: str,
        password: str,
        env: dict[str, str],
        recorder: "LiveRecorder | None" = None,
    ):
        self.binary = binary
        self.cwd = cwd
        self.password = password
        self.env = env
        # Optional: when present, server stderr becomes durable as it arrives
        # instead of only reaching disk through the final runner result.
        self.recorder = recorder
        self.process: subprocess.Popen | None = None
        self.url: str | None = None
        self.server_stderr = ""
        self._stderr_chunks: list[str] = []
        self._stderr_thread: threading.Thread | None = None
        self.collection_deadline: float | None = None
        self.raw_pages: list[dict[str, Any]] = []
        self._raw_page_sizes: list[int] = []
        self._raw_page_digests: list[str] = []
        self._raw_page_hashes: set[str] = set()
        self._raw_page_bytes = 0
        self.raw_pages_dropped = 0
        self.raw_pages_deduplicated = 0
        self.raw_pages_oversize = 0

    def _record_raw_page(self, record: dict[str, Any]) -> None:
        """Keep a bounded, de-duplicated tail of raw HTTP page evidence."""
        encoded = json.dumps(
            record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
        digest = hashlib.sha256(encoded).hexdigest()
        if digest in self._raw_page_hashes:
            self.raw_pages_deduplicated += 1
            return
        if len(encoded) > RAW_PAGE_BYTES_LIMIT:
            self.raw_pages_oversize += 1
            record = {
                "endpoint": str(record.get("endpoint") or "")[:2048],
                "status": record.get("status"),
                "oversize_page": {
                    "encoded_bytes": len(encoded),
                    "sha256": digest,
                    "response_omitted": True,
                },
            }
            encoded = json.dumps(
                record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode()
        self.raw_pages.append(record)
        self._raw_page_sizes.append(len(encoded))
        self._raw_page_digests.append(digest)
        self._raw_page_hashes.add(digest)
        self._raw_page_bytes += len(encoded)
        while len(self.raw_pages) > 1 and (
            len(self.raw_pages) > RAW_PAGE_LIMIT
            or self._raw_page_bytes > RAW_PAGE_BYTES_LIMIT
        ):
            self.raw_pages.pop(0)
            removed_size = self._raw_page_sizes.pop(0)
            removed_digest = self._raw_page_digests.pop(0)
            self._raw_page_hashes.discard(removed_digest)
            self._raw_page_bytes -= removed_size
            self.raw_pages_dropped += 1

    def _request_timeout(self, default: float) -> float:
        if self.collection_deadline is None:
            return default
        return max(0.05, min(default, self.collection_deadline - time.monotonic()))

    def _api_url(self, path: str) -> str:
        if not self.url:
            raise RuntimeError("OpenCode server has not started")
        return self.url.rstrip("/") + "/" + path.lstrip("/")

    def start(self) -> str:
        command = [
            self.binary,
            "serve",
            "--stdio",
            "--hostname",
            "127.0.0.1",
            "--port",
            "0",
        ]
        self.process = subprocess.Popen(
            command,
            cwd=self.cwd,
            env=self.env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        self._stderr_chunks = []

        def drain_stderr() -> None:
            stream = self.process.stderr if self.process else None
            if stream is None:
                return
            try:
                for line in stream:
                    self._stderr_chunks.append(line)
                    if self.recorder is not None:
                        self.recorder.server_stderr_line(line)
            except (OSError, ValueError):
                pass

        self._stderr_thread = threading.Thread(
            target=drain_stderr, name="opencode-v2-server-stderr", daemon=True
        )
        self._stderr_thread.start()
        try:
            self.url = wait_for_server(self.process)
        except Exception as error:
            try:
                self.stop()
            except Exception as cleanup_error:
                error.add_note(
                    "OpenCode server cleanup after readiness failure also failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
            raise
        # Fail fast when the URL could never answer a benchmark request, and
        # do not leak the server if readiness itself fails.
        try:
            status, _ = http_get(self._api_url("api/info"), self.password, timeout=10.0)
            if status != 200:
                raise RuntimeError(
                    f"OpenCode server {self.url} failed its readiness check (HTTP {status})"
                )
        except Exception as error:
            try:
                self.stop()
            except Exception as cleanup_error:
                error.add_note(
                    "OpenCode server cleanup after health failure also failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
            raise
        return self.url

    # -- paged collection --------------------------------------------------

    def page_sessions(
        self, parent_id: str | None = None, cursor: str | None = None
    ) -> tuple[list[dict], str | None]:
        query: list[tuple[str, str | int]] = [("limit", SESSION_PAGE_SIZE)]
        if parent_id:
            query.append(("parentID", parent_id))
        if cursor:
            query.append(("cursor", cursor))
        assert self.url is not None
        status, payload = http_get_json(
            self._api_url(f"api/session?{urllib.parse.urlencode(query)}"),
            self.password,
            timeout=self._request_timeout(60.0),
        )
        self._record_raw_page(
            {
                "endpoint": "/api/session",
                "query": dict(query),
                "status": status,
                "response": payload,
            }
        )
        if status != 200 or not isinstance(payload, dict):
            raise RuntimeError(f"GET /api/session failed (HTTP {status}): {payload!r}")
        data = payload.get("data")
        if not isinstance(data, list) or any(
            not isinstance(item, dict) for item in data
        ):
            raise RuntimeError("GET /api/session returned malformed session records")
        return data, (payload.get("cursor") or {}).get("next")

    def page_messages(
        self, session_id: str, cursor: str | None = None
    ) -> tuple[list[dict], str | None]:
        # The message cursor is opaque and order-sensitive: never send `order`
        # together with it, and always pass `type` through unchanged when the
        # caller filters (Pier collects unfiltered, so nothing to carry).
        query: list[tuple[str, str | int]] = [("limit", MESSAGE_PAGE_SIZE)]
        if cursor:
            query.append(("cursor", cursor))
        else:
            query.append(("order", "asc"))
        assert self.url is not None
        status, payload = http_get_json(
            self._api_url(
                f"api/session/{urllib.parse.quote(session_id, safe='')}/message?"
                f"{urllib.parse.urlencode(query)}"
            ),
            self.password,
            timeout=self._request_timeout(60.0),
        )
        self._record_raw_page(
            {
                "endpoint": f"/api/session/{session_id}/message",
                "query": dict(query),
                "status": status,
                "response": payload,
            }
        )
        if status != 200 or not isinstance(payload, dict):
            raise RuntimeError(
                f"GET /api/session/{session_id}/message failed "
                f"(HTTP {status}): {payload!r}"
            )
        data = payload.get("data")
        if not isinstance(data, list) or any(
            not isinstance(item, dict) for item in data
        ):
            raise RuntimeError(
                f"GET /api/session/{session_id}/message returned malformed records"
            )
        return data, (payload.get("cursor") or {}).get("next")

    def collect_sessions(self, parent_id: str | None = None) -> list[dict]:
        """All sessions, following every cursor page.

        A failed page is an evidence gap, not an empty page. Callers preserve
        the exception in the manifest and must not report a complete tree.
        """
        sessions: list[dict] = []
        cursor = None
        seen: set[str] = set()
        while True:
            page, next_cursor = self.page_sessions(parent_id=parent_id, cursor=cursor)
            sessions.extend(page)
            if not next_cursor:
                break
            if next_cursor in seen:
                raise RuntimeError(
                    f"session pagination cursor repeated: {next_cursor!r}"
                )
            seen.add(next_cursor)
            cursor = next_cursor
        return sessions

    def collect_messages(self, session_id: str) -> list[dict]:
        """All messages of one session, following every cursor page."""
        messages: list[dict] = []
        cursor = None
        seen: set[str] = set()
        while True:
            page, next_cursor = self.page_messages(session_id, cursor=cursor)
            messages.extend(page)
            if not next_cursor:
                break
            if next_cursor in seen:
                raise RuntimeError(
                    f"message pagination cursor repeated: {next_cursor!r}"
                )
            seen.add(next_cursor)
            cursor = next_cursor
        return messages

    # -- runtime state -----------------------------------------------------

    def running_session_ids(self) -> set[str]:
        """Session ids the server reports as active or busy right now."""
        if not self.url:
            return set()
        status, payload = http_get_json(
            self._api_url("api/session/active"),
            self.password,
            timeout=self._request_timeout(10.0),
        )
        if status != 200 or not isinstance(payload, dict):
            raise RuntimeError(
                f"GET /api/session/active failed (HTTP {status}): {payload!r}"
            )
        active = payload.get("data")
        if not isinstance(active, dict):
            raise RuntimeError("GET /api/session/active returned invalid data")
        return {
            session_id
            for session_id, info in active.items()
            if isinstance(info, dict) and info.get("type") == "running"
        }

    def interrupt_session(self, session_id: str) -> None:
        if not self.url:
            return
        http_post(
            self._api_url(f"api/session/{session_id}/interrupt"),
            self.password,
            {},
            timeout=self._request_timeout(10.0),
        )

    def wait_session(
        self, session_id: str, timeout: float = SESSION_WAIT_REQUEST_SECONDS
    ) -> bool:
        """Use the native wait contract to establish an idle session."""
        if not self.url:
            return False
        return http_post(
            self._api_url(f"api/experimental/session/{session_id}/wait"),
            self.password,
            {},
            timeout=self._request_timeout(timeout),
        ) in {200, 204}

    def interrupt_all(self, session_ids) -> None:
        for session_id in sorted(session_ids):
            self.interrupt_session(session_id)

    def inbox_items(self, session_id: str) -> list[dict] | None:
        if not self.url:
            return None
        status, payload = http_get_json(
            self._api_url(f"api/session/{session_id}/inbox"),
            self.password,
            timeout=self._request_timeout(10.0),
        )
        if status != 200 or not isinstance(payload, dict):
            return None
        data = payload.get("data")
        return (
            data
            if isinstance(data, list) and all(isinstance(item, dict) for item in data)
            else None
        )

    def get_session(self, session_id: str) -> dict | None:
        """Fetch one session without relying on global-list pagination."""
        if not self.url:
            return None
        status, payload = http_get_json(
            self._api_url(f"api/session/{session_id}"),
            self.password,
            timeout=self._request_timeout(60.0),
        )
        if status != 200 or not isinstance(payload, dict):
            return None
        data = payload.get("data")
        return data if isinstance(data, dict) else None

    # -- lifecycle -----------------------------------------------------------

    def read_server_stderr(self) -> str:
        process = self.process
        if process is None or not process.stderr:
            return "".join(self._stderr_chunks)
        if process.poll() is not None and self._stderr_thread is not None:
            self._stderr_thread.join(timeout=1)
        return "".join(self._stderr_chunks)

    def stop(self) -> None:
        """Close stdin, then terminate the server's whole process group."""
        process = self.process
        if process is None:
            return
        process_group = process.pid

        def group_exists() -> bool:
            try:
                os.killpg(process_group, 0)
            except ProcessLookupError:
                return False
            except PermissionError:
                return True
            return True

        try:
            if process.stdin and not process.stdin.closed:
                process.stdin.close()
        except OSError:
            pass
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            pass
        # The lease-owning leader may exit while a helper remains in the
        # server's session. Do not return until the whole owned process group
        # is gone.
        for sig in (signal.SIGTERM, signal.SIGKILL):
            if not group_exists():
                break
            try:
                os.killpg(process_group, sig)
            except (ProcessLookupError, PermissionError):
                pass
            deadline = time.monotonic() + 3
            while group_exists() and time.monotonic() < deadline:
                try:
                    process.wait(timeout=0.1)
                except subprocess.TimeoutExpired:
                    pass
                time.sleep(0.05)
        if group_exists():
            raise RuntimeError(
                f"OpenCode server process group {process_group} survived SIGKILL"
            )
        self.process = None

    def inspect_session(
        self, session: dict, active_ids: set[str] | None = None
    ) -> dict:
        return inspect_session(self, session, active_ids=active_ids)

    def collect_descendants(self, root_id: str) -> list[dict]:
        """Recursively enumerate a root and every direct child page."""
        found: list[dict] = []
        processed: set[str] = set()
        enqueued: set[str] = {root_id}
        queue = [root_id]
        while queue:
            parent_id = queue.pop(0)
            if parent_id in processed:
                continue
            processed.add(parent_id)
            roots = self.collect_sessions(parent_id=parent_id)
            for session in roots:
                session_id = str(session.get("id") or "")
                if not session_id or session_id in enqueued:
                    continue
                enqueued.add(session_id)
                found.append(session)
                queue.append(session_id)
        root_pages = self.collect_sessions()
        root = next(
            (item for item in root_pages if str(item.get("id")) == root_id), None
        )
        if root is None:
            # A newly-created root can fall outside the first page of the
            # global session list. The native point lookup avoids guessing
            # from whichever historical session happens to be newest.
            root = self.get_session(root_id)
        if root is None:
            raise RuntimeError(f"root session {root_id} was not returned")
        found.insert(0, root)
        return found


def resolve_binary(explicit: str | None) -> str:
    """Locate the OpenCode binary.

    ``--binary`` wins; otherwise ``OPENCODE_BINARY``; otherwise PATH.
    """
    if explicit:
        return os.path.abspath(explicit)
    from_env = os.environ.get("OPENCODE_BINARY")
    if from_env:
        return os.path.abspath(from_env)
    found = shutil.which("opencode")
    if found:
        return found
    raise RuntimeError(
        "opencode binary not found: pass --binary or set OPENCODE_BINARY"
    )


def dump_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    )


def _persist_root_candidates(
    path: Path, payload: dict[str, Any], errors: list[str]
) -> None:
    """Retain root candidates without interrupting CLI stdout collection."""
    try:
        path.write_text(json.dumps(payload, indent=2) + "\n")
    except OSError as error:
        errors.append(f"root candidate persistence: {type(error).__name__}: {error}")


# Bare key names that are always a credential.
_CREDENTIAL_KEYS = frozenset({"token", "auth", "key", "apikey", "secret", "password"})
# Substrings that only appear in credential-bearing key names. A bare "token"
# is deliberately absent: model metadata uses `max_tokens`, `maxTokensField`
# and `outputTokens`, and redacting those would destroy the output-cap
# evidence the preflight record exists to prove (PA1 #40/#46).
_CREDENTIAL_MARKERS = (
    "apikey",
    "authorization",
    "password",
    "passwd",
    "secret",
    "credential",
    "accesstoken",
    "authtoken",
    "bearertoken",
    "idtoken",
    "refreshtoken",
    "sessiontoken",
    "apitoken",
)


def _redact(value: Any) -> Any:
    """Remove credential-shaped values from persisted preflight evidence."""
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            normalized = "".join(char for char in key.lower() if char.isalnum())
            if normalized in {"baseurl", "url"}:
                result[key] = "<redacted-url>"
            elif normalized in _CREDENTIAL_KEYS or any(
                marker in normalized for marker in _CREDENTIAL_MARKERS
            ):
                result[key] = "<redacted>"
            else:
                result[key] = _redact(item)
        return result
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _location_data(server: OpenCodeV2Server, endpoint: str) -> list[dict]:
    query = urllib.parse.urlencode({"location[directory]": server.cwd})
    status, payload = http_get_json(
        server._api_url(f"api/{endpoint}?{query}"),
        server.password,
        timeout=server._request_timeout(15.0),
    )
    if status != 200 or not isinstance(payload, dict):
        raise RuntimeError(f"GET /api/{endpoint} failed (HTTP {status}): {payload!r}")
    data = payload.get("data")
    if not isinstance(data, list):
        raise RuntimeError(f"GET /api/{endpoint} returned invalid data")
    return [item for item in data if isinstance(item, dict)]


def _model_selection(value: str, *, label: str) -> tuple[str, str, str]:
    if "/" not in value:
        raise RuntimeError(f"{label} must be provider/model[#variant]")
    provider_id, model_ref = value.split("/", 1)
    model_id, separator, variant = model_ref.partition("#")
    if not provider_id or not model_id or (separator and not variant):
        raise RuntimeError(f"{label} must be provider/model[#variant]")
    return provider_id, model_id, variant if separator else ""


def preflight_runtime(
    server: OpenCodeV2Server,
    *,
    model_spec: str | None,
    config_file: str | None,
    restrict_model: bool,
    timeout: float = 15.0,
) -> dict[str, Any]:
    """Resolve and verify the selected model and all configured active agents."""
    if not model_spec:
        raise RuntimeError("preflight requires provider/model[#variant]")
    provider_id, model_id, variant = _model_selection(
        model_spec, label="preflight model"
    )

    config: dict[str, Any] = {}
    if config_file:
        value = json.loads(Path(config_file).read_text())
        if not isinstance(value, dict):
            raise RuntimeError("OpenCode config is not an object")
        config = value
    expected_model = (
        ((config.get("providers") or {}).get(provider_id) or {}).get("models") or {}
    ).get(model_id) or {}
    expected_body = expected_model.get("body") or {}
    expected_limit = expected_model.get("limit") or {}
    configured_agents = config.get("agents") or {}

    deadline = time.monotonic() + timeout
    last_error = "configured entries were not visible"
    while time.monotonic() < deadline:
        try:
            models = _location_data(server, "model")
            agents = _location_data(server, "agent")
            selected = next(
                (
                    item
                    for item in models
                    if item.get("providerID") == provider_id
                    and item.get("id") == model_id
                ),
                None,
            )
            if selected is None:
                raise RuntimeError("selected model is absent from /api/model")
            variants = {
                str(item.get("id"))
                for item in selected.get("variants") or []
                if isinstance(item, dict) and item.get("id")
            }
            if variant and variant not in variants:
                raise RuntimeError(f"selected variant {variant!r} is absent")
            if restrict_model:
                identities = {
                    (str(item.get("providerID")), str(item.get("id")))
                    for item in models
                }
                if identities != {(provider_id, model_id)}:
                    raise RuntimeError(
                        "restricted model catalog exposed unexpected models: "
                        + ", ".join(
                            f"{provider}/{model}"
                            for provider, model in sorted(identities)
                        )
                    )
                if variant and variants != {variant}:
                    raise RuntimeError(
                        "restricted model catalog exposed unexpected variants: "
                        + ", ".join(sorted(variants))
                    )
            resolved_body = selected.get("body") or {}
            for key, value in expected_body.items():
                if resolved_body.get(key) != value:
                    raise RuntimeError(
                        f"resolved model body {key!r} is {resolved_body.get(key)!r}, "
                        f"expected {value!r}"
                    )
            resolved_limit = selected.get("limit") or {}
            for key, value in expected_limit.items():
                if resolved_limit.get(key) != value:
                    raise RuntimeError(
                        f"resolved model limit {key!r} is "
                        f"{resolved_limit.get(key)!r}, expected {value!r}"
                    )
            context_limit = resolved_limit.get("context")
            input_limit = resolved_limit.get("input")
            output_limit = resolved_limit.get("output")
            if (
                isinstance(context_limit, int)
                and isinstance(input_limit, int)
                and isinstance(output_limit, int)
                and input_limit + output_limit > context_limit
            ):
                raise RuntimeError(
                    "resolved model limits are contradictory: "
                    f"input ({input_limit}) + output ({output_limit}) exceeds "
                    f"context ({context_limit})"
                )

            by_id = {str(item.get("id")): item for item in agents if item.get("id")}
            resolved_agents: list[dict[str, Any]] = []
            for agent_id, expected in configured_agents.items():
                if not isinstance(expected, dict) or expected.get("disabled"):
                    continue
                agent = by_id.get(str(agent_id))
                if agent is None:
                    raise RuntimeError(f"configured agent {agent_id!r} is absent")
                actual = agent.get("model") or {}
                expected_spec = model_spec
                if not restrict_model:
                    expected_spec = (
                        expected.get("model") or config.get("model") or model_spec
                    )
                if not isinstance(expected_spec, str):
                    raise RuntimeError(
                        f"configured agent {agent_id!r} model must be a string"
                    )
                (
                    expected_provider,
                    expected_model_id,
                    expected_variant,
                ) = _model_selection(
                    expected_spec, label=f"configured agent {agent_id!r} model"
                )
                if (
                    actual.get("providerID") != expected_provider
                    or actual.get("id") != expected_model_id
                    or (expected_variant and actual.get("variant") != expected_variant)
                ):
                    raise RuntimeError(
                        f"agent {agent_id!r} resolved unexpected model {actual!r}; "
                        f"expected {expected_spec!r}"
                    )
                resolved_agents.append(agent)

            sanitized_model = _redact(selected)
            encoded = json.dumps(
                sanitized_model, sort_keys=True, separators=(",", ":")
            ).encode()
            return {
                "selection": model_spec,
                "resolved_model": sanitized_model,
                "resolved_model_sha256": hashlib.sha256(encoded).hexdigest(),
                "resolved_agents": _redact(resolved_agents),
            }
        except Exception as error:  # entries can appear shortly after readiness
            last_error = f"{type(error).__name__}: {error}"
            time.sleep(min(0.25, max(0.0, deadline - time.monotonic())))
    raise RuntimeError(f"OpenCode runtime preflight failed: {last_error}")


def inspect_session(
    server: OpenCodeV2Server,
    session: dict,
    *,
    active_ids: set[str] | None = None,
) -> dict:
    """Collect one session's messages and native running/inbox state."""
    session_id = str(session.get("id"))
    if active_ids is None:
        active_ids = server.running_session_ids()
    return {
        "session": session,
        "messages": server.collect_messages(session_id),
        "active": {
            "type": "running",
        }
        if session_id in active_ids
        else None,
        "inbox": server.inbox_items(session_id),
    }


def _bounded_cli_line(line: str) -> str:
    encoded = line.encode("utf-8", errors="replace")
    if len(encoded) <= CLI_CAPTURE_MAX_LINE_BYTES:
        return line
    return encoded[:CLI_CAPTURE_MAX_LINE_BYTES].decode("utf-8", errors="ignore")


def _truncate(value: Any, limit: int = 2000) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= limit else text[:limit] + "...<truncated>"


def _note_cli_error(recorder: LiveRecorder, event: dict[str, Any]) -> None:
    """Persist tool/provider failures the CLI reports, as they are reported.

    Detection is structural rather than a text search: the agent frequently
    writes code and tests that merely mention errors, and those must not be
    recorded as failures of the run.
    """
    recorder.observe_event(event)
    event_type = str(event.get("type") or "")
    session_id = event.get("sessionID")
    if "error" in event_type.lower():
        recorder.note(
            "cli-error-event",
            _truncate(event.get("error") or event),
            event_type=event_type,
            sessionID=session_id,
        )
        return
    if event.get("error"):
        recorder.note(
            "cli-error",
            _truncate(event["error"]),
            event_type=event_type,
            sessionID=session_id,
        )
        return
    part = event.get("part")
    if not isinstance(part, dict):
        return
    state = part.get("state")
    if not isinstance(state, dict):
        return
    if str(state.get("status") or "") == "error":
        recorder.note(
            "tool-error",
            _truncate(state.get("error") or state.get("output") or state),
            tool=part.get("tool"),
            sessionID=session_id,
        )


def _events(stdout_lines: Iterable[str]) -> list[dict]:
    result: list[dict] = []
    for line in stdout_lines:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError:
            result.append({"type": "cli-stdout", "raw": stripped})
        else:
            result.append(
                value
                if isinstance(value, dict)
                else {"type": "cli-stdout", "raw": stripped}
            )
    return result


def _root_id(
    events: list[dict],
    sessions: list[dict],
    before: set[str],
    candidate_ids: set[str] | None = None,
) -> str | None:
    """Resolve the newly-created root without guessing from global history."""
    event_ids = {str(event["sessionID"]) for event in events if event.get("sessionID")}
    event_ids.update(candidate_ids or ())
    candidates = [
        session
        for session in sessions
        if str(session.get("id") or "") not in before
        and not session.get("parentID")
        and (not event_ids or str(session.get("id")) in event_ids)
    ]
    if len(candidates) == 1:
        return str(candidates[0]["id"])
    # The CLI emits the root ID on every event. Use it only when the server
    # metadata confirms it is a root and there is one unambiguous candidate.
    event_roots = [
        session
        for session in sessions
        if str(session.get("id")) not in before
        and str(session.get("id")) in event_ids
        and not session.get("parentID")
    ]
    if len(event_roots) == 1:
        return str(event_roots[0]["id"])
    return None


def _snapshot_signature(inspections: list[dict]) -> tuple:
    return tuple(
        sorted(
            (
                str(item.get("session", {}).get("id")),
                item.get("session", {}).get("status")
                or item.get("session", {}).get("outcome"),
                json.dumps(
                    item.get("messages", []), sort_keys=True, separators=(",", ":")
                ),
                (item.get("active") or {}).get("type"),
                tuple(
                    sorted(str(inbox.get("id")) for inbox in (item.get("inbox") or []))
                ),
            )
            for item in inspections
        )
    )


def _referenced_session_ids(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {
                "sessionID",
                "sessionId",
                "childSessionID",
                "childSessionId",
            } and isinstance(item, str):
                found.add(item)
            found.update(_referenced_session_ids(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_referenced_session_ids(item))
    return found


def _unfinished_records(inspections: list[dict]) -> list[str]:
    """Identify active persisted work that native wait alone cannot prove done."""
    unfinished: list[str] = []
    for inspection in inspections:
        session_id = str((inspection.get("session") or {}).get("id") or "unknown")
        messages = inspection.get("messages") or []
        for message in messages:
            if not isinstance(message, dict):
                continue
            record_id = str(message.get("id") or "unknown")
            message_type = message.get("type")
            if message_type == "assistant":
                if (message.get("time") or {}).get("completed") is None:
                    unfinished.append(f"{session_id}:{record_id}:assistant")
                for part in message.get("content") or []:
                    if not isinstance(part, dict) or part.get("type") != "tool":
                        continue
                    status = str((part.get("state") or {}).get("status") or "")
                    if status not in {"completed", "error"}:
                        unfinished.append(
                            f"{session_id}:{record_id}:tool:{part.get('id', 'unknown')}:{status or 'unknown'}"
                        )
            elif message_type in {"compaction", "shell"}:
                status = str(message.get("status") or "")
                if status not in {"completed", "failed", "error"}:
                    unfinished.append(
                        f"{session_id}:{record_id}:{message_type}:{status or 'unknown'}"
                    )
    return unfinished


def _sessions_without_terminal_outcome(inspections: list[dict]) -> list[str]:
    missing: list[str] = []
    for inspection in inspections:
        session_outcome = (inspection.get("session") or {}).get("outcome")
        messages = inspection.get("messages") or []
        terminal = session_outcome in {"succeeded", "failed", "interrupted"} or any(
            isinstance(message, dict)
            and message.get("type") == "idle"
            and message.get("outcome") in {"succeeded", "failed", "interrupted"}
            for message in messages
        )
        if not terminal:
            missing.append(
                str((inspection.get("session") or {}).get("id") or "unknown")
            )
    return missing


def _collection_complete(
    *,
    settled: bool,
    errors: list[str],
    root_id: str | None,
    server_exit_code: int | None,
) -> bool:
    """Whether collection proved a stable tree while its owned server lived."""
    return settled and not errors and bool(root_id) and server_exit_code is None


def _raise_runner_failure(
    pending_error: BaseException | None,
    run_error: str | None,
) -> None:
    """Fail only the agent execution; collection gaps stay in the manifest."""
    if pending_error is not None:
        raise pending_error
    if run_error:
        raise SystemExit(run_error)
    # Observational gaps withhold complete aggregates through runner-result.


def _collect_tree(
    server: OpenCodeV2Server,
    root_id: str,
    *,
    deadline: float,
    errors: list[str],
    recorder: LiveRecorder | None = None,
) -> tuple[list[dict], bool]:
    """Wait and collect two identical terminal snapshots of the whole tree."""
    previous: tuple | None = None
    stable = False
    inspections: list[dict] = []
    last_blockers: list[str] = []
    poll_interval = SETTLE_INTERVAL_SECONDS
    iterations = 0
    server.collection_deadline = deadline
    try:
        while time.monotonic() < deadline:
            iterations += 1
            if recorder is not None:
                recorder.update(settle_iterations=iterations)
            blockers: list[str] = []
            try:
                sessions = server.collect_descendants(root_id)
                if not sessions:
                    raise RuntimeError(f"root session {root_id} was not returned")
                ids = [str(item.get("id")) for item in sessions if item.get("id")]
                for session_id in ids:
                    if not server.wait_session(session_id):
                        blockers.append(f"session.wait failed for {session_id}")
                active = server.running_session_ids()
                inspections = [
                    server.inspect_session(session, active_ids=active)
                    for session in sessions
                ]
                if any(item.get("inbox") is None for item in inspections):
                    blockers.append("session inbox state unavailable")
                pending_inbox = any(item.get("inbox") for item in inspections)
                if pending_inbox:
                    blockers.append("session tree retained queued background work")
                known = {str(item.get("session", {}).get("id")) for item in inspections}
                missing = _referenced_session_ids(inspections) - known
                if missing:
                    blockers.append(
                        "session references absent from discovery: "
                        + ", ".join(sorted(missing))
                    )
                unfinished = _unfinished_records(inspections)
                if unfinished:
                    blockers.append(
                        "unfinished persisted work: " + ", ".join(unfinished)
                    )
                missing_terminal = _sessions_without_terminal_outcome(inspections)
                if missing_terminal:
                    blockers.append(
                        "sessions lack terminal idle outcome: "
                        + ", ".join(missing_terminal)
                    )
                signature = _snapshot_signature(inspections)
                if active:
                    blockers.append("active sessions: " + ", ".join(sorted(active)))
                if not blockers and previous == signature:
                    stable = True
                    break
                if previous != signature:
                    poll_interval = SETTLE_INTERVAL_SECONDS
                    # Only rewrite when the tree actually changed: settling can
                    # take minutes and the snapshot is large.
                    if recorder is not None:
                        recorder.write_partial_sessions(inspections)
                previous = signature
            except Exception as error:  # preserve partial records and retry discovery
                blockers = [f"collection: {type(error).__name__}: {error}"]
            last_blockers = blockers
            time.sleep(min(poll_interval, max(0.0, deadline - time.monotonic())))
            poll_interval = min(SETTLE_MAX_INTERVAL_SECONDS, poll_interval * 2)
    finally:
        server.collection_deadline = None
    if not stable:
        errors.extend(last_blockers or ["session tree did not become terminal"])
        # A settlement deadline is a task timeout, not a clean shutdown. Stop
        # every known/newly-discoverable session, then take one final partial
        # snapshot while the owned server is still reachable.
        known_ids = {root_id}
        known_ids.update(
            str(item.get("session", {}).get("id"))
            for item in inspections
            if item.get("session", {}).get("id")
        )
        server.collection_deadline = time.monotonic() + 5
        try:
            sessions = server.collect_descendants(root_id)
            known_ids.update(
                str(session.get("id")) for session in sessions if session.get("id")
            )
            server.interrupt_all(known_ids)
            inspections = [server.inspect_session(session) for session in sessions]
            if recorder is not None:
                recorder.write_partial_sessions(inspections)
        except Exception as error:
            server.interrupt_all(known_ids)
            errors.append(f"timeout interruption: {type(error).__name__}: {error}")
        finally:
            server.collection_deadline = None
    return inspections, stable


def _preserve_private_state(env: dict[str, str], logs_dir: Path) -> list[str]:
    """Copy safe private state after failure without parsing credential stores."""
    copied: list[str] = []
    target = logs_dir / "private-state"
    excluded_names = {"auth.json", "credentials.json"}
    for label, env_key in (("state", "XDG_STATE_HOME"), ("data", "XDG_DATA_HOME")):
        source_text = env.get(env_key)
        if not source_text:
            continue
        source = Path(source_text)
        if not source.is_dir():
            continue
        for item in source.rglob("*"):
            if item.is_symlink() or not item.is_file():
                continue
            relative = item.relative_to(source)
            if relative.name.casefold() in excluded_names:
                continue
            # OpenCode's data tree can contain a full Git snapshot of the task.
            # It is not the session database and duplicating it into logs can
            # consume gigabytes. Preserve database/log/metadata state, while
            # the task repository remains available through normal artifacts.
            if "snapshot" in relative.parts:
                continue
            destination = target / label / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, destination)
            copied.append(str(destination))
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instruction-file", required=True)
    parser.add_argument("--logs-dir", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--binary", default=None)
    parser.add_argument("--model", default=None, help="provider/model[#variant]")
    parser.add_argument("--variant", default=None)
    parser.add_argument("--restrict-model", action="store_true")
    parser.add_argument("--title", default="pier-benchmark")
    parser.add_argument("--config-file", default=None)
    parser.add_argument("--agent", default=None, help="Primary agent name")
    parser.add_argument(
        "--settle-timeout",
        type=float,
        default=float(
            os.environ.get("PIER_OPENCODE_V2_SETTLE_TIMEOUT", DEFAULT_SETTLE_SECONDS)
        ),
    )
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir).resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)
    # Start recording before anything else can fail, so even an immediate
    # crash leaves a stage behind.
    recorder = LiveRecorder(logs_dir)
    work_dir = Path(args.work_dir).resolve()
    instruction = Path(args.instruction_file).read_text()
    binary = resolve_binary(args.binary)

    home = os.environ.get("HOME") or str(Path.home())
    # Generate the authentication secret inside the trial runner. It is shared
    # only with the owned server and CLI subprocesses and never crosses Pier's
    # debug-logged exec environment.
    password = secrets.token_urlsafe(32)

    # Private XDG dirs under the run's home. The generated config file is
    # pointed at with OPENCODE_CONFIG (an absolute path inside the sandbox) and
    # project-level config discovery is disabled, so the repository's own
    # opencode.json cannot replace the benchmark configuration.
    env = dict(os.environ)
    env["HOME"] = home
    env.setdefault("XDG_CONFIG_HOME", os.path.join(home, ".config"))
    env.setdefault("XDG_DATA_HOME", os.path.join(home, ".local", "share"))
    env.setdefault("XDG_STATE_HOME", os.path.join(home, ".local", "state"))
    env.setdefault("OPENCODE_CONFIG_PROJECT_DISABLE", "1")
    env.setdefault("OPENCODE_DISABLE_MODELS_FETCH", "1")
    env.setdefault("OPENCODE_DISABLE_AUTOUPDATE", "1")
    env["OPENCODE_PASSWORD"] = password
    env["OPENCODE_SERVER_PASSWORD"] = password
    env["PWD"] = str(work_dir)
    for key in ("NO_PROXY", "no_proxy"):
        entries = [item.strip() for item in env.get(key, "").split(",") if item.strip()]
        for host in ("127.0.0.1", "localhost", "::1"):
            if host not in entries:
                entries.append(host)
        env[key] = ",".join(entries)
    if args.config_file:
        env["OPENCODE_CONFIG"] = os.path.abspath(args.config_file)

    server = OpenCodeV2Server(
        binary=binary,
        cwd=str(work_dir),
        password=password,
        env=env,
        recorder=recorder,
    )

    stdout_lines: deque[str] = deque(maxlen=CLI_CAPTURE_MAX_LINES)
    cli_stderr = ""
    cli_returncode: int | None = None
    server_stderr = ""
    run_error: str | None = None
    collection_errors: list[str] = RecordedErrors(recorder)
    inspections: list[dict] = []
    interrupted: list[str] = []
    cancelled = False
    root_id: str | None = None
    cli_process: subprocess.Popen | None = None
    pending_error: BaseException | None = None
    before_ids: set[str] = set()
    events: list[dict] = []
    settled = False
    preflight: dict[str, Any] | None = None
    private_state_artifacts: list[str] = []
    server_exit_code: int | None = None
    early_root_candidates: set[str] = set()
    early_root_events: deque[dict[str, str]] = deque(maxlen=CLI_CAPTURE_MAX_LINES)

    def handle_termination(signum, _frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    signal.signal(signal.SIGTERM, handle_termination)

    try:
        recorder.stage("server-starting")
        server.start()
        assert server.url is not None
        recorder.stage("server-started", server_started=True, server_url_known=True)
        model_spec = args.model
        if model_spec and args.variant and "#" not in model_spec:
            model_spec = f"{model_spec}#{args.variant}"
        preflight = preflight_runtime(
            server,
            model_spec=model_spec,
            config_file=args.config_file,
            restrict_model=args.restrict_model,
        )
        (logs_dir / "opencode-v2-preflight.json").write_text(
            json.dumps(preflight, indent=2) + "\n"
        )
        recorder.stage("preflight-ok")
        try:
            before_ids = {
                str(item.get("id"))
                for item in server.collect_sessions()
                if item.get("id")
            }
        except Exception as error:
            collection_errors.append(
                f"initial discovery: {type(error).__name__}: {error}"
            )

        cli_command = [
            binary,
            "run",
            "--server",
            server.url,
            "--format",
            "json",
            "--thinking",
            "--auto",
            "--title",
            args.title,
        ]
        if model_spec:
            cli_command += ["-m", model_spec]
        if args.agent:
            cli_command += ["--agent", args.agent]
        cli_command += ["--", instruction]

        cli_process = subprocess.Popen(
            cli_command,
            cwd=str(work_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        stderr_lines: deque[str] = deque(maxlen=CLI_CAPTURE_MAX_LINES)

        def drain(
            stream, destination: deque[str], *, capture_root: bool = False
        ) -> None:
            if stream is None:
                return
            try:
                for line in stream:
                    if capture_root:
                        try:
                            event = json.loads(line)
                        except json.JSONDecodeError:
                            event = None
                        session_id = (
                            event.get("sessionID") if isinstance(event, dict) else None
                        )
                        if isinstance(event, dict):
                            _note_cli_error(recorder, event)
                        if isinstance(session_id, str) and session_id:
                            early_root_events.append({"sessionID": session_id})
                            if (
                                session_id not in early_root_candidates
                                and len(early_root_candidates) < MAX_ROOT_CANDIDATES
                            ):
                                early_root_candidates.add(session_id)
                                # Persist the CLI-supplied ID while execution
                                # is in progress. Final collection validates it
                                # against private server metadata.
                                recorder.update(
                                    root_candidates=sorted(early_root_candidates)
                                )
                                _persist_root_candidates(
                                    logs_dir / "opencode-v2-root-candidates.json",
                                    {
                                        "source": "cli-event",
                                        "candidate_session_ids": sorted(
                                            early_root_candidates
                                        ),
                                        "validated": False,
                                    },
                                    collection_errors,
                                )
                    destination.append(_bounded_cli_line(line))
                    # Durable at observation time. The bounded deque above
                    # still backs the existing cli-events artifact unchanged.
                    if capture_root:
                        recorder.cli_stdout_line(line.rstrip("\n"))
                    else:
                        recorder.cli_stderr_line(line.rstrip("\n"))
            except (OSError, ValueError):
                pass

        stdout_thread = threading.Thread(
            target=drain,
            args=(cli_process.stdout, stdout_lines),
            kwargs={"capture_root": True},
            name="opencode-v2-cli-stdout",
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=drain,
            args=(cli_process.stderr, stderr_lines),
            name="opencode-v2-cli-stderr",
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()
        recorder.stage("cli-started", cli_started=True, cli_pid=cli_process.pid)
        try:
            cli_returncode = cli_process.wait()
            recorder.stage("cli-exited", cli_returncode=cli_returncode)
        except BaseException as error:
            # Cancellation or runner shutdown: kill the CLI, then interrupt
            # every session the server has discovered before tearing down.
            cancelled = True
            recorder.stage(
                "cli-cancelled",
                cancelled=True,
                cancel_cause=f"{type(error).__name__}: {error}",
            )
            try:
                os.killpg(cli_process.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                cli_process.kill()
            try:
                cli_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(cli_process.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    try:
                        cli_process.kill()
                    except OSError:
                        pass
                try:
                    cli_process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    collection_errors.append(
                        "CLI process did not exit after cancellation"
                    )
            stdout_thread.join(timeout=1)
            stderr_thread.join(timeout=1)
            cli_stderr = "".join(stderr_lines)
            events = _events(stdout_lines)
            events.extend(early_root_events)
            dump_jsonl(logs_dir / "opencode-v2-cli-events.jsonl", events)
            try:
                discovered = server.collect_sessions()
            except Exception as discover_error:
                discovered = []
                collection_errors.append(
                    f"interruption discovery: {type(discover_error).__name__}: {discover_error}"
                )
            for session in discovered:
                session_id = session.get("id")
                if session_id:
                    server.interrupt_session(str(session_id))
                    interrupted.append(str(session_id))
            raise error
        stdout_thread.join(timeout=2)
        stderr_thread.join(timeout=2)
        cli_stderr = "".join(stderr_lines)
        events = _events(stdout_lines)
        events.extend(early_root_events)
        if cli_returncode not in (None, 0):
            run_error = f"OpenCode CLI exited with status {cli_returncode}"
        try:
            dump_jsonl(logs_dir / "opencode-v2-cli-events.jsonl", events)
            sessions = server.collect_sessions()
            root_id = _root_id(
                events, sessions, before_ids, candidate_ids=early_root_candidates
            )
            if root_id is None:
                collection_errors.append(
                    "could not resolve an unambiguous newly-created root session"
                )
            else:
                (logs_dir / "opencode-v2-root-candidates.json").write_text(
                    json.dumps(
                        {
                            "source": "cli-event",
                            "candidate_session_ids": sorted(early_root_candidates),
                            "root_id": root_id,
                            "validated": True,
                        },
                        indent=2,
                    )
                    + "\n"
                )
                recorder.stage("collecting", root_id=root_id)
                inspections, settled = _collect_tree(
                    server,
                    root_id,
                    deadline=time.monotonic() + max(0.1, args.settle_timeout),
                    errors=collection_errors,
                    recorder=recorder,
                )
                recorder.stage("collected", settled=settled)
                if not settled:
                    collection_errors.append(
                        "session tree did not reach two identical terminal snapshots"
                    )
        except Exception as error:
            collection_errors.append(
                f"post-run collection: {type(error).__name__}: {error}"
            )
    except BaseException as error:  # noqa: BLE001 - preserve evidence before re-raising
        pending_error = error
        run_error = run_error or f"{type(error).__name__}: {error}"
        cancelled = cancelled or isinstance(error, (KeyboardInterrupt, SystemExit))
        recorder.stage(
            "aborted",
            cancelled=cancelled,
            run_error=run_error,
        )
        recorder.note("runner-aborted", run_error)
        # On cancellation, interrupt the entire currently-discovered tree and
        # take one partial snapshot before closing the server.
        try:
            discovered = server.collect_sessions()
            discovered_ids = {
                str(item.get("id")) for item in discovered if item.get("id")
            }
            for session_id in sorted(discovered_ids):
                server.interrupt_session(session_id)
                interrupted.append(session_id)
            if root_id is None:
                root_id = _root_id(
                    events,
                    discovered,
                    before_ids,
                    candidate_ids=early_root_candidates,
                )
            if root_id:
                inspections, _ = _collect_tree(
                    server,
                    root_id,
                    deadline=time.monotonic() + 5,
                    errors=collection_errors,
                    recorder=recorder,
                )
            else:
                inspections = [server.inspect_session(item) for item in discovered]
        except Exception as cleanup_error:
            collection_errors.append(
                f"partial collection: {type(cleanup_error).__name__}: {cleanup_error}"
            )
    finally:
        recorder.stage("shutting-down")
        process = server.process
        if process is not None:
            server_exit_code = process.poll()
            recorder.update(server_exit_code=server_exit_code)
            if server_exit_code is not None:
                collection_errors.append(
                    "OpenCode server exited unexpectedly before owned shutdown "
                    f"(status {server_exit_code})"
                )
        try:
            server.stop()
        except Exception as error:  # preserve artifacts if process cleanup misbehaves
            collection_errors.append(
                f"server shutdown: {type(error).__name__}: {error}"
            )
        server_stderr = server.read_server_stderr()
        if run_error or collection_errors or server_exit_code is not None:
            try:
                private_state_artifacts = _preserve_private_state(env, logs_dir)
            except Exception as error:
                collection_errors.append(
                    f"private state preservation: {type(error).__name__}: {error}"
                )

    # Written atomically: a half-written final dump would otherwise be
    # preferred over the intact partial snapshot, which defeats the point of
    # keeping the snapshot at all.
    sessions_dump = logs_dir / "opencode-v2-sessions.jsonl"
    sessions_temporary = sessions_dump.with_suffix(".jsonl.tmp")
    sessions_temporary.write_text(
        "".join(
            json.dumps(inspection, ensure_ascii=False) + "\n"
            for inspection in inspections
        )
    )
    os.replace(sessions_temporary, sessions_dump)
    dump_jsonl(logs_dir / "opencode-v2-raw-pages.jsonl", server.raw_pages)

    try:
        version_result = subprocess.run(
            [binary, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        binary_version = (version_result.stdout or version_result.stderr).strip()
    except (OSError, subprocess.SubprocessError) as error:
        binary_version = f"unavailable: {type(error).__name__}: {error}"

    result = {
        "instruction_file": args.instruction_file,
        "binary": binary,
        "binary_version": binary_version,
        "model": args.model,
        "variant": args.variant,
        "server_url": server.url,
        "cli_returncode": cli_returncode,
        "cli_stderr": cli_stderr[-20000:],
        "server_stderr": server_stderr[-20000:],
        "run_error": run_error,
        "cancelled": cancelled,
        "root_id": root_id,
        "collection_complete": _collection_complete(
            settled=settled,
            errors=collection_errors,
            root_id=root_id,
            server_exit_code=server_exit_code,
        ),
        "collection_errors": collection_errors,
        "discovered_session_ids": [
            str(item.get("session", {}).get("id")) for item in inspections
        ],
        "interrupted_sessions": interrupted,
        "server_exit_code_before_shutdown": server_exit_code,
        "raw_pages_dropped": server.raw_pages_dropped,
        "raw_pages_deduplicated": server.raw_pages_deduplicated,
        "raw_pages_oversize": server.raw_pages_oversize,
        "resolved_model_sha256": (preflight or {}).get("resolved_model_sha256"),
        "private_state_artifacts": private_state_artifacts,
        "session_count": len(inspections),
        "message_count": sum(len(inspection["messages"]) for inspection in inspections),
        # A pointer to the write-through evidence, so a reader that only has
        # runner-result.json knows the live record exists and how far it got.
        "live_status": recorder.snapshot(),
    }
    (logs_dir / "runner-result.json").write_text(
        json.dumps(result, indent=2, default=str)
    )
    recorder.stage(
        "finished",
        root_id=root_id,
        cli_returncode=cli_returncode,
        session_count=len(inspections),
        collection_complete=result["collection_complete"],
    )
    recorder.close()

    _raise_runner_failure(pending_error, run_error)


if __name__ == "__main__":
    main()
