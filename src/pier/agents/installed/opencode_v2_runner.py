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
from pathlib import Path
from typing import Any

READY_TIMEOUT_SECONDS = 30
AUTH_USERNAME = "opencode"
SESSION_PAGE_SIZE = 200
MESSAGE_PAGE_SIZE = 200
INSPECT_ATTEMPTS = 3
DEFAULT_SETTLE_SECONDS = 3600.0
SESSION_WAIT_REQUEST_SECONDS = 30.0
SETTLE_INTERVAL_SECONDS = 1.0


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

    def __init__(self, binary: str, cwd: str, password: str, env: dict[str, str]):
        self.binary = binary
        self.cwd = cwd
        self.password = password
        self.env = env
        self.process: subprocess.Popen | None = None
        self.url: str | None = None
        self.server_stderr = ""
        self._stderr_chunks: list[str] = []
        self._stderr_thread: threading.Thread | None = None
        self.collection_deadline: float | None = None
        self.raw_pages: list[dict[str, Any]] = []

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
            except (OSError, ValueError):
                pass

        self._stderr_thread = threading.Thread(
            target=drain_stderr, name="opencode-v2-server-stderr", daemon=True
        )
        self._stderr_thread.start()
        try:
            self.url = wait_for_server(self.process)
        except Exception:
            self.stop()
            raise
        # Fail fast when the URL could never answer a benchmark request, and
        # do not leak the server if readiness itself fails.
        try:
            status, _ = http_get(
                self._api_url("api/health"), self.password, timeout=10.0
            )
            if status != 200:
                raise RuntimeError(
                    f"OpenCode server {self.url} failed its readiness check (HTTP {status})"
                )
        except Exception:
            self.stop()
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
        self.raw_pages.append(
            {
                "endpoint": "/api/session",
                "query": dict(query),
                "status": status,
                "response": payload,
            }
        )
        if status != 200 or not isinstance(payload, dict):
            raise RuntimeError(f"GET /api/session failed (HTTP {status}): {payload!r}")
        return (
            payload.get("data") or [],
            (payload.get("cursor") or {}).get("next"),
        )

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
        self.raw_pages.append(
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
        return (
            payload.get("data") or [],
            (payload.get("cursor") or {}).get("next"),
        )

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
            if not next_cursor or next_cursor in seen:
                break
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
            if not next_cursor or next_cursor in seen:
                break
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
            self._api_url(f"api/session/{session_id}/wait"),
            self.password,
            {},
            timeout=self._request_timeout(timeout),
        ) in {200, 204}

    def interrupt_all(self, session_ids) -> None:
        for session_id in sorted(session_ids):
            self.interrupt_session(session_id)

    # -- terminal snapshots --------------------------------------------------

    def terminal_snapshot(self, session_id: str) -> dict | None:
        """Deprecated: V2 has no terminal-session endpoint."""
        return None

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
        return data if isinstance(data, list) else []

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

    def inspect_session(self, session: dict) -> dict:
        return inspect_session(self, session)

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
        if root is not None:
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


def _redact(value: Any) -> Any:
    """Remove credential-shaped values from persisted preflight evidence."""
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            normalized = "".join(char for char in key.lower() if char.isalnum())
            if normalized in {"baseurl", "url"}:
                result[key] = "<redacted-url>"
            elif normalized.endswith("apikey") or any(
                marker in normalized
                for marker in (
                    "authorization",
                    "password",
                    "secret",
                    "token",
                    "credential",
                )
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


def preflight_runtime(
    server: OpenCodeV2Server,
    *,
    model_spec: str | None,
    config_file: str | None,
    timeout: float = 15.0,
) -> dict[str, Any]:
    """Resolve and verify the selected model and all configured active agents."""
    if not model_spec or "/" not in model_spec:
        raise RuntimeError("preflight requires provider/model[#variant]")
    provider_id, model_ref = model_spec.split("/", 1)
    model_id, separator, variant = model_ref.partition("#")
    variant = variant if separator else ""

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
            resolved_body = selected.get("body") or {}
            for key, value in expected_body.items():
                if resolved_body.get(key) != value:
                    raise RuntimeError(
                        f"resolved model body {key!r} is {resolved_body.get(key)!r}, "
                        f"expected {value!r}"
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
                if (
                    actual.get("providerID") != provider_id
                    or actual.get("id") != model_id
                    or (variant and actual.get("variant") != variant)
                ):
                    raise RuntimeError(
                        f"agent {agent_id!r} resolved unexpected model {actual!r}"
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


def inspect_session(server: OpenCodeV2Server, session: dict) -> dict:
    """Collect one session's messages, running state, and terminal snapshot."""
    session_id = str(session.get("id"))
    return {
        "session": session,
        "messages": server.collect_messages(session_id),
        "active": {
            "type": "running",
        }
        if session_id in server.running_session_ids()
        else None,
        "inbox": server.inbox_items(session_id),
        # V2.0.3 has no terminal-session endpoint; inbox + active + wait are
        # the native completion contract.
        "terminal": None,
    }


def _events(stdout_lines: list[str]) -> list[dict]:
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


def _root_id(events: list[dict], sessions: list[dict], before: set[str]) -> str | None:
    """Resolve the newly-created root without guessing from global history."""
    event_ids = {str(event["sessionID"]) for event in events if event.get("sessionID")}
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
        if str(session.get("id")) in event_ids and not session.get("parentID")
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
                tuple(
                    sorted(
                        str(message.get("id")) for message in item.get("messages", [])
                    )
                ),
                json.dumps(
                    item.get("messages", []), sort_keys=True, separators=(",", ":")
                ),
                (item.get("active") or {}).get("type"),
                tuple(
                    sorted(str(inbox.get("id")) for inbox in (item.get("inbox") or []))
                ),
                json.dumps(item.get("terminal"), sort_keys=True, separators=(",", ":")),
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
        messages = inspection.get("messages") or []
        terminal = any(
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


def _collect_tree(
    server: OpenCodeV2Server,
    root_id: str,
    *,
    deadline: float,
    errors: list[str],
) -> tuple[list[dict], bool]:
    """Wait and collect two identical terminal snapshots of the whole tree."""
    previous: tuple | None = None
    stable = False
    inspections: list[dict] = []
    last_blockers: list[str] = []
    server.collection_deadline = deadline
    try:
        while time.monotonic() < deadline:
            blockers: list[str] = []
            try:
                sessions = server.collect_descendants(root_id)
                if not sessions:
                    raise RuntimeError(f"root session {root_id} was not returned")
                ids = [str(item.get("id")) for item in sessions if item.get("id")]
                for session_id in ids:
                    if not server.wait_session(session_id):
                        blockers.append(f"session.wait failed for {session_id}")
                inspections = [server.inspect_session(session) for session in sessions]
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
                active = server.running_session_ids()
                if active:
                    blockers.append("active sessions: " + ", ".join(sorted(active)))
                if not blockers and previous == signature:
                    stable = True
                    break
                previous = signature
            except Exception as error:  # preserve partial records and retry discovery
                blockers = [f"collection: {type(error).__name__}: {error}"]
            last_blockers = blockers
            time.sleep(
                min(SETTLE_INTERVAL_SECONDS, max(0.0, deadline - time.monotonic()))
            )
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
        except Exception as error:
            server.interrupt_all(known_ids)
            errors.append(f"timeout interruption: {type(error).__name__}: {error}")
        finally:
            server.collection_deadline = None
    return inspections, stable


def _preserve_private_state(env: dict[str, str], logs_dir: Path) -> list[str]:
    """Copy private OpenCode state after a diagnostic failure without parsing it."""
    copied: list[str] = []
    target = logs_dir / "private-state"
    for label, env_key in (("state", "XDG_STATE_HOME"), ("data", "XDG_DATA_HOME")):
        source_text = env.get(env_key)
        if not source_text:
            continue
        source = Path(source_text)
        if not source.is_dir():
            continue
        for item in source.rglob("*"):
            if not item.is_file():
                continue
            relative = item.relative_to(source)
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
        binary=binary, cwd=str(work_dir), password=password, env=env
    )

    stdout_lines: list[str] = []
    cli_stderr = ""
    cli_returncode: int | None = None
    server_stderr = ""
    run_error: str | None = None
    collection_errors: list[str] = []
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

    def handle_termination(signum, _frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    signal.signal(signal.SIGTERM, handle_termination)

    try:
        server.start()
        assert server.url is not None
        model_spec = args.model
        if model_spec and args.variant and "#" not in model_spec:
            model_spec = f"{model_spec}#{args.variant}"
        preflight = preflight_runtime(
            server,
            model_spec=model_spec,
            config_file=args.config_file,
        )
        (logs_dir / "opencode-v2-preflight.json").write_text(
            json.dumps(preflight, indent=2) + "\n"
        )
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
        stderr_lines: list[str] = []

        def drain(
            stream, destination: list[str], *, capture_root: bool = False
        ) -> None:
            if stream is None:
                return
            try:
                for line in stream:
                    destination.append(line)
                    if not capture_root:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    session_id = (
                        event.get("sessionID") if isinstance(event, dict) else None
                    )
                    if not isinstance(session_id, str) or not session_id:
                        continue
                    early_root_candidates.add(session_id)
                    # Persist the CLI-supplied ID while execution is still in
                    # progress. Final collection validates it against private
                    # server metadata before accepting it as the root.
                    (logs_dir / "opencode-v2-root-candidates.json").write_text(
                        json.dumps(
                            {
                                "source": "cli-event",
                                "candidate_session_ids": sorted(early_root_candidates),
                                "validated": False,
                            },
                            indent=2,
                        )
                        + "\n"
                    )
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
        try:
            cli_returncode = cli_process.wait()
        except BaseException as error:
            # Cancellation or runner shutdown: kill the CLI, then interrupt
            # every session the server has discovered before tearing down.
            cancelled = True
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
        dump_jsonl(logs_dir / "opencode-v2-cli-events.jsonl", events)
        sessions = server.collect_sessions()
        root_id = _root_id(events, sessions, before_ids)
        if root_id is None:
            raise RuntimeError(
                "could not resolve an unambiguous newly-created root session"
            )
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
        inspections, settled = _collect_tree(
            server,
            root_id,
            deadline=time.monotonic() + max(0.1, args.settle_timeout),
            errors=collection_errors,
        )
        if not settled:
            collection_errors.append(
                "session tree did not reach two identical terminal snapshots"
            )
        if cli_returncode not in (None, 0):
            run_error = f"OpenCode CLI exited with status {cli_returncode}"
    except BaseException as error:  # noqa: BLE001 - preserve evidence before re-raising
        pending_error = error
        run_error = run_error or f"{type(error).__name__}: {error}"
        cancelled = cancelled or isinstance(error, (KeyboardInterrupt, SystemExit))
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
                root_id = _root_id(events, discovered, before_ids)
            if root_id:
                inspections, _ = _collect_tree(
                    server,
                    root_id,
                    deadline=time.monotonic() + 5,
                    errors=collection_errors,
                )
            else:
                inspections = [server.inspect_session(item) for item in discovered]
        except Exception as cleanup_error:
            collection_errors.append(
                f"partial collection: {type(cleanup_error).__name__}: {cleanup_error}"
            )
    finally:
        process = server.process
        if process is not None:
            server_exit_code = process.poll()
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

    sessions_dump = logs_dir / "opencode-v2-sessions.jsonl"
    sessions_dump.write_text(
        "".join(
            json.dumps(inspection, ensure_ascii=False) + "\n"
            for inspection in inspections
        )
    )
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
        "collection_complete": settled and not collection_errors and bool(root_id),
        "collection_errors": collection_errors,
        "discovered_session_ids": [
            str(item.get("session", {}).get("id")) for item in inspections
        ],
        "interrupted_sessions": interrupted,
        "server_exit_code_before_shutdown": server_exit_code,
        "resolved_model_sha256": (preflight or {}).get("resolved_model_sha256"),
        "private_state_artifacts": private_state_artifacts,
        "session_count": len(inspections),
        "message_count": sum(len(inspection["messages"]) for inspection in inspections),
    }
    (logs_dir / "runner-result.json").write_text(json.dumps(result, indent=2))

    if pending_error is not None:
        raise pending_error
    if run_error or collection_errors:
        raise SystemExit(run_error or "; ".join(collection_errors))


if __name__ == "__main__":
    main()
