import json
import shlex
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import toml

from pier.agents.installed.base import (
    BaseInstalledAgent,
    CliFlag,
    with_prompt_template,
)
from pier.agents.network import allowlist_from_urls, collect_url_values
from pier.environments.base import BaseEnvironment
from pier.models.agent.context import AgentContext
from pier.models.agent.install import AgentInstallSpec, InstallStep
from pier.models.agent.name import AgentName
from pier.models.agent.network import NetworkAllowlist
from pier.models.trajectories import (
    Agent,
    FinalMetrics,
    Metrics,
    Observation,
    ObservationResult,
    Step,
    ToolCall,
    Trajectory,
)
from pier.models.trial.paths import EnvironmentPaths
from pier.utils.env import parse_bool_env_value
from pier.utils.trajectory_metrics import (
    extra_with_context_metrics,
    populate_context_from_final_metrics,
)
from pier.utils.trajectory_utils import format_trajectory_json

_COMPACTION_DROP_TOKEN_THRESHOLD = 10_000


class Codex(BaseInstalledAgent):
    """
    The Codex agent uses OpenAI's Codex CLI tool to solve tasks.
    """

    SUPPORTS_ATIF: bool = True
    _OUTPUT_FILENAME = "codex.txt"
    _REMOTE_CODEX_HOME = PurePosixPath("/tmp/codex-home")
    _REMOTE_MODEL_CATALOG = _REMOTE_CODEX_HOME / "model-catalog.json"
    _REMOTE_CODEX_SECRETS_DIR = PurePosixPath("/tmp/codex-secrets")

    CLI_FLAGS = [
        CliFlag(
            "reasoning_effort",
            cli="-c",
            type="str",
            default="high",
            format="-c model_reasoning_effort={value}",
        ),
        CliFlag(
            "reasoning_summary",
            cli="-c",
            type="enum",
            choices=["auto", "concise", "detailed", "none"],
            format="-c model_reasoning_summary={value}",
        ),
    ]

    def __init__(
        self,
        *args,
        command_model_name: str | None = None,
        config_toml: str | None = None,
        config_toml_file: str | None = None,
        restrict_model_catalog: bool = False,
        model_catalog_file: str | None = None,
        **kwargs,
    ):
        self._command_model_name = command_model_name
        self._config_toml = config_toml
        if config_toml_file:
            self._config_toml = Path(config_toml_file).read_text()

        # Benchmark isolation is opt-in: by default Codex keeps its stock model
        # catalog and its delegation behaviour is left completely untouched.
        self._restrict_model_catalog = restrict_model_catalog
        self._model_catalog_file = model_catalog_file
        if model_catalog_file and not restrict_model_catalog:
            raise ValueError(
                "model_catalog_file requires restrict_model_catalog=true; "
                "without restriction the supplied catalog would be ignored."
            )
        super().__init__(*args, **kwargs)

    @staticmethod
    def name() -> str:
        return AgentName.CODEX.value

    @property
    def _trajectory_path(self) -> PurePosixPath:
        return PurePosixPath(EnvironmentPaths.agent_dir / "trajectory.json")

    def get_version_command(self) -> str | None:
        return "if [ -s ~/.nvm/nvm.sh ]; then . ~/.nvm/nvm.sh; fi; codex --version"

    def parse_version(self, stdout: str) -> str:
        text = stdout.strip()
        for line in text.splitlines():
            line = line.strip()
            if line:
                return line.removeprefix("codex-cli").strip()
        return text

    def network_allowlist(self) -> NetworkAllowlist:
        urls: list[str] = []
        for key in ("OPENAI_BASE_URL", "OPENAI_API_BASE"):
            if value := self._get_env(key):
                urls.append(value)

        if self._config_toml:
            try:
                parsed = toml.loads(self._config_toml)
            except toml.TomlDecodeError:
                parsed = {}
            urls.extend(collect_url_values(parsed))

        return allowlist_from_urls(urls, default_domains=["api.openai.com"])

    def install_spec(self) -> AgentInstallSpec:
        version_spec = f"@{self._version}" if self._version else "@latest"
        root_run = (
            "if ldd --version 2>&1 | grep -qi musl || [ -f /etc/alpine-release ]; then"
            "  apk add --no-cache curl bash nodejs npm ripgrep;"
            " elif command -v apt-get &>/dev/null; then"
            "  apt-get update && apt-get install -y curl ripgrep;"
            " elif command -v yum &>/dev/null; then"
            "  yum install -y curl ripgrep;"
            " else"
            '  echo "Warning: No known package manager found, assuming curl is available" >&2;'
            " fi"
        )
        agent_run = (
            "set -euo pipefail; "
            "if ldd --version 2>&1 | grep -qi musl || [ -f /etc/alpine-release ]; then"
            f"  npm install -g @openai/codex{version_spec};"
            " else"
            "  curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh | bash &&"
            '  export NVM_DIR="$HOME/.nvm" &&'
            '  \\. "$NVM_DIR/nvm.sh" || true &&'
            "  command -v nvm &>/dev/null || { echo 'Error: NVM failed to load' >&2; exit 1; } &&"
            "  nvm install 22 && nvm alias default 22 && npm -v &&"
            f"  npm install -g @openai/codex{version_spec};"
            " fi && "
            "codex --version"
        )
        symlink_run = (
            "for bin in node codex; do"
            '  BIN_PATH="$(which "$bin" 2>/dev/null || true)";'
            '  if [ -n "$BIN_PATH" ] && [ "$BIN_PATH" != "/usr/local/bin/$bin" ]; then'
            '    ln -sf "$BIN_PATH" "/usr/local/bin/$bin";'
            "  fi;"
            " done"
        )
        return AgentInstallSpec(
            agent_name=self.name(),
            version=self._version,
            steps=[
                InstallStep(
                    user="root",
                    env={"DEBIAN_FRONTEND": "noninteractive"},
                    run=root_run,
                ),
                InstallStep(user="agent", run=agent_run),
                InstallStep(user="root", run=symlink_run),
            ],
        )

    @staticmethod
    def _extract_message_text(content: list[Any]) -> str:
        """Extract joined text from Codex content blocks."""
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)

    @staticmethod
    def _parse_output_blob(raw: Any) -> tuple[str | None, dict[str, Any] | None]:
        """Extract textual output and metadata from Codex tool outputs."""
        if raw is None:
            return None, None

        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                return raw, None
        else:
            parsed = raw

        if isinstance(parsed, dict):
            output = parsed.get("output")
            if output is None and parsed:
                # dumping remaining structure if output missing
                output = json.dumps(parsed, ensure_ascii=False)
            metadata = parsed.get("metadata")
            return output, metadata if isinstance(metadata, dict) else None

        return str(parsed), None

    @staticmethod
    def _group_events_by_api_call_id(
        normalized_events: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Merge assistant events from the same Codex model request into one step."""
        result: list[dict[str, Any]] = []
        groups: dict[str, dict[str, Any]] = {}
        group_order: list[str] = []

        def flush() -> None:
            for group_id in group_order:
                group = groups.pop(group_id, None)
                if group is None:
                    continue
                group["tool_calls"].sort(key=lambda tc: tc.get("tool_order", 0))
                message_parts = [
                    part
                    for part in group.pop("message_parts")
                    if isinstance(part, str) and part
                ]
                group["text"] = "\n\n".join(message_parts)
                result.append(group)
            group_order.clear()

        for event in normalized_events:
            api_call_id = event.get("api_call_id")
            kind = event.get("kind")
            role = event.get("role")

            if kind == "message" and role != "assistant":
                flush()
                result.append(event)
                continue

            if not isinstance(api_call_id, str):
                flush()
                result.append(event)
                continue

            if api_call_id not in groups:
                groups[api_call_id] = {
                    "kind": "bundled",
                    "api_call_id": api_call_id,
                    "codex_turn_id": event.get("codex_turn_id"),
                    "timestamp": event.get("timestamp"),
                    "message_parts": [],
                    "reasoning": None,
                    "tool_calls": [],
                    "metrics": event.get("metrics"),
                }
                group_order.append(api_call_id)

            group = groups[api_call_id]
            if kind == "message":
                text = event.get("text")
                if isinstance(text, str) and text:
                    group["message_parts"].append(text)
                if event.get("reasoning"):
                    group["reasoning"] = event["reasoning"]
                if event.get("timestamp"):
                    group["timestamp"] = event["timestamp"]
            elif kind == "tool_call":
                group["tool_calls"].append(event)
                if not group["reasoning"] and event.get("reasoning"):
                    group["reasoning"] = event["reasoning"]
                if not group.get("metrics") and event.get("metrics"):
                    group["metrics"] = event["metrics"]

        flush()
        return result

    @staticmethod
    def _metrics_from_token_count_payload(
        payload: dict[str, Any],
    ) -> dict[str, Any] | None:
        info = payload.get("info")
        if not isinstance(info, dict):
            return None

        last_usage = info.get("last_token_usage")
        if not isinstance(last_usage, dict):
            return None

        prompt_tokens = last_usage.get("input_tokens")
        completion_tokens = last_usage.get("output_tokens")
        cached_tokens = last_usage.get("cached_input_tokens")
        reasoning_tokens = last_usage.get("reasoning_output_tokens")
        total_tokens = last_usage.get("total_tokens")

        return {
            "prompt_tokens": prompt_tokens if prompt_tokens else None,
            "completion_tokens": completion_tokens or None,
            "cached_tokens": cached_tokens or None,
            "extra": {
                "reasoning_output_tokens": reasoning_tokens,
                "total_tokens": total_tokens,
            },
        }

    @staticmethod
    def _final_cumulative_usage(
        raw_events: list[dict[str, Any]],
        initial_total_usage: dict[str, Any] | None = None,
    ) -> tuple[dict[str, int] | None, bool]:
        """Read a thread's final cumulative Codex usage snapshot.

        This deliberately follows Harbor's Codex adapter: final token totals come
        from the last ``token_count.info.total_token_usage`` snapshot, rather than
        summing ``last_token_usage`` records. Codex re-emits token snapshots for
        non-inference events such as rate-limit updates, so summing every
        ``last_token_usage`` can double count a model call.

        Pier's only extension is for full-history subagents. Codex seeds those
        child counters with the copied parent history, so ``initial_total_usage``
        is subtracted from the child's final cumulative snapshot. The baseline is
        captured from the copied block before that block is removed from the child
        trajectory. This is the minimum delta needed to make Harbor-style
        cumulative accounting composable across a trajectory tree.

        Returns ``(usage, complete)``. A negative child delta means the cumulative
        counter was reset or the inherited baseline is otherwise incompatible;
        in that case the caller must withhold token totals instead of guessing.
        """
        final_total_usage: dict[str, Any] | None = None
        for event in reversed(raw_events):
            if event.get("type") != "event_msg":
                continue
            payload = event.get("payload")
            if not isinstance(payload, dict) or payload.get("type") != "token_count":
                continue
            info = payload.get("info")
            if not isinstance(info, dict):
                continue
            cumulative = info.get("total_token_usage")
            if not isinstance(cumulative, dict):
                continue
            final_total_usage = cumulative
            break

        if final_total_usage is None:
            return None, True

        field_map = {
            "prompt_tokens": "input_tokens",
            "completion_tokens": "output_tokens",
            "cached_tokens": "cached_input_tokens",
            "reasoning_tokens": "reasoning_output_tokens",
            "total_tokens": "total_tokens",
        }
        usage: dict[str, int] = {}
        for target, source in field_map.items():
            value = final_total_usage.get(source)
            if value is None:
                value = 0
            if not isinstance(value, int):
                return None, False

            baseline = 0
            if initial_total_usage is not None:
                inherited = initial_total_usage.get(source)
                if inherited is not None:
                    if not isinstance(inherited, int):
                        return None, False
                    baseline = inherited

            delta = value - baseline
            if delta < 0:
                return None, False
            usage[target] = delta

        return usage, True

    @staticmethod
    def _extract_context_metrics(
        raw_events: list[dict[str, Any]],
    ) -> tuple[int | None, int | None]:
        """Return best-effort (peak_context_tokens, summarization_count)."""
        peak_context_tokens: int | None = None
        window_peak: int | None = None
        token_drop_summarization_count = 0
        compacted_item_count = sum(
            1 for event in raw_events if event.get("type") == "compacted"
        )
        context_compacted_event_count = 0
        saw_usage = False

        for event in raw_events:
            if event.get("type") != "event_msg":
                continue
            payload = event.get("payload", {})
            if isinstance(payload, dict) and payload.get("type") == "context_compacted":
                context_compacted_event_count += 1
            if not isinstance(payload, dict) or payload.get("type") != "token_count":
                continue
            info = payload.get("info")
            if not isinstance(info, dict):
                continue
            last_usage = info.get("last_token_usage")
            if not isinstance(last_usage, dict):
                continue

            context_tokens = last_usage.get("total_tokens")
            if not isinstance(context_tokens, int):
                context_tokens = last_usage.get("input_tokens")
            if not isinstance(context_tokens, int):
                continue

            saw_usage = True
            peak_context_tokens = (
                context_tokens
                if peak_context_tokens is None
                else max(peak_context_tokens, context_tokens)
            )
            if (
                window_peak is not None
                and context_tokens < window_peak - _COMPACTION_DROP_TOKEN_THRESHOLD
            ):
                token_drop_summarization_count += 1
                window_peak = context_tokens
            else:
                window_peak = (
                    context_tokens
                    if window_peak is None
                    else max(window_peak, context_tokens)
                )

        if not saw_usage:
            summarization_count = compacted_item_count or context_compacted_event_count
            return None, summarization_count if summarization_count else None

        summarization_count = (
            compacted_item_count
            or context_compacted_event_count
            or token_drop_summarization_count
        )
        return peak_context_tokens, summarization_count

    def _convert_event_to_step(self, event: dict[str, Any], step_id: int) -> Step:
        """Convert a normalized Codex event dictionary into an ATIF step."""
        kind = event.get("kind")
        timestamp = event.get("timestamp")

        if kind == "message":
            role = event.get("role", "user")
            text = event.get("text", "")
            reasoning = event.get("reasoning")
            source: Literal["system", "user", "agent"]
            if role == "assistant":
                source = "agent"
            elif role == "user":
                source = "user"
            else:
                source = "system"

            extra = event.get("extra")

            return Step(
                step_id=step_id,
                timestamp=timestamp,
                source=source,
                message=text,
                reasoning_content=reasoning
                if source == "agent" and reasoning
                else None,
                model_name=self.model_name
                if source == "agent" and self.model_name
                else None,
                llm_call_count=1 if source == "agent" else None,
                extra=extra if extra else None,
            )

        if kind == "tool_call":
            call_id = event.get("call_id", "")
            tool_name = event.get("tool_name", "")
            reasoning = event.get("reasoning")
            arguments = event.get("arguments") or {}
            if not isinstance(arguments, dict):
                arguments = {"value": arguments}

            tool_call = ToolCall(
                tool_call_id=call_id,
                function_name=tool_name,
                arguments=arguments,
            )

            observation: Observation | None = None
            output_text = event.get("output")
            if output_text is not None:
                observation = Observation(
                    results=[
                        ObservationResult(
                            source_call_id=call_id or None,
                            content=output_text,
                        )
                    ]
                )

            metrics_payload = event.get("metrics")
            metrics: Metrics | None = None
            if isinstance(metrics_payload, dict):
                metrics = Metrics(**metrics_payload)

            extra: dict[str, Any] | None = None
            metadata = event.get("metadata")
            if metadata:
                extra = {"tool_metadata": metadata}
            raw_arguments = event.get("raw_arguments")
            if raw_arguments:
                extra = extra or {}
                extra["raw_arguments"] = raw_arguments
            status = event.get("status")
            if status:
                extra = extra or {}
                extra["status"] = status
            api_call_id = event.get("api_call_id")
            if api_call_id:
                extra = extra or {}
                extra["api_call_id"] = api_call_id
            codex_turn_id = event.get("codex_turn_id")
            if codex_turn_id:
                extra = extra or {}
                extra["codex_turn_id"] = codex_turn_id

            message_text = event.get("message") or ""

            return Step(
                step_id=step_id,
                timestamp=timestamp,
                source="agent",
                message=message_text,
                tool_calls=[tool_call],
                observation=observation,
                model_name=self.model_name if self.model_name else None,
                reasoning_content=reasoning if reasoning else None,
                metrics=metrics,
                llm_call_count=1,
                extra=extra,
            )

        if kind == "bundled":
            text = event.get("text", "")
            reasoning = event.get("reasoning")

            tool_calls: list[ToolCall] = []
            observation_results: list[ObservationResult] = []
            for tc in event.get("tool_calls", []):
                call_id = tc.get("call_id", "")
                arguments = tc.get("arguments") or {}
                if not isinstance(arguments, dict):
                    arguments = {"value": arguments}

                tool_calls.append(
                    ToolCall(
                        tool_call_id=call_id,
                        function_name=tc.get("tool_name", ""),
                        arguments=arguments,
                    )
                )
                observation_results.append(
                    ObservationResult(
                        source_call_id=call_id or None,
                        content=tc.get("output"),
                    )
                )

            extra: dict[str, Any] | None = None
            api_call_id = event.get("api_call_id")
            if api_call_id:
                extra = {"api_call_id": api_call_id}
            codex_turn_id = event.get("codex_turn_id")
            if codex_turn_id:
                extra = extra or {}
                extra["codex_turn_id"] = codex_turn_id

            tool_details: dict[str, Any] = {}
            for tc in event.get("tool_calls", []):
                call_id = tc.get("call_id", "")
                details: dict[str, Any] = {}
                for source_key, target_key in (
                    ("metadata", "metadata"),
                    ("raw_arguments", "raw_arguments"),
                    ("status", "status"),
                ):
                    value = tc.get(source_key)
                    if value:
                        details[target_key] = value
                if details:
                    tool_details[call_id] = details
            if tool_details:
                extra = extra or {}
                extra["tool_call_details"] = tool_details

            observation = (
                Observation(results=observation_results)
                if observation_results
                else None
            )

            return Step(
                step_id=step_id,
                timestamp=event.get("timestamp"),
                source="agent",
                message=text,
                model_name=self.model_name if self.model_name else None,
                reasoning_content=reasoning if reasoning else None,
                tool_calls=tool_calls or None,
                observation=observation,
                metrics=Metrics(**event["metrics"]) if event.get("metrics") else None,
                llm_call_count=1,
                extra=extra,
            )

        raise ValueError(f"Unsupported event kind '{kind}'")

    def _compute_cost_from_pricing(
        self,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        cached_tokens: int | None,
        model_name: str | None = None,
    ) -> float | None:
        """Compute total cost in USD from token counts via LiteLLM pricing.

        ``model_name`` defaults to the run's model but should be the model the
        rollout being priced actually used: with an unrestricted catalog Codex may
        delegate to a different (differently priced) model.
        """
        model_name = model_name or self.model_name
        if not model_name:
            return None

        try:
            import litellm
        except ImportError:
            self.logger.warning("litellm not available; leaving codex cost_usd as None")
            return None

        pricing: dict[str, Any] | None = None
        for key in (model_name, model_name.split("/", 1)[-1]):
            entry = litellm.model_cost.get(key)
            if entry:
                pricing = entry
                break

        if pricing is None:
            self.logger.warning(
                "No LiteLLM pricing entry for model '%s'; leaving codex "
                "cost_usd as None",
                model_name,
            )
            return None

        input_rate = pricing.get("input_cost_per_token") or 0.0
        output_rate = pricing.get("output_cost_per_token") or 0.0
        cache_read_rate = pricing.get("cache_read_input_token_cost", input_rate)
        if cache_read_rate is None:
            cache_read_rate = input_rate

        uncached_input = max(0, (prompt_tokens or 0) - (cached_tokens or 0))
        cached = cached_tokens or 0
        output = completion_tokens or 0

        return (
            uncached_input * input_rate
            + cached * cache_read_rate
            + output * output_rate
        )

    def _convert_events_to_trajectory(self, session_dir: Path) -> Trajectory | None:
        """Convert every Codex rollout in ``session_dir`` into one ATIF trajectory.

        Codex writes each conversation thread to its own rollout JSONL file, so a
        run that delegates produces one file for the root thread plus one per
        spawned subagent. The root thread becomes the returned trajectory and its
        descendants are embedded in ``subagent_trajectories``, keeping their real
        parent/child nesting.

        Based on harbor-framework/harbor#2366, with Pier extensions: rollouts are
        discovered recursively (a run crossing midnight, or a subagent started
        after it, lands in a different ``<YYYY>/<MM>/<DD>`` directory), the root
        is identified from the run's stdout rather than by file order, and only
        threads actually descended from that root are embedded.
        """
        session_files = sorted(session_dir.rglob("*.jsonl"))

        if not session_files:
            self.logger.debug(f"No Codex session files found in {session_dir}")
            return None

        # Parse every rollout first: a forked child can only be stripped of the
        # history it inherited by comparing it against its direct parent.
        rollouts: list[dict[str, Any]] = []
        for session_file in session_files:
            events = self._read_rollout_events(session_file)
            if not events:
                continue
            thread_id, parent_id, _ = self._rollout_thread_ids(events)
            rollouts.append(
                {
                    "file": session_file,
                    "events": events,
                    "thread_id": thread_id,
                    "parent_id": parent_id,
                }
            )

        events_by_thread = {
            rollout["thread_id"]: rollout["events"]
            for rollout in rollouts
            if rollout["thread_id"]
        }

        # Known before conversion so the root thread (and only the root) can fall
        # back to the stdout usage when its rollout carries no token events.
        stdout_root_id = self._root_thread_id_from_stdout()

        trajectories: list[Trajectory] = []
        for rollout in rollouts:
            parent_events = events_by_thread.get(rollout["parent_id"])
            local_events, complete, inherited_total_usage = self._local_rollout_events(
                rollout["events"], parent_events
            )
            if not complete:
                self.logger.warning(
                    "Codex rollout %s looks forked from %s but that parent rollout "
                    "is unavailable; its metrics are incomplete",
                    rollout["thread_id"],
                    rollout["parent_id"],
                )
            trajectory = self._convert_session_file_to_trajectory(
                rollout["file"],
                local_events,
                metrics_complete=complete,
                is_root_thread=rollout["thread_id"] == stdout_root_id,
                initial_total_usage=inherited_total_usage,
            )
            if trajectory is not None:
                trajectories.append(trajectory)

        if not trajectories:
            return None

        root = self._select_root_trajectory(trajectories)
        if root is None:
            return None

        return self._embed_descendants(root, trajectories)

    def _read_rollout_events(self, session_file: Path) -> list[dict[str, Any]]:
        """Parse one rollout JSONL file, skipping malformed lines."""
        raw_events: list[dict[str, Any]] = []
        try:
            with open(session_file, "r") as handle:
                for line in handle:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        raw_events.append(json.loads(stripped))
                    except json.JSONDecodeError as exc:
                        self.logger.debug(
                            f"Skipping malformed JSONL line in {session_file}: {exc}"
                        )
        except OSError as exc:
            self.logger.debug(f"Failed to read rollout {session_file}: {exc}")
        return raw_events

    @staticmethod
    def _rollout_thread_ids(
        raw_events: list[dict[str, Any]],
    ) -> tuple[str | None, str | None, bool]:
        """Return ``(thread_id, parent_thread_id, has_copied_session_meta)``.

        A full-history fork copies the parent's rollout items into the child, so a
        forked rollout carries a second ``session_meta`` after its own canonical
        one. That duplicate marks where the inherited block starts.
        """
        metas = [e for e in raw_events if e.get("type") == "session_meta"]
        if not metas:
            return None, None, False

        payload = metas[0].get("payload") or {}
        thread_id = payload.get("id")
        parent_id = payload.get("parent_thread_id")
        if parent_id is None:
            source = payload.get("source")
            if isinstance(source, dict):
                subagent_source = source.get("subagent")
                if isinstance(subagent_source, dict):
                    spawn = subagent_source.get("thread_spawn")
                    if isinstance(spawn, dict):
                        parent_id = spawn.get("parent_thread_id")

        return (
            thread_id if isinstance(thread_id, str) else None,
            parent_id if isinstance(parent_id, str) else None,
            len(metas) > 1,
        )

    @staticmethod
    def _same_rollout_event(left: dict[str, Any], right: dict[str, Any]) -> bool:
        return left.get("type") == right.get("type") and left.get(
            "payload"
        ) == right.get("payload")

    def _local_rollout_events(
        self,
        raw_events: list[dict[str, Any]],
        parent_events: list[dict[str, Any]] | None,
    ) -> tuple[list[dict[str, Any]], bool, dict[str, Any] | None]:
        """Drop history a full-history fork copied in from the parent thread.

        Codex's ``retain_forked_item`` keeps ``EventMsg`` and ``SessionMeta`` items
        when seeding a forked child, so a child rollout physically contains the
        parent's ``token_count`` events. Counting them would attribute the parent's
        usage to the child a second time.

        Inheritance is an *ordered copied segment*, never an arbitrary set of
        matching events: the copied items appear in parent order, immediately after
        the child's canonical ``session_meta``. Matching stops at the first child
        event that does not continue that sequence, so an identical payload the
        child emits later on its own is kept.

        Returns ``(local_events, complete, inherited_total_usage)``. The final
        cumulative token snapshot in the copied segment is retained as an
        accounting baseline even though the copied event itself is removed.
        ``complete`` is False when the rollout looks forked but the parent's
        rollout was unavailable, in which case the caller must not publish totals
        that would include inherited usage.
        """
        if not raw_events:
            return raw_events, True, None

        _, _, has_copied_meta = self._rollout_thread_ids(raw_events)

        if not parent_events:
            # Nothing to compare against. Only a copied session_meta proves that
            # inherited history is present; without the parent we cannot strip it.
            return raw_events, not has_copied_meta, None

        start = 1
        if has_copied_meta:
            for position, event in enumerate(raw_events[1:], start=1):
                if event.get("type") == "session_meta":
                    start = position
                    break

        index = start
        # The copied session_meta belongs to the inherited block, so consume it
        # before order-matching. It is only a marker: a malformed or partially
        # written one still delimits the block, and the ordered comparison below
        # is what actually decides where the block ends.
        if has_copied_meta and raw_events[index].get("type") == "session_meta":
            index += 1

        parent_cursor = 0
        while index < len(raw_events):
            match_at = None
            for candidate in range(parent_cursor, len(parent_events)):
                if self._same_rollout_event(
                    raw_events[index], parent_events[candidate]
                ):
                    match_at = candidate
                    break
            if match_at is None:
                break
            index += 1
            parent_cursor = match_at + 1

        if index == start:
            return raw_events, True, None

        inherited_total_usage: dict[str, Any] | None = None
        for event in raw_events[start:index]:
            if event.get("type") != "event_msg":
                continue
            payload = event.get("payload")
            if not isinstance(payload, dict) or payload.get("type") != "token_count":
                continue
            info = payload.get("info")
            if not isinstance(info, dict):
                continue
            cumulative = info.get("total_token_usage")
            if isinstance(cumulative, dict):
                inherited_total_usage = cumulative

        stripped = raw_events[:start] + raw_events[index:]
        self.logger.debug(
            "Stripped %d inherited event(s) copied from the parent rollout",
            index - start,
        )
        return stripped, True, inherited_total_usage

    def _root_thread_id_from_stdout(self) -> str | None:
        """Return the root thread id that ``codex exec --json`` reported.

        ``codex exec`` emits ``{"type":"thread.started","thread_id":...}`` for the
        thread it starts, which is by definition the run's root. This is the only
        authoritative root identifier: rollout files sort by start time (so a
        subagent can sort first) and an aborted attempt or a retry can leave extra
        ``thread_source: user`` rollouts behind.
        """
        stdout_path = self.logs_dir / self._OUTPUT_FILENAME
        if not stdout_path.exists():
            return None

        try:
            with open(stdout_path, "r") as handle:
                for line in handle:
                    stripped = line.strip()
                    if not stripped.startswith("{"):
                        continue
                    try:
                        event = json.loads(stripped)
                    except json.JSONDecodeError:
                        continue
                    if event.get("type") != "thread.started":
                        continue
                    thread_id = event.get("thread_id")
                    if isinstance(thread_id, str) and thread_id:
                        return thread_id
        except OSError as exc:
            self.logger.debug(f"Failed to read {stdout_path}: {exc}")

        return None

    def _select_root_trajectory(
        self, trajectories: list[Trajectory]
    ) -> Trajectory | None:
        """Identify the thread that owns the run.

        Order of resolution: the id reported on stdout, then the parent graph (the
        user thread that spawned subagents). If neither resolves a single thread
        the run is ambiguous, and picking one anyway would silently attribute a
        whole trajectory and its token usage to the wrong agent, so this fails
        loudly instead.
        """
        by_id = {
            trajectory.trajectory_id: trajectory
            for trajectory in trajectories
            if trajectory.trajectory_id
        }

        root_id = self._root_thread_id_from_stdout()
        if root_id:
            root = by_id.get(root_id)
            if root is not None:
                return root
            self.logger.warning(
                "Codex stdout reported root thread %s but no matching rollout was "
                "found; falling back to the parent graph",
                root_id,
            )

        def extra_of(trajectory: Trajectory) -> dict[str, Any]:
            return trajectory.extra or {}

        candidates = [
            trajectory
            for trajectory in trajectories
            if not extra_of(trajectory).get("is_subagent")
        ] or list(trajectories)

        if len(candidates) == 1:
            return candidates[0]

        parent_ids = {
            extra_of(trajectory).get("parent_thread_id")
            for trajectory in trajectories
            if extra_of(trajectory).get("is_subagent")
        }
        parent_ids.discard(None)

        referenced = [
            trajectory
            for trajectory in candidates
            if trajectory.trajectory_id in parent_ids
        ]

        if len(referenced) == 1:
            return referenced[0]

        self.logger.error(
            "Ambiguous Codex root thread: %d candidate root rollouts (%s) and no "
            "usable thread.started id in %s. Refusing to guess.",
            len(candidates),
            ", ".join(sorted(str(t.trajectory_id) for t in candidates)),
            self._OUTPUT_FILENAME,
        )
        return None

    def _embed_descendants(
        self, root: Trajectory, trajectories: list[Trajectory]
    ) -> Trajectory:
        """Nest the root's descendants under it and drop unrelated rollouts.

        Only threads reachable from the root through ``parent_thread_id`` belong to
        this run. Harbor#2366 embeds every remaining rollout, which silently adopts
        the orphan left behind by an aborted attempt and corrupts both the tree and
        its totals.
        """
        children_by_parent: dict[str, list[Trajectory]] = {}
        for trajectory in trajectories:
            if trajectory is root:
                continue
            parent_id = (trajectory.extra or {}).get("parent_thread_id")
            if isinstance(parent_id, str):
                children_by_parent.setdefault(parent_id, []).append(trajectory)

        attached: set[int] = {id(root)}

        def attach(trajectory: Trajectory) -> Trajectory:
            children = children_by_parent.get(trajectory.trajectory_id or "", [])
            children = [child for child in children if id(child) not in attached]
            if not children:
                return trajectory
            for child in children:
                attached.add(id(child))
            data = trajectory.to_json_dict()
            data["subagent_trajectories"] = [
                attach(child).to_json_dict() for child in children
            ]
            self._link_subagent_refs(data)
            return Trajectory.model_validate(data)

        embedded_root = attach(root)

        orphans = [
            trajectory for trajectory in trajectories if id(trajectory) not in attached
        ]
        if orphans:
            self.logger.warning(
                "Ignoring %d Codex rollout(s) not descended from root thread %s: %s",
                len(orphans),
                root.trajectory_id,
                ", ".join(
                    sorted(
                        f"{t.trajectory_id} ({(t.extra or {}).get('session_file')})"
                        for t in orphans
                    )
                ),
            )

        return self._with_tree_metrics(embedded_root)

    def _convert_session_file_to_trajectory(
        self,
        session_file: Path,
        raw_events: list[dict[str, Any]] | None = None,
        metrics_complete: bool = True,
        is_root_thread: bool = False,
        initial_total_usage: dict[str, Any] | None = None,
    ) -> Trajectory | None:
        """Convert a single Codex rollout into an ATIF trajectory.

        ``raw_events`` may be supplied already parsed and already stripped of any
        history inherited from a parent thread; see ``_local_rollout_events``.
        """
        session_dir = session_file.parent

        if raw_events is None:
            raw_events = self._read_rollout_events(session_file)

        if not raw_events:
            return None

        session_meta = next(
            (e for e in raw_events if e.get("type") == "session_meta"), None
        )
        session_payload = (
            session_meta.get("payload", {})
            if session_meta and isinstance(session_meta, dict)
            else {}
        )
        raw_thread_id = session_payload.get("id")
        trajectory_id = (
            raw_thread_id if isinstance(raw_thread_id, str) else session_file.stem
        )
        raw_session_id = session_payload.get("session_id")
        session_id = (
            raw_session_id
            if isinstance(raw_session_id, str)
            else (trajectory_id if raw_thread_id else session_dir.name)
        )

        agent_version = "unknown"
        agent_extra: dict[str, Any] | None = None
        default_model_name: str | None = None
        trajectory_extra: dict[str, Any] = {"session_file": session_file.name}

        if session_meta:
            payload = session_payload
            agent_version = payload.get("cli_version") or agent_version
            extra: dict[str, Any] = {}
            for key in ("originator", "cwd", "git", "instructions"):
                value = payload.get(key)
                if value is not None:
                    extra[key] = value
            agent_extra = extra or None

            # Codex records a spawned thread's lineage either as flat keys or
            # nested under `source.subagent.thread_spawn`, depending on version.
            source = payload.get("source")
            spawn: dict[str, Any] = {}
            if isinstance(source, dict):
                subagent_source = source.get("subagent")
                if isinstance(subagent_source, dict):
                    thread_spawn = subagent_source.get("thread_spawn")
                    if isinstance(thread_spawn, dict):
                        spawn = thread_spawn

            for key in ("parent_thread_id", "agent_nickname", "agent_role", "depth"):
                value = payload.get(key)
                if value is None:
                    value = spawn.get(key)
                if value is not None:
                    trajectory_extra[key] = value

            for key in ("thread_source", "forked_from_id"):
                value = payload.get(key)
                if value is not None:
                    trajectory_extra[key] = value

            trajectory_extra["is_subagent"] = bool(
                payload.get("thread_source") == "subagent"
                or (isinstance(source, dict) and "subagent" in source)
            )

        if agent_version == "unknown" and self._version:
            agent_version = self._version

        for event in raw_events:
            if event.get("type") == "turn_context":
                model_name = event.get("payload", {}).get("model")
                if isinstance(model_name, str):
                    default_model_name = model_name
                    break

        if default_model_name is None:
            default_model_name = self.model_name

        # normalize events to a structure suitable for conversion into Steps
        normalized_events: list[dict[str, Any]] = []
        pending_calls: dict[str, dict[str, Any]] = {}
        pending_reasoning: str | None = None
        codex_turn_id: str | None = None
        api_call_index = 1
        current_api_call_id = f"api_call_{api_call_index}"
        api_call_metrics: dict[str, dict[str, Any]] = {}
        saw_model_output_in_api_call = False
        tool_order_counter = 0

        def record_model_output() -> None:
            nonlocal saw_model_output_in_api_call
            saw_model_output_in_api_call = True

        def finish_api_call(token_count_payload: dict[str, Any]) -> None:
            nonlocal api_call_index, current_api_call_id, saw_model_output_in_api_call
            nonlocal tool_order_counter

            if not saw_model_output_in_api_call:
                return

            metrics = self._metrics_from_token_count_payload(token_count_payload)
            if metrics:
                api_call_metrics[current_api_call_id] = metrics

            api_call_index += 1
            current_api_call_id = f"api_call_{api_call_index}"
            saw_model_output_in_api_call = False
            tool_order_counter = 0

        for event in raw_events:
            etype = event.get("type")
            payload = event.get("payload", {})
            timestamp = event.get("timestamp")

            if etype == "event_msg" and isinstance(payload, dict):
                event_type = payload.get("type")
                if event_type in {"task_started", "turn_started"}:
                    turn_id = payload.get("turn_id")
                    codex_turn_id = turn_id if isinstance(turn_id, str) else None
                elif event_type in {"task_complete", "turn_complete", "turn_aborted"}:
                    codex_turn_id = None
                elif event_type == "token_count":
                    finish_api_call(payload)
                continue

            if etype == "turn_context":
                turn_id = payload.get("turn_id") if isinstance(payload, dict) else None
                if isinstance(turn_id, str) and codex_turn_id is None:
                    codex_turn_id = turn_id
                continue

            if etype != "response_item":
                continue

            payload_type = payload.get("type")
            if payload_type == "reasoning":
                summary = payload.get("summary")
                if isinstance(summary, list) and summary:
                    reasoning_parts: list[str] = []
                    for item in summary:
                        if isinstance(item, str):
                            reasoning_parts.append(item)
                        elif isinstance(item, dict):
                            text = item.get("text")
                            if isinstance(text, str):
                                reasoning_parts.append(text)
                    pending_reasoning = (
                        "\n".join(reasoning_parts) if reasoning_parts else None
                    )
                else:
                    pending_reasoning = None
                continue

            if payload_type == "message":
                content = payload.get("content", [])
                text = (
                    self._extract_message_text(content)
                    if isinstance(content, list)
                    else ""
                )
                normalized_events.append(
                    {
                        "kind": "message",
                        "api_call_id": current_api_call_id,
                        "codex_turn_id": codex_turn_id,
                        "timestamp": timestamp,
                        "role": payload.get("role", "user"),
                        "text": text,
                        "reasoning": pending_reasoning
                        if payload.get("role") == "assistant"
                        else None,
                    }
                )
                if payload.get("role") == "assistant":
                    record_model_output()
                pending_reasoning = None
                continue

            if payload_type == "web_search_call":
                action = payload.get("action") or {}
                action_type = action.get("type", "")
                arguments: dict[str, Any] = {"action_type": action_type}
                if "query" in action:
                    arguments["query"] = action["query"]
                if "queries" in action:
                    arguments["queries"] = action["queries"]
                if "url" in action:
                    arguments["url"] = action["url"]

                normalized_events.append(
                    {
                        "kind": "tool_call",
                        "api_call_id": current_api_call_id,
                        "codex_turn_id": codex_turn_id,
                        "tool_order": tool_order_counter,
                        "timestamp": timestamp,
                        "call_id": "",
                        "tool_name": "web_search_call",
                        "arguments": arguments,
                        "raw_arguments": None,
                        "reasoning": pending_reasoning,
                        "status": payload.get("status"),
                        "message": None,
                    }
                )
                tool_order_counter += 1
                record_model_output()
                pending_reasoning = None
                continue

            if payload_type in {"function_call", "custom_tool_call"}:
                call_id = payload.get("call_id")
                if not call_id:
                    continue

                raw_args_key = (
                    "arguments" if payload_type == "function_call" else "input"
                )
                raw_arguments = payload.get(raw_args_key)
                try:
                    parsed_args = json.loads(raw_arguments)
                except (json.JSONDecodeError, TypeError):
                    if isinstance(raw_arguments, str):
                        parsed_args = {"input": raw_arguments}
                    elif raw_arguments is None:
                        parsed_args = {}
                    else:
                        parsed_args = {"value": raw_arguments}

                pending_calls[call_id] = {
                    "kind": "tool_call",
                    "api_call_id": current_api_call_id,
                    "codex_turn_id": codex_turn_id,
                    "tool_order": tool_order_counter,
                    "timestamp": timestamp,
                    "call_id": call_id,
                    "tool_name": payload.get("name") or "",
                    "arguments": parsed_args,
                    "raw_arguments": raw_arguments,
                    "reasoning": pending_reasoning,
                    "status": payload.get("status"),
                    "message": None,
                }
                tool_order_counter += 1
                record_model_output()
                pending_reasoning = None
                continue

            if payload_type in {"function_call_output", "custom_tool_call_output"}:
                call_id = payload.get("call_id")
                output_text, metadata = self._parse_output_blob(payload.get("output"))

                call_info = pending_calls.pop(call_id, None) if call_id else None

                if call_info is None:
                    call_info = {
                        "kind": "tool_call",
                        "api_call_id": current_api_call_id,
                        "codex_turn_id": codex_turn_id,
                        "tool_order": tool_order_counter,
                        "timestamp": timestamp,
                        "call_id": call_id or "",
                        "tool_name": payload.get("name", "") or "",
                        "arguments": {},
                        "raw_arguments": None,
                        "reasoning": pending_reasoning,
                        "status": None,
                        "message": None,
                    }
                    tool_order_counter += 1

                call_info["output"] = output_text
                call_info["metadata"] = metadata
                call_info["timestamp"] = call_info.get("timestamp") or timestamp
                normalized_events.append(call_info)
                pending_reasoning = None
                continue

        for event in normalized_events:
            api_call_id = event.get("api_call_id")
            if isinstance(api_call_id, str) and api_call_id in api_call_metrics:
                event["metrics"] = api_call_metrics[api_call_id]

        grouped_events = self._group_events_by_api_call_id(normalized_events)

        steps: list[Step] = []
        for idx, norm_event in enumerate(grouped_events, start=1):
            try:
                step = self._convert_event_to_step(norm_event, idx)
            except ValueError as exc:
                self.logger.debug(f"Skipping event during step conversion: {exc}")
                continue

            # Provide default model name if not set for agent steps
            if step.source == "agent" and not step.model_name and default_model_name:
                step.model_name = default_model_name

            steps.append(step)

        if not steps:
            self.logger.debug("No valid steps produced from Codex session")
            return None

        peak_context_tokens, summarization_count = self._extract_context_metrics(
            raw_events
        )

        spawned = self._collect_spawned_threads(raw_events)
        if spawned:
            trajectory_extra["spawned_threads"] = spawned

        # Match Harbor's authoritative final-metrics source: the latest
        # cumulative total_token_usage snapshot. Full-history children subtract
        # the cumulative baseline copied from their parent so tree totals remain
        # child-local instead of counting inherited history again.
        usage, usage_complete = self._final_cumulative_usage(
            raw_events, initial_total_usage
        )
        metrics_complete = metrics_complete and usage_complete
        if usage is None and usage_complete and is_root_thread:
            usage = self._root_usage_from_stdout()
            if usage is not None:
                self.logger.debug(
                    "Recovered root usage from %s; the rollout had no token_count "
                    "events",
                    self._OUTPUT_FILENAME,
                )

        final_extra: dict[str, Any] = {}
        if summarization_count:
            final_extra["compacted"] = True
        final_extra = dict(
            extra_with_context_metrics(
                final_extra,
                peak_context_tokens=peak_context_tokens,
                summarization_count=summarization_count,
            )
            or {}
        )

        total_metrics: FinalMetrics | None = None
        if not metrics_complete:
            final_extra["metrics_complete"] = False
            total_metrics = FinalMetrics(
                total_prompt_tokens=None,
                total_completion_tokens=None,
                total_cached_tokens=None,
                total_cost_usd=None,
                total_steps=len(steps),
                extra=final_extra,
            )
        elif usage is not None:
            total_cost_usd = self._compute_cost_from_pricing(
                prompt_tokens=usage["prompt_tokens"],
                completion_tokens=usage["completion_tokens"],
                cached_tokens=usage["cached_tokens"],
                model_name=default_model_name,
            )
            final_extra["reasoning_output_tokens"] = usage["reasoning_tokens"] or None
            final_extra["total_tokens"] = usage["total_tokens"] or None

            total_metrics = FinalMetrics(
                total_prompt_tokens=usage["prompt_tokens"] or None,
                total_completion_tokens=usage["completion_tokens"] or None,
                total_cached_tokens=usage["cached_tokens"] or None,
                total_cost_usd=total_cost_usd,
                total_steps=len(steps),
                extra=final_extra,
            )

        trajectory = Trajectory(
            schema_version="ATIF-v1.7",
            session_id=session_id,
            trajectory_id=trajectory_id,
            agent=Agent(
                name="codex",
                version=agent_version,
                model_name=default_model_name,
                extra=agent_extra,
            ),
            steps=steps,
            final_metrics=total_metrics,
            extra=trajectory_extra or None,
        )

        return trajectory

    @staticmethod
    def _collect_spawned_threads(
        raw_events: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Record which thread each ``spawn_agent`` call produced, and when.

        Codex reports the spawn as a ``CollabAgentToolCall`` item carrying the
        receiver thread ids. Keeping the timestamp lets the embedded child be
        linked back to the step that spawned it.
        """
        spawned: list[dict[str, Any]] = []
        for event in raw_events:
            if event.get("type") != "event_msg":
                continue
            payload = event.get("payload")
            if not isinstance(payload, dict):
                continue
            item = payload.get("item")
            if not isinstance(item, dict):
                continue
            if item.get("type") != "CollabAgentToolCall":
                continue
            if item.get("tool") != "spawn_agent":
                continue
            receivers = item.get("receiver_thread_ids")
            if not isinstance(receivers, list) or not receivers:
                continue
            spawned.append(
                {
                    "thread_ids": [r for r in receivers if isinstance(r, str)],
                    "timestamp": event.get("timestamp"),
                }
            )
        return spawned

    def _link_subagent_refs(self, data: dict[str, Any]) -> None:
        """Point each spawning step at the child trajectory it created.

        ATIF v1.7 resolves an embedded child through ``subagent_trajectory_ref``
        keyed by ``trajectory_id``. Harbor#2366 embeds children without the refs,
        which leaves a consumer to guess which tool call produced which child.

        The spawn is matched to the last step that could have issued it: Codex may
        invoke the tool from inside a code-mode script, so the step's own tool name
        is not reliably ``spawn_agent`` and the timestamp ordering is what links
        them.
        """
        children = data.get("subagent_trajectories") or []
        if not children:
            return

        by_thread = {
            child.get("trajectory_id"): child
            for child in children
            if child.get("trajectory_id")
        }
        spawned = (data.get("extra") or {}).get("spawned_threads") or []
        steps = data.get("steps") or []

        for record in spawned:
            timestamp = record.get("timestamp")
            candidates = [
                step
                for step in steps
                if step.get("source") == "agent"
                and step.get("observation")
                and (timestamp is None or (step.get("timestamp") or "") <= timestamp)
            ]
            if not candidates:
                continue
            observation = candidates[-1].get("observation") or {}
            results = observation.get("results") or []
            if not results:
                continue

            refs = results[0].get("subagent_trajectory_ref") or []
            for thread_id in record.get("thread_ids", []):
                child = by_thread.get(thread_id)
                if child is None:
                    continue
                refs.append(
                    {
                        "trajectory_id": thread_id,
                        "session_id": child.get("session_id"),
                    }
                )
            if refs:
                results[0]["subagent_trajectory_ref"] = refs

    def _root_usage_from_stdout(self) -> dict[str, int] | None:
        """Recover the root thread's usage from `codex exec --json` stdout.

        Backport of the Codex half of harbor-framework/harbor#970: when a rollout
        carries no `token_count` events, usage would otherwise be reported as
        nothing at all. Codex emits a cumulative `turn.completed.usage` on stdout,
        and the last one is the run's total.

        Measured against a real delegating run, that figure covers the ROOT
        thread only - it matched the root exactly and excluded the child - so it
        can restore the root's own metrics but can never stand in for the tree.
        """
        stdout_path = self.logs_dir / self._OUTPUT_FILENAME
        if not stdout_path.exists():
            return None

        usage: dict[str, int] | None = None
        try:
            with open(stdout_path, "r") as handle:
                for line in handle:
                    stripped = line.strip()
                    if not stripped.startswith("{"):
                        continue
                    try:
                        event = json.loads(stripped)
                    except json.JSONDecodeError:
                        continue
                    if event.get("type") != "turn.completed":
                        continue
                    payload = event.get("usage")
                    if not isinstance(payload, dict):
                        continue
                    usage = {
                        "prompt_tokens": payload.get("input_tokens") or 0,
                        "completion_tokens": payload.get("output_tokens") or 0,
                        "cached_tokens": payload.get("cached_input_tokens") or 0,
                        "reasoning_tokens": payload.get("reasoning_output_tokens") or 0,
                        "total_tokens": 0,
                    }
        except OSError as exc:
            self.logger.debug(f"Failed to read {stdout_path}: {exc}")

        return usage

    def _with_tree_metrics(self, root: Trajectory) -> Trajectory:
        """Put whole-tree totals on the root, keeping root-only figures alongside.

        Pier-specific. Run-level statistics must cover every agent in the tree so
        that a delegating Codex run is comparable with the flat Claude Code
        implementation, while the child trajectories stay structurally separate.

        ATIF expects ``final_metrics.total_steps`` to match the trajectory's own
        step count unless the difference is documented, so the root carries a note
        explaining the divergence and ``extra.self_only`` keeps the root's own
        metrics.
        """

        def walk(trajectory: Trajectory) -> list[Trajectory]:
            found = [trajectory]
            for child in trajectory.subagent_trajectories or []:
                found.extend(walk(child))
            return found

        nodes = walk(root)
        self_metrics = root.final_metrics
        root_incomplete = ((self_metrics.extra or {}) if self_metrics else {}).get(
            "metrics_complete"
        ) is False
        if len(nodes) == 1 and not root_incomplete:
            return root
        complete = True
        totals = {
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_cached_tokens": 0,
            "total_steps": 0,
        }
        total_cost: float | None = None
        peak_context: int | None = None
        summarizations: int | None = None

        cost_complete = True
        for node in nodes:
            metrics = node.final_metrics
            if metrics is None:
                # A thread with no metrics at all contributes unknown usage.
                # Skipping it silently would publish a total that omits a whole
                # agent while still claiming to describe the tree.
                complete = False
                cost_complete = False
                continue
            extra = metrics.extra or {}
            if extra.get("metrics_complete") is False:
                complete = False
                cost_complete = False
            if metrics.total_cost_usd is None and (
                metrics.total_prompt_tokens or metrics.total_completion_tokens
            ):
                # Usage that could not be priced (no pricing entry for that
                # thread's model) must not be summed into a tree cost as if the
                # thread were free.
                cost_complete = False
            for field in (
                "total_prompt_tokens",
                "total_completion_tokens",
                "total_cached_tokens",
                "total_steps",
            ):
                totals[field] += getattr(metrics, field) or 0
            if metrics.total_cost_usd is not None:
                total_cost = (total_cost or 0.0) + metrics.total_cost_usd
            node_peak = extra.get("peak_context_tokens")
            if isinstance(node_peak, int):
                peak_context = (
                    node_peak if peak_context is None else max(peak_context, node_peak)
                )
            node_summarizations = extra.get("summarization_count")
            if isinstance(node_summarizations, int):
                summarizations = (summarizations or 0) + node_summarizations

        tree_extra: dict[str, Any] = dict(
            (self_metrics.extra or {}) if self_metrics else {}
        )
        tree_extra.pop("metrics_complete", None)
        if self_metrics is not None:
            tree_extra["self_only"] = {
                "total_prompt_tokens": self_metrics.total_prompt_tokens,
                "total_completion_tokens": self_metrics.total_completion_tokens,
                "total_cached_tokens": self_metrics.total_cached_tokens,
                "total_cost_usd": self_metrics.total_cost_usd,
                "total_steps": self_metrics.total_steps,
            }
        tree_extra["subagent_count"] = len(nodes) - 1
        tree_extra["tree_metrics_complete"] = complete
        tree_extra["tree_cost_complete"] = cost_complete
        if peak_context is not None:
            tree_extra["peak_context_tokens"] = peak_context
        if summarizations is not None:
            tree_extra["summarization_count"] = summarizations

        # An incomplete tree must not publish aggregates that would silently omit
        # (or double count) a thread's usage; keep the root-only view instead.
        aggregate_steps = sum(len(node.steps) for node in nodes) or None
        aggregate = (
            {
                "total_prompt_tokens": totals["total_prompt_tokens"] or None,
                "total_completion_tokens": totals["total_completion_tokens"] or None,
                "total_cached_tokens": totals["total_cached_tokens"] or None,
                "total_cost_usd": total_cost if cost_complete else None,
                "total_steps": aggregate_steps,
            }
            if complete
            else {
                "total_prompt_tokens": None,
                "total_completion_tokens": None,
                "total_cached_tokens": None,
                "total_cost_usd": None,
                "total_steps": aggregate_steps,
            }
        )

        data = root.to_json_dict()
        data["final_metrics"] = {**aggregate, "extra": tree_extra}
        notes = data.get("notes")
        note = (
            "final_metrics aggregates this trajectory and its embedded "
            "subagent_trajectories; final_metrics.extra.self_only holds the root "
            "thread's own metrics."
        )
        data["notes"] = f"{notes}\n{note}" if notes else note
        return Trajectory.model_validate(data)

    def populate_context_post_run(self, context: AgentContext) -> None:
        """
        Populate the agent context after Codex finishes executing.

        Converts the Codex session JSONL file into an ATIF trajectory, persists it,
        and propagates usage metrics back to the Pier context.
        """
        # Scan the whole sessions tree. The previous single-directory lookup
        # required exactly one deepest `<YYYY>/<MM>/<DD>` directory, which a run
        # crossing midnight (or a subagent starting after it) violates.
        sessions_dir = self.logs_dir / "sessions"
        if not sessions_dir.exists():
            self.logger.debug("No Codex session directory found")
            return

        try:
            trajectory = self._convert_events_to_trajectory(sessions_dir)
        except Exception:
            self.logger.exception("Failed to convert Codex events to trajectory")
            return

        if not trajectory:
            self.logger.debug("Failed to convert Codex session to trajectory")
            return

        trajectory_path = self.logs_dir / "trajectory.json"
        try:
            with open(trajectory_path, "w") as handle:
                handle.write(format_trajectory_json(trajectory.to_json_dict()))
            self.logger.debug(f"Wrote Codex trajectory to {trajectory_path}")
        except OSError as exc:
            self.logger.debug(
                f"Failed to write trajectory file {trajectory_path}: {exc}"
            )

        if trajectory.final_metrics:
            populate_context_from_final_metrics(context, trajectory.final_metrics)

    def _benchmark_reasoning_effort(self) -> str | None:
        """Reasoning effort configured for this run, if any."""
        value = self._resolved_flags.get("reasoning_effort")
        return value if isinstance(value, str) and value else None

    @staticmethod
    def narrow_model_catalog(
        catalog: dict[str, Any], model: str, effort: str | None
    ) -> dict[str, Any]:
        """Reduce a Codex model catalog to the single benchmark configuration.

        Restricting the catalog is what actually enforces the benchmark model on
        spawned agents: Codex resolves a requested subagent model against the
        catalog and rejects anything absent from it, so a one-model catalog is a
        hard allowlist rather than a hidden tool-schema field. Trimming the
        reasoning levels closes the matching gap for effort, since one model still
        leaves every effort level of that model reachable.

        Raises ``ValueError`` when the model has no entry: its metadata (context
        window, tool mode, transport) is model-specific behaviour, and deriving it
        from an unrelated entry would silently misconfigure the benchmark. Such
        models must be described by an explicit ``model_catalog_file``.
        """
        entries = catalog.get("models") or []
        matches = [entry for entry in entries if entry.get("slug") == model]
        if not matches:
            # `available_slugs` is carried when the catalog was pre-filtered in the
            # sandbox, so the diagnostic can still name what Codex does know.
            slugs = catalog.get("available_slugs") or [
                entry.get("slug") for entry in entries
            ]
            known = ", ".join(sorted(str(slug) for slug in slugs))
            raise ValueError(
                f"restrict_model_catalog is enabled but '{model}' is not in the "
                f"Codex bundled catalog, so its metadata cannot be derived. "
                f"Supply model_catalog_file with an explicit one-model catalog. "
                f"Known models: {known}"
            )

        entry = dict(matches[0])
        if effort:
            levels = [
                level
                for level in entry.get("supported_reasoning_levels") or []
                if level.get("effort") == effort
            ]
            if not levels:
                raise ValueError(
                    f"model '{model}' does not support reasoning effort '{effort}'"
                )
            entry["supported_reasoning_levels"] = levels
            entry["default_reasoning_level"] = effort

        # Deliberately no transport mutation here: measured against codex-cli
        # 0.149.1, `prefer_websockets = false` does not prevent the WebSocket
        # attempts anyway, and a future release that does honour it would make
        # restrict_model_catalog silently change transport semantics as well as
        # model availability. This narrowing constrains model and effort only.
        return {"models": [entry]}

    async def _install_model_catalog(
        self, environment: BaseEnvironment, model: str, env: dict[str, str]
    ) -> None:
        """Generate and install the narrowed catalog inside the sandbox.

        The source is `codex debug models --bundled`, which skips the remote
        refresh: the metadata then depends only on the installed Codex version and
        never on the gateway or the credentials in play.
        """
        if self._model_catalog_file:
            # A supplied catalog is narrowed exactly like a derived one. It is the
            # path used for models Codex ships no metadata for, so it is also the
            # path that most needs the guarantee: uploading it verbatim would let
            # a multi-entry file leave other models (and other reasoning efforts)
            # spawnable, which is the enforcement this option exists to provide.
            # Every other field of the chosen entry is preserved untouched,
            # including a context window Codex knows nothing about.
            try:
                supplied = json.loads(Path(self._model_catalog_file).read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"Could not read model_catalog_file "
                    f"{self._model_catalog_file}: {exc}"
                ) from exc

            narrowed = self.narrow_model_catalog(
                supplied, model, self._benchmark_reasoning_effort()
            )
            await self._write_model_catalog(environment, narrowed, model, env)
            return

        # The full bundled catalog is hundreds of kilobytes (every model carries
        # its complete instructions template), which is too much to pull back
        # through the exec channel. Select the one entry in the sandbox and do the
        # narrowing here, where it is unit-tested.
        # Node is guaranteed present (Codex itself is installed through npm),
        # whereas the task image may not ship python.
        selector = (
            "let raw='';"
            "process.stdin.on('data',c=>raw+=c).on('end',()=>{"
            "const cat=JSON.parse(raw);"
            "const all=cat.models||[];"
            "process.stdout.write('<<CATALOG>>'+JSON.stringify({"
            "models:all.filter(e=>e.slug===process.argv[1]),"
            "available_slugs:all.map(e=>e.slug)})+'<<END>>');"
            "});"
        )
        result = await self.exec_as_agent(
            environment,
            command=(
                "if [ -s ~/.nvm/nvm.sh ]; then . ~/.nvm/nvm.sh; fi; "
                "codex debug models --bundled | "
                f"node -e {shlex.quote(selector)} {shlex.quote(model)}"
            ),
            env=env,
        )
        # The environment merges stderr into stdout, and Codex prints warnings
        # there, so the payload is delimited rather than parsed off the raw stream.
        stdout = getattr(result, "stdout", None) or ""
        start = stdout.find("<<CATALOG>>")
        end = stdout.find("<<END>>")
        if start == -1 or end == -1:
            raise RuntimeError(
                "Could not read the bundled Codex model catalog required by "
                f"restrict_model_catalog. Output was: {stdout[:400]}"
            )
        try:
            catalog = json.loads(stdout[start + len("<<CATALOG>>") : end])
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Could not parse the bundled Codex model catalog required by "
                f"restrict_model_catalog: {exc}"
            ) from exc

        narrowed = self.narrow_model_catalog(
            catalog, model, self._benchmark_reasoning_effort()
        )
        await self._write_model_catalog(environment, narrowed, model, env)

    async def _write_model_catalog(
        self,
        environment: BaseEnvironment,
        catalog: dict[str, Any],
        model: str,
        env: dict[str, str],
    ) -> None:
        """Write a narrowed catalog into the sandbox."""
        payload = json.dumps(catalog)
        await self.exec_as_agent(
            environment,
            command=(
                f"cat >{shlex.quote(self._REMOTE_MODEL_CATALOG.as_posix())} "
                f"<<'CATALOG'\n{payload}\nCATALOG"
            ),
            env=env,
        )
        self.logger.debug("Installed narrowed Codex catalog for %s", model)

    def _benchmark_config_overrides(self, model: str) -> list[str]:
        """`-c` overrides that pin delegation to the benchmark configuration.

        Passed as CLI overrides rather than appended to config.toml: appending
        risks duplicate keys or a key landing inside a previously opened table,
        and `-c` outranks any user-supplied config by design.

        Emitted only under restriction, so an unrestricted run is stock Codex.
        """
        if not self._restrict_model_catalog:
            return []

        overrides = [
            f"model_catalog_json={self._REMOTE_MODEL_CATALOG.as_posix()}",
            f"agents.default_subagent_model={model}",
            # V2-only, inert for a V1 model, correct when a V2 model is benchmarked.
            "features.multi_agent_v2.expose_spawn_agent_model_overrides=false",
        ]
        if effort := self._benchmark_reasoning_effort():
            overrides.append(f"agents.default_subagent_reasoning_effort={effort}")
        return overrides

    def _build_register_skills_command(self) -> str | None:
        """Return a shell command that copies skills to Codex's skills directory."""
        if not self.skills_dir:
            return None
        return (
            f"mkdir -p $HOME/.agents/skills && "
            f"cp -r {shlex.quote(self.skills_dir)}/* "
            f"$HOME/.agents/skills/ 2>/dev/null || true"
        )

    def _build_register_mcp_servers_command(self) -> str | None:
        """Return a shell command that writes MCP config to $CODEX_HOME/config.toml."""
        if not self.mcp_servers:
            return None
        lines: list[str] = []
        for server in self.mcp_servers:
            lines.append(f"[mcp_servers.{server.name}]")
            if server.transport == "stdio":
                cmd_parts = [server.command] + server.args if server.command else []
                lines.append(f'command = "{shlex.join(cmd_parts)}"')
            else:
                lines.append(f'url = "{server.url}"')
            lines.append("")
        escaped_config = shlex.quote("\n".join(lines))
        return f'echo {escaped_config} >> "$CODEX_HOME/config.toml"'

    def _resolve_auth_json_path(self) -> Path | None:
        """Resolve which auth.json to inject, if any.

        Defaults to None (OPENAI_API_KEY auth). Opt into auth.json auth via:
          - CODEX_AUTH_JSON_PATH=<path> → use that specific file
          - CODEX_FORCE_AUTH_JSON=<truthy> → use ~/.codex/auth.json
        """
        explicit = self._get_env("CODEX_AUTH_JSON_PATH")
        if explicit:
            p = Path(explicit)
            if not p.is_file():
                raise ValueError(
                    f"CODEX_AUTH_JSON_PATH points to non-existent file: {explicit}"
                )
            return p

        if parse_bool_env_value(
            self._get_env("CODEX_FORCE_AUTH_JSON"),
            name="CODEX_FORCE_AUTH_JSON",
            default=False,
        ):
            default = Path.home() / ".codex" / "auth.json"
            if not default.is_file():
                raise ValueError(
                    f"CODEX_FORCE_AUTH_JSON is set but {default} does not exist"
                )
            return default

        return None

    @with_prompt_template
    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        escaped_instruction = shlex.quote(instruction)

        if not self.model_name:
            raise ValueError("Model name is required")

        model = self._command_model_name or self.model_name.split("/")[-1]

        # Build command with optional CLI config flags from descriptors.
        cli_flags = self.build_cli_flags()
        cli_flags_arg = (cli_flags + " ") if cli_flags else ""

        config_overrides: list[str] = []
        if base_url := self._get_env("OPENAI_BASE_URL"):
            config_overrides.append(f"openai_base_url={base_url}")
        config_overrides.extend(self._benchmark_config_overrides(model))
        overrides_arg = "".join(
            f"-c {shlex.quote(override)} " for override in config_overrides
        )

        # Auth resolution:
        #   1. CODEX_AUTH_JSON_PATH=<path> → use that specific auth.json file
        #   2. CODEX_FORCE_AUTH_JSON=<truthy> → use ~/.codex/auth.json
        #   3. Default: use OPENAI_API_KEY
        auth_json_path = self._resolve_auth_json_path()

        remote_codex_home = self._REMOTE_CODEX_HOME.as_posix()
        remote_secrets_dir = self._REMOTE_CODEX_SECRETS_DIR.as_posix()
        remote_auth_path = (self._REMOTE_CODEX_SECRETS_DIR / "auth.json").as_posix()

        env = self.build_process_env({"CODEX_HOME": remote_codex_home})

        await self.exec_as_agent(
            environment,
            command=(
                f'mkdir -p "$CODEX_HOME" {shlex.quote(remote_secrets_dir)} '
                f"{shlex.quote(EnvironmentPaths.agent_dir.as_posix())}"
            ),
            env=env,
        )

        if auth_json_path:
            self.logger.debug("Codex auth: using auth.json from %s", auth_json_path)
            await environment.upload_file(auth_json_path, remote_auth_path)
            # upload_file copies as root; fix ownership so the agent user can read it
            if environment.default_user is not None:
                await self.exec_as_root(
                    environment,
                    command=f"chown {environment.default_user} {remote_auth_path}",
                )
            setup_command = (
                f'ln -sf {shlex.quote(remote_auth_path)} "$CODEX_HOME/auth.json"\n'
            )
        else:
            self.logger.debug("Codex auth: using OPENAI_API_KEY")
            env.setdefault("OPENAI_API_KEY", self._get_env("OPENAI_API_KEY") or "")
            setup_command = (
                f"cat >{shlex.quote(remote_auth_path)} <<EOF\n"
                '{\n  "OPENAI_API_KEY": "${OPENAI_API_KEY}"\n}\nEOF\n'
                f"ln -sf {shlex.quote(remote_auth_path)} "
                '"$CODEX_HOME/auth.json"\n'
            )

        if openai_base_url := self._get_env("OPENAI_BASE_URL"):
            env["OPENAI_BASE_URL"] = openai_base_url

        # codex only honors openai_base_url from config, not the env var. It is
        # passed as a `-c` override rather than appended to config.toml: appending
        # can place a top-level key inside a table opened by user-supplied config,
        # which silently changes its meaning.
        config_toml_block = ""
        if self._config_toml:
            escaped_toml = shlex.quote(self._config_toml)
            config_toml_block += (
                f'\nprintf "%s\\n" {escaped_toml} >> "$CODEX_HOME/config.toml"\n'
            )

        setup_command += config_toml_block

        skills_command = self._build_register_skills_command()
        if skills_command:
            setup_command += f"\n{skills_command}"

        mcp_command = self._build_register_mcp_servers_command()
        if mcp_command:
            setup_command += f"\n{mcp_command}"

        if setup_command.strip():
            await self.exec_as_agent(
                environment,
                command=setup_command,
                env=env,
            )

        if self._restrict_model_catalog:
            await self._install_model_catalog(environment, model, env)
        try:
            await self.exec_as_agent(
                environment,
                command=(
                    "if [ -s ~/.nvm/nvm.sh ]; then . ~/.nvm/nvm.sh; fi; "
                    "codex exec "
                    "--dangerously-bypass-approvals-and-sandbox "
                    "--skip-git-repo-check "
                    f"--model {model} "
                    "--json "
                    "--enable unified_exec "
                    f"{overrides_arg}"
                    f"{cli_flags_arg}"
                    "-- "  # end of flags
                    f"{escaped_instruction} "
                    f"2>&1 </dev/null | tee {
                        EnvironmentPaths.agent_dir / self._OUTPUT_FILENAME
                    }"
                ),
                env=env,
            )
        finally:
            try:
                await self.exec_as_agent(
                    environment,
                    command=(
                        f"mkdir -p {EnvironmentPaths.agent_dir.as_posix()}\n"
                        'if [ -d "$CODEX_HOME/sessions" ]; then\n'
                        f"  rm -rf {
                            (EnvironmentPaths.agent_dir / 'sessions').as_posix()
                        }\n"
                        f'  cp -R "$CODEX_HOME/sessions" {
                            (EnvironmentPaths.agent_dir / "sessions").as_posix()
                        }\n'
                        "fi"
                    ),
                    env=env,
                )
            except Exception:
                pass
            # cleanup - best effort
            try:
                await self.exec_as_agent(
                    environment,
                    command=f'rm -rf {shlex.quote(remote_secrets_dir)} "$CODEX_HOME"',
                    env=env,
                )
            except Exception:
                pass
