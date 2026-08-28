import copy
import json
import shlex
from datetime import datetime, timezone
from typing import Any

from pier.agents.installed.base import (
    BaseInstalledAgent,
    CliFlag,
    NonZeroAgentExitCodeError,
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
from pier.utils.trajectory_metrics import (
    extra_with_context_metrics,
    peak_context_tokens_from_steps,
    populate_context_from_final_metrics,
)
from pier.utils.trajectory_utils import format_trajectory_json

_CURRENT_PI_PACKAGE = "@earendil-works/pi-coding-agent"
_LEGACY_PI_PACKAGE = "@mariozechner/pi-coding-agent"
# Version at which the npm package was renamed to the @earendil-works scope.
_PI_PACKAGE_RENAME_VERSION = (0, 74, 0)


def _version_tuple(version: str) -> tuple[int, ...] | None:
    """Parse the leading numeric components of a version string.

    Deliberately avoids a ``packaging`` dependency: only the leading dotted
    integers matter for the package-rename comparison, and anything unparseable
    falls back to the current package name.
    """
    parts: list[int] = []
    for chunk in version.strip().lstrip("v").split("."):
        digits = ""
        for char in chunk:
            if not char.isdigit():
                break
            digits += char
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts) or None


class Pi(BaseInstalledAgent):
    """
    The Pi agent uses the pi coding agent (https://pi.dev) to solve tasks.

    Parses the JSON lines emitted by ``pi --print --mode json`` (captured to
    ``pi.txt``) into an ATIF trajectory.

    Stdout event types consumed:
        session               - header line carrying the session UUID
        turn_end              - one API turn: assistant message plus its tool results
        message_end           - authoritative assistant message; used as the
                                fallback for a final turn cut short by a timeout
        compaction_end        - context compaction, including the summary's own usage
        auto_retry_start      - provider error that pi retried internally

    ``message_update`` (delta-only), ``entry_appended`` and ``agent_end`` are
    filtered out of the captured stream: each re-serialises message content that
    is already covered by ``turn_end`` / ``message_end``.
    """

    SUPPORTS_ATIF: bool = True

    _OUTPUT_FILENAME = "pi.txt"
    _SESSION_DIR = "/logs/agent/pi/sessions"
    _MODELS_JSON_PATH = "$HOME/.pi/agent/models.json"

    CLI_FLAGS = [
        CliFlag(
            "thinking",
            cli="--thinking",
            type="enum",
            choices=["off", "minimal", "low", "medium", "high", "xhigh", "max"],
        ),
    ]

    # Env vars forwarded to the pi subprocess per provider, using pi's own names
    # (see the pi docs, "Providers"). Only the requested provider's vars are passed.
    _PROVIDER_ENVS: dict[str, tuple[str, ...]] = {
        "amazon-bedrock": (
            "AWS_BEARER_TOKEN_BEDROCK",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
            "AWS_REGION",
        ),
        "anthropic": (
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_OAUTH_TOKEN",
            "ANTHROPIC_BASE_URL",
        ),
        "azure-openai-responses": ("AZURE_OPENAI_API_KEY",),
        "cerebras": ("CEREBRAS_API_KEY",),
        "deepseek": ("DEEPSEEK_API_KEY",),
        "github-copilot": ("GITHUB_TOKEN",),
        "google": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        "groq": ("GROQ_API_KEY",),
        "mistral": ("MISTRAL_API_KEY",),
        "openai": ("OPENAI_API_KEY", "OPENAI_BASE_URL"),
        "openrouter": ("OPENROUTER_API_KEY",),
        "xai": ("XAI_API_KEY",),
    }

    # Env vars holding a provider endpoint override, used to build models.json and
    # to widen the network allowlist to a gateway/proxy host.
    _PROVIDER_BASE_URL_ENVS: dict[str, tuple[str, ...]] = {
        "anthropic": ("ANTHROPIC_BASE_URL",),
        "openai": ("OPENAI_BASE_URL",),
    }

    # Env var pi reads for each provider's API key, referenced from models.json by
    # name (``$VAR``) so the secret itself is never written to disk.
    _PROVIDER_API_KEY_ENVS: dict[str, str] = {
        "anthropic": "ANTHROPIC_API_KEY",
        "azure-openai-responses": "AZURE_OPENAI_API_KEY",
        "cerebras": "CEREBRAS_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "google": "GEMINI_API_KEY",
        "groq": "GROQ_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "openai": "OPENAI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "xai": "XAI_API_KEY",
    }

    _DEFAULT_PROVIDER_DOMAINS: dict[str, list[str]] = {
        "anthropic": ["api.anthropic.com"],
        "deepseek": ["api.deepseek.com"],
        "google": [".googleapis.com"],
        "groq": ["api.groq.com"],
        "mistral": ["api.mistral.ai"],
        "openai": ["api.openai.com"],
        "openrouter": ["openrouter.ai"],
        "xai": ["api.x.ai"],
    }

    def __init__(
        self,
        *args,
        pi_config: dict[str, Any] | None = None,
        provider_base_url_env: str | None = None,
        provider_api_key_env: str | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._pi_config: dict[str, Any] = pi_config or {}
        self._provider_base_url_env = provider_base_url_env
        self._provider_api_key_env = provider_api_key_env
        self._instruction: str | None = None

    @staticmethod
    def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        """Merge *override* into *base* in place, recursing into nested dicts."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                Pi._deep_merge(base[key], value)
            else:
                base[key] = value
        return base

    @staticmethod
    def name() -> str:
        return AgentName.PI.value

    def get_version_command(self) -> str | None:
        return ". ~/.nvm/nvm.sh; pi --version"

    def parse_version(self, stdout: str) -> str:
        return stdout.strip().splitlines()[-1].strip()

    def _package_name(self) -> str:
        if self._version and (parsed := _version_tuple(self._version)):
            if parsed < _PI_PACKAGE_RENAME_VERSION:
                return _LEGACY_PI_PACKAGE
        return _CURRENT_PI_PACKAGE

    def install_spec(self) -> AgentInstallSpec:
        version_spec = f"@{self._version}" if self._version else "@latest"
        package_name = self._package_name()
        return AgentInstallSpec(
            agent_name=self.name(),
            version=self._version,
            steps=[
                InstallStep(
                    user="root",
                    env={"DEBIAN_FRONTEND": "noninteractive"},
                    run="apt-get update && apt-get install -y curl",
                ),
                InstallStep(
                    user="agent",
                    run=(
                        "set -euo pipefail; "
                        "curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh | bash && "
                        'export NVM_DIR="$HOME/.nvm" && '
                        '\\. "$NVM_DIR/nvm.sh" || true && '
                        "command -v nvm &>/dev/null || { echo 'Error: NVM failed to load' >&2; exit 1; } && "
                        "nvm install 22 && npm -v && "
                        f"npm install -g --ignore-scripts {package_name}{version_spec} && "
                        "pi --version"
                    ),
                ),
            ],
            verification_command=self.get_version_command(),
        )

    def _provider(self) -> str | None:
        if not self.model_name or "/" not in self.model_name:
            return None
        return self.model_name.split("/", 1)[0]

    def _base_url_values(self) -> list[str]:
        """Endpoint overrides set for the requested provider, if any."""
        provider = self._provider()
        if provider is None:
            return []
        values: list[str] = []
        env_names = list(self._PROVIDER_BASE_URL_ENVS.get(provider, ()))
        if self._provider_base_url_env:
            env_names.insert(0, self._provider_base_url_env)
        for env_name in env_names:
            if value := self._get_env(env_name):
                values.append(value)
        return values

    def _build_models_config(self) -> dict[str, Any]:
        """Build the ``models.json`` contents for this run.

        Pi resolves ``--model`` against its built-in catalog, so a gateway or
        proxy needs a provider entry. Only ``baseUrl``/``apiKey`` are generated:
        that routes pi's *built-in* models through the endpoint while keeping
        their shipped cost and capability metadata. Declaring a ``models`` entry
        for an id that matches a built-in would replace it and silently drop that
        metadata, so custom model entries are left to ``pi_config``, where the
        caller states the id, ``api`` and ``cost`` explicitly.
        """
        config: dict[str, Any] = {}
        provider = self._provider()

        if provider is not None:
            provider_entry: dict[str, Any] = {}
            base_urls = self._base_url_values()
            if base_urls:
                provider_entry["baseUrl"] = base_urls[0]
            key_env = self._provider_api_key_env or self._PROVIDER_API_KEY_ENVS.get(provider)
            if key_env and self._has_env(key_env):
                # Reference the variable by name so the key is not written to disk.
                provider_entry["apiKey"] = f"${key_env}"
            if provider_entry:
                config["providers"] = {provider: provider_entry}

        return self._deep_merge(copy.deepcopy(config), self._pi_config)

    def _build_register_models_command(self) -> str | None:
        """Return a shell command writing models.json, or None when not needed."""
        config = self._build_models_config()
        if not config:
            return None
        escaped = shlex.quote(json.dumps(config, indent=2))
        return (
            f'mkdir -p "$(dirname {self._MODELS_JSON_PATH})" && '
            f"printf '%s\\n' {escaped} > {self._MODELS_JSON_PATH}"
        )

    def _build_register_skills_command(self) -> str | None:
        """Return a shell command that copies skills to Pi's skills directory."""
        if not self.skills_dir:
            return None
        return (
            f"mkdir -p $HOME/.agents/skills && "
            f"cp -r {shlex.quote(self.skills_dir)}/* "
            f"$HOME/.agents/skills/ 2>/dev/null || true"
        )

    def network_allowlist(self) -> NetworkAllowlist:
        provider = self._provider()
        if provider is None:
            return NetworkAllowlist()

        # Pick up endpoints from the environment and from any baseUrl declared in
        # pi_config, so a gateway host is allowlisted without extra configuration.
        # pi spells the key `baseUrl`, which is not one of collect_url_values'
        # defaults, so it has to be requested explicitly.
        urls = list(self._base_url_values())
        urls.extend(
            collect_url_values(
                self._build_models_config(),
                keys={"baseUrl", "base_url", "baseurl", "baseURL", "url"},
            )
        )
        return allowlist_from_urls(
            urls,
            default_domains=self._DEFAULT_PROVIDER_DOMAINS.get(provider, []),
        )

    @with_prompt_template
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        if not self.model_name or "/" not in self.model_name:
            raise ValueError("Model name must be in the format provider/model_name")

        provider, model_id = self.model_name.split("/", 1)
        self._instruction = instruction
        escaped_instruction = shlex.quote(instruction)

        if self.mcp_servers:
            self.logger.warning(
                "pi has no built-in MCP support; ignoring %d configured MCP server(s)",
                len(self.mcp_servers),
            )

        # PI_OFFLINE/PI_SKIP_VERSION_CHECK suppress pi's startup update check and
        # catalog refresh, which would be denied by the egress proxy on
        # no-network tasks and only add latency. Job-level agent.env can override.
        env = self.build_process_env({"PI_OFFLINE": "1", "PI_SKIP_VERSION_CHECK": "1"})
        for key in self._PROVIDER_ENVS.get(provider, ()):
            if value := self._get_env(key):
                env[key] = value

        if models_command := self._build_register_models_command():
            await self.exec_as_agent(environment, command=models_command, env=env)

        if skills_command := self._build_register_skills_command():
            await self.exec_as_agent(environment, command=skills_command, env=env)

        cli_flags = self.build_cli_flags()
        cli_flags_arg = (cli_flags + " ") if cli_flags else ""

        # The instruction is piped on stdin rather than passed as a positional
        # argument. pi's parser has no `--` end-of-options separator (a bare `--`
        # consumes the next argument as a flag value), treats a leading `-` as an
        # unknown option, and reads a leading `@` as a file reference. Print mode
        # merges piped stdin into the initial prompt, so this passes the
        # instruction through verbatim whatever it starts with.
        await self.exec_as_agent(
            environment,
            command=(
                ". ~/.nvm/nvm.sh; "
                f"mkdir -p {self._SESSION_DIR}; "
                f"printf '%s' {escaped_instruction} | "
                f"pi --print --mode json --session-dir {self._SESSION_DIR} "
                f"--provider {shlex.quote(provider)} --model {shlex.quote(model_id)} "
                f"{cli_flags_arg}"
                "2>&1 "
                '| grep -Ev \'"type":"(message_update|entry_appended|agent_end)"\' '
                f"| stdbuf -oL tee /logs/agent/{self._OUTPUT_FILENAME}"
            ),
            env=env,
        )

    def _parse_stdout(self) -> list[dict[str, Any]]:
        """Read and parse JSON lines from the pi stdout capture."""
        output_path = self.logs_dir / self._OUTPUT_FILENAME
        if not output_path.exists():
            return []

        events: list[dict[str, Any]] = []
        for line in output_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                events.append(event)
        return events

    @staticmethod
    def _millis_to_iso(timestamp_ms: int | float | None) -> str | None:
        """Convert a millisecond Unix timestamp to an ISO 8601 string."""
        if not isinstance(timestamp_ms, (int, float)):
            return None
        try:
            return datetime.fromtimestamp(
                timestamp_ms / 1000, tz=timezone.utc
            ).isoformat()
        except (OSError, ValueError, OverflowError):
            return None

    @staticmethod
    def _message_key(message: dict[str, Any]) -> tuple[Any, Any]:
        return message.get("responseId"), message.get("timestamp")

    @staticmethod
    def _content_text(content: Any) -> str:
        """Join the text blocks of a message content array."""
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return ""
        parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "\n".join(part for part in parts if part)

    def _records(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Reduce the event stream to ordered step records.

        ``turn_end`` carries an assistant message together with its tool results
        and is the primary source. A ``message_end`` is held back until the next
        event decides its fate: the matching ``turn_end`` supersedes it, while
        end-of-stream means the turn never completed (agent timeout or abort) and
        the message becomes a step on its own.
        """
        records: list[dict[str, Any]] = []
        pending: dict[str, Any] | None = None

        def flush() -> None:
            nonlocal pending
            if pending is not None:
                records.append(pending)
                pending = None

        for event in events:
            event_type = event.get("type")

            if event_type == "message_end":
                message = event.get("message")
                if not isinstance(message, dict):
                    continue
                if message.get("role") != "assistant":
                    continue
                flush()
                pending = {"kind": "agent", "message": message, "tool_results": []}

            elif event_type == "turn_end":
                message = event.get("message")
                if not isinstance(message, dict):
                    continue
                tool_results = [
                    result
                    for result in (event.get("toolResults") or [])
                    if isinstance(result, dict)
                ]
                if pending is not None and self._message_key(
                    pending["message"]
                ) == self._message_key(message):
                    pending = None
                else:
                    flush()
                records.append(
                    {
                        "kind": "agent",
                        "message": message,
                        "tool_results": tool_results,
                    }
                )

            elif event_type == "compaction_end":
                if event.get("aborted") is True:
                    continue
                result = event.get("result")
                if not isinstance(result, dict):
                    continue
                flush()
                records.append(
                    {
                        "kind": "compaction",
                        "reason": event.get("reason"),
                        "result": result,
                    }
                )

        flush()
        return records

    def _agent_step(
        self, step_id: int, record: dict[str, Any]
    ) -> tuple[Step, dict[str, int | float]]:
        """Build one ATIF step for an API turn, plus its usage contribution."""
        message = record["message"]
        content = message.get("content")
        usage = message.get("usage") if isinstance(message.get("usage"), dict) else {}

        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for block in content if isinstance(content, list) else []:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                if text := block.get("text"):
                    text_parts.append(text)
            elif block_type == "thinking":
                if thinking := block.get("thinking"):
                    reasoning_parts.append(thinking)
            elif block_type == "toolCall":
                arguments = block.get("arguments")
                if not isinstance(arguments, dict):
                    arguments = {"value": arguments} if arguments is not None else {}
                tool_calls.append(
                    ToolCall(
                        tool_call_id=str(block.get("id") or ""),
                        function_name=str(block.get("name") or ""),
                        arguments=arguments,
                    )
                )

        observation_results: list[ObservationResult] = []
        tool_input = tool_output = 0
        tool_cached = tool_cache_write = tool_cache_write_1h = 0
        tool_reasoning = 0
        tool_cost = 0.0
        for result in record["tool_results"]:
            observation_results.append(
                ObservationResult(
                    source_call_id=str(result.get("toolCallId") or "") or None,
                    content=self._content_text(result.get("content")),
                    extra={"is_error": True} if result.get("isError") else None,
                )
            )
            # Tools may perform their own LLM work; pi bills it, so it belongs in
            # the run totals even though it is not part of the main context.
            if isinstance(result.get("usage"), dict):
                tool_usage = result["usage"]
                tool_input += int(tool_usage.get("input") or 0)
                tool_output += int(tool_usage.get("output") or 0)
                tool_cached += int(tool_usage.get("cacheRead") or 0)
                tool_cache_write += int(tool_usage.get("cacheWrite") or 0)
                tool_cache_write_1h += int(tool_usage.get("cacheWrite1h") or 0)
                tool_reasoning += int(tool_usage.get("reasoning") or 0)
                cost = tool_usage.get("cost")
                if isinstance(cost, dict):
                    tool_cost += float(cost.get("total") or 0.0)

        input_tokens = int(usage.get("input") or 0)
        output_tokens = int(usage.get("output") or 0)
        cache_read = int(usage.get("cacheRead") or 0)
        cache_write = int(usage.get("cacheWrite") or 0)
        cache_write_1h = int(usage.get("cacheWrite1h") or 0)
        reasoning_tokens = int(usage.get("reasoning") or 0)
        cost_total = 0.0
        if isinstance(usage.get("cost"), dict):
            cost_total = float(usage["cost"].get("total") or 0.0)

        # pi reports `input` net of cache, so the full prompt is the sum of all
        # three components (matching provider-reported prompt_tokens).
        prompt_tokens = input_tokens + cache_read + cache_write

        metrics: Metrics | None = None
        if prompt_tokens or output_tokens:
            metrics_extra = {
                key: value
                for key, value in {
                    "reasoning_tokens": reasoning_tokens,
                    "cache_write_tokens": cache_write,
                    "cache_write_1h_tokens": cache_write_1h,
                }.items()
                if value
            }
            metrics = Metrics(
                prompt_tokens=prompt_tokens,
                completion_tokens=output_tokens,
                cached_tokens=cache_read or None,
                cost_usd=cost_total or None,
                extra=metrics_extra or None,
            )

        step_extra: dict[str, Any] = {}
        if stop_reason := message.get("stopReason"):
            step_extra["stop_reason"] = stop_reason
        if error_message := message.get("errorMessage"):
            step_extra["error_message"] = error_message
        if provider := message.get("provider"):
            step_extra["provider"] = provider

        step_kwargs: dict[str, Any] = {
            "step_id": step_id,
            "timestamp": self._millis_to_iso(message.get("timestamp")),
            "source": "agent",
            "message": "\n".join(text_parts),
            "model_name": message.get("responseModel")
            or message.get("model")
            or self.model_name,
            "llm_call_count": 1,
        }
        if reasoning_effort := self._resolved_flags.get("thinking"):
            step_kwargs["reasoning_effort"] = reasoning_effort
        if reasoning_parts:
            step_kwargs["reasoning_content"] = "\n\n".join(reasoning_parts)
        if tool_calls:
            step_kwargs["tool_calls"] = tool_calls
        if observation_results:
            step_kwargs["observation"] = Observation(results=observation_results)
        if metrics:
            step_kwargs["metrics"] = metrics
        if step_extra:
            step_kwargs["extra"] = step_extra

        totals: dict[str, int | float] = {
            "prompt": prompt_tokens + tool_input + tool_cached + tool_cache_write,
            "completion": output_tokens + tool_output,
            "cached": cache_read + tool_cached,
            "cache_write": cache_write + tool_cache_write,
            "cache_write_1h": cache_write_1h + tool_cache_write_1h,
            "reasoning": reasoning_tokens + tool_reasoning,
            "cost": cost_total + tool_cost,
        }
        return Step(**step_kwargs), totals

    def _compaction_step(
        self, step_id: int, record: dict[str, Any]
    ) -> tuple[Step, dict[str, int | float]]:
        """Build a system step recording one context compaction."""
        result = record["result"]
        usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}

        input_tokens = int(usage.get("input") or 0)
        output_tokens = int(usage.get("output") or 0)
        cache_read = int(usage.get("cacheRead") or 0)
        cache_write = int(usage.get("cacheWrite") or 0)
        cache_write_1h = int(usage.get("cacheWrite1h") or 0)
        reasoning_tokens = int(usage.get("reasoning") or 0)
        cost_total = 0.0
        if isinstance(usage.get("cost"), dict):
            cost_total = float(usage["cost"].get("total") or 0.0)
        prompt_tokens = input_tokens + cache_read + cache_write

        compaction_extra = {
            key: value
            for key, value in {
                "reason": record.get("reason"),
                "tokens_before": result.get("tokensBefore"),
                "estimated_tokens_after": result.get("estimatedTokensAfter"),
            }.items()
            if value is not None
        }

        # ATIF restricts `metrics`/`model_name` to `source: "agent"` steps, so the
        # summarization call's usage is reported here as plain numbers. It is still
        # folded into FinalMetrics below, because pi bills it to the run.
        if prompt_tokens or output_tokens:
            compaction_extra["usage"] = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": output_tokens,
                "cached_tokens": cache_read,
                "cache_write_tokens": cache_write,
                "cache_write_1h_tokens": cache_write_1h,
                "reasoning_tokens": reasoning_tokens,
                "cost_usd": cost_total,
            }

        step_kwargs: dict[str, Any] = {
            "step_id": step_id,
            "source": "system",
            "message": str(result.get("summary") or ""),
            "extra": {"compaction": compaction_extra},
        }

        totals: dict[str, int | float] = {
            "prompt": prompt_tokens,
            "completion": output_tokens,
            "cached": cache_read,
            "cache_write": cache_write,
            "cache_write_1h": cache_write_1h,
            "reasoning": reasoning_tokens,
            "cost": cost_total,
        }
        return Step(**step_kwargs), totals

    def _compute_cost_from_pricing(
        self,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        cached_tokens: int | None,
    ) -> float | None:
        """Fall back to LiteLLM's bundled price table when pi reports no cost."""
        if not self.model_name:
            return None

        try:
            import litellm
        except ImportError:
            self.logger.warning("litellm not available; cost_usd left as None")
            return None

        pricing: dict[str, Any] | None = None
        for key in (self.model_name, self.model_name.split("/", 1)[-1]):
            entry = litellm.model_cost.get(key)
            if entry:
                pricing = entry
                break

        if pricing is None:
            self.logger.warning(
                "No LiteLLM pricing for '%s'; cost_usd left as None. Add `cost` "
                "rates to the model's pi_config entry to price a custom slug.",
                self.model_name,
            )
            return None

        input_rate = pricing.get("input_cost_per_token") or 0.0
        output_rate = pricing.get("output_cost_per_token") or 0.0
        cache_read_rate = pricing.get("cache_read_input_token_cost") or input_rate

        uncached = max(0, (prompt_tokens or 0) - (cached_tokens or 0))
        cached = cached_tokens or 0
        output = completion_tokens or 0

        return uncached * input_rate + cached * cache_read_rate + output * output_rate

    def _convert_events_to_trajectory(
        self, events: list[dict[str, Any]]
    ) -> Trajectory | None:
        """Convert pi stdout JSON events into an ATIF trajectory."""
        records = self._records(events)
        if not records:
            return None

        session_id: str | None = None
        for event in events:
            if event.get("type") == "session" and event.get("id"):
                session_id = str(event["id"])
                break

        retry_count = sum(
            1 for event in events if event.get("type") == "auto_retry_start"
        )

        steps: list[Step] = []
        totals = {
            "prompt": 0,
            "completion": 0,
            "cached": 0,
            "cache_write": 0,
            "cache_write_1h": 0,
            "reasoning": 0,
            "cost": 0.0,
        }
        summarization_count = 0
        compaction_reasons: list[str] = []

        # The instruction is piped through stdin rather than passed as a CLI
        # argument or emitted as an event, so the opening user step is synthesised
        # from what was sent.
        if self._instruction:
            steps.append(Step(step_id=1, source="user", message=self._instruction))

        for record in records:
            step_id = len(steps) + 1
            if record["kind"] == "agent":
                step, contribution = self._agent_step(step_id, record)
            else:
                step, contribution = self._compaction_step(step_id, record)
                summarization_count += 1
                if reason := record.get("reason"):
                    compaction_reasons.append(str(reason))
            steps.append(step)
            for key, value in contribution.items():
                totals[key] += value

        if not any(step.source != "user" for step in steps):
            return None

        total_cost = totals["cost"] if totals["cost"] > 0 else None
        if total_cost is None:
            # A zero result means no usage to price, which is absent rather than free.
            total_cost = (
                self._compute_cost_from_pricing(
                    totals["prompt"], totals["completion"], totals["cached"]
                )
                or None
            )

        final_extra: dict[str, Any] = {}
        if totals["cache_write"]:
            final_extra["total_cache_write_tokens"] = totals["cache_write"]
        if totals["cache_write_1h"]:
            final_extra["total_cache_write_1h_tokens"] = totals["cache_write_1h"]
        if totals["reasoning"]:
            final_extra["total_reasoning_tokens"] = totals["reasoning"]
        if summarization_count:
            final_extra["compacted"] = True
        if compaction_reasons:
            final_extra["compaction_reasons"] = compaction_reasons
        if retry_count:
            final_extra["api_retry_count"] = retry_count
        if stop_reason := self._final_stop_reason(records):
            final_extra["stop_reason"] = stop_reason

        final_metrics = FinalMetrics(
            total_prompt_tokens=totals["prompt"] or None,
            total_completion_tokens=totals["completion"] or None,
            total_cached_tokens=totals["cached"] or None,
            total_cost_usd=total_cost,
            total_steps=len(steps),
            extra=extra_with_context_metrics(
                final_extra or None,
                peak_context_tokens=peak_context_tokens_from_steps(steps),
                summarization_count=summarization_count,
            ),
        )

        return Trajectory(
            schema_version="ATIF-v1.7",
            session_id=session_id or "unknown",
            agent=Agent(
                name=self.name(),
                version=self.version() or "unknown",
                model_name=self.model_name,
            ),
            steps=steps,
            final_metrics=final_metrics,
        )

    @staticmethod
    def _final_stop_reason(records: list[dict[str, Any]]) -> str | None:
        for record in reversed(records):
            if record["kind"] == "agent":
                stop_reason = record["message"].get("stopReason")
                return str(stop_reason) if stop_reason else None
        return None

    def _final_error(self, events: list[dict[str, Any]]) -> str | None:
        """Return the error message when the run ended on a failed turn.

        Only the final turn is considered: pi retries provider errors internally
        (``auto_retry_*``), so an intermediate error does not mean the run failed.
        """
        records = self._records(events)
        for record in reversed(records):
            if record["kind"] != "agent":
                continue
            message = record["message"]
            if message.get("stopReason") == "error":
                return str(message.get("errorMessage") or "pi reported a failed turn")
            return None
        return None

    def populate_context_post_run(self, context: AgentContext) -> None:
        """Parse pi stdout, write the ATIF trajectory, and fill run metrics."""
        events = self._parse_stdout()
        if not events:
            return

        try:
            trajectory = self._convert_events_to_trajectory(events)
        except Exception:
            self.logger.exception("Failed to convert pi events to trajectory")
            trajectory = None

        if trajectory is not None:
            trajectory_path = self.logs_dir / "trajectory.json"
            try:
                trajectory_path.write_text(
                    format_trajectory_json(trajectory.to_json_dict())
                )
                self.logger.debug(f"Wrote pi trajectory to {trajectory_path}")
            except OSError as exc:
                self.logger.debug(
                    f"Failed to write trajectory file {trajectory_path}: {exc}"
                )

            if trajectory.final_metrics:
                populate_context_from_final_metrics(context, trajectory.final_metrics)

        if error := self._final_error(events):
            raise NonZeroAgentExitCodeError(f"pi ended on a failed turn: {error}")
