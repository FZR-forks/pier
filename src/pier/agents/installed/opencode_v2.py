"""Pier adapter for OpenCode V2 (``@opencode/cli`` 2.x).

Unlike the V1 adapter, which parses ``opencode run --format=json`` stdout, the
V2 adapter runs against an OpenCode server the runner owns. ``opencode run
--server <url>`` creates the run's sessions on that server, and the runner
pages the V2 REST API afterwards:

- ``GET /api/session`` — every session of the run (the server is private to
  the run, so unscoped is scoped enough), optionally ``?parentID=<id>`` for a
  session's direct children.
- ``GET /api/session/<id>/message`` — a session's messages; both endpoints
  return opaque ``cursor.next`` pages that must be followed without an
  ``order`` parameter (combining them is a client error).
- ``GET /api/session/active`` and each message's tool state distinguish a
  session that was still running at collection time from a quiet one.

The persisted messages are converted into one ATIF trajectory per unique
session, nested by their real ``parentID``.
"""

import copy
import ipaddress
import json
import re
import shlex
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from pier.agents.installed.base import (
    BaseInstalledAgent,
    NonZeroAgentExitCodeError,
    with_prompt_template,
)
from pier.agents.installed.opencode import OpenCode
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

_RUNNER_FILENAME = "opencode_v2_runner.py"

# Terminal finishes that still carry authoritative usage when the message
# completed: a truncated (`length`) or content-filtered reply, or a provider
# error, records real tokens and must not be dropped.
_NON_STOP_FINISHES = ("error", "length", "content-filter")

_PROVIDER_ENV_KEYS: dict[str, tuple[str, ...]] = {
    "amazon-bedrock": (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
    ),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "azure": ("AZURE_RESOURCE_NAME", "AZURE_API_KEY"),
    "deepseek": ("DEEPSEEK_API_KEY",),
    "github-copilot": ("GITHUB_TOKEN",),
    "google": (
        "GEMINI_API_KEY",
        "GOOGLE_GENERATIVE_AI_API_KEY",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_CLOUD_LOCATION",
        "GOOGLE_GENAI_USE_VERTEXAI",
        "GOOGLE_API_KEY",
    ),
    "groq": ("GROQ_API_KEY",),
    "huggingface": ("HF_TOKEN",),
    "llama": ("LLAMA_API_KEY",),
    "mistral": ("MISTRAL_API_KEY",),
    "openai": ("OPENAI_API_KEY", "OPENAI_BASE_URL"),
    "opencode": ("OPENCODE_API_KEY",),
    "openrouter": ("OPENROUTER_API_KEY",),
    "xai": ("XAI_API_KEY",),
}


def _iso(timestamp_ms: Any) -> str | None:
    """Convert a millisecond Unix timestamp to an ISO 8601 string."""
    if not isinstance(timestamp_ms, (int, float)) or isinstance(timestamp_ms, bool):
        return None
    try:
        return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).isoformat()
    except (OSError, ValueError, OverflowError):
        return None


def _env_template_values(config: dict[str, Any]) -> dict[str, str]:
    """``{env:NAME}`` placeholders the caller's environment must satisfy.

    OpenCode V2 resolves these templates itself, so they must be left intact in
    the generated config — but Pier can still validate up front that the
    referenced variables exist, failing the run before any tokens are spent.
    """
    values: dict[str, str] = {}

    def walk(node: Any) -> None:
        if isinstance(node, str):
            if node.startswith("{env:") and node.endswith("}"):
                name = node[5:-1]
                values[name] = node
        elif isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(config)
    return values


class OpenCodeV2(BaseInstalledAgent):
    """OpenCode V2 agent, driven through its own server process."""

    SUPPORTS_ATIF: bool = True

    _REMOTE_WORKDIR = Path("/tmp/opencode-v2-work")
    _REMOTE_BINARY = Path("/installed-agent/opencode-v2-bin")
    _RUNNER_PATH = "/installed-agent/opencode_v2_runner.py"
    _OUTPUT_FILENAME = Path("/logs/agent/opencode-v2.txt")
    _RUNNER_LOG_DIR = Path("/logs/agent/opencode-v2")
    _RUNNER_OUTPUT = _RUNNER_LOG_DIR / "runner-result.json"
    _SESSIONS_OUTPUT = _RUNNER_LOG_DIR / "opencode-v2-sessions.jsonl"
    _CLI_EVENTS_OUTPUT = _RUNNER_LOG_DIR / "opencode-v2-cli-events.jsonl"
    _INSTRUCTION_PATH = _RUNNER_LOG_DIR / "instruction.txt"

    # These values define the adapter's private process/config boundary.  A
    # caller-supplied agent.env value must not be able to replace them after
    # BaseInstalledAgent._exec merges its extra environment.
    _RESERVED_ENV_KEYS = frozenset(
        {
            "HOME",
            "PWD",
            "XDG_CONFIG_HOME",
            "XDG_DATA_HOME",
            "XDG_STATE_HOME",
            "OPENCODE_CONFIG",
            "OPENCODE_CONFIG_DIR",
            "OPENCODE_CONFIG_PROJECT_DISABLE",
            "OPENCODE_DISABLE_MODELS_FETCH",
            "OPENCODE_DISABLE_AUTOUPDATE",
            "OPENCODE_PASSWORD",
            "OPENCODE_SERVER_PASSWORD",
            "NO_PROXY",
            "no_proxy",
        }
    )

    # V2 default egress, mirroring V1's allowlist defaults.
    _DEFAULT_PROVIDER_DOMAINS: dict[str, list[str]] = OpenCode._DEFAULT_PROVIDER_DOMAINS

    def __init__(
        self,
        *args,
        opencode_v2_config: dict[str, Any] | None = None,
        opencode_v2_checksums: dict[str, str] | None = None,
        restrict_model: bool = False,
        variant: str | None = None,
        **kwargs,
    ):
        extra_env = kwargs.get("extra_env") or {}
        reserved_env = set(extra_env).intersection(self._RESERVED_ENV_KEYS)
        if reserved_env:
            if reserved_env.intersection(
                {"OPENCODE_PASSWORD", "OPENCODE_SERVER_PASSWORD"}
            ):
                message = (
                    "runner-owned server passwords cannot be supplied through agent env"
                )
            else:
                message = (
                    "adapter-owned environment cannot be supplied through agent env"
                )
            raise ValueError(
                "OpenCode V2 " + message + ": " + ", ".join(sorted(reserved_env))
            )
        if not isinstance(restrict_model, bool):
            raise ValueError("restrict_model must be a boolean")
        if variant is not None and (
            not isinstance(variant, str) or not variant or "#" in variant
        ):
            raise ValueError("variant must be a non-empty string without '#'")
        if opencode_v2_config is not None and not isinstance(opencode_v2_config, dict):
            raise ValueError("opencode_v2_config must be an object")
        if opencode_v2_checksums is not None and not isinstance(
            opencode_v2_checksums, dict
        ):
            raise ValueError("opencode_v2_checksums must be an object")
        checksums = copy.deepcopy(opencode_v2_checksums or {})
        for target, digest in checksums.items():
            if not re.fullmatch(r"linux-(?:x64|arm64)", target):
                raise ValueError(
                    "opencode_v2_checksums keys must be linux-x64 or linux-arm64"
                )
            if not isinstance(digest, str) or not re.fullmatch(
                r"[0-9a-fA-F]{64}", digest
            ):
                raise ValueError(
                    f"opencode_v2_checksums.{target} must be a SHA-256 hex digest"
                )
        super().__init__(*args, **kwargs)
        self._opencode_v2_config: dict[str, Any] = copy.deepcopy(
            opencode_v2_config or {}
        )
        self._opencode_v2_checksums = checksums
        self._restrict_model = restrict_model
        self._variant = variant
        self._instruction: str | None = None
        # A floating `latest` install cannot safely reuse a Docker image whose
        # static build command may have resolved an older release. Keep this
        # stable for one agent instance but unique across trials, and use it as
        # the unpinned install cache key below.
        self._unpinned_install_token = uuid.uuid4().hex
        self._log_redacted_env_keys = {
            key for keys in _PROVIDER_ENV_KEYS.values() for key in keys
        }
        self._log_redacted_env_keys.update(
            key
            for key in set(extra_env) | set(self._resolved_env_vars)
            if self._looks_like_credential_env_key(key)
        )
        self._log_redacted_env_keys.update(
            _env_template_values(self._opencode_v2_config)
        )

    @staticmethod
    def _looks_like_credential_env_key(key: str) -> bool:
        normalized = key.upper()
        return normalized.endswith("_BASE_URL") or any(
            marker in normalized
            for marker in (
                "API_KEY",
                "TOKEN",
                "SECRET",
                "PASSWORD",
                "AUTH",
                "CREDENTIAL",
            )
        )

    @staticmethod
    def name() -> str:
        return AgentName.OPENCODE_V2.value

    def get_version_command(self) -> str | None:
        return f"{self._REMOTE_BINARY.as_posix()} --version"

    def parse_version(self, stdout: str) -> str:
        value = stdout.strip()
        match = re.fullmatch(r"(?:opencode\s+)?v?(.+)", value)
        return match.group(1) if match else value

    def _process_env_for_logging(self, env: dict[str, str] | None) -> dict[str, str]:
        logged = super()._process_env_for_logging(env)
        for key in self._log_redacted_env_keys:
            if key in logged:
                logged[key] = "<redacted>"
        return logged

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def _deep_merge(self, base: dict[str, Any], override: dict[str, Any]):
        """Merge *override* into *base* in place, recursing into nested dicts."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
        return base

    def _model_parts(self) -> tuple[str, str, str | None]:
        """Split model_name into provider/model and return the kwarg variant."""
        if not self.model_name or "/" not in self.model_name:
            raise ValueError("Model name must be in the format provider/model_name")
        spec = self.model_name
        if "#" in spec:
            raise ValueError(
                "model_name must not include '#'; pass variant=... instead"
            )
        provider, model_id = spec.split("/", 1)
        if not provider or not model_id:
            raise ValueError("Model name must include non-empty provider and model")
        return provider, model_id, self._variant

    def _model_selection(self) -> str:
        """Return OpenCode's native provider/model#variant selection reference."""
        provider, model_id, variant = self._model_parts()
        return f"{provider}/{model_id}" + (f"#{variant}" if variant else "")

    def _resolved_variant(self) -> str | None:
        _, _, variant = self._model_parts()
        return variant

    def _build_providers_config(self) -> dict[str, Any]:
        """The provider settings, models and variants the run needs.

        Built fresh on every call: the benchmark model body and the
        caller-supplied ``opencode_v2_config`` must never leak between runs.
        """
        provider, model_id, _variant = self._model_parts()
        return {provider: {"models": {model_id: {}}}}

    def _validate_selected_model_config(self, config: dict[str, Any]) -> None:
        """Validate, but never synthesize, transport-specific output fields."""
        provider, model_id, _ = self._model_parts()
        providers = config.get("providers")
        if not isinstance(providers, dict):
            raise ValueError("providers must be an object")
        provider_entry = providers.get(provider)
        if provider_entry is None:
            provider_entry = {}
        if not isinstance(provider_entry, dict):
            raise ValueError(f"providers.{provider} must be an object")
        models = provider_entry.get("models")
        if models is None:
            models = {}
        if not isinstance(models, dict):
            raise ValueError(f"providers.{provider}.models must be an object")
        model_entry = models.get(model_id)
        if model_entry is None:
            model_entry = {}
        if not isinstance(model_entry, dict):
            raise ValueError(
                f"providers.{provider}.models.{model_id} must be an object"
            )
        limit = model_entry.get("limit")
        if limit is None:
            limit = {}
        if not isinstance(limit, dict):
            raise ValueError(
                f"providers.{provider}.models.{model_id}.limit must be an object"
            )
        output = limit.get("output")
        if output is None:
            output_limit = None
        elif isinstance(output, bool) or not isinstance(output, int) or output <= 0:
            raise ValueError(
                f"providers.{provider}.models.{model_id}.limit.output must be a positive integer"
            )
        else:
            output_limit = output

        body = model_entry.get("body")
        if body is None:
            body = {}
        if not isinstance(body, dict):
            raise ValueError(
                f"providers.{provider}.models.{model_id}.body must be an object"
            )
        output_keys = {
            key
            for key in ("max_tokens", "max_completion_tokens", "max_output_tokens")
            if key in body
        }
        if len(output_keys) > 1:
            raise ValueError(
                "output limit has competing model body fields: "
                + ", ".join(sorted(output_keys))
            )
        for key in output_keys:
            value = body[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                or (output_limit is not None and value > output_limit)
            ):
                suffix = (
                    " no greater than limit.output" if output_limit is not None else ""
                )
                raise ValueError(
                    f"providers.{provider}.models.{model_id}.body.{key} must be "
                    f"a positive integer{suffix}"
                )

        package = str(
            model_entry.get("npm")
            or model_entry.get("package")
            or provider_entry.get("npm")
            or provider_entry.get("package")
            or ""
        )
        if (
            package.endswith("/responses")
            and output_keys
            and output_keys != {"max_output_tokens"}
        ):
            raise ValueError(
                f"providers.{provider}.models.{model_id} uses a Responses package; "
                "its explicit output field must be max_output_tokens"
            )

    def _build_agents_config(self) -> dict[str, Any]:
        """Agent definitions that pin the benchmark model and variant.

        Restricting the root (``build``), subagent (``general``, ``explore``),
        and hidden housekeeping (``title``, ``summary``, ``compaction``)
        agents keeps a spawned subagent (or compaction) from resolving a
        different model: without an explicit ``model``, V2 falls back to
        whatever the agent's own default is.
        """
        if not self._restrict_model or not self.model_name:
            return {}
        # `provider/model#variant` is the native V2 selection syntax; the same
        # identity must be used by the root agent, every subagent, and
        # compaction so all requests stay on one benchmark model.
        selection = self._model_selection()
        return {
            agent_id: {
                "model": selection,
                **({"disabled": True} if agent_id in {"title", "summary"} else {}),
            }
            for agent_id in (
                "build",
                "plan",
                "general",
                "explore",
                "title",
                "summary",
                "compaction",
            )
        }

    def _pin_restricted_model(self, config: dict[str, Any]) -> None:
        """Apply the model lock after all caller configuration is merged.

        A caller may define useful custom agents and permissions. Under the
        benchmark lock those definitions remain, but every agent model is
        overwritten with the selected identity. This closes custom delegation
        and model-switch paths without mutating the caller's object or
        broadening any native permissions.
        """
        if not self._restrict_model or not self.model_name:
            return
        config["model"] = self._model_selection()
        agents = config.setdefault("agents", {})
        if not isinstance(agents, dict):
            raise ValueError("agents must be an object")
        required = {
            "build",
            "plan",
            "general",
            "explore",
            "title",
            "summary",
            "compaction",
        }
        for agent_id in required | set(agents):
            agent_config = agents.setdefault(agent_id, {})
            if not isinstance(agent_config, dict):
                raise ValueError(f"agents.{agent_id} must be an object")
            agent_config["model"] = self._model_selection()
            if agent_id in {"title", "summary"}:
                agent_config["disabled"] = True

    def _build_runtime_config(self, *, include_mcp: bool) -> dict[str, Any]:
        """The opencode.json V2 config for this run.

        Layers: benchmark model/agents → MCP → caller ``opencode_v2_config``.
        Built fresh on every call so caller dicts are never mutated.
        """
        configured_providers = self._opencode_v2_config.get("providers") or {}
        for provider_id, provider_config in configured_providers.items():
            if isinstance(provider_config, dict) and "options" in provider_config:
                raise ValueError(
                    f"providers.{provider_id}.options is V1 compatibility syntax; "
                    "use native V2 settings/body fields"
                )

        config: dict[str, Any] = {
            "update": "disable",
            "share": "disabled",
            "websearch": False,
            "providers": self._build_providers_config(),
        }
        agents = self._build_agents_config()
        if agents:
            config["agents"] = agents

        if include_mcp and self.mcp_servers:
            mcp: dict[str, dict[str, Any]] = {}
            for server in self.mcp_servers:
                if server.transport == "stdio":
                    cmd_list = [server.command] + server.args if server.command else []
                    mcp[server.name] = {"type": "local", "command": cmd_list}
                else:  # sse or streamable-http
                    mcp[server.name] = {"type": "remote", "url": server.url}
            config["mcp"] = mcp

        # Provider- and transport-specific model bodies belong to the caller's
        # configuration. Pier preserves and validates them without deriving a
        # wire field from generic ``limit.output`` metadata.
        config = self._deep_merge(config, copy.deepcopy(self._opencode_v2_config))
        self._validate_selected_model_config(config)

        self._pin_restricted_model(config)
        if self._restrict_model:
            self._assert_single_model(config)
        return config

    def _assert_single_model(self, config: dict[str, Any]) -> None:
        """Fail if the effective config could resolve any other model.

        PA1 policy: a trial measures exactly one model. Every provider entry,
        the top-level default selection, and every agent override must either
        be the benchmark model or absent (inherit it).

        Agent models have already been pinned at this point. Rechecking them
        here is deliberate defense in depth against future merge-order changes.
        """
        provider, model_id, _ = self._model_parts()
        selection = self._model_selection()
        offenders: list[str] = []

        providers = config.get("providers")
        if not isinstance(providers, dict) or provider not in providers:
            offenders.append(f"providers.{provider} missing")
        else:
            models = (providers[provider] or {}).get("models")
            if not isinstance(models, dict) or model_id not in models:
                offenders.append(f"providers.{provider}.models.{model_id} missing")
            for other in models or {}:
                if other != model_id:
                    offenders.append(f"providers.{provider}.models.{other}")

        for provider_id, provider_config in (
            providers.items() if isinstance(providers, dict) else ()
        ):
            if provider_id == provider or not isinstance(provider_config, dict):
                continue
            models = provider_config.get("models") or {}
            if models:
                offenders.append(f"providers.{provider_id} has models")

        for agent_id, agent_config in (config.get("agents") or {}).items():
            if isinstance(agent_config, dict):
                agent_model = str(agent_config.get("model") or "")
                if agent_model and agent_model != selection:
                    offenders.append(f"agents.{agent_id}.model={agent_model}")

        top_level = config.get("model")
        if top_level and top_level != selection:
            offenders.append(f"model={top_level}")

        if offenders:
            raise ValueError(
                "restrict_model: the effective OpenCode V2 config could "
                f"resolve models other than {selection}: "
                + "; ".join(sorted(set(offenders)))
            )

    def _effective_provider_ids(self, config: dict[str, Any]) -> set[str]:
        """Providers reachable by the selected root and enabled configured agents."""
        provider, _, _ = self._model_parts()
        provider_ids = {provider}
        top_level_model = config.get("model")
        if top_level_model is not None:
            if not isinstance(top_level_model, str) or "/" not in top_level_model:
                raise ValueError(
                    "top-level OpenCode model must use provider/model syntax"
                )
            top_level_provider, _ = top_level_model.split("/", 1)
            if not top_level_provider:
                raise ValueError("top-level OpenCode model has no provider")
            provider_ids.add(top_level_provider)

        agents = config.get("agents") or {}
        if not isinstance(agents, dict):
            raise ValueError("agents must be an object")
        for agent_id, agent_config in agents.items():
            if (
                not isinstance(agent_config, dict)
                or agent_config.get("disabled") is True
            ):
                continue
            model = agent_config.get("model")
            if model is None:
                continue
            if not isinstance(model, str) or "/" not in model:
                raise ValueError(
                    f"enabled OpenCode agent {agent_id!r} model must use "
                    "provider/model syntax"
                )
            agent_provider, _ = model.split("/", 1)
            if not agent_provider:
                raise ValueError(
                    f"enabled OpenCode agent {agent_id!r} model has no provider"
                )
            provider_ids.add(agent_provider)
        return provider_ids

    def network_allowlist(self) -> NetworkAllowlist:
        config = self._build_runtime_config(include_mcp=True)
        provider_ids = self._effective_provider_ids(config)
        providers = config.get("providers") or {}
        if not isinstance(providers, dict):
            raise ValueError("providers must be an object")
        provider_urls: list[str] = []
        default_domains: set[str] = set()
        for provider_id in provider_ids:
            provider_config = providers.get(provider_id) or {}
            configured_urls = collect_url_values(provider_config)
            provider_urls.extend(configured_urls)
            default_domains.update(self._DEFAULT_PROVIDER_DOMAINS.get(provider_id, ()))
            if provider_id == "openai" and (
                base_url := self._get_env("OPENAI_BASE_URL")
            ):
                provider_urls.append(base_url)
            if provider_id == "amazon-bedrock" and not configured_urls:
                region = self._get_env("AWS_REGION") or self._get_env(
                    "AWS_DEFAULT_REGION"
                )
                if region is None:
                    raise ValueError(
                        "amazon-bedrock requires an explicit provider URL or "
                        "AWS_REGION/AWS_DEFAULT_REGION for the egress allowlist"
                    )
                if not re.fullmatch(r"[a-z0-9-]+", region):
                    raise ValueError(
                        "AWS region for amazon-bedrock contains unsupported characters"
                    )
                provider_urls.append(f"https://bedrock-runtime.{region}.amazonaws.com")
        urls = self._resolve_network_urls(provider_urls, kind="provider")
        urls.extend(
            self._resolve_network_urls(
                collect_url_values(config.get("mcp") or {}), kind="MCP"
            )
        )
        return allowlist_from_urls(
            urls,
            default_domains=default_domains,
        )

    def _resolve_network_urls(self, values: list[str], *, kind: str) -> list[str]:
        """Resolve URL templates early enough to construct the egress policy."""
        resolved: list[str] = []
        for value in values:
            match = re.fullmatch(r"\{env:([^{}]+)\}", value)
            if match:
                value = self._get_env(match.group(1)) or ""
                if not value:
                    raise ValueError(
                        f"{kind} URL template {{env:{match.group(1)}}} cannot be "
                        "resolved while constructing the network allowlist; "
                        "provide it through agent.env"
                    )
            elif value.startswith("${"):
                raise ValueError(
                    f"{kind} URL template {value!r} is unresolved while "
                    "constructing the network allowlist"
                )
            if "{env:" in value:
                raise ValueError(
                    f"{kind} URL template {value!r} is only partially templated; "
                    "use a whole-value {env:NAME} reference"
                )
            if kind == "provider":
                self._validate_provider_url(value)
            resolved.append(value)
        return resolved

    @staticmethod
    def _validate_provider_url(value: str) -> None:
        """Keep provider credentials off cleartext remote connections."""
        parsed = urlparse(value)
        hostname = (parsed.hostname or "").lower().rstrip(".")
        if not hostname:
            raise ValueError(f"Provider URLs must include a hostname, got {value!r}")
        loopback = hostname == "localhost"
        if hostname:
            try:
                loopback = loopback or ipaddress.ip_address(hostname).is_loopback
            except ValueError:
                pass
        if parsed.scheme == "https" or (parsed.scheme == "http" and loopback):
            return
        raise ValueError(
            "Provider URLs must use HTTPS; only explicit loopback HTTP endpoints "
            f"are allowed, got {value!r}"
        )

    # ------------------------------------------------------------------
    # Installation
    # ------------------------------------------------------------------

    def install_spec(self) -> AgentInstallSpec:
        version = self._version
        if version and not re.fullmatch(r"[0-9A-Za-z][0-9A-Za-z._+-]*", version):
            raise ValueError("OpenCode V2 version contains unsupported characters")
        if not version and self._opencode_v2_checksums:
            raise ValueError(
                "OpenCode V2 checksums require an explicit version in agent config"
            )
        version_command = f"version={shlex.quote(version)}; " if version else ""
        if not version:
            latest_version_script = (
                'import json,sys; print(json.load(sys.stdin)["dist-tags"]["latest"])'
            )
            version_command = (
                'metadata_url="https://registry.npmjs.org/'
                f'@opencode%2fcli-${{target}}"; version="$(curl -fsSL '
                f'"${{metadata_url}}" | python3 -c '
                f'{shlex.quote(latest_version_script)})"; '
            )
        checksum_cases = "".join(
            f"{target}) expected_sha={shlex.quote(digest.lower())} ;; "
            for target, digest in sorted(self._opencode_v2_checksums.items())
        )
        checksum_command = ""
        if checksum_cases:
            checksum_command = (
                f'case "$target" in {checksum_cases}*) '
                'echo "No configured OpenCode V2 checksum for $target" >&2; exit 1 ;; '
                "esac; "
                'printf \'%s  %s\\n\' "$expected_sha" "$archive" | sha256sum -c -; '
            )
        return AgentInstallSpec(
            agent_name=self.name(),
            version=version,
            cache_key=(
                f"opencode-v2-unpinned-{self._unpinned_install_token}"
                if not version
                else None
            ),
            steps=[
                InstallStep(
                    user="root",
                    env={"DEBIAN_FRONTEND": "noninteractive"},
                    run="apt-get update && apt-get install -y curl python3",
                ),
                InstallStep(
                    user="root",
                    run=(
                        "set -euo pipefail; "
                        "install -d -m 755 /installed-agent; "
                        'machine="$(uname -m)"; '
                        'case "$machine" in '
                        "x86_64|amd64) target=linux-x64 ;; "
                        "aarch64|arm64) target=linux-arm64 ;; "
                        '*) echo "Unsupported OpenCode V2 architecture: $machine" >&2; '
                        "exit 1 ;; esac; "
                        f"{version_command}"
                        'package_dir="$(mktemp -d /tmp/opencode-v2-pkg.XXXXXX)"; '
                        "trap 'rm -rf -- \"$package_dir\"' EXIT; "
                        'archive="$package_dir/cli.tgz"; '
                        'archive_url="https://registry.npmjs.org/'
                        '@opencode%2fcli-${target}/-/cli-${target}-${version}.tgz"; '
                        'curl -fsSL -o "$archive" "$archive_url"; '
                        f"{checksum_command}"
                        'tar -xzf "$archive" -C "$package_dir"; '
                        f'install -m 755 "$package_dir/package/bin/opencode" '
                        f"{self._REMOTE_BINARY.as_posix()}; "
                        f'installed_version="$({self._REMOTE_BINARY.as_posix()} '
                        '--version)"; '
                        'if [ "$installed_version" != "opencode v$version" ] '
                        '&& [ "$installed_version" != "$version" ]; then '
                        'echo "OpenCode version mismatch: expected $version, '
                        'got $installed_version" >&2; exit 1; fi; '
                        'printf "%s\\n" "$installed_version"'
                    ),
                ),
            ],
            verification_command=self.get_version_command(),
        )

    async def setup(self, environment: BaseEnvironment) -> None:
        await super().setup(environment)
        if self._version:
            result = await self.exec_as_agent(
                environment, command=self.get_version_command() or ""
            )
            actual = self.parse_version(result.stdout)
            if actual != self._version:
                raise RuntimeError(
                    f"OpenCode version mismatch: expected {self._version}, got {actual}"
                )
        runner = Path(__file__).with_name(_RUNNER_FILENAME)
        await environment.upload_file(runner, self._RUNNER_PATH)
        await self.exec_as_root(
            environment, command=f"chmod a+r {shlex.quote(self._RUNNER_PATH)}"
        )

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    @with_prompt_template
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        self._instruction = instruction
        runtime_config = self._build_runtime_config(include_mcp=True)
        self._log_redacted_env_keys.update(_env_template_values(runtime_config))
        # Standard trial creation builds the policy before start; repeat the
        # resolution here so direct adapter use cannot accept a URL template
        # that the filtered-egress policy would be unable to resolve.
        self.network_allowlist()

        # Forward ambient credentials for every provider that the effective
        # unrestricted config can select, not just the root benchmark provider.
        env = self.build_process_env()
        for provider_id in self._effective_provider_ids(runtime_config):
            for key in _PROVIDER_ENV_KEYS.get(provider_id, ()):
                if value := self._get_env(key):
                    env[key] = value

        # The generated config resolves `{env:NAME}` templates against the
        # process environment, so every referenced variable must be present.
        for name, placeholder in _env_template_values(runtime_config).items():
            value = self._get_env(name)
            if value is not None:
                # build_process_env deliberately excludes ambient host values;
                # explicitly forward every value referenced by the config.
                env[name] = value
            elif name not in environment.persistent_env:
                raise ValueError(
                    f"opencode_v2_config references {placeholder} but {name} is not set"
                )

        config = runtime_config
        config_json = json.dumps(config, indent=2)
        # Every trial gets private global/config/data/state directories. The
        # task workdir remains the environment's actual cwd so OpenCode edits
        # the repository supplied by the harness rather than a scratch path.
        session_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", environment.session_id)
        remote_home = Path(f"/tmp/opencode-v2-home-{session_id}")
        config_path = remote_home / "config" / "opencode" / "opencode.json"
        try:
            pwd_result = await self.exec_as_agent(environment, command="pwd", env=env)
            remote_workdir = Path(
                (pwd_result.stdout or "").strip() or self._REMOTE_WORKDIR.as_posix()
            )
        except Exception:
            # Preserve the normal setup failure semantics; the fallback only
            # applies to minimal test environments without a meaningful cwd.
            remote_workdir = self._REMOTE_WORKDIR
        remote_workdir_text = remote_workdir.as_posix()
        env["HOME"] = remote_home.as_posix()
        env["XDG_CONFIG_HOME"] = (remote_home / "config").as_posix()
        env["XDG_DATA_HOME"] = (remote_home / "data").as_posix()
        env["XDG_STATE_HOME"] = (remote_home / "state").as_posix()
        env["OPENCODE_CONFIG_DIR"] = (remote_home / "config" / "opencode").as_posix()
        env["OPENCODE_CONFIG"] = config_path.as_posix()
        env["OPENCODE_CONFIG_PROJECT_DISABLE"] = "1"
        env["OPENCODE_DISABLE_MODELS_FETCH"] = "1"
        env["OPENCODE_DISABLE_AUTOUPDATE"] = "1"
        env["PWD"] = remote_workdir_text
        # The copied runner generates its fresh server password internally so
        # Pier never passes it through BaseInstalledAgent._exec, whose debug
        # metadata intentionally records process environments.
        # Explicitly shadow task-level persistent values. The runner creates
        # its own password for the child server/CLI; an empty value here keeps
        # a task password out of setup and runner processes.
        for key in ("OPENCODE_PASSWORD", "OPENCODE_SERVER_PASSWORD"):
            if key in environment.persistent_env:
                env[key] = ""
        for key in ("NO_PROXY", "no_proxy"):
            current = env.get(key, "")
            entries = [item.strip() for item in current.split(",") if item.strip()]
            for host in ("127.0.0.1", "localhost", "::1"):
                if host not in entries:
                    entries.append(host)
            env[key] = ",".join(entries)

        # Keep the repository's own AGENTS.md discovery working. Project config
        # disabling suppresses opencode.json discovery, not instruction files.
        setup_command = (
            f"mkdir -p {shlex.quote(config_path.parent.as_posix())} "
            f"{shlex.quote(remote_workdir_text)} "
            f"{self._RUNNER_OUTPUT.parent.as_posix()}"
        )

        skills_command = self._build_register_skills_command()
        if skills_command:
            setup_command += f"\n{skills_command}"

        mcp_servers = self.mcp_servers
        if mcp_servers:
            mcp_names = ", ".join(server.name for server in mcp_servers)
            self.logger.debug(
                "OpenCode V2 MCP servers are registered through opencode.json: %s",
                mcp_names,
            )

        try:
            await self.exec_as_agent(environment, command=setup_command, env=env)
            # Transfer the JSON through the environment file API instead of
            # embedding credentials in a command that BaseInstalledAgent logs.
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                prefix="pier-opencode-v2-",
                suffix=".json",
                delete=False,
            ) as config_file:
                config_file.write(config_json)
                local_config = Path(config_file.name)
            try:
                await environment.upload_file(local_config, config_path.as_posix())
            finally:
                local_config.unlink(missing_ok=True)

            # The runner owns the server; the CLI only talks to it via --server.
            # A non-zero runner exit must fail the run and all raw artifacts
            # are downloaded in the finally block below.
            runner_log_dir = self._RUNNER_LOG_DIR.as_posix()
            model_spec = self.model_name
            run_command = (
                f"mkdir -p {shlex.quote(self._OUTPUT_FILENAME.parent.as_posix())}; "
                f"python3 {self._RUNNER_PATH} "
                f"--instruction-file {self._INSTRUCTION_PATH.as_posix()} "
                f"--logs-dir {runner_log_dir} "
                f"--work-dir {shlex.quote(remote_workdir_text)} "
                f"--binary {self._REMOTE_BINARY.as_posix()} "
                f"--config-file {shlex.quote(config_path.as_posix())} "
                + (
                    f"--model {shlex.quote(model_spec.split('#', 1)[0])} "
                    if model_spec
                    else ""
                )
                + (
                    f"--variant {shlex.quote(variant)} "
                    if (variant := self._resolved_variant())
                    else ""
                )
                + ("--restrict-model " if self._restrict_model else "")
                + "--title pier-benchmark "
                f"2>&1 </dev/null | stdbuf -oL tee "
                f"{self._OUTPUT_FILENAME.as_posix()}"
            )

            instruction_path = self._INSTRUCTION_PATH
            escaped_instruction = shlex.quote(instruction)
            write_instruction = (
                f"mkdir -p {instruction_path.parent.as_posix()} && "
                f"printf '%s' {escaped_instruction} > {instruction_path.as_posix()}"
            )
            await self.exec_as_agent(environment, command=write_instruction, env=env)
            await self.exec_as_agent(
                environment,
                command=(
                    f"set -o pipefail; {run_command}; "
                    f"rc=$?; "
                    f"mkdir -p {runner_log_dir}; exit $rc"
                ),
                env=env,
            )
        finally:
            # Preserve the primary execution error even when artifact
            # collection fails; the runner keeps its JSONL evidence on disk
            # precisely so a failed run can still be graded.
            try:
                await self._collect_runner_artifacts(environment)
            except Exception:
                self.logger.exception("OpenCode V2 artifact collection failed")

    def _build_register_skills_command(self) -> str | None:
        if not self.skills_dir:
            return None
        return (
            'mkdir -p "$OPENCODE_CONFIG_DIR/skills" && '
            f"cp -r {shlex.quote(self.skills_dir)}/* "
            '"$OPENCODE_CONFIG_DIR/skills/" 2>/dev/null || true'
        )

    async def _collect_runner_artifacts(self, environment: BaseEnvironment) -> None:
        """Best-effort download of the runner's JSONL artifacts."""
        local_dir = self.logs_dir / "opencode-v2"
        try:
            await environment.download_dir(self._RUNNER_LOG_DIR.as_posix(), local_dir)
            return
        except Exception:
            # Minimal/test environments may only implement download_file. Keep
            # the three post-run contract files available there too.
            self.logger.debug("Could not download OpenCode V2 artifact directory")
        for remote, local in (
            (self._RUNNER_OUTPUT, self.logs_dir / "opencode-v2" / "runner-result.json"),
            (
                self._SESSIONS_OUTPUT,
                self.logs_dir / "opencode-v2" / "opencode-v2-sessions.jsonl",
            ),
            (
                self._CLI_EVENTS_OUTPUT,
                self.logs_dir / "opencode-v2" / "opencode-v2-cli-events.jsonl",
            ),
        ):
            try:
                await environment.download_file(remote.as_posix(), local)
            except Exception:
                self.logger.debug("Could not download %s", remote)

    # ------------------------------------------------------------------
    # Conversion to ATIF
    # ------------------------------------------------------------------

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        records: list[dict[str, Any]] = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                records.append({"type": "cli-stdout", "raw": line})
                continue
            if isinstance(record, dict):
                records.append(record)
        return records

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            value = json.loads(path.read_text())
        except json.JSONDecodeError:
            return None
        return value if isinstance(value, dict) else None

    @staticmethod
    def _usage_fields(usage: dict[str, Any] | None) -> dict[str, int]:
        """Normalize one V2 usage record into plain token counters.

        V2 zeroes absent counters, so zero and missing are distinct inputs here
        only before this point — the distinction lives in the caller.
        """
        usage = usage or {}
        cache = usage.get("cache") if isinstance(usage.get("cache"), dict) else {}

        def value(mapping: dict[str, Any], key: str) -> int:
            raw = mapping.get(key)
            if isinstance(raw, bool) or not isinstance(raw, (int, float)) or raw < 0:
                return 0
            return int(raw)

        return {
            "input": value(usage, "input"),
            "output": value(usage, "output"),
            "reasoning": value(usage, "reasoning"),
            "cache_read": value(cache, "read"),
            "cache_write": value(cache, "write"),
        }

    @staticmethod
    def _usage_record_complete(usage: Any) -> bool:
        """Whether all normalized V2 categories are explicitly reported."""
        if not isinstance(usage, dict):
            return False
        cache = usage.get("cache")
        if not isinstance(cache, dict):
            return False
        values = [
            usage.get("input"),
            usage.get("output"),
            usage.get("reasoning"),
            cache.get("read"),
            cache.get("write"),
        ]
        return all(
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and value >= 0
            for value in values
        )

    @staticmethod
    def _partial_usage_fields(usage: Any) -> dict[str, int | None]:
        """Preserve reported categories without turning absent values into zero."""
        source = usage if isinstance(usage, dict) else {}
        cache = source.get("cache") if isinstance(source.get("cache"), dict) else {}

        def value(mapping: dict[str, Any], key: str) -> int | None:
            raw = mapping.get(key)
            if isinstance(raw, bool) or not isinstance(raw, (int, float)) or raw < 0:
                return None
            return int(raw)

        return {
            "input": value(source, "input"),
            "output": value(source, "output"),
            "reasoning": value(source, "reasoning"),
            "cache_read": value(cache, "read"),
            "cache_write": value(cache, "write"),
        }

    def _convert_session_to_trajectory(
        self,
        inspection: dict[str, Any],
        parent_ids: dict[str, str | None],
    ) -> Trajectory | None:
        """One ATIF trajectory for one session's recorded messages."""
        session = inspection.get("session") or {}
        session_id = session.get("id") or "unknown"
        raw_messages = inspection.get("messages")
        messages = self._dedupe_messages(
            raw_messages if isinstance(raw_messages, list) else []
        )
        malformed_messages = bool(inspection.get("_malformed_messages")) or (
            not isinstance(raw_messages, list)
            or any(not isinstance(message, dict) for message in raw_messages)
        )

        steps: list[Step] = []
        totals = {
            "prompt": 0,
            "completion": 0,
            "cached": 0,
            "cache_write": 0,
            "reasoning": 0,
            "cost": 0.0,
        }
        summarization_count = 0
        incomplete = malformed_messages
        usage_seen = False
        cost_complete = True
        failed_attempts = 0
        model_provenance_complete = True
        model_mismatches: list[str] = []
        outcome: str | None = (
            str(session.get("outcome")) if session.get("outcome") else None
        )

        for message in messages:
            message_type = message.get("type")

            if message_type == "assistant":
                usage_seen = usage_seen or isinstance(message.get("tokens"), dict)
                if message.get("finish") == "error":
                    failed_attempts += 1
                if self._restrict_model and self.model_name:
                    actual_model = self._assistant_model_name(message)
                    model_status = self._model_identity_status(
                        actual_model, self._model_selection()
                    )
                    if model_status == "unknown":
                        model_provenance_complete = False
                        model_mismatches.append(
                            ("missing" if actual_model is None else "missing_variant")
                            + f":{message.get('id', 'unknown')}"
                        )
                    elif model_status == "mismatch":
                        model_provenance_complete = False
                        model_mismatches.append(
                            f"{message.get('id', 'unknown')}:{actual_model}"
                        )
                step, contribution, message_incomplete = self._assistant_step(message)
                incomplete = incomplete or message_incomplete
            elif message_type == "compaction":
                usage_seen = usage_seen or isinstance(message.get("tokens"), dict)
                if str(message.get("status") or "") in {"failed", "error"}:
                    failed_attempts += 1
                if self._restrict_model and self.model_name:
                    actual_model = self._compaction_model_name(message)
                    model_status = self._model_identity_status(
                        actual_model, self._model_selection()
                    )
                    if model_status == "unknown":
                        model_provenance_complete = False
                        model_mismatches.append(
                            ("missing" if actual_model is None else "missing_variant")
                            + f":{message.get('id', 'unknown')}"
                        )
                    elif model_status == "mismatch":
                        model_provenance_complete = False
                        model_mismatches.append(
                            f"{message.get('id', 'unknown')}:{actual_model}"
                        )
                step, contribution, message_incomplete = self._compaction_step(message)
                summarization_count += 1
                incomplete = incomplete or message_incomplete
            elif message_type == "idle":
                if message.get("outcome"):
                    outcome = str(message["outcome"])
                continue
            elif message_type == "user":
                text = str(message.get("text") or "")
                if not text:
                    continue
                step = Step(
                    step_id=len(steps) + 1,
                    timestamp=_iso((message.get("time") or {}).get("created")),
                    source="user",
                    message=text,
                )
                contribution = None
            else:
                continue

            if contribution:
                for key in (
                    "prompt",
                    "completion",
                    "cached",
                    "cache_write",
                    "reasoning",
                ):
                    totals[key] += contribution[key]
                if contribution["cost"] is None:
                    cost_complete = False
                else:
                    totals["cost"] += contribution["cost"]
            steps.append(step)
            steps[-1].step_id = len(steps)

        if not any(step.source != "user" for step in steps):
            # Preserve an empty/partial discovered session as trajectory
            # evidence. It must not disappear from the tree or turn a missing
            # child into a plausible complete aggregate.
            incomplete = True
            if not steps:
                steps.append(
                    Step(
                        step_id=1,
                        source="system",
                        message=(
                            "OpenCode session was discovered without collectible "
                            "message records."
                        ),
                        extra={"collection_gap": True},
                    )
                )

        final_extra: dict[str, Any] = {}
        if usage_seen:
            final_extra["total_cache_write_input_tokens"] = totals["cache_write"]
            final_extra["total_reasoning_tokens"] = totals["reasoning"]
        if not cost_complete:
            final_extra["cost_complete"] = False
        if summarization_count:
            final_extra["compacted"] = True
        if failed_attempts:
            final_extra["failed_attempts"] = failed_attempts
        if outcome:
            final_extra["outcome"] = outcome
        if inspection.get("active"):
            # The session was still running at collection time: its usage can
            # only grow, so the totals below are a lower bound.
            final_extra["interrupted"] = True
        if incomplete:
            final_extra["unfinished"] = True

        final_metrics = FinalMetrics(
            total_prompt_tokens=totals["prompt"] if usage_seen else None,
            total_completion_tokens=totals["completion"] if usage_seen else None,
            total_cached_tokens=totals["cached"] if usage_seen else None,
            total_cost_usd=totals["cost"] if usage_seen and cost_complete else None,
            total_steps=len(steps),
            extra=extra_with_context_metrics(
                final_extra or None,
                peak_context_tokens=peak_context_tokens_from_steps(steps),
                summarization_count=summarization_count
                if summarization_count
                else None,
            ),
        )
        if incomplete:
            final_metrics.extra = dict(final_metrics.extra or {})
            final_metrics.extra["metrics_complete"] = False
            final_metrics.total_prompt_tokens = None
            final_metrics.total_completion_tokens = None
            final_metrics.total_cached_tokens = None
            final_metrics.total_cost_usd = None
        if self._restrict_model and self.model_name and not model_provenance_complete:
            final_metrics.extra = dict(final_metrics.extra or {})
            final_metrics.extra["model_provenance_complete"] = False
            final_metrics.extra["model_mismatches"] = model_mismatches

        parent_id = parent_ids.get(str(session_id))
        is_subagent = bool(parent_id)
        trajectory_extra: dict[str, Any] = {"session_id": session_id}
        if parent_id:
            trajectory_extra["parent_session_id"] = parent_id
        if is_subagent:
            trajectory_extra["is_subagent"] = True

        return Trajectory(
            schema_version="ATIF-v1.7",
            session_id=session_id,
            trajectory_id=session_id,
            agent=Agent(
                name=self.name(),
                version=self.version() or "unknown",
                model_name=self.model_name,
            ),
            steps=steps,
            final_metrics=final_metrics,
            extra=trajectory_extra or None,
        )

    def _assistant_step(
        self, message: dict[str, Any]
    ) -> tuple[Step, dict[str, float] | None, bool]:
        """One ATIF agent step from one assistant message.

        Returns ``(step, usage_contribution, incomplete)``. A message whose
        usage record is absent is incomplete (no totals can be trusted); a
        ``length``/``content-filter``/``error`` finish that still completed
        keeps its authoritative usage but marks the reply as truncated.
        """
        tokens = message.get("tokens")
        # A usage record that is present at all counts as present — a
        # reported zero is data, an absent record is missing.
        usage_present = isinstance(tokens, dict)
        usage_complete = self._usage_record_complete(tokens)
        usage = tokens if isinstance(tokens, dict) else {}
        counts = self._usage_fields(usage)
        # V2 reports `input` net of cache: the full prompt is the sum. Its
        # normalized `output` excludes hidden reasoning; ATIF completion is
        # the billable output plus reasoning, with reasoning also retained in
        # the extra metrics for reconciliation.
        prompt_tokens = counts["input"] + counts["cache_read"] + counts["cache_write"]
        completion_tokens = counts["output"] + counts["reasoning"]

        finished = (message.get("time") or {}).get("completed") is not None
        finish = message.get("finish")
        # An unfinished message or an errored one has no trustworthy totals;
        # a truncated (`length`) or filtered (`content-filter`) reply that
        # completed still does. A recorded all-zero usage record is data (the
        # provider reported zeros), not a missing record.
        # A persisted provider error can still carry a billable token record;
        # count that failed attempt. Only a missing usage record or unfinished
        # message withholds aggregate totals.
        incomplete = not finished or not usage_complete

        cost: float | None = None
        if isinstance(message.get("cost"), (int, float)) and not isinstance(
            message.get("cost"), bool
        ):
            cost = float(message["cost"])

        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        observation_results: list[ObservationResult] = []

        for part in message.get("content") or []:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type == "text":
                if part.get("text"):
                    text_parts.append(str(part["text"]))
            elif part_type == "reasoning":
                if part.get("text"):
                    reasoning_parts.append(str(part["text"]))
            elif part_type == "tool":
                state = part.get("state") or {}
                arguments = state.get("input") or {}
                if not isinstance(arguments, dict):
                    arguments = {"value": arguments} if arguments else {}
                tool_call = ToolCall(
                    tool_call_id=str(part.get("id") or ""),
                    function_name=str(part.get("name") or ""),
                    arguments=arguments,
                )
                tool_calls.append(tool_call)
                status = state.get("status")
                if status in ("completed", "error"):
                    output = ""
                    for content in state.get("content") or []:
                        if isinstance(content, dict) and content.get("type") == "text":
                            output += str(content.get("text") or "")
                    observation_results.append(
                        ObservationResult(
                            source_call_id=tool_call.tool_call_id or None,
                            content=output,
                            extra={"is_error": status == "error"}
                            if status == "error"
                            else None,
                        )
                    )

        metrics: Metrics | None = None
        if usage_complete:
            metrics = Metrics(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cached_tokens=counts["cache_read"],
                cost_usd=cost,
                extra={
                    "reasoning_tokens": counts["reasoning"],
                    "cache_write_tokens": counts["cache_write"],
                },
            )

        step_kwargs: dict[str, Any] = {
            "step_id": 1,  # renumbered by the caller
            "timestamp": _iso((message.get("time") or {}).get("created")),
            "source": "agent",
            "message": "\n".join(text_parts),
            "model_name": self._assistant_model_name(message),
            "llm_call_count": 1,
        }
        if reasoning_parts:
            step_kwargs["reasoning_content"] = "\n\n".join(reasoning_parts)
        if tool_calls:
            step_kwargs["tool_calls"] = tool_calls
        if observation_results:
            step_kwargs["observation"] = Observation(results=observation_results)
        if metrics:
            step_kwargs["metrics"] = metrics
        # Preserve partial/unfinished/error states without failing the run: a
        # `length` finish is a truncated-but-real reply, `content-filter` a
        # refused one, `error` a failed turn.
        step_extra: dict[str, Any] = {}
        if finish:
            step_extra["finish"] = str(finish)
        if usage_present and not usage_complete:
            step_extra["usage_partial"] = self._partial_usage_fields(tokens)
        if step_extra:
            step_kwargs["extra"] = step_extra

        return (
            Step(**step_kwargs),
            (
                {
                    "prompt": prompt_tokens,
                    "completion": completion_tokens,
                    "cached": counts["cache_read"],
                    "cache_write": counts["cache_write"],
                    "reasoning": counts["reasoning"],
                    "cost": cost,
                }
                if usage_complete
                else None
            ),
            incomplete,
        )

    def _assistant_model_name(self, message: dict[str, Any]) -> str | None:
        model = message.get("model") or {}
        if isinstance(model, dict) and model.get("providerID") and model.get("id"):
            name = f"{model['providerID']}/{model['id']}"
            variant = model.get("variant") or message.get("variant")
            return f"{name}#{variant}" if variant else name
        # Missing model provenance is unknown, not an implicit match with the
        # configured benchmark model.
        return None

    def _compaction_model_name(self, message: dict[str, Any]) -> str | None:
        model = message.get("model") or {}
        if (
            not isinstance(model, dict)
            or not model.get("providerID")
            or not model.get("id")
        ):
            return None
        name = f"{model['providerID']}/{model['id']}"
        variant = model.get("variant") or message.get("variant")
        return f"{name}#{variant}" if variant else name

    def _compaction_step(
        self, message: dict[str, Any]
    ) -> tuple[Step, dict[str, float] | None, bool]:
        """One ATIF system step recording one context compaction.

        Only a *completed* compaction's summary is authoritative. A compaction
        record is emitted once per compaction; the runner's stable snapshots
        already deduplicate, so a second record with the same id means the
        first one was superseded and only the final state is converted (the
        caller's ledger replacement handles this).
        """
        status = str(message.get("status") or "completed")
        tokens = message.get("tokens")
        # A usage record that is present at all counts as present — a
        # reported zero is data, an absent record is missing.
        duplicate_of = message.get("_usage_duplicate_of")
        usage_present = isinstance(tokens, dict)
        usage_complete = self._usage_record_complete(tokens)
        counts = self._usage_fields(tokens if isinstance(tokens, dict) else {})
        prompt_tokens = counts["input"] + counts["cache_read"] + counts["cache_write"]
        completion_tokens = counts["output"] + counts["reasoning"]
        cost: float | None = None
        if isinstance(message.get("cost"), (int, float)) and not isinstance(
            message.get("cost"), bool
        ):
            cost = float(message["cost"])

        compaction_extra: dict[str, Any] = {
            "reason": message.get("reason"),
            "status": status,
        }
        model = message.get("model")
        if isinstance(model, dict) and model.get("providerID") and model.get("id"):
            compaction_extra["model"] = f"{model['providerID']}/{model['id']}"
            if model.get("variant"):
                compaction_extra["variant"] = str(model["variant"])
        # A completed compaction carries usage; a failed one carries an error
        # and (in V2) a token record for what it managed to spend. A completed
        # compaction without any usage record cannot be priced or counted, so
        # it withholds the session's totals too.
        # Terminal failed compactions may have their own persisted billable
        # attempt. Count those records; only an active/nonterminal or missing
        # usage record makes accounting incomplete.
        incomplete = status not in {"completed", "failed", "error"} or not (
            usage_complete or duplicate_of
        )
        if message.get("error"):
            compaction_extra["error"] = message["error"]
            incomplete = incomplete or not (usage_complete or duplicate_of)
        if duplicate_of:
            compaction_extra["usage_duplicate_of"] = duplicate_of

        # ATIF restricts `metrics` to agent steps, so the summarization call's
        # usage is reported as plain numbers, but it is still folded into the
        # FinalMetrics below because V2 bills it to the run.
        if usage_complete:
            compaction_extra["usage"] = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cached_tokens": counts["cache_read"],
                "cache_write_tokens": counts["cache_write"],
                "reasoning_tokens": counts["reasoning"],
                "cost_usd": cost,
            }
        elif usage_present:
            compaction_extra["usage_partial"] = self._partial_usage_fields(tokens)

        step = Step(
            step_id=1,
            timestamp=_iso((message.get("time") or {}).get("created")),
            source="system",
            message=str(message.get("summary") or ""),
            extra={"compaction": compaction_extra},
        )
        return (
            step,
            None
            if duplicate_of or not usage_complete
            else {
                "prompt": prompt_tokens,
                "completion": completion_tokens,
                "cached": counts["cache_read"],
                "cache_write": counts["cache_write"],
                "reasoning": counts["reasoning"],
                "cost": cost,
            },
            incomplete,
        )

    # -- ledger ----------------------------------------------------------------

    @staticmethod
    def _assistant_references(record: dict[str, Any]) -> set[str]:
        """Return explicit assistant-record references carried by a wrapper.

        A compaction/event wrapper may repeat the usage of an assistant record
        while adding summary metadata. Only an explicit record reference is
        safe evidence for dropping that wrapper; equal token counts are not.
        """
        reference_keys = {
            "assistantmessageid",
            "assistantid",
            "sourceassistantmessageid",
            "sourcemessageid",
            "sourceid",
        }
        found: set[str] = set()

        def visit(value: Any) -> None:
            if isinstance(value, dict):
                for key, item in value.items():
                    normalized = str(key).replace("_", "").lower()
                    if normalized in reference_keys and isinstance(item, str):
                        found.add(item)
                    visit(item)
            elif isinstance(value, list):
                for item in value:
                    visit(item)

        visit(record)
        return found

    @staticmethod
    def _dedupe_messages(messages: list[Any]) -> list[dict[str, Any]]:
        """Replace re-observed records by (sessionID, type, id) ledger key.

        The runner re-snapshots sessions until the server goes quiet, so a
        resumed session can report the same message twice. The LAST snapshot
        of a record wins (it is the most recent observation), and order
        follows first appearance.
        """
        replaced: dict[tuple[Any, Any, Any], dict[str, Any]] = {}
        order: list[tuple[Any, Any, Any]] = []
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                continue
            if not message.get("id"):
                key = ("__missing__", message.get("type"), index)
                replaced[key] = message
                order.append(key)
                continue
            key = (
                message.get("sessionID"),
                message.get("type"),
                message.get("id"),
            )
            if key not in replaced:
                order.append(key)
            replaced[key] = message
        deduped = [replaced[key] for key in order]
        assistant_ids = {
            str(message.get("id"))
            for message in deduped
            if message.get("type") == "assistant" and message.get("id")
        }
        # A compaction wrapper that explicitly points at an assistant record is
        # still useful trajectory evidence, but its repeated usage is not a
        # second paid call. Preserve the wrapper and annotate the reference;
        # never infer duplicates from equal token numbers.
        result: list[dict[str, Any]] = []
        for message in deduped:
            references = (
                assistant_ids.intersection(OpenCodeV2._assistant_references(message))
                if message.get("type") == "compaction"
                else set()
            )
            if references:
                message = copy.deepcopy(message)
                message.pop("tokens", None)
                message.pop("cost", None)
                message["_usage_duplicate_of"] = sorted(references)[0]
            result.append(message)
        return result

    @staticmethod
    def _message_has_malformed_nested_fields(message: dict[str, Any]) -> bool:
        """Reject nested shapes that conversion accesses as mappings."""
        message_type = message.get("type")
        if message_type in {"assistant", "compaction", "user"}:
            if "time" in message and not isinstance(message["time"], dict):
                return True

        if message_type == "assistant":
            content = message.get("content")
            if content is not None and not isinstance(content, list):
                return True
            for part in content or []:
                if not isinstance(part, dict):
                    return True
                if part.get("type") == "tool":
                    state = part.get("state")
                    if "state" in part and not isinstance(state, dict):
                        return True
                    if isinstance(state, dict) and "content" in state:
                        state_content = state["content"]
                        if not isinstance(state_content, list) or any(
                            not isinstance(content, dict) for content in state_content
                        ):
                            return True
        elif message_type == "tool":
            state = message.get("state")
            if "state" in message and not isinstance(state, dict):
                return True
        return False

    # -- tree assembly -------------------------------------------------------

    def _convert_inspections_to_trajectories(
        self, inspections: list[dict[str, Any]]
    ) -> list[Trajectory]:
        parent_ids: dict[str, str | None] = {}
        for inspection in inspections:
            session = inspection.get("session") or {}
            session_id = session.get("id")
            if session_id:
                parent_ids[str(session_id)] = session.get("parentID")

        trajectories: list[Trajectory] = []
        for inspection in inspections:
            inspection = copy.deepcopy(inspection)
            session_id = str((inspection.get("session") or {}).get("id") or "")
            raw_messages = inspection.get("messages")
            malformed_messages = not isinstance(raw_messages, list)
            valid_messages: list[dict[str, Any]] = []
            for message in raw_messages if isinstance(raw_messages, list) else []:
                if not isinstance(
                    message, dict
                ) or self._message_has_malformed_nested_fields(message):
                    malformed_messages = True
                    continue
                if session_id:
                    message.setdefault("sessionID", session_id)
                valid_messages.append(message)
            inspection["_malformed_messages"] = malformed_messages
            inspection["messages"] = self._dedupe_messages(valid_messages)
            trajectory = self._convert_session_to_trajectory(inspection, parent_ids)
            if trajectory is not None:
                trajectories.append(trajectory)
        return trajectories

    def _select_root(
        self, trajectories: list[Trajectory], recorded_root_id: str | None = None
    ) -> Trajectory | None:
        """Identify the session that owns the run.

        A root names no parent (a child always names its parent) — or, when
        every session names a parent, the session nobody names as its child
        chain head: pick the parent-side root by graph reachability. Zero or
        multiple roots is ambiguous, and picking one anyway would attribute
        the whole tree to the wrong session, so this refuses to guess.
        """
        if not trajectories:
            return None
        if recorded_root_id:
            matches = [
                trajectory
                for trajectory in trajectories
                if trajectory.trajectory_id == recorded_root_id
            ]
            if len(matches) == 1:
                return matches[0]
            self.logger.error(
                "Recorded OpenCode V2 root %s is absent from collected trajectories",
                recorded_root_id,
            )
            return None
        roots = [
            trajectory
            for trajectory in trajectories
            if not (trajectory.extra or {}).get("parent_session_id")
        ]
        # Every session names a parent: a cycle or an uncollected parent.
        # Fall back to the session no other session names as a child-head —
        # for a single orphan this resolves it; for a true cycle refuse.
        if not roots:
            child_heads = {
                (trajectory.extra or {}).get("parent_session_id")
                for trajectory in trajectories
                if (trajectory.extra or {}).get("parent_session_id")
                in {t.trajectory_id for t in trajectories}
            }
            roots = [
                trajectory
                for trajectory in trajectories
                if trajectory.trajectory_id not in child_heads
            ]
        if len(roots) == 1:
            return roots[0]

        self.logger.error(
            "Ambiguous OpenCode V2 root session: %d candidates (%s). Refusing to guess.",
            len(roots),
            ", ".join(sorted(str(t.trajectory_id) for t in roots)) or "none",
        )
        return None

    def _embed_descendants(
        self,
        root: Trajectory,
        trajectories: list[Trajectory],
        *,
        subagent_count: int | None = None,
    ) -> Trajectory:
        children_by_parent: dict[str, list[Trajectory]] = {}
        for trajectory in trajectories:
            if trajectory is root:
                continue
            parent_id = (trajectory.extra or {}).get("parent_session_id")
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
            return Trajectory.model_validate(data)

        embedded_root = attach(root)
        return self._with_tree_metrics(
            embedded_root,
            len(trajectories) - 1 if subagent_count is None else subagent_count,
        )

    def _with_tree_metrics(self, root: Trajectory, subagent_count: int) -> Trajectory:
        """Whole-tree totals on the root, with root-only figures in extra.

        Follows Codex: the tree aggregate covers every embedded subagent, so a
        delegating V2 run is comparable with a flat one, while
        ``extra.self_only`` preserves the root's own metrics. When any session
        withholds totals, the aggregate is withheld too.
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
        if len(nodes) == 1 and subagent_count == 0 and not root_incomplete:
            return root

        complete = True
        cost_complete = True
        cache_writes_complete = True
        model_provenance_complete = True
        totals = {
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_cached_tokens": 0,
        }
        total_cost: float | None = None
        peak_context: int | None = None
        summarizations: int | None = None
        cache_writes: int | None = None
        reasoning_tokens: int | None = None

        for node in nodes:
            metrics = node.final_metrics
            if metrics is None:
                complete = False
                cost_complete = False
                continue
            extra = metrics.extra or {}
            if extra.get("metrics_complete") is False:
                complete = False
                cost_complete = False
            if extra.get("model_provenance_complete") is False:
                model_provenance_complete = False
            if extra.get("cost_complete") is False or (
                metrics.total_cost_usd is None
                and any(
                    value is not None
                    for value in (
                        metrics.total_prompt_tokens,
                        metrics.total_completion_tokens,
                        metrics.total_cached_tokens,
                    )
                )
            ):
                cost_complete = False
            for field in (
                "total_prompt_tokens",
                "total_completion_tokens",
                "total_cached_tokens",
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
            node_cache_writes = extra.get("total_cache_write_input_tokens")
            if isinstance(node_cache_writes, int):
                cache_writes = (cache_writes or 0) + node_cache_writes
            else:
                cache_writes_complete = False
            node_reasoning = extra.get("total_reasoning_tokens")
            if isinstance(node_reasoning, int):
                reasoning_tokens = (reasoning_tokens or 0) + node_reasoning

        tree_extra: dict[str, Any] = dict(
            (self_metrics.extra or {}) if self_metrics else {}
        )
        # A root-only success marker must not survive on a whole-tree
        # aggregate. A complete root plus an incomplete child would otherwise
        # publish metrics_complete=true beside tree_metrics_complete=false.
        # Preserve an explicit false from an incomplete root as local evidence.
        if root_incomplete:
            tree_extra["metrics_complete"] = False
        else:
            tree_extra.pop("metrics_complete", None)
        if self_metrics is not None:
            tree_extra["self_only"] = {
                "total_prompt_tokens": self_metrics.total_prompt_tokens,
                "total_completion_tokens": self_metrics.total_completion_tokens,
                "total_cached_tokens": self_metrics.total_cached_tokens,
                "total_cache_write_input_tokens": (self_metrics.extra or {}).get(
                    "total_cache_write_input_tokens"
                ),
                "total_cost_usd": self_metrics.total_cost_usd,
                "total_steps": self_metrics.total_steps,
            }
        tree_extra["subagent_count"] = subagent_count
        tree_extra["tree_metrics_complete"] = complete
        tree_extra["tree_cost_complete"] = cost_complete
        tree_extra["tree_model_provenance_complete"] = model_provenance_complete
        if peak_context is not None:
            tree_extra["peak_context_tokens"] = peak_context
        if summarizations is not None:
            tree_extra["summarization_count"] = summarizations
        tree_extra.pop("total_cache_write_input_tokens", None)
        if complete and cache_writes_complete and cache_writes is not None:
            tree_extra["total_cache_write_input_tokens"] = cache_writes
        tree_extra.pop("total_reasoning_tokens", None)
        if complete and reasoning_tokens is not None:
            tree_extra["total_reasoning_tokens"] = reasoning_tokens

        aggregate_steps = sum(len(node.steps) for node in nodes) or None
        aggregate = (
            {
                # Keep an explicitly reported zero distinct from unknown.  The
                # per-session converter already preserves zero-valued usage;
                # tree aggregation must not turn an all-zero subtree into
                # missing data.
                "total_prompt_tokens": totals["total_prompt_tokens"],
                "total_completion_tokens": totals["total_completion_tokens"],
                "total_cached_tokens": totals["total_cached_tokens"],
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
            "session's own metrics."
        )
        data["notes"] = f"{notes}\n{note}" if notes else note
        return Trajectory.model_validate(data)

    @staticmethod
    def _mark_collection_incomplete(root: Trajectory) -> Trajectory:
        """Withhold aggregates when the runner could not prove full collection."""
        data = root.to_json_dict()
        metrics = data.get("final_metrics") or {}
        extra = dict(metrics.get("extra") or {})
        extra.update(
            {
                "metrics_complete": False,
                "tree_metrics_complete": False,
                "tree_cost_complete": False,
                "collection_incomplete": True,
            }
        )
        metrics.update(
            {
                "total_prompt_tokens": None,
                "total_completion_tokens": None,
                "total_cached_tokens": None,
                "total_cost_usd": None,
                "extra": extra,
            }
        )
        data["final_metrics"] = metrics
        return Trajectory.model_validate(data)

    @staticmethod
    def _populate_context_if_usage_known(
        context: AgentContext, metrics: FinalMetrics | None
    ) -> None:
        """Keep withheld token totals unknown instead of coercing them to zero."""
        if metrics is None or any(
            value is None
            for value in (
                metrics.total_prompt_tokens,
                metrics.total_completion_tokens,
                metrics.total_cached_tokens,
            )
        ):
            return
        populate_context_from_final_metrics(context, metrics)

    # -- entry point -------------------------------------------------------

    def _model_restriction_mismatches(
        self, inspections: list[dict[str, Any]]
    ) -> list[str]:
        if not self._restrict_model or not self.model_name:
            return []
        mismatches: list[str] = []
        for inspection in inspections:
            raw_messages = inspection.get("messages")
            messages = self._dedupe_messages(
                raw_messages if isinstance(raw_messages, list) else []
            )
            for message in messages:
                if not isinstance(message, dict):
                    continue
                message_type = message.get("type")
                if message_type == "assistant":
                    actual = self._assistant_model_name(message)
                elif message_type == "compaction":
                    actual = self._compaction_model_name(message)
                else:
                    continue
                record_id = message.get("id", "unknown")
                # Missing persisted provenance (including a missing variant)
                # is unknown and remains visible through trajectory metadata.
                # Only an observed conflict proves the restriction failed.
                if (
                    self._model_identity_status(actual, self._model_selection())
                    == "mismatch"
                ):
                    mismatches.append(f"{record_id}:{actual}")
        return list(dict.fromkeys(mismatches))

    @staticmethod
    def _model_identity_status(actual: str | None, expected: str) -> str:
        """Return match, mismatch, or unknown for persisted provenance."""
        if actual is None:
            return "unknown"
        expected_model, separator, expected_variant = expected.partition("#")
        actual_model, actual_separator, actual_variant = actual.partition("#")
        if actual_model != expected_model:
            return "mismatch"
        if not separator:
            return "match"
        if not actual_separator or not actual_variant:
            return "unknown"
        return "match" if actual_variant == expected_variant else "mismatch"

    @staticmethod
    def _raise_model_restriction_failure(mismatches: list[str]) -> None:
        if mismatches:
            raise NonZeroAgentExitCodeError(
                "OpenCode V2 model restriction failed: " + "; ".join(mismatches[:10])
            )

    def populate_context_post_run(self, context: AgentContext) -> None:
        """Convert the runner's recorded sessions into ATIF trajectories."""
        runner_result = self._read_json(
            self.logs_dir / "opencode-v2" / "runner-result.json"
        )
        inspections = self._read_jsonl(
            self.logs_dir / "opencode-v2" / "opencode-v2-sessions.jsonl"
        )
        valid_inspections = [
            record
            for record in inspections
            if isinstance(record.get("session"), dict) and record["session"].get("id")
        ]
        if len(valid_inspections) != len(inspections):
            self.logger.warning(
                "Ignoring %d malformed OpenCode V2 session inspection record(s)",
                len(inspections) - len(valid_inspections),
            )
        inspections = valid_inspections
        if not inspections:
            raw_collection_errors = (runner_result or {}).get("collection_errors")
            collection_errors = [
                str(error)
                for error in (
                    raw_collection_errors
                    if isinstance(raw_collection_errors, list)
                    else []
                )
            ]
            if not collection_errors:
                collection_errors = [
                    "runner result manifest missing"
                    if runner_result is None
                    else "no collectible session records"
                ]
            self.logger.error(
                "No OpenCode V2 session inspections found: %s",
                "; ".join(collection_errors),
            )
            recorded_root_id = (
                str(runner_result.get("root_id"))
                if runner_result and runner_result.get("root_id")
                else None
            )
            stub = Trajectory(
                schema_version="ATIF-v1.7",
                session_id=recorded_root_id,
                trajectory_id=recorded_root_id,
                agent=Agent(
                    name=self.name(),
                    version=self.version() or "unknown",
                    model_name=self.model_name,
                ),
                steps=[
                    Step(
                        step_id=1,
                        source="system",
                        message=(
                            "OpenCode completed without collectible session records."
                        ),
                        extra={
                            "collection_gap": True,
                            "collection_errors": collection_errors,
                        },
                    )
                ],
                final_metrics=FinalMetrics(
                    total_steps=1,
                    extra={
                        "metrics_complete": False,
                        "tree_metrics_complete": False,
                        "tree_cost_complete": False,
                        "collection_incomplete": True,
                        "collection_errors": collection_errors,
                    },
                ),
            )
            trajectory_path = self.logs_dir / "trajectory.json"
            try:
                trajectory_path.write_text(format_trajectory_json(stub.to_json_dict()))
            except OSError as exc:
                self.logger.error(
                    "Failed to write incomplete trajectory file %s: %s",
                    trajectory_path,
                    exc,
                )
            self._populate_context_if_usage_known(context, stub.final_metrics)
            return

        restriction_mismatches = self._model_restriction_mismatches(inspections)
        try:
            trajectories = self._convert_inspections_to_trajectories(inspections)
        except Exception:
            self.logger.exception("Failed to convert OpenCode V2 sessions")
            self._raise_model_restriction_failure(restriction_mismatches)
            return

        if not trajectories:
            self._raise_model_restriction_failure(restriction_mismatches)
            return

        recorded_root_id = (
            str(runner_result.get("root_id"))
            if runner_result and runner_result.get("root_id")
            else None
        )
        root = self._select_root(trajectories, recorded_root_id)
        if root is None:
            self._raise_model_restriction_failure(restriction_mismatches)
            return
        discovered_ids = {
            str(item)
            for item in (runner_result or {}).get("discovered_session_ids", [])
            if item
        }
        converted_ids = {
            str(trajectory.trajectory_id)
            for trajectory in trajectories
            if trajectory.trajectory_id
        }
        embedded = self._embed_descendants(
            root,
            trajectories,
            subagent_count=max(0, len(discovered_ids) - 1) if discovered_ids else None,
        )

        def embedded_ids(trajectory: Trajectory) -> set[str]:
            ids = {str(trajectory.trajectory_id)} if trajectory.trajectory_id else set()
            for child in trajectory.subagent_trajectories or []:
                ids.update(embedded_ids(child))
            return ids

        attachment_complete = embedded_ids(embedded) == converted_ids
        # A session dump without the runner manifest cannot prove that all
        # pages, children, and shutdown diagnostics were collected. Preserve
        # the partial trajectory but withhold its aggregate totals.
        if (
            runner_result is None
            or runner_result.get("collection_complete") is not True
            or (discovered_ids and discovered_ids != converted_ids)
            or not attachment_complete
        ):
            embedded = self._mark_collection_incomplete(embedded)

        trajectory_path = self.logs_dir / "trajectory.json"
        try:
            trajectory_path.write_text(format_trajectory_json(embedded.to_json_dict()))
            self.logger.debug(f"Wrote OpenCode V2 trajectory to {trajectory_path}")
        except OSError as exc:
            self.logger.debug(
                f"Failed to write trajectory file {trajectory_path}: {exc}"
            )

        self._populate_context_if_usage_known(context, embedded.final_metrics)
        self._raise_model_restriction_failure(restriction_mismatches)
