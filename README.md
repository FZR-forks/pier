# pier

Pier is a [Harbor](https://www.harborframework.com/docs/tasks)-compatible framework for evaluating coding agents in sandboxed environments. It reads Harbor's task format and runs trials against it.

```bash
pier run -p path/to/task --agent claude-code --env modal
```

## Why pier

Pier is a fork. We wanted a smaller, more opinionated base to build on. On top of Harbor, Pier adds:

- **Installed agents in air-gapped tasks (`allow_internet = false`).** When the agent runs *inside* the sandbox (Claude Code, Codex, etc.), both the install step and the inference call need the network. Pier lets agents declare their install scripts and a network allowlist, which `docker` and `modal` environments honor when setting up the sandbox.
- **Augmented ATIF v1.7.** Strict one step per API turn, strict reasoning vs agent message separation, no fabricated assistant text, `peak_context_tokens`, `summarization_count`, `llm_call_count`, real upstream timestamps.
- **A chat-style trajectory viewer** (`pier view`).
- **`pier critique run`** for inspecting completed trials with a fresh agent in a fresh sandbox.

## What works today

- **Task format:** Harbor-compatible.
- **Environments:** `docker`, `modal`. Per-agent install specs and network allowlists are honored on both, so installed agents work under `allow_internet = false`.
- **Agents:** `nop`, `oracle`, `antigravity-sdk`, `claude-code`, `codex`, `cursor-cli`, `gemini-cli`, `opencode`, `mini-swe-agent`, `pi`. All emit augmented ATIF v1.7.
- **Datasets:** local Harbor-format task directories via `-p` / `--path`.
- **CLI:** `pier run`, `pier job`, `pier view`, `pier critique run`, `pier check` / `pier analyze` (vendored from Harbor)

Pier does not currently resolve or download Harbor registry datasets directly.

## Install

```bash
uv tool install datacurve-pier
# or
pip install datacurve-pier
```

## Run

```bash
export ANTHROPIC_API_KEY=...
pier run -p path/to/task --agent claude-code --env modal --env-file .env
```

Run a local dataset, optionally a deterministic random subset:

```bash
pier run -p path/to/dataset --agent claude-code --env modal
pier run -p path/to/dataset --n-tasks 10 --sample-seed 0
```

To use a Harbor registry dataset, download it with Harbor first, then point Pier at it:

```bash
uv run --directory ~/code/harbor harbor download swebenchpro -o ~/code/pier/datasets
uv run pier run -p datasets/swebenchpro --n-tasks 10 --sample-seed 0
```

Trials land under `jobs/<timestamp_or_name>/<trial_id>/`. See `pier run --help`, `pier job --help`, `pier critique --help`, and `pier view --help` for everything else.

## Private certificate authorities

If your inference gateway is served with an internal CA, point `PIER_EXTRA_CA_CERTS` at a PEM bundle on the host:

```bash
PIER_EXTRA_CA_CERTS=/path/to/internal-roots.pem pier run -p path/to/task --agent codex
```

Pier appends a root install step to every installed agent that registers those certificates with the sandbox's OS trust store, and exports the CA variables each runtime reads: `SSL_CERT_FILE`, `REQUESTS_CA_BUNDLE`, `CURL_CA_BUNDLE` and `GIT_SSL_CAINFO` get a bundle of the distro roots plus yours (these replace the default store), while `NODE_EXTRA_CA_CERTS` and `CODEX_CA_CERTIFICATE` get your certificates alone (these add to it). Codex needs `CODEX_CA_CERTIFICATE` specifically -- the npm package is only a launcher shim around a static Rust binary, so `NODE_EXTRA_CA_CERTS` does nothing for it.

Public roots are preserved, so agent installs that reach npm or GitHub keep working. The step runs after the agent's own install steps, which is where `ca-certificates` comes from on images that don't ship it (bare `debian:12` has no trust store at all); if no public roots are found it falls back to certifi and warns in the build log. The certificates are baked into the image at build time and counted in the install fingerprint, so rotating the CA invalidates cached sandbox images. Works on both `docker` and `modal`, and under `allow_internet = false` -- the egress proxy tunnels HTTPS with `CONNECT` and never re-signs the origin certificate, so the agent validates the gateway's real cert in every network mode.

The variables are injected as defaults; an explicit `agent.env` entry for the same key still wins.

## Agent runtime configuration

Use `agent.model_name` for trial metadata, `agent.env` for runtime env vars, and agent-specific `kwargs` for tool config. Pier's network allowlist also reads URLs out of those configs (Codex `config_toml`, OpenCode `opencode_config`, mini-swe `config_yaml`), so any base URL you set is allowlisted without code changes.

A few things we've learned plumbing this through Respan and OpenRouter:

**Claude Code** routes through the Anthropic face from Respan. Plan mode is disabled by default (`--disallowedTools EnterPlanMode`).

```yaml
- name: claude-code
  model_name: claude-opus-4-7
  env:
    ANTHROPIC_AUTH_TOKEN: ${RESPAN_API_KEY}
    ANTHROPIC_BASE_URL: https://endpoint.respan.ai/api/anthropic
    ANTHROPIC_CUSTOM_HEADERS: "X-Respan-Route-Provider: vertex_ai"
  kwargs:
    reasoning_effort: max
```

**Codex** needs a `[model_providers.<name>]` block with `wire_api = "responses"` (not WebSockets, which Codex defaults to and Respan doesn't speak).

```yaml
- name: codex
  model_name: openai/gpt-5.5
  env: { RESPAN_API_KEY: ${RESPAN_API_KEY} }
  kwargs:
    config_toml: |
      model_provider = "respan"
      [model_providers.respan]
      name = "Respan Gateway"
      base_url = "https://endpoint.respan.ai/api/"
      wire_api = "responses"
      env_key = "RESPAN_API_KEY"
    reasoning_effort: xhigh
```

**Gemini CLI**:

```yaml
- name: gemini-cli
  model_name: gemini/gemini-3.1-pro-preview
  env:
    GEMINI_API_KEY: ${RESPAN_API_KEY}
    GOOGLE_GENERATIVE_AI_API_KEY: ${RESPAN_API_KEY}
    GEMINI_API_BASE: https://endpoint.respan.ai/api/google/vertexai/v1beta
    GOOGLE_GEMINI_BASE_URL: https://endpoint.respan.ai/api/google/vertexai/
```

**Antigravity SDK** runs Google's Python SDK with its platform-specific local
harness in an isolated Python 3.12 environment with hash-verified, fully locked
dependencies. It supports Pier skills, stdio and streamable-HTTP MCP servers,
and live ATIF checkpoints. `reasoning_effort` accepts `minimal`, `low`, `medium`,
or `high`; `None` uses `medium`. SSE MCP servers are not supported by
google-antigravity 0.1.9.

```yaml
- name: antigravity-sdk
  model_name: google/gemini-3.6-flash
  env:
    GEMINI_API_KEY: ${GEMINI_API_KEY}
  kwargs:
    reasoning_effort: high
```

**Cursor CLI** uses the installed `cursor-agent` binary, so it fits the same
inside-the-sandbox path as Claude Code, Codex, Gemini CLI, and OpenCode. Use
`cursor/composer-2.5` for Composer 2.5 trial metadata and pass `CURSOR_API_KEY`
through your env file.

```yaml
- name: cursor-cli
  model_name: cursor/composer-2.5
  env:
    CURSOR_API_KEY: ${CURSOR_API_KEY}
```

**OpenCode** uses `opencode_config` to add unknown providers or override known ones. To redirect Google to Respan, override just `options.baseURL`; to add a fully custom provider, use `opencode_config.provider.<name>` with the npm package, options, and models.

**Pi** resolves `--model` against its own built-in catalog, so a gateway or proxy needs a provider entry in `models.json`. Setting the provider's base-URL env var (`OPENAI_BASE_URL`, `ANTHROPIC_BASE_URL`) generates one automatically, which routes pi's built-in models through the endpoint while keeping their shipped cost and capability metadata. Use `pi_config` to declare a slug that is not in the catalog; supply its `cost` (per million tokens) so pi can price it, otherwise Pier falls back to the LiteLLM price table and leaves `cost_usd` unset for a private slug. `pi_config` is deep-merged over the generated config, and any `baseUrl` in it is added to the network allowlist.

Pi has no built-in MCP support (configured MCP servers are ignored with a warning) and no resume support. Skills are copied to `~/.agents/skills`. `PI_OFFLINE=1` and `PI_SKIP_VERSION_CHECK=1` are set by default so pi's startup update check does not hit the egress proxy on no-network tasks; override them via `env` if needed.

```yaml
- name: pi
  model_name: openai/my-proxy-slug
  env:
    OPENAI_API_KEY: ${OPENAI_API_KEY}
    OPENAI_BASE_URL: ${OPENAI_BASE_URL}
  kwargs:
    thinking: medium
    pi_config:
      providers:
        openai:
          api: openai-completions
          models:
            - id: my-proxy-slug
              reasoning: true
              contextWindow: 400000
              cost: { input: 1.25, output: 10.0, cacheRead: 0.125 }
```

**mini-swe-agent** picks a native adapter from the model-name prefix: `openai/...` → `litellm_response` (OpenAI Responses end-to-end), `openrouter/...` → `openrouter` (BYOK costs from `cost_details.upstream_inference_cost`), everything else → LiteLLM auto.

For Gemini 3 via mini-swe-agent/LiteLLM, omitting `reasoning_effort` uses the Gemini API default high/dynamic thinking level, but it does not request readable thought summaries. Set `kwargs.reasoning_effort: high` explicitly when you want LiteLLM to send `includeThoughts` and preserve returned summaries as reasoning content.

```yaml
- name: mini-swe-agent
  model_name: openrouter/qwen/qwen3.6-plus
  env: { OPENROUTER_API_KEY: ${OPENROUTER_API_KEY} }
  kwargs:
    set_cache_control: default_end
```
