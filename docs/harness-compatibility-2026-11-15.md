# Harness compatibility review — 2026-11-15

## Scope and baseline

The last four-harness modernization was `1fba391` (November 1), merged in
[PR #20](https://github.com/FZR-forks/pier/pull/20) as `5bb5a03` on November 2.
This review includes the subsequent shutdown/accounting fixes in
[PR #22](https://github.com/FZR-forks/pier/pull/22), merged as `b5f6014`, the
current committed `main`.

Only the four requested harnesses and their stable releases are in scope.
Versions were checked against their current npm packages, including the V2
package `@opencode/cli` and renamed Pi package
`@earendil-works/pi-coding-agent`, rather than their legacy package names.

## Results

| Harness | Modernization baseline → current stable | Assessment |
| --- | --- | --- |
| OpenCode V2 | `2.2.5` → `2.3.0` | Consumed API contracts remain compatible, including background-by-default subagents. Upstream still downloads missing `rg` at runtime; Pier now installs ripgrep in its agent layer to make search available under restricted egress. |
| Claude Code | `2.1.316` → `2.1.325` | No required invocation/parser change found. **Gateway context limits need attention:** unrecognized model IDs now default to a 1M window, not 200K. Declare a smaller real window explicitly. |
| Codex | `0.166.0` → `0.169.0` | No required adapter change found. Usage records and exec JSON event definitions are unchanged. Project-local config now requires explicit trust; Pier-generated config/MCP entries are user-level in its private `CODEX_HOME`, so they remain loaded. |
| Pi | `1.4.0` → `1.5.2` | Execution/event contracts remain compatible. Provider-billed zero costs can still be replaced with fallback pricing; pricing changes are outside this change's scope because this benchmark does not rely on built-in cost calculations. |

This is a source/contract review with local regression tests, **not a new
end-to-end certification of every provider or harness feature**. In particular,
it does not make benchmark conditions identical across versions.

### OpenCode V2

Compared the release-tag sources and OpenAPI documents for the routes Pier
uses: info, session discovery, messages, active sessions, inboxes, interrupt,
wait, model/provider/agent discovery, and events.

- Session IDs, `parentID`, session totals, assistant content/tool states,
  `TokenUsage.Info`, and the `data`/`cursor` response shapes remain compatible.
  The added optional `time.firstToken` does not invalidate Pier's parsing.
- Optional query parameters no longer accept explicit null values; Pier omits
  absent parameters. Cursor requests already omit `order`.
- The native `run` command waits for descendant jobs, queued reports and active
  executions. Pier additionally discovers descendants and requires two identical
  terminal snapshots, no active sessions and no queued inbox work. The new
  regression covers a child finishing after the root, queuing a report and
  waking the parent before final collection.
- `calculateCeiling` and its input-window rule are unchanged. Pier's existing
  15% / minimum-16K reserve calculation still matches 2.3.0. The same numerical
  preflight tests now run for both 2.2.5 and 2.3.0. Background summary scheduling
  and retained-tail policy did change; usage remains in session counters even
  when a precomputed summary is never applied.
- Local shell commands can now run through the persistent process daemon.
  The packaged executable embeds that helper, so Pier's binary-only install
  does not need another download. The daemon inherits the server environment,
  and commands receive the supplied session environment; Pier's inherited
  ownership tag and detached-process sweep remain applicable. This path was
  inspected, not newly exercised with the real 2.3.0 binary.
- Follow-up search-dependency check: `RipgrepBinary` and the ripgrep execution
  layer are identical in 2.2.5 and 2.3.0. They use a system/cached `rg`, or
  download ripgrep 15.1.0 from GitHub and verify its hash. This is **not** an
  embedded binary or a non-ripgrep search fallback. If `rg` is absent and that
  download is blocked, the built-in grep tool still fails. Pier's V2 installer
  now installs `ripgrep` alongside `curl` and `python3` and verifies
  `rg --version` during installation. The dependency is available before
  runtime egress restrictions, without editing task Dockerfiles or publishing
  replacement base images. Changed install commands invalidate pinned-agent
  cache fingerprints automatically. Native OpenCode grep was not newly tested
  against a network-denied binary.

#### DeepSWE image coverage for the install change

The local DeepSWE checkout contains 113 Linux task definitions, all using
`public.ecr.aws/x8v8d7g8/mars-base:latest` in their environment Dockerfiles,
with no custom agent user or Compose setup. The cached base images cover all
20 tasks in PA1's current combined selection (19 distinct image IDs).

In disposable root containers from **all 19 distinct cached images**, ran the
command `apt-get update && apt-get install -y curl python3 ripgrep`,
then checked `rg --version` and a JSON-output literal search. **19/19 passed**;
all are Debian 12 amd64 and installed ripgrep 13.0.0. Original images and task
files were not modified. This verifies installation and basic rg execution
for the complete current 20-task selection, not the native OpenCode grep tool
end to end. The remaining 93 task-image tags were not downloaded or tested;
shared base definitions alone do not guarantee their apt repositories remain
healthy. Installing the package still needs network access during setup/build,
not during the restricted agent run.

### Claude Code

The stream's cumulative `modelUsage` and `total_cost_usd` remain the adapter's
native aggregate sources. New hook/progress messages do not replace transcript
messages or the final result. Changes to plugin-model-call accounting can add
usage without a transcript step; Pier already retains that difference as
`unattributed_usage`. Code Mode is opt-in, not a new adapter default.

The notable operational change is the unknown-model context window in
2.1.323. For a gateway model whose actual window is below 1M, supply the real
limit in **`agent.env`**, for example (only if 272,000 is the route's real limit):

```yaml
env:
  CLAUDE_CODE_MAX_CONTEXT_TOKENS: "272000"
```

Pier already forwards this value; a regression now checks that behavior.
Do not assume a variable set only in the host shell will be forwarded.
`CLAUDE_CODE_DISABLE_1M_CONTEXT=1` restores the earlier 200K sizing when that
is appropriate. For recognized Claude IDs or IDs containing `[1m]`, follow
the upstream model-window documentation rather than assuming the override
applies in every case. Pier does not guess the window from the model name.

Subagent progress-summary requests and their timing also changed. Existing
model pinning, refusal-fallback suppression, cumulative usage reconciliation
and owned-process cleanup remain in place. No global suppression of these
native requests was added.

### Codex

Compared the native `TokenUsage`, `TokenUsageRecord`, `TokenUsageInfo` and
`TokenCountEvent` definitions and the entire `exec_events.rs`: all are
unchanged. Rollout changes add metadata or alter permission representations
that Pier does not require. Bounded Code Mode output still uses content items
that the existing converter joins as text.

Codex 0.169.0 no longer implicitly trusts project configuration just because
an exec session is writable. Pier's own configuration, provider settings and
MCP registration are written to `$CODEX_HOME/config.toml`, not the task's
`.codex/config.toml`; these still apply. A task that intentionally relies on
project-local Codex configuration must provide an explicit trust decision.
This review does not add blanket `--trust`, which would authorize task-local
configuration and MCP commands automatically.

`features.multi_agent` is deprecated upstream in favor of `agents.enabled`.
Pier does not force the deprecated switch. The existing rejection of explicit
`features.async_compaction` remains: its unrecorded background usage cannot be
counted reliably. The new `async_compaction_root_only` modifier does not enable
async compaction by itself.

### Pi

Pi 1.5 adds `usage.cost.totalFromProvider`. Missing/false means estimated;
true means its total is provider-reported, including zero. Pier's existing
fallback can replace that zero with catalog pricing. This review does not
change Pi's pricing behavior or its tests; the benchmark reprices usage
separately and does not depend on these built-in cost figures.

1.5.1 changes idle cache warming, Anthropic message threads, tool validation,
OpenAI-compatible empty-tool requests and shutdown handling. The JSON event
envelope and token fields Pier consumes remain compatible. Print mode still
disposes the session after its prompts; it does not intentionally sit idle for
the interactive mode's 30-minute warming period. Older proxies that require
an empty tools array after tool history can set
`compat.requiresToolsWithToolHistory: true` through `pi_config`.

## Validation

- Added coverage for the OpenCode 2.3.0 compaction ceiling and background-child
  parent wakeup, plus forwarding Claude Code's gateway context-window override.
- Added install regressions for root-level ripgrep provisioning, shell syntax,
  and cache invalidation when adding the dependency to a pinned install.
- DeepSWE prerequisite-install smoke checks: **19/19 distinct cached images
  passed**, covering all 20 tasks in the current benchmark selection.
- No paid inference, model-backed benchmark rerun, production credential use,
  benchmark-output rewrite, or harness version-pin change was needed.

## Upstream references

- [OpenCode 2.3.0 release](https://github.com/anomalyco/opencode/releases/tag/v2.3.0)
- [OpenCode release comparison](https://github.com/anomalyco/opencode/compare/v2.2.5...v2.3.0)
- [OpenCode V2 API documentation](https://opencode.ai/docs/api)
- [Claude Code changelog](https://github.com/anthropics/claude-code/blob/main/CHANGELOG.md)
- [Claude Code gateway context windows](https://code.claude.com/docs/en/model-config#correct-the-window-for-a-gateway-or-custom-model-id)
- [Codex release comparison](https://github.com/openai/codex/compare/rust-v0.166.0...rust-v0.169.0)
- [Codex 0.169.0 release](https://github.com/openai/codex/releases/tag/rust-v0.169.0)
- [Pi 1.5.2 coding-agent changelog](https://github.com/earendil-works/pi/blob/v1.5.2/packages/coding-agent/CHANGELOG.md)
- [Pi 1.5.2 AI changelog](https://github.com/earendil-works/pi/blob/v1.5.2/packages/ai/CHANGELOG.md)
