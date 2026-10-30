# Issue #17 validation evidence

The implementation was tested with DeepSeek V4.1 Flash, low reasoning, on PA1's
existing one-file `opencode-v2-smoke` task. The three harnesses were Pi 0.84.4,
Codex 0.151.0 and OpenCode V2 2.0.8. Each trial passed the verifier with reward 1,
ran once, and had no whole-trial retries or exception.

## Native inference reconciliation

Job: `issue-17-native-usage-final`, 2026-10-29. The read-only
`scripts/audit_native_usage.py` reports `ok: true`, with no corrections applied
to the job output.

| Harness | Input (including cache) | Cache hits | Output (including reasoning) | Peak input | Inference cost USD |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pi | 6,847 | 1,610 | 258 | 1,885 | 0.001890360 |
| Codex | 33,466 | 24,366 | 345 | 8,630 | 0.003290196 |
| OpenCode V2 | 12,736 | 6,150 | 196 | 6,456 | 0.002247900 |
| Canonical job total | 53,049 | 32,126 | 799 | — | 0.007428456 |

All three reported zero successful compactions. Native token/cache/output totals
agree with trajectory, trial and job output. OpenCode's message ledger also
agrees with its independent saved session totals. Pi costs agree with the native
message usage costs. Codex does not report dollars; its price estimate was
independently recomputed using the selected standard provider pricing entry.

The test used published Fireworks standard inference prices for DeepSeek V4.1
Flash: $0.30/M fresh input, $0.006/M cached input, $1.20/M output. Pi/OpenCode
catalogs were configured consistently; Codex used the explicit pricing alias
`fireworks_ai/accounts/fireworks/models/deepseek-v4p1-flash`. Gateway response
caching and gateway charges are deliberately excluded from this comparison.

The native audit records SHA-256 hashes of the saved Pi stdout, Codex rollout,
and OpenCode session dump. Full run artifacts remain on the validation host at
`/tmp/opencode/pier-metrics-live/issue-17-native-usage-final`; do not commit
credentials or full session prompts as test evidence.

A repeat on PR head (`issue-17-native-usage-pr-head`, revision `3cf8528`) also
passed all three verifiers and the native audit with zero discrepancies:

| Harness | Input | Cache hits | Output | Peak input | Inference cost USD |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pi | 6,797 | 3,220 | 201 | 1,835 | 0.001333620 |
| Codex | 24,919 | 16,248 | 258 | 8,516 | 0.003008388 |
| OpenCode V2 | 19,133 | 12,254 | 184 | 6,486 | 0.002358024 |
| Canonical job total | 50,849 | 31,722 | 643 | — | 0.006700032 |

Its artifacts are in the same parent directory, under the repeat job name.
This live evidence is specifically from `3cf8528`, not the later review head.
Subsequent adapter/completeness and lifecycle changes were checked with unit
tests and read-only replay of saved native artifacts; they were not all part
of that live invocation.

## Read-only historical replay

Reconverted saved PA1 native records in memory, without rewriting historical
results or invoking models:

- 62 OpenCode V2 sessions checked; 9 had incomplete message collection. There
  were **no contradictory session/message token or cost totals** among complete
  observations. Running session totals remain available for incomplete ones.
- 81 Codex rollouts checked. Every emitted peak equalled the maximum step input,
  and every compaction count equalled the number of explicit trajectory markers.
  Exactly **8 compactions** were found, matching the original issue's audit.

## Regression coverage

The test suite covers literal U+0085/U+2028/U+2029 JSON strings, CRLF, long
reasoning without compaction, explicit compaction markers, interrupted/partial
session trees, contradictory usage, zero versus unknown, provider cache rates,
cache-creation input, reasoning output, canonical retry selection, archive
failure, cancellation during retry backoff, resume provenance, incomplete job
and multi-step aggregates, and rejected request input diagnostics.

PR review regressions additionally cover evidence-only Codex child sessions,
missing/malformed/context-full audit snapshots, custom Pi default-zero versus
explicitly free model rates, cancellation inside the real trial START hook,
retry callback/backoff cancellation with resume-count verification, and viewer
peak/step averages with partially missing observations. Provider error regex
tests confirm the adjacent string literals form one two-capture pattern.

A further review pass covers absent and withheld cost provenance across Claude
Code, Codex, Gemini CLI, mini-swe-agent and both OpenCode adapters, including
explicit zero-cost observations and incomplete child trees. Partial OpenCode
session audits now retain best-known values and per-category missing counts,
returning a non-green audit report instead of raising on absent counters.

Lifecycle review regressions also exercise the real filtered-resume command
with identical task slots, multi-attempt archives and a retained canonical
trial; START-lock cancellation reconciles retries again on completion; ordinary
START-hook faults propagate without benchmark observations; and cancellation
of blocked executor writes (including repeated cancellation) keeps final job
accounting serialized until the worker finishes.

## Independent Opus review follow-up live run

On 2026-10-30 the Opus-reviewed follow-up working tree (based on `010bcbd`)
was tested again as `issue-17-opus-review-validation`. All three DeepSeek Low
trials passed on attempt 1, with no exception or retry. The independent native
audit reported `ok: true`, no missing categories and no withheld native values.

| Harness | Input | Cache hits | Output | Peak input | Inference cost USD |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pi | 3,268 | 1,438 | 105 | 1,700 | 0.000683628 |
| Codex | 42,545 | 33,180 | 535 | 8,847 | 0.003650580 |
| OpenCode V2 | 19,139 | 12,032 | 164 | 6,482 | 0.002401092 |
| Canonical job total | 64,952 | 46,650 | 804 | — | 0.006735300 |

Artifacts remain under `/tmp/opencode/pier-metrics-live/issue-17-opus-review-validation`.
The subsequent neutral-zero cost-source fix applies only to jobs with trials
whose agents never started, not this run. PA1's checkout remains unchanged.

An independent Opus 5.5 High review additionally led to explicit retry-policy
overrides being honored (safe defaults retained), contradiction qualification
surviving aggregate usage, unreadable retry archives remaining visible without
crashing a job, and known-zero usage for trials that never entered agent
execution. The unrelated reasoning-effort viewer fallback was removed. The
OpenCode audit mapping is independent of the adapter, and the V1 reasoning
output fix was verified against `opencode-ai` v1.18.25 `Session.getUsage`:
`tokens.output = outputTokens - reasoningTokens`, with reasoning separately
charged as output. These are generic Pier fixes; PA1 sources and data were not
modified.
