# Collected usage and retry results

Pier preserves native harness usage. Input includes provider cache reads and
creation; reasoning is part of the harness's billable output. Missing totals
remain null rather than being converted to zero. Explicitly reported zero token
totals remain zero. A trial/job category is null if an agent that ran did not
report that category; a trial without an agent context contributes nothing.

Codex `peak_context_tokens` is the largest step prompt/input count, not input +
output. Compaction counts use explicit `compacted` items (or `context_compacted`
events when items are absent), and the trajectory contains corresponding system
markers. Changing output or reasoning length cannot create a compaction.

OpenCode V2 saved running totals are retained in
`agent_result.session_reported_usage` and `final_metrics.extra.session_reported_usage`.
They have `source`, `complete`, and input/cache/output/cost fields. When
`complete: false`, these are available observations only: an in-flight request
or uncollected child may be missing. Exact run totals remain null. Missing
categories are not filled with zero. Root session usage covers collected child
sessions once each, identified by their native IDs.
Collected sessions whose intermediate parent is missing still contribute to
these observed totals; inability to attach them to the trajectory tree keeps
`complete: false`. Failed summarizer requests retain usage, but only completed
compactions increment `summarization_count`.

Billed requests that have no message of their own are part of the totals:
Claude Code compaction summaries (from the stream's cumulative `modelUsage`),
OpenCode V2 background summaries that were never applied (from the complete
session counters), Pi cache-warming requests (`usage` session entries) and
Codex responses whose `token_count` never arrived (`token_usage_record`). Steps
are not invented for them. Claude Code and OpenCode V2 record the difference as
`final_metrics.extra.unattributed_usage`, and Pi records per-kind
`usage_entries`. The native totals are adopted only when every component
(uncached input, cache reads, output) is at least the steps' figure. Otherwise
the step sums stay, and Claude Code records `native_usage_mismatch`. A compacted
Claude Code run without native totals (for example a timeout) is flagged with
`compaction_usage_missing`.

`rejected_request_tokens` records input sizes explicitly reported in errors,
including their source. They are separate from successful input usage and peak
context: a gateway may have transformed the rejected request.

Whole-trial retries still default to zero. If enabled without an explicit
`include_exceptions`, only the infrastructure allowlist is eligible. An empty
include set disables eligibility; explicit include/exclude configuration is
honored, with exclusions taking precedence. A generic nonzero harness exit is
not an infrastructure classification and is not retried by default.
Once verification has produced a result (or finished), teardown errors cannot
trigger another benchmark attempt, even if their exception type is retryable.
The error remains recorded without replacing the reward.

The kept trial is the ordinary job child directory. Discarded infrastructure
attempts stay under `.retry-attempts/<trial>/attempt-N/`; `TrialResult.attempt`
identifies the attempt within that execution. Failed archival preserves the original observation
instead of deleting it and rerunning. Job rewards and primary usage totals use
the queue's kept results. `stats.retry_overhead` separately records archived
attempt count and usage; an unreadable archive counts but makes its totals
unknown. Existing lifecycle hooks and explicit resume behavior are unchanged.
Backoff leaves the current trial directory in place. If loading a replacement
is cancelled, the original observation is restored before cancellation escapes,
so a resumed job finds the same canonical identity rather than inventing a trial.

Pi aggregates a cost only when every recorded usage contribution includes its
cost. Missing costs remain unknown instead of being summed as free usage.
Explicitly configured all-zero model rates are preserved; Pi's default zeros
for an unpriced model continue to use the standard pricing fallback.
Claude Code's reported cost is likewise unknown when any model in its
`modelUsage` has `costBasis: "unknown"`: Claude Code had no price for that
model ID and priced it at the default model's rate.

These are inference observations, not gateway invoices. Standard provider
prices and upstream LLM caching are in scope; gateway billing, gateway response
caches and request rewriting are not used to adjust these figures. Live
reconciliation evidence is attached to the PR rather than shipped as an audit
subsystem. No historical benchmark output is rewritten.
