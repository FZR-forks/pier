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

`rejected_request_tokens` records input sizes explicitly reported in errors,
including their source. They are separate from successful input usage and peak
context: a gateway may have transformed the rejected request.

Whole-trial retries still default to zero. If enabled without an explicit
`include_exceptions`, only the infrastructure allowlist is eligible. An empty
include set disables eligibility; explicit include/exclude configuration is
honored, with exclusions taking precedence. A generic nonzero harness exit is
not an infrastructure classification and is not retried by default.

The kept trial is the ordinary job child directory. Discarded infrastructure
attempts stay under `.retry-attempts/<trial>/attempt-N/`; `TrialResult.attempt`
identifies the attempt within that execution. Failed archival preserves the original observation
instead of deleting it and rerunning. Job rewards and primary usage totals use
the queue's kept results. `stats.retry_overhead` separately records archived
attempt count and usage; an unreadable archive counts but makes its totals
unknown. Existing lifecycle hooks and explicit resume behavior are unchanged.

These are inference observations, not gateway invoices. Standard provider
prices and upstream LLM caching are in scope; gateway billing, gateway response
caches and request rewriting are not used to adjust these figures. Live
reconciliation evidence is attached to the PR rather than shipped as an audit
subsystem. No historical benchmark output is rewritten.
