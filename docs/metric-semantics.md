# Trustworthy benchmark metrics

Pier reports native harness observations. It does not retokenize prompts, infer
missing usage from text length, or treat unknown usage as zero. Provider prompt
cache reads and writes are in scope. Gateway request rewriting, gateway response
caching and gateway-specific billing are not part of these metrics.

## Token and cost semantics

- `n_input_tokens` / `prompt_tokens` includes fresh input, cache reads and cache
  creation exactly once. `n_cache_tokens` is the cache-hit subset, not extra input.
- `n_output_tokens` / `completion_tokens` includes billable reasoning output.
  Native harnesses differ: Codex and Pi already include reasoning in output;
  OpenCode V2 splits it out, so Pier adds it once. Gemini's tool-prompt count is
  not generated output.
- Cache creation and reasoning totals are promoted when reported. Unknown
  categories stay null; an explicitly reported zero stays zero.
- `cost_source` distinguishes `harness_reported` from `pricing_estimate` (or
  `mixed`/`unknown` in an aggregate). Harness prices and configured standard
  provider prices must be correct for the model being evaluated. Pier does not
  replace these figures with a gateway invoice or a gateway's response-cache cost.
- Codex custom slugs can use `kwargs.pricing_model_name` to explicitly select
  the standard provider's LiteLLM price-table entry without changing the model
  sent to the harness. This only reprices that selected model, not differently
  priced subagents. Cache discounts and cache-creation prices are preserved.
- Per-call costs are unknown when only a session cost exists; session cost is
  never allocated to calls by output length.

`peak_context_tokens` is the maximum observed model input, never input + output.
Rejected requests have separate `rejected_request_tokens` observations carrying
their source and measurement type; those are not billed usage or successful
context peaks. Codex compactions are counted from explicit native events and
appear as system trajectory markers. Failed OpenCode summarizer requests retain
their usage but do not count as successful context compactions.

## Incomplete observations

OpenCode V2's saved running totals survive timeout, interruption and partial
collection as `agent_result.session_reported_usage` and in trajectory final
metrics extras. `complete: false` means they must not be substituted for exact
trial totals. `lower_bound: true` indicates recorded usage can omit an in-flight
request or child. `usage_discrepancies` exposes contradictory native totals.

Job `stats.session_reported_usage` aggregates the same canonical trial
observations. It carries completeness, lower-bound qualification, per-category
`missing_observations`, source and cost provenance. Ordinary job token/cost
fields are null for a category if any canonical trial lacks that category;
they never silently sum around an unknown trial. The same rule applies to
multi-step trial totals. CLI output warns about missing usage.

## Attempts and provenance

Whole-trial retries use a conservative positive infrastructure allowlist.
Generic nonzero harness exits, timeouts, context overflow, interactive questions,
deterministic failures, and already graded trials do not automatically receive
another model attempt. `retry.include_exceptions` can specify explicit
infrastructure exception types; an empty set disables eligibility. Exclusions
take precedence, and generic harness exits/cancellation remain non-retryable.

Each trial has `attempt`, `canonical`, and `retry_attempts` (IDs and archive
paths). `stats.canonical_trials` identifies exactly the observations underlying
job metrics. Completion hooks see canonical results only; discarded attempts
emit `TrialEvent.RETRY`. Retry usage is separately labelled `stats.retry_overhead`
with `retry_attempt_paths`. Archive failure preserves the original observation
and stops retrying instead of deleting evidence. Explicit resume filters also
archive prior observations rather than deleting them.

Existing historical output is not rewritten automatically. These semantics apply
to newly generated files; retain the `lock.json` with every benchmark job.

## Reconciliation

Run the regression suite with `uv run pytest -q tests`. Example task verifiers
under `examples/tasks` are sandbox tests, not part of the host unit suite.

For a simple Pi/Codex/OpenCode V2 smoke job, independently compare native records,
trajectory metrics, trial fields, context peaks and canonical job totals:

```sh
uv run python scripts/audit_native_usage.py /path/to/job > native-audit.json
```

This command is read-only and exits nonzero for a discrepancy or unsupported
native structure. It does not fetch billing, alter observations or perform data
corrections. It records source hashes so audit evidence can be traced back to
the saved harness files.
