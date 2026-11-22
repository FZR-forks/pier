# Pi cleanup failure hardening

## Observed failures

Haiku Pi's Python state-machine trial completed its native session, then the
process sweep found a UID-0 process whose environment it could not read.
OXVG reached its execution deadline and a cleanup failure masked that timeout.
Both errors skipped local usage population by design: unconfirmed sandboxes must
not be read or graded. The original run artifacts remain unmodified.

This is shared Pier infrastructure, not Haiku-specific code. A model's tool
choices can expose it differently, but no retained evidence identifies the
unreadable process or establishes why it occurred only in those observations.

Container UID 0 does not imply CAP_SYS_PTRACE. A process with dumpability disabled
can deny `/proc/<pid>/environ` even to another same-UID container-root process.
The isolated Docker regression reproduces that mechanism, not the lost PID's
exact historical state. Exec/exit races can cause temporary read failures too.

## Changes

- Retry only the distinct unreadable-process status a bounded number of times.
  Re-run ownership checks: never infer ownership from age, UID, or ancestry.
  Persistent permission failures remain errors and stop the sandbox.
- Keep the execution failure, cleanup failure, and cancellation request as
  independent facts. An agent deadline remains an agent timeout; an operator
  cancellation still propagates even if the body previously failed. Trial results
  retain `agent_cleanup_error`, `agent_cleanup_cancelled`, and whether fallback
  shutdown was independently confirmed. Cleanup-affected trials are never retried,
  including when their primary failure is normally retryable.
- Docker stop is best-effort and logs some errors instead of raising. Query all
  containers in this exact Compose project separately and require no running,
  paused, or restarting process before considering mounted logs stable.
- Only after strict confirmation may Pi parse already-local stdout. No sandbox
  download, artifact extraction, or verification is allowed on this branch.
- Keep parsed usage in `session_reported_usage` with `complete: false` and null
  canonical totals. Preserve warming usage; do not mistake a partial response's
  missing usage for zero. Write the trajectory for later audit.

## First review follow-up

- Preserve known contributions if the interrupted final response lacks usage;
  exact-total conversion is unchanged. Recovery is populated independently of a
  best-effort trajectory write, so a read-only output cannot erase accounting.
- Use the same effective Docker client environment for Compose and both runtime
  state queries; per-run `DOCKER_HOST`/`DOCKER_CONTEXT` selection must not query
  a different daemon and mistake an empty result for stopped containers.
- Preserve cancellation arriving during cleanup, and execution errors replaced
  by cleanup inside an adapter's `finally` block. Do not replace an existing
  primary error with another cleanup exception or confuse an operator cancel
  with an execution deadline.
- Recover into the active multi-step context when applicable, while leaving
  prior steps unchanged. Shutdown confirmation and mounted-log guards still
  apply, and verification/downloads remain prohibited.

## Scoped untangle and independent review

Pi's normal trajectory and cleanup recovery use one `PiUsage` ledger for usage
source selection, warming deduplication, validation and known subtotals. Their
shared price policy distinguishes unknown pricing from explicitly free models.
File-output failure cannot erase already-populated accounting. The recovery
view preserves known individual input/cache components if an interrupted record
omits another component, but never claims exact canonical totals.

`AgentProcessCleanupError` owns primary-error preservation and a separate late
cancellation request. `TrialExecution` attaches the failing context and translates
execution deadlines. Trial serialization does not infer deadlines from arbitrary
error types or dynamically assigned current-step attributes. Failed multi-step
local logs are relocated after confirmed stop; evidence must not be moved while
the sandbox might still be writing it.

Adapters retain the actual cleanup error rather than a second mutable failed
boolean. OpenCode re-raises that stored error instead of replacing it and losing
shutdown proof. Already-failed adapter cleanup does not trigger a second sandbox
fallback. The adjacent Codex/Gemini changes only consume this shared error state;
they do not change routing, task execution or tool configuration.

Cancellation, retry-policy, cost and adapter error-replacement boundaries have
behavior-level regressions. Redundant test variants for a second fallback
were removed once the second fallback was removed; timeout/error/cancel coverage
remains. Tests bypassing `TrialExecution` now exercise the real execution boundary.
Deadline-versus-operator cancellation uses the pending task cancellation count
after `asyncio.timeout` removes its own request. Saved statistics, live job state
and the existing resume filter all use `TrialResult.is_cancelled`; that does not
authorize any automatic resumption of a stopped benchmark. Usage records with no
warming payload retain the normal converter's prior ignore behavior. Invalid
negative/boolean token counts remain unknown rather than usable accounting.
Exact cost requires all records; incomplete recovery retains positive known-cost
subtotals as observed values and never labels default missing pricing as free.
Adapter cleanup snapshots an already-expired execution deadline before awaiting
shutdown. A later operator cancel is independent: the primary timeout is retained
and cancellation still propagates. A failed move of stable multi-step logs is
logged without replacing either failure; files not yet moved remain available in
the original agent directory.

## Safety limits

This patch does not make permanent unreadable-process cleanup a successful
trial, and cannot guarantee no cleanup error will recur. Doing that by ignoring
the process, extending benchmark limits, or granting all agent tools ptrace
privileges would hide an integrity failure or change the benchmark environment.
The stronger long-term solution is a supervisor-owned kernel boundary (e.g.
per-run cgroup) with verified membership/termination, not environment-tag scans.
Non-Docker backends default to unconfirmed until they implement their own strict
shutdown-state check. No model routing, verifier retry, or task limit is changed.

## Tests

From this isolated worktree (uses the existing runner's Python dependencies):

```bash
PYTHONPATH=src /home/admin1/pier/.venv/bin/python -m pytest \
  tests/test_pi.py tests/test_agent_processes.py tests/test_job_usage.py \
  tests/test_pi_cleanup_accounting.py tests/test_docker_shutdown_confirmation.py -q
PIER_TEST_DOCKER=1 PYTHONPATH=src /home/admin1/pier/.venv/bin/python -m pytest \
  tests/test_agent_processes.py tests/test_pi_cleanup_docker_regression.py -q
```

The new regression containers use a cached image with `--pull=never` and
`--network=none`; they make no provider calls. Tests remove only their own
uniquely named containers. No running benchmark or bridge is restarted.

This worktree is not the deployed benchmark runner. Applying the fix requires
review and a deliberate runner update; historical results must not be regenerated
or relabeled as successes merely because a fix exists.
