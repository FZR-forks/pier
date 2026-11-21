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
- Keep the primary execution failure separately from a secondary cleanup error.
  An agent deadline remains an agent timeout; an operator cancellation still
  propagates. Trial results retain `agent_cleanup_error` and whether fallback
  shutdown was independently confirmed.
- Docker stop is best-effort and logs some errors instead of raising. Query all
  containers in this exact Compose project separately and require no running,
  paused, or restarting process before considering mounted logs stable.
- Only after strict confirmation may Pi parse already-local stdout. No sandbox
  download, artifact extraction, or verification is allowed on this branch.
- Keep parsed usage in `session_reported_usage` with `complete: false` and null
  canonical totals. Preserve warming usage; do not mistake a partial response's
  missing usage for zero. Write the trajectory for later audit.
- End the inference deadline after a successful adapter return, rather than
  accidentally extending it into shutdown work.

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
