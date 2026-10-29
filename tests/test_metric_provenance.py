import json
from datetime import datetime
from pathlib import Path

import pytest

from pier.agents.installed.codex import Codex
from pier.agents.installed.opencode_v2 import OpenCodeV2
from pier.agents.installed.pi import Pi
from pier.job import Job
from pier.models.agent.context import AgentContext
from pier.models.agent.usage import ReportedUsage
from pier.models.job.config import JobConfig, RetryConfig
from pier.models.job.result import JobResult, JobStats
from pier.models.trial.config import TrialConfig, TaskConfig
from pier.models.trial.result import AgentInfo, ExceptionInfo, TrialResult
from pier.models.task.id import LocalTaskId
from pier.models.verifier.result import VerifierResult
from pier.models.trajectories import FinalMetrics
from pier.utils.trajectory_metrics import populate_context_from_final_metrics
from pier.trial.queue import TrialQueue
from pier.trial.trial import Trial
from pier.trial.hooks import TrialEvent
from pier.utils.rejected_request import rejected_requests_from_errors


FIXTURES = Path(__file__).parent / "fixtures" / "opencode_v2"


def result_for(tmp_path, *, error=None, usage=None, reward=None):
    config = TrialConfig(
        trial_name="trial", trials_dir=tmp_path, task=TaskConfig(path=tmp_path)
    )
    return TrialResult(
        task_name="task",
        trial_name="trial",
        trial_uri=(tmp_path / "trial").as_uri(),
        task_id=LocalTaskId(path=tmp_path),
        task_checksum="test",
        config=config,
        agent_info=AgentInfo(name="test", version="1"),
        agent_result=usage,
        verifier_result=VerifierResult(rewards={"reward": reward})
        if reward is not None
        else None,
        exception_info=ExceptionInfo(
            exception_type=error,
            exception_message=error,
            exception_traceback="",
            occurred_at=datetime.now(),
        )
        if error
        else None,
    )


def test_codex_context_ignores_output_and_input_drops(tmp_path):
    events = []
    for prompt, output in [(20000, 100000), (21000, 1), (100, 1)]:
        events.extend(
            [
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "done"}],
                    },
                },
                {
                    "type": "event_msg",
                    "payload": {
                        "type": "token_count",
                        "info": {
                            "last_token_usage": {
                                "input_tokens": prompt,
                                "output_tokens": output,
                                "reasoning_output_tokens": output - 1,
                                "total_tokens": prompt + output,
                            },
                            "total_token_usage": {
                                "input_tokens": prompt,
                                "output_tokens": output,
                                "cached_input_tokens": 0,
                                "total_tokens": prompt + output,
                            },
                        },
                    },
                },
            ]
        )
    agent = Codex(logs_dir=tmp_path, model_name="test")
    assert agent._extract_context_metrics(events) == (21000, 0)
    trajectory = agent._convert_session_file_to_trajectory(
        tmp_path / "test.jsonl", events
    )
    assert (
        trajectory.final_metrics.extra["peak_context_tokens"]
        == max(step.metrics.prompt_tokens for step in trajectory.steps if step.metrics)
        == 21000
    )
    assert trajectory.final_metrics.extra["summarization_count"] == 0

    events += [
        {"type": "event_msg", "payload": {"type": "context_compacted"}},
        {"type": "compacted", "payload": {}},
    ]
    trajectory = agent._convert_session_file_to_trajectory(
        tmp_path / "test.jsonl", events
    )
    markers = [
        step for step in trajectory.steps if (step.extra or {}).get("compaction")
    ]
    assert len(markers) == trajectory.final_metrics.extra["summarization_count"] == 1


def test_jsonl_unicode_separators_roundtrip(tmp_path):
    text = "NEL: \u0085; LS: \u2028; PS: \u2029"
    # Use actual separators, not JSON escapes, including a CRLF record boundary.
    record = json.loads((FIXTURES / "unicode_separators.jsonl").read_text())
    record["messages"][0]["content"][0]["text"] = text
    path = tmp_path / "unicode.jsonl"
    path.write_bytes((json.dumps(record, ensure_ascii=False) + "\r\n").encode())
    records = OpenCodeV2._read_jsonl(path)
    assert len(records) == 1
    agent = OpenCodeV2(logs_dir=tmp_path, model_name="test/test")
    trajectory = agent._convert_session_to_trajectory(records[0], {})
    assert trajectory.steps[0].message == text
    assert trajectory.final_metrics.total_prompt_tokens == 19


@pytest.mark.parametrize("manifest", [False, True])
def test_incomplete_opencode_preserves_session_totals_in_trial_output(
    tmp_path, manifest
):
    inspections = list(
        json.loads((FIXTURES / "root_child_inspections.json").read_text()).values()
    )
    # Message collection is partial, but the running session totals survive.
    inspections[1]["messages"][1].pop("tokens")
    base = tmp_path / "opencode-v2"
    base.mkdir()
    (base / "opencode-v2-sessions.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in inspections)
    )
    if manifest:
        (base / "runner-result.json").write_text(
            json.dumps(
                {
                    "root_id": inspections[0]["session"]["id"],
                    "collection_complete": False,
                }
            )
        )
    context = AgentContext()
    OpenCodeV2(
        logs_dir=tmp_path, model_name="litellm/kimi-k3"
    ).populate_context_post_run(context)
    assert context.n_input_tokens is None
    usage = context.session_reported_usage
    assert usage.complete is False
    assert (usage.n_input_tokens, usage.n_cache_tokens, usage.n_output_tokens) == (
        223,
        30,
        120,
    )
    assert usage.cost_usd == 4.5
    assert len(usage.session_ids) == 2
    result = result_for(tmp_path, usage=context)
    saved = json.loads(result.model_dump_json())
    assert saved["agent_result"]["session_reported_usage"]["cost_usd"] == 4.5
    assert result.compute_token_cost_totals() == (None, None, None, None)


def test_session_partial_categories_remain_unknown():
    usage = OpenCodeV2._session_reported_usage(
        {
            "id": "partial",
            "tokens": {"input": 10, "cache": {"read": 0}},
        },
        complete=False,
    )
    assert usage.n_input_tokens is None
    assert usage.n_output_tokens is None
    assert usage.n_cache_tokens == 0
    assert usage.cost_usd is None


@pytest.mark.parametrize(
    "error",
    [
        "NonZeroAgentExitCodeError",
        "ValueError",
        "RuntimeError",
        "AgentTimeoutError",
        "CancelledError",
        "ContextWindowExceededError",
        "QuestionError",
    ],
)
def test_evaluated_failures_never_retried_by_default(error):
    assert not TrialQueue(1, RetryConfig(max_retries=1))._should_retry_exception(error)


def test_retry_allowlist_is_positive_and_excludes_generic_harness_exit():
    assert TrialQueue(1)._should_retry_exception("ConnectionError")
    assert not TrialQueue(
        1, RetryConfig(include_exceptions=set())
    )._should_retry_exception("ConnectionError")
    assert not TrialQueue(
        1, RetryConfig(include_exceptions={"NonZeroAgentExitCodeError"})
    )._should_retry_exception("NonZeroAgentExitCodeError")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error,reward,expected_calls",
    [
        ("ConnectionError", None, 2),
        ("NonZeroAgentExitCodeError", None, 1),
        ("ConnectionError", 1, 1),
    ],
)
async def test_canonical_attempt_and_retry_provenance(
    tmp_path, monkeypatch, error, reward, expected_calls
):
    trial_dir = tmp_path / "trial"
    calls = []
    first = result_for(
        tmp_path,
        error=error,
        reward=reward,
        usage=AgentContext(
            n_input_tokens=100,
            n_cache_tokens=10,
            n_output_tokens=20,
            cost_usd=0.1,
        ),
    )
    second = result_for(
        tmp_path,
        usage=AgentContext(
            n_input_tokens=200,
            n_cache_tokens=20,
            n_output_tokens=40,
            cost_usd=0.2,
        ),
    )

    class FakeTrial:
        def __init__(self):
            self.trial_dir = trial_dir

        async def run(self, **kwargs):
            trial_dir.mkdir(exist_ok=True)
            calls.append(True)
            return first if len(calls) == 1 else second

    async def create(cls, config):
        return FakeTrial()

    monkeypatch.setattr(Trial, "create", classmethod(create))
    queue = TrialQueue(1, RetryConfig(max_retries=1, min_wait_sec=0, max_wait_sec=0))
    completed, retried = [], []

    async def on_end(event):
        completed.append(event.result.id)

    async def on_retry(event):
        retried.append(event.result.id)

    queue.add_hook(TrialEvent.END, on_end)
    queue.add_hook(TrialEvent.RETRY, on_retry)
    result = await queue._execute_trial_with_retries(first.config)
    assert len(calls) == expected_calls
    canonical = TrialResult.model_validate_json((trial_dir / "result.json").read_text())
    assert canonical.canonical is True
    assert canonical.id == result.id
    assert canonical.attempt == expected_calls
    assert completed == [canonical.id]
    assert retried == ([first.id] if expected_calls == 2 else [])
    if expected_calls == 2:
        archived = TrialResult.model_validate_json(
            (tmp_path / canonical.retry_attempts[0].result_path).read_text()
        )
        assert archived.canonical is False
        assert archived.id == first.id
        assert JobStats.from_trial_results([canonical]).cost_usd == 0.2
        with pytest.raises(ValueError, match="Discarded"):
            JobStats.from_trial_results([archived])


def test_rejected_request_sizes_are_not_successful_context_usage(tmp_path):
    error = "Max Input Tokens=1048576, Got=1054336"
    events = [
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "content": [],
                "stopReason": "error",
                "errorMessage": error,
            },
        }
    ]
    assert rejected_requests_from_errors({"text": error}, source="test") == []
    agent = Pi(logs_dir=tmp_path, model_name="deepseek/deepseek-v4p1-flash")
    trajectory = agent._convert_events_to_trajectory(events)
    assert (
        trajectory.final_metrics.extra["rejected_request_tokens"][0]["input_tokens"]
        == 1054336
    )
    assert trajectory.final_metrics.extra.get("peak_context_tokens") is None


def test_job_summary_separates_retry_overhead_and_lower_bound_usage(tmp_path):
    canonical = result_for(
        tmp_path,
        usage=AgentContext(
            session_reported_usage=ReportedUsage(
                source="opencode.session.tokens/cost",
                complete=False,
                n_input_tokens=500,
                cost_usd=0.5,
            )
        ),
    )
    archived = result_for(
        tmp_path,
        usage=AgentContext(
            n_input_tokens=100,
            n_cache_tokens=0,
            n_output_tokens=20,
            cost_usd=0.1,
        ),
    )
    archive_dir = tmp_path / ".retry-attempts" / "trial" / "attempt-1"
    archive_dir.mkdir(parents=True)
    (archive_dir / "result.json").write_text(archived.model_dump_json())
    job = Job.__new__(Job)
    job.config = JobConfig(job_name=tmp_path.name, jobs_dir=tmp_path.parent)
    job._previous_trial_results = {canonical.trial_name: canonical}
    job._job_result = JobResult(
        id=canonical.id,
        started_at=datetime.now(),
        n_total_trials=1,
        stats=JobStats.from_trial_results([canonical]),
    )
    job._write_job_result(exclude_trial_results=True)
    stats = json.loads((tmp_path / "result.json").read_text())["stats"]
    assert stats["cost_usd"] is None
    assert stats["n_trials_missing_usage"] == 1
    assert stats["session_reported_usage"]["cost_usd"] == 0.5
    assert stats["session_reported_usage"]["complete"] is False
    assert stats["retry_overhead"]["cost_usd"] == 0.1
    assert stats["canonical_trials"]["trial"]["id"] == str(canonical.id)


def test_shared_population_preserves_unknown_and_explicit_zero():
    context = AgentContext()
    populate_context_from_final_metrics(
        context,
        FinalMetrics(
            total_prompt_tokens=None,
            total_completion_tokens=0,
            total_cached_tokens=0,
            total_cost_usd=0,
        ),
    )
    assert context.n_input_tokens is None
    assert context.n_cache_tokens == context.n_output_tokens == context.cost_usd == 0


def test_job_exact_totals_do_not_silently_omit_missing_trials(tmp_path):
    known = result_for(
        tmp_path,
        usage=AgentContext(
            n_input_tokens=100,
            n_cache_tokens=0,
            n_output_tokens=20,
            cost_usd=0.1,
        ),
    )
    unknown = result_for(
        tmp_path,
        usage=AgentContext(
            n_input_tokens=200,
            n_cache_tokens=0,
            n_output_tokens=30,
            cost_usd=None,
        ),
    )
    unknown.trial_name = "unknown"
    stats = JobStats.from_trial_results([known, unknown])
    assert stats.n_input_tokens == 300
    assert stats.cost_usd is None
    assert stats.n_trials_missing_usage == 1
    stats = JobStats.model_validate_json(stats.model_dump_json())
    stats.remove_trial(unknown)
    assert stats.cost_usd == 0.1
    assert stats.n_trials_missing_usage == 0


def test_multi_step_trial_preserves_missing_category(tmp_path):
    from pier.models.trial.result import StepResult

    result = result_for(tmp_path)
    result.step_results = [
        StepResult(step_name="one", agent_result=AgentContext(n_input_tokens=100)),
        StepResult(step_name="two", agent_result=AgentContext(n_output_tokens=10)),
    ]
    assert result.compute_token_cost_totals() == (None, None, None, None)
    usage = result.reported_usage()
    assert usage.n_input_tokens == 100
    assert usage.complete is False
    assert usage.missing_observations["n_input_tokens"] == 1


def test_opencode_usage_disagreement_is_explicit_not_silently_selected(tmp_path):
    inspection = json.loads((FIXTURES / "unicode_separators.jsonl").read_text())
    inspection["session"]["tokens"]["input"] = 1000
    agent = OpenCodeV2(logs_dir=tmp_path, model_name="test/test")
    trajectory = agent._convert_session_to_trajectory(inspection, {})
    assert trajectory.final_metrics.total_prompt_tokens is None
    assert trajectory.final_metrics.extra["usage_discrepancies"]["n_input_tokens"] == {
        "messages": 19,
        "session": 1009,
    }
    context = AgentContext()
    agent._populate_context_if_usage_known(context, trajectory.final_metrics)
    assert context.usage_discrepancies
    assert context.session_reported_usage.n_input_tokens == 1009
    assert context.session_reported_usage.complete is False
    assert context.session_reported_usage.lower_bound is False


def test_pi_zero_cost_and_unicode_are_preserved(tmp_path):
    text = "\u0085\u2028\u2029"
    event = {
        "type": "turn_end",
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
            "usage": {
                "input": 1,
                "output": 0,
                "cacheRead": 0,
                "cacheWrite": 0,
                "cost": {"total": 0},
            },
        },
        "toolResults": [],
    }
    (tmp_path / "pi.txt").write_text(json.dumps(event, ensure_ascii=False) + "\n")
    context = AgentContext()
    Pi(logs_dir=tmp_path, model_name="test/test").populate_context_post_run(context)
    assert context.n_cache_tokens == context.n_output_tokens == context.cost_usd == 0
    assert context.cost_source == "harness_reported"
    trajectory = json.loads((tmp_path / "trajectory.json").read_text())
    assert trajectory["steps"][0]["message"] == text


def test_codex_rejected_request_exposed_without_successful_usage(tmp_path):
    events = [
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "task"}],
            },
        },
        {
            "type": "event_msg",
            "payload": {
                "type": "error",
                "message": "Max Input Tokens=1048576, Got=1054336",
            },
        },
    ]
    trajectory = Codex(
        logs_dir=tmp_path, model_name="test"
    )._convert_session_file_to_trajectory(tmp_path / "test.jsonl", events)
    assert (
        trajectory.final_metrics.extra["rejected_request_tokens"][0]["input_tokens"]
        == 1054336
    )
    assert trajectory.final_metrics.extra.get("peak_context_tokens") is None


@pytest.mark.asyncio
async def test_cancellation_during_retry_wait_keeps_canonical_observation(
    tmp_path, monkeypatch
):
    import asyncio

    trial_dir = tmp_path / "trial"
    result = result_for(tmp_path, error="ConnectionError")

    class FakeTrial:
        def __init__(self):
            self.trial_dir = trial_dir

        async def run(self, **kwargs):
            trial_dir.mkdir()
            return result

    async def create(cls, config):
        return FakeTrial()

    async def cancel(delay):
        raise asyncio.CancelledError()

    monkeypatch.setattr(Trial, "create", classmethod(create))
    monkeypatch.setattr("pier.trial.queue.asyncio.sleep", cancel)
    with pytest.raises(asyncio.CancelledError):
        await TrialQueue(1, RetryConfig(max_retries=1))._execute_trial_with_retries(
            result.config
        )
    saved = TrialResult.model_validate_json((trial_dir / "result.json").read_text())
    assert saved.canonical is True
    assert saved.id == result.id
    assert not (tmp_path / ".retry-attempts" / "trial" / "attempt-1").exists()


def test_ungraded_trial_is_still_counted_in_job_denominator(tmp_path):
    stats = JobStats.from_trial_results([result_for(tmp_path, error="RuntimeError")])
    assert stats.evals["test__adhoc"].n_trials == 1
    assert stats.evals["test__adhoc"].n_errors == 1


@pytest.mark.asyncio
async def test_job_hooks_and_disk_summary_select_only_canonical_attempt(
    tmp_path, monkeypatch
):
    from collections import defaultdict
    from pier.metrics.mean import Mean
    from pier.models.trial.config import AgentConfig
    from pier.trial.hooks import TrialHookEvent
    from pier.models.agent.name import AgentName

    config = JobConfig(
        job_name="job",
        jobs_dir=tmp_path,
        quiet=True,
        agents=[AgentConfig(name=AgentName.NOP)],
        retry=RetryConfig(max_retries=1, min_wait_sec=0, max_wait_sec=0),
    )
    job = Job(
        config,
        _task_configs=[TaskConfig(path=tmp_path)],
        _metrics=defaultdict(lambda: [Mean()]),
    )
    attempts = []

    class FakeTrial:
        def __init__(self, config):
            self.config = config
            self.trial_dir = config.trials_dir / config.trial_name
            self.hooks = defaultdict(list)

        def add_hook(self, event, hook):
            self.hooks[event].append(hook)

        async def run(self, **kwargs):
            self.trial_dir.mkdir()
            result = result_for(
                tmp_path,
                error="ConnectionError" if not attempts else None,
                reward=None if not attempts else 1,
                usage=AgentContext(
                    n_input_tokens=100 if not attempts else 200,
                    n_cache_tokens=0,
                    n_output_tokens=10,
                    cost_usd=0.1 if not attempts else 0.2,
                ),
            )
            result.trial_name = self.config.trial_name
            result.config = self.config
            result.attempt = kwargs["attempt"]
            result.retry_attempts = list(kwargs["retry_attempts"])
            attempts.append(result)
            for event in (TrialEvent.START, TrialEvent.END):
                for hook in self.hooks[event]:
                    await hook(
                        TrialHookEvent(
                            event=event,
                            trial_id=result.trial_name,
                            task_name=result.task_name,
                            config=self.config,
                            result=result,
                        )
                    )
            (self.trial_dir / "result.json").write_text(result.model_dump_json())
            return result

    async def create(cls, config):
        return FakeTrial(config)

    monkeypatch.setattr(Trial, "create", classmethod(create))
    result = await job.run()
    assert result.stats.n_completed_trials == 1
    assert result.stats.n_retries == 1
    assert result.stats.n_input_tokens == 200
    assert result.stats.cost_usd == 0.2
    assert result.stats.retry_overhead.cost_usd == 0.1
    assert result.stats.retry_overhead.n_input_tokens == 100
    saved = json.loads((job.job_dir / "result.json").read_text())["stats"]
    canonical = next(iter(saved["canonical_trials"].values()))
    assert canonical["attempt"] == 2
    assert canonical["id"] == str(attempts[1].id)
    assert next(iter(saved["evals"].values()))["metrics"] == [{"mean": 1.0}]
