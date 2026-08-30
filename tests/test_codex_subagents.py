import json
import logging
from pathlib import Path
from typing import Any

import pytest

from pier.agents.installed.codex import Codex
from pier.trial.trial import _agent_step_count_from_trajectory_path

ROOT = "root-thread"
CHILD = "child-thread"
GRANDCHILD = "grandchild-thread"
ORPHAN = "orphan-thread"


def _session_meta(
    thread_id: str,
    *,
    parent_thread_id: str | None = None,
    nickname: str | None = None,
    source: Any = "exec",
    thread_source: str = "user",
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": thread_id,
        "thread_source": thread_source,
        "source": source,
        "cli_version": "0.149.1",
    }
    if parent_thread_id is not None:
        payload["parent_thread_id"] = parent_thread_id
        if isinstance(source, dict):
            payload["source"] = {
                "subagent": {
                    "thread_spawn": {
                        "parent_thread_id": parent_thread_id,
                        "depth": 1,
                        "agent_nickname": nickname or "Euclid",
                        "agent_role": None,
                        "agent_path": None,
                    }
                }
            }
    if nickname is not None:
        payload["agent_nickname"] = nickname
    return {"type": "session_meta", "payload": payload}


def _token_count(
    input_tokens: int,
    output_tokens: int,
    *,
    cached_input_tokens: int = 0,
    total_input_tokens: int | None = None,
    total_output_tokens: int | None = None,
    context_tokens: int | None = None,
) -> dict[str, Any]:
    last_usage: dict[str, int] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_input_tokens": cached_input_tokens,
    }
    if context_tokens is not None:
        last_usage["total_tokens"] = context_tokens

    total_usage = {
        "input_tokens": (
            input_tokens if total_input_tokens is None else total_input_tokens
        ),
        "output_tokens": (
            output_tokens if total_output_tokens is None else total_output_tokens
        ),
        "cached_input_tokens": cached_input_tokens,
    }
    return {
        "type": "event_msg",
        "payload": {
            "type": "token_count",
            "info": {
                "last_token_usage": last_usage,
                "total_token_usage": total_usage,
            },
        },
    }


def _turns(
    thread_id: str,
    usages: list[tuple[int, int]],
    *,
    contexts: list[int | None] | None = None,
    total_usage: list[tuple[int, int] | None] | None = None,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    contexts = contexts or [None] * len(usages)
    total_usage = total_usage or [None] * len(usages)
    for index, ((input_tokens, output_tokens), context, cumulative) in enumerate(
        zip(usages, contexts, total_usage, strict=True), start=1
    ):
        events.extend(
            [
                {
                    "type": "event_msg",
                    "payload": {
                        "type": "turn_started",
                        "turn_id": f"{thread_id}-turn-{index}",
                    },
                },
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": f"{thread_id}-{index}"}
                        ],
                    },
                },
                _token_count(
                    input_tokens,
                    output_tokens,
                    total_input_tokens=cumulative[0] if cumulative else None,
                    total_output_tokens=cumulative[1] if cumulative else None,
                    context_tokens=context,
                ),
            ]
        )
    return events


def _thread_events(
    thread_id: str,
    usages: list[tuple[int, int]],
    *,
    model: str = "root-model",
    parent_thread_id: str | None = None,
    nickname: str | None = None,
    source: Any = "exec",
    thread_source: str = "user",
    contexts: list[int | None] | None = None,
    total_usage: list[tuple[int, int] | None] | None = None,
    compacted: int = 0,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = [
        _session_meta(
            thread_id,
            parent_thread_id=parent_thread_id,
            nickname=nickname,
            source=source,
            thread_source=thread_source,
        ),
        {"type": "turn_context", "payload": {"model": model}},
    ]
    events.extend(
        _turns(
            thread_id,
            usages,
            contexts=contexts,
            total_usage=total_usage,
        )
    )
    events.extend({"type": "compacted"} for _ in range(compacted))
    return events


def _forked_events(
    parent_events: list[dict[str, Any]],
    thread_id: str,
    usages: list[tuple[int, int]],
    *,
    model: str = "child-model",
    parent_thread_id: str = ROOT,
    nickname: str = "Euclid",
    source: Any = None,
    thread_source: str = "subagent",
    contexts: list[int | None] | None = None,
    total_usage: list[tuple[int, int] | None] | None = None,
    compacted: int = 0,
    include_copied_session_meta: bool = True,
) -> list[dict[str, Any]]:
    """Build Codex's full-history layout: canonical child meta, then copied parent."""
    child_meta = _session_meta(
        thread_id,
        parent_thread_id=parent_thread_id,
        nickname=nickname,
        source=(
            {
                "subagent": {
                    "thread_spawn": {
                        "parent_thread_id": parent_thread_id,
                        "depth": 1,
                        "agent_nickname": nickname,
                        "agent_role": None,
                        "agent_path": None,
                    }
                }
            }
            if source is None
            else source
        ),
        thread_source=thread_source,
    )
    copied = parent_events if include_copied_session_meta else parent_events[1:]
    return [
        child_meta,
        *copied,
        *_turns(thread_id, usages, contexts=contexts, total_usage=total_usage),
        *({"type": "compacted"} for _ in range(compacted)),
    ]


def _write_rollout(
    logs_dir: Path,
    filename: str,
    events: list[dict[str, Any]],
    *,
    date: tuple[str, str, str] = ("2026", "01", "01"),
) -> Path:
    session_dir = logs_dir / "sessions" / date[0] / date[1] / date[2]
    session_dir.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(event) for event in events) + "\n"
    # Keep the fixture self-checking: every line written here must be standalone JSON.
    assert all(isinstance(json.loads(line), dict) for line in text.splitlines())
    path = session_dir / filename
    path.write_text(text)
    return path


def _write_stdout(logs_dir: Path, thread_id: str) -> None:
    (logs_dir / "codex.txt").write_text(
        "progress\n"
        + json.dumps({"type": "thread.started", "thread_id": thread_id})
        + "\n"
    )


def _convert(logs_dir: Path) -> Any:
    trajectory = Codex(
        logs_dir=logs_dir, model_name="root-model"
    )._convert_events_to_trajectory(logs_dir / "sessions")
    assert trajectory is not None
    return trajectory


def _convert_optional(logs_dir: Path) -> Any:
    return Codex(
        logs_dir=logs_dir, model_name="root-model"
    )._convert_events_to_trajectory(logs_dir / "sessions")


def _metrics(trajectory: Any) -> Any:
    assert trajectory is not None
    assert trajectory.final_metrics is not None
    return trajectory.final_metrics


def _children(trajectory: Any) -> list[Any]:
    assert trajectory.subagent_trajectories is not None
    return trajectory.subagent_trajectories


# Fork group


def test_f1_full_history_fork_excludes_copied_parent_usage(tmp_path: Path):
    root_events = _thread_events(ROOT, [(10, 2)], total_usage=[(10, 2)])
    child_events = _forked_events(
        root_events,
        CHILD,
        [(5, 1)],
        total_usage=[(15, 3)],
    )
    _write_rollout(tmp_path, "rollout-2000-root.jsonl", root_events)
    _write_rollout(tmp_path, "rollout-2001-child.jsonl", child_events)
    _write_stdout(tmp_path, ROOT)

    trajectory = _convert(tmp_path)
    child = _children(trajectory)[0]

    assert _metrics(child).total_prompt_tokens == 5
    assert _metrics(child).total_completion_tokens == 1
    assert _metrics(trajectory).total_prompt_tokens == 15
    assert _metrics(trajectory).total_completion_tokens == 3


def test_f2_identical_later_token_payload_is_not_globally_deduplicated(
    tmp_path: Path,
):
    root_events = _thread_events(ROOT, [(7, 3)], total_usage=[(7, 3)])
    child_events = _forked_events(
        root_events,
        CHILD,
        [(7, 3)],
        total_usage=[(14, 6)],
    )
    assert (
        child_events[-1]["payload"]["info"]["last_token_usage"]
        == root_events[-1]["payload"]["info"]["last_token_usage"]
    )
    _write_rollout(tmp_path, "rollout-root.jsonl", root_events)
    _write_rollout(tmp_path, "rollout-child.jsonl", child_events)
    _write_stdout(tmp_path, ROOT)

    trajectory = _convert(tmp_path)
    child = _children(trajectory)[0]

    assert _metrics(child).total_prompt_tokens == 7
    assert _metrics(child).total_completion_tokens == 3
    assert _metrics(trajectory).total_prompt_tokens == 14
    assert _metrics(trajectory).total_completion_tokens == 6


def test_f3_nested_forks_report_only_each_threads_own_calls(tmp_path: Path):
    root_events = _thread_events(ROOT, [(10, 1)], total_usage=[(10, 1)])
    child_events = _forked_events(
        root_events,
        CHILD,
        [(20, 2)],
        total_usage=[(30, 3)],
    )
    grandchild_events = _forked_events(
        child_events,
        GRANDCHILD,
        [(30, 3)],
        parent_thread_id=CHILD,
        nickname="Archimedes",
        model="grandchild-model",
        total_usage=[(60, 6)],
    )
    _write_rollout(tmp_path, "rollout-root.jsonl", root_events)
    _write_rollout(tmp_path, "rollout-child.jsonl", child_events)
    _write_rollout(tmp_path, "rollout-grandchild.jsonl", grandchild_events)
    _write_stdout(tmp_path, ROOT)

    trajectory = _convert(tmp_path)
    child = _children(trajectory)[0]
    grandchild = _children(child)[0]

    assert _metrics(trajectory).total_prompt_tokens == 60
    assert _metrics(child).total_prompt_tokens == 20
    assert _metrics(grandchild).total_prompt_tokens == 30
    assert _metrics(trajectory).total_completion_tokens == 6
    assert _metrics(child).total_completion_tokens == 2
    assert _metrics(grandchild).total_completion_tokens == 3


def test_f4_duplicated_session_meta_preserves_child_identity(tmp_path: Path):
    root_events = _thread_events(ROOT, [(2, 1)])
    child_events = _forked_events(root_events, CHILD, [(3, 1)])
    _write_rollout(tmp_path, "rollout-root.jsonl", root_events)
    _write_rollout(tmp_path, "rollout-child.jsonl", child_events)
    _write_stdout(tmp_path, ROOT)

    trajectory = _convert(tmp_path)
    child = _children(trajectory)[0]

    assert trajectory.trajectory_id == ROOT
    assert child.trajectory_id == CHILD
    assert (child.extra or {})["is_subagent"] is True
    assert (child.extra or {})["parent_thread_id"] == ROOT
    assert (child.extra or {})["agent_nickname"] == "Euclid"


def test_f5_missing_or_malformed_fork_metadata_uses_ordered_parent_fallback(
    tmp_path: Path,
):
    root_events = _thread_events(ROOT, [(11, 2)], total_usage=[(11, 2)])
    cases = {
        "missing": root_events[1:],
        "malformed": [
            {"type": "session_meta", "payload": {"id": None, "source": []}},
            *root_events[1:],
        ],
    }

    for label, copied_segment in cases.items():
        case_dir = tmp_path / label
        child_events = _forked_events(
            [root_events[0], *copied_segment],
            CHILD,
            [(4, 1)],
            total_usage=[(15, 3)],
            include_copied_session_meta=False,
        )
        _write_rollout(case_dir, "rollout-root.jsonl", root_events)
        _write_rollout(case_dir, "rollout-child.jsonl", child_events)
        _write_stdout(case_dir, ROOT)

        trajectory = _convert(case_dir)
        child = _children(trajectory)[0]
        assert _metrics(child).total_prompt_tokens == 4, label
        assert _metrics(child).total_completion_tokens == 1, label


def test_f6_unavailable_ancestor_marks_metrics_incomplete(tmp_path: Path):
    parent_events = _thread_events(ROOT, [(12, 2)], total_usage=[(12, 2)])
    child_events = _forked_events(
        parent_events,
        CHILD,
        [(4, 1)],
        total_usage=[(16, 3)],
    )
    _write_rollout(tmp_path, "rollout-child.jsonl", child_events)

    trajectory = _convert(tmp_path)
    metrics = _metrics(trajectory)

    assert metrics.extra is not None
    assert metrics.extra["tree_metrics_complete"] is False
    assert metrics.extra["tree_cost_complete"] is False
    assert metrics.extra["self_only"]["total_prompt_tokens"] is None
    assert metrics.extra["self_only"]["total_completion_tokens"] is None
    assert metrics.extra["self_only"]["total_cost_usd"] is None
    assert metrics.extra["self_only"]["total_steps"] == 2
    assert metrics.total_prompt_tokens is None
    assert metrics.total_completion_tokens is None
    assert metrics.total_cached_tokens is None
    assert metrics.total_cost_usd is None
    assert metrics.total_steps == 2


# Structure group


def test_s1_subagent_sorted_first_does_not_win_over_stdout_root(tmp_path: Path):
    root_events = _thread_events(ROOT, [(2, 1)])
    child_events = _thread_events(
        CHILD,
        [(3, 1)],
        parent_thread_id=ROOT,
        nickname="Euclid",
        source={"subagent": {}},
        thread_source="subagent",
    )
    _write_rollout(tmp_path, "rollout-0000-child.jsonl", child_events)
    _write_rollout(tmp_path, "rollout-9999-root.jsonl", root_events)
    _write_stdout(tmp_path, ROOT)

    trajectory = _convert(tmp_path)

    assert trajectory.trajectory_id == ROOT
    assert [child.trajectory_id for child in _children(trajectory)] == [CHILD]


def test_s2_rollouts_split_across_date_directories_are_both_found(tmp_path: Path):
    root_events = _thread_events(ROOT, [(2, 1)])
    child_events = _thread_events(
        CHILD,
        [(3, 1)],
        parent_thread_id=ROOT,
        nickname="Euclid",
        source={"subagent": {}},
        thread_source="subagent",
    )
    _write_rollout(
        tmp_path, "rollout-root.jsonl", root_events, date=("2026", "01", "01")
    )
    _write_rollout(
        tmp_path, "rollout-child.jsonl", child_events, date=("2026", "01", "02")
    )

    trajectory = _convert(tmp_path)

    assert trajectory.trajectory_id == ROOT
    assert _children(trajectory)[0].trajectory_id == CHILD


def test_s3_orphan_user_rollout_is_excluded_and_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    root_events = _thread_events(ROOT, [(2, 1)])
    child_events = _thread_events(
        CHILD,
        [(3, 1)],
        parent_thread_id=ROOT,
        nickname="Euclid",
        source={"subagent": {}},
        thread_source="subagent",
    )
    orphan_events = _thread_events(ORPHAN, [(99, 99)])
    _write_rollout(tmp_path, "rollout-root.jsonl", root_events)
    _write_rollout(tmp_path, "rollout-child.jsonl", child_events)
    _write_rollout(tmp_path, "rollout-orphan.jsonl", orphan_events)
    caplog.set_level(logging.WARNING)

    trajectory = _convert(tmp_path)

    assert trajectory.trajectory_id == ROOT
    assert [child.trajectory_id for child in _children(trajectory)] == [CHILD]
    assert ORPHAN in caplog.text
    assert "Ignoring" in caplog.text


def test_s4_stdout_graph_fallback_and_ambiguous_multi_root_are_safe(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    graph_case = tmp_path / "graph"
    root_events = _thread_events(ROOT, [(2, 1)])
    child_events = _thread_events(
        CHILD,
        [(3, 1)],
        parent_thread_id=ROOT,
        nickname="Euclid",
        source={"subagent": {}},
        thread_source="subagent",
    )
    _write_rollout(graph_case, "rollout-root.jsonl", root_events)
    _write_rollout(graph_case, "rollout-child.jsonl", child_events)
    assert _convert(graph_case).trajectory_id == ROOT

    stdout_case = tmp_path / "stdout"
    _write_rollout(stdout_case, "rollout-0000-child.jsonl", child_events)
    _write_rollout(stdout_case, "rollout-9999-root.jsonl", root_events)
    _write_stdout(stdout_case, ROOT)
    assert _convert(stdout_case).trajectory_id == ROOT
    (stdout_case / "codex.txt").unlink()
    assert _convert(stdout_case).trajectory_id == ROOT

    ambiguous_case = tmp_path / "ambiguous"
    _write_rollout(
        ambiguous_case, "rollout-a.jsonl", _thread_events("root-a", [(1, 1)])
    )
    _write_rollout(
        ambiguous_case, "rollout-b.jsonl", _thread_events("root-b", [(2, 2)])
    )
    caplog.set_level(logging.ERROR)
    assert _convert_optional(ambiguous_case) is None
    assert "Ambiguous Codex root thread" in caplog.text


def test_s5_grandchild_is_nested_under_real_parent(tmp_path: Path):
    root_events = _thread_events(ROOT, [(1, 1)])
    child_events = _thread_events(
        CHILD,
        [(2, 1)],
        parent_thread_id=ROOT,
        nickname="Euclid",
        source={"subagent": {}},
        thread_source="subagent",
    )
    grandchild_events = _thread_events(
        GRANDCHILD,
        [(3, 1)],
        parent_thread_id=CHILD,
        nickname="Archimedes",
        source={"subagent": {}},
        thread_source="subagent",
    )
    _write_rollout(tmp_path, "rollout-root.jsonl", root_events)
    _write_rollout(tmp_path, "rollout-child.jsonl", child_events)
    _write_rollout(tmp_path, "rollout-grandchild.jsonl", grandchild_events)

    trajectory = _convert(tmp_path)
    child = _children(trajectory)[0]

    assert [item.trajectory_id for item in _children(trajectory)] == [CHILD]
    assert [item.trajectory_id for item in child.subagent_trajectories] == [GRANDCHILD]


def test_s6_single_rollout_keeps_shape_and_own_totals(tmp_path: Path):
    root_events = _thread_events(ROOT, [(4, 1), (6, 2)], total_usage=[(4, 1), (10, 3)])
    _write_rollout(tmp_path, "rollout-root.jsonl", root_events)

    trajectory = _convert(tmp_path)
    metrics = _metrics(trajectory)

    assert trajectory.subagent_trajectories is None
    assert metrics.total_prompt_tokens == 10
    assert metrics.total_completion_tokens == 3
    assert metrics.total_steps == 2


# Metrics group


def test_m0_rate_limit_snapshot_does_not_repeat_last_usage(tmp_path: Path):
    root_events = _thread_events(ROOT, [(14_727, 93)], total_usage=[(14_727, 93)])
    repeated = json.loads(json.dumps(root_events[-1]))
    repeated["payload"]["rate_limits"] = {
        "limit_id": "premium",
        "rate_limit_reached_type": "workspace_member_credits_depleted",
    }
    root_events.append(repeated)
    _write_rollout(tmp_path, "rollout-root.jsonl", root_events)

    metrics = _metrics(_convert(tmp_path))

    assert metrics.total_prompt_tokens == 14_727
    assert metrics.total_completion_tokens == 93
    assert metrics.total_steps == 1


def test_m0_two_identical_incremental_calls_are_both_counted(tmp_path: Path):
    root_events = _thread_events(
        ROOT,
        [(7, 3), (7, 3)],
        total_usage=[(7, 3), (14, 6)],
    )
    _write_rollout(tmp_path, "rollout-root.jsonl", root_events)

    metrics = _metrics(_convert(tmp_path))

    assert metrics.total_prompt_tokens == 14
    assert metrics.total_completion_tokens == 6


def test_m0_local_context_snapshot_is_not_billed_as_model_usage(tmp_path: Path):
    root_events = _thread_events(ROOT, [(10, 2)], total_usage=[(10, 2)])
    root_events.append(
        {
            "type": "event_msg",
            "payload": {
                "type": "token_count",
                "info": {
                    "total_token_usage": {
                        "input_tokens": 10,
                        "output_tokens": 2,
                        "cached_input_tokens": 0,
                        "total_tokens": 12,
                    },
                    "last_token_usage": {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cached_input_tokens": 0,
                        "total_tokens": 50_000,
                    },
                },
            },
        }
    )
    _write_rollout(tmp_path, "rollout-root.jsonl", root_events)

    metrics = _metrics(_convert(tmp_path))

    assert metrics.total_prompt_tokens == 10
    assert metrics.total_completion_tokens == 2
    assert metrics.total_steps == 1


def test_m0_context_window_full_snapshot_preserves_prior_billed_usage(
    tmp_path: Path,
):
    root_events = _thread_events(ROOT, [(10, 2)], total_usage=[(10, 2)])
    root_events.append(
        {
            "type": "event_msg",
            "payload": {
                "type": "token_count",
                "info": {
                    "model_context_window": 272_000,
                    "total_token_usage": {
                        "input_tokens": 0,
                        "cached_input_tokens": 0,
                        "cache_write_input_tokens": 0,
                        "output_tokens": 0,
                        "reasoning_output_tokens": 0,
                        "total_tokens": 272_000,
                    },
                    "last_token_usage": {
                        "input_tokens": 0,
                        "cached_input_tokens": 0,
                        "cache_write_input_tokens": 0,
                        "output_tokens": 0,
                        "reasoning_output_tokens": 0,
                        "total_tokens": 271_988,
                    },
                },
            },
        }
    )
    _write_rollout(tmp_path, "rollout-root.jsonl", root_events)

    metrics = _metrics(_convert(tmp_path))

    assert metrics.total_prompt_tokens == 10
    assert metrics.total_completion_tokens == 2
    assert metrics.total_steps == 1


def test_m0_context_window_full_without_prior_usage_does_not_invent_usage(
    tmp_path: Path,
):
    root_events = [
        _session_meta(ROOT),
        {
            "type": "event_msg",
            "payload": {
                "type": "token_count",
                "info": {
                    "model_context_window": 272_000,
                    "total_token_usage": {
                        "input_tokens": 0,
                        "cached_input_tokens": 0,
                        "cache_write_input_tokens": 0,
                        "output_tokens": 0,
                        "reasoning_output_tokens": 0,
                        "total_tokens": 272_000,
                    },
                    "last_token_usage": {
                        "input_tokens": 0,
                        "cached_input_tokens": 0,
                        "cache_write_input_tokens": 0,
                        "output_tokens": 0,
                        "reasoning_output_tokens": 0,
                        "total_tokens": 272_000,
                    },
                },
            },
        },
    ]
    usage, complete = Codex._final_cumulative_usage(root_events)

    assert usage is None
    assert complete is True


def test_m0_full_history_child_uses_copied_total_as_snapshot_baseline(
    tmp_path: Path,
):
    root_events = _thread_events(ROOT, [(10, 2)], total_usage=[(10, 2)])
    child_events = _forked_events(
        root_events,
        CHILD,
        [(5, 1)],
        total_usage=[(15, 3)],
    )
    # A rate-limit-only emission can repeat the parent's last non-zero usage
    # before the child's first real model response. It is local to the child and
    # therefore is not removed with the copied history block.
    repeated_parent_snapshot = json.loads(json.dumps(root_events[-1]))
    repeated_parent_snapshot["payload"]["rate_limits"] = {"limit_id": "premium"}
    child_events.insert(1 + len(root_events), repeated_parent_snapshot)
    _write_rollout(tmp_path, "rollout-root.jsonl", root_events)
    _write_rollout(tmp_path, "rollout-child.jsonl", child_events)
    _write_stdout(tmp_path, ROOT)

    trajectory = _convert(tmp_path)
    child = _children(trajectory)[0]

    assert _metrics(child).total_prompt_tokens == 5
    assert _metrics(child).total_completion_tokens == 1
    assert _metrics(trajectory).total_prompt_tokens == 15
    assert _metrics(trajectory).total_completion_tokens == 3


def test_m0_full_history_child_counter_reset_is_incomplete(tmp_path: Path):
    root_events = _thread_events(ROOT, [(10, 2)], total_usage=[(10, 2)])
    child_events = _forked_events(
        root_events,
        CHILD,
        [(5, 1)],
        # Lower than the inherited parent baseline: the child cumulative counter
        # cannot be converted to a trustworthy child-local delta.
        total_usage=[(5, 1)],
    )
    _write_rollout(tmp_path, "rollout-root.jsonl", root_events)
    _write_rollout(tmp_path, "rollout-child.jsonl", child_events)
    _write_stdout(tmp_path, ROOT)

    trajectory = _convert(tmp_path)
    child_metrics = _metrics(_children(trajectory)[0])
    tree_metrics = _metrics(trajectory)

    assert child_metrics.extra is not None
    assert child_metrics.extra["metrics_complete"] is False
    assert child_metrics.total_prompt_tokens is None
    assert child_metrics.total_completion_tokens is None
    assert child_metrics.total_cost_usd is None
    assert child_metrics.total_steps == 1
    assert tree_metrics.extra["tree_metrics_complete"] is False
    assert tree_metrics.total_prompt_tokens is None
    assert tree_metrics.total_completion_tokens is None
    assert tree_metrics.total_cost_usd is None
    assert tree_metrics.total_steps == 2


def test_m1_root_has_tree_totals_self_only_and_scoped_child_metrics(tmp_path: Path):
    root_events = _thread_events(ROOT, [(4, 1)], total_usage=[(4, 1)])
    child_events = _thread_events(
        CHILD,
        [(6, 2)],
        parent_thread_id=ROOT,
        nickname="Euclid",
        source={"subagent": {}},
        thread_source="subagent",
        total_usage=[(6, 2)],
    )
    _write_rollout(tmp_path, "rollout-root.jsonl", root_events)
    _write_rollout(tmp_path, "rollout-child.jsonl", child_events)

    trajectory = _convert(tmp_path)
    root_metrics = _metrics(trajectory)
    child_metrics = _metrics(_children(trajectory)[0])
    self_only = root_metrics.extra["self_only"]

    assert root_metrics.total_prompt_tokens == 10
    assert root_metrics.total_completion_tokens == 3
    assert root_metrics.total_steps == 2
    assert self_only["total_prompt_tokens"] == 4
    assert self_only["total_completion_tokens"] == 1
    assert child_metrics.total_prompt_tokens == 6
    assert child_metrics.total_completion_tokens == 2
    assert child_metrics.total_steps == 1


def test_m2_peak_context_is_max_and_summarization_is_sum(tmp_path: Path):
    root_events = _thread_events(
        ROOT,
        [(2, 1)],
        contexts=[100],
        compacted=1,
    )
    child_events = _thread_events(
        CHILD,
        [(3, 1)],
        parent_thread_id=ROOT,
        nickname="Euclid",
        source={"subagent": {}},
        thread_source="subagent",
        contexts=[500],
        compacted=2,
    )
    _write_rollout(tmp_path, "rollout-root.jsonl", root_events)
    _write_rollout(tmp_path, "rollout-child.jsonl", child_events)

    trajectory = _convert(tmp_path)
    extra = _metrics(trajectory).extra

    assert extra is not None
    assert extra["peak_context_tokens"] == 500
    assert extra["summarization_count"] == 3


def test_m3_each_thread_cost_uses_its_rollout_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import litellm

    monkeypatch.setitem(
        litellm.model_cost,
        "root-model",
        {
            "input_cost_per_token": 1e-6,
            "output_cost_per_token": 2e-6,
            "cache_read_input_token_cost": 1e-6,
        },
    )
    monkeypatch.setitem(
        litellm.model_cost,
        "child-model",
        {
            "input_cost_per_token": 10e-6,
            "output_cost_per_token": 20e-6,
            "cache_read_input_token_cost": 10e-6,
        },
    )
    root_events = _thread_events(ROOT, [(10, 2)], model="root-model")
    child_events = _thread_events(
        CHILD,
        [(10, 2)],
        model="child-model",
        parent_thread_id=ROOT,
        nickname="Euclid",
        source={"subagent": {}},
        thread_source="subagent",
    )
    _write_rollout(tmp_path, "rollout-root.jsonl", root_events)
    _write_rollout(tmp_path, "rollout-child.jsonl", child_events)

    trajectory = _convert(tmp_path)
    root_cost = _metrics(trajectory).extra["self_only"]["total_cost_usd"]
    child_cost = _metrics(_children(trajectory)[0]).total_cost_usd
    assert root_cost == pytest.approx(14e-6)
    assert child_cost == pytest.approx(140e-6)
    assert _metrics(trajectory).total_cost_usd == pytest.approx(154e-6)


def test_m4_agent_step_count_recurses_through_nested_subagents(tmp_path: Path):
    root_events = _thread_events(ROOT, [(1, 1)])
    child_events = _thread_events(
        CHILD,
        [(1, 1)],
        parent_thread_id=ROOT,
        nickname="Euclid",
        source={"subagent": {}},
        thread_source="subagent",
    )
    grandchild_events = _thread_events(
        GRANDCHILD,
        [(1, 1)],
        parent_thread_id=CHILD,
        nickname="Archimedes",
        source={"subagent": {}},
        thread_source="subagent",
    )
    _write_rollout(tmp_path, "rollout-root.jsonl", root_events)
    _write_rollout(tmp_path, "rollout-child.jsonl", child_events)
    _write_rollout(tmp_path, "rollout-grandchild.jsonl", grandchild_events)
    trajectory = _convert(tmp_path)
    trajectory_path = tmp_path / "trajectory.json"
    trajectory_path.write_text(json.dumps(trajectory.to_json_dict()))

    assert _agent_step_count_from_trajectory_path(trajectory_path) == 3
