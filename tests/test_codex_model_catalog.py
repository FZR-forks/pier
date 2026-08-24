import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from pier.agents.installed.codex import Codex

MODEL = "gpt-5.6-luna"
OTHER_MODEL = "gpt-5.6-orbit"
THIRD_MODEL = "gpt-5.5"


@pytest.fixture
def mini_catalog() -> dict[str, object]:
    """A small catalog with the metadata Codex uses at runtime."""
    return {
        "models": [
            {
                "slug": MODEL,
                "supported_reasoning_levels": [
                    {"effort": "low", "description": "Fast"},
                    {"effort": "xhigh", "description": "Deep"},
                ],
                "default_reasoning_level": "xhigh",
                "prefer_websockets": True,
                "context_window": 272000,
                "tool_mode": "code_mode_only",
                "input_modalities": ["text"],
                "supports_reasoning_summaries": True,
            },
            {
                "slug": OTHER_MODEL,
                "supported_reasoning_levels": [
                    {"effort": "medium", "description": "Balanced"},
                ],
                "default_reasoning_level": "medium",
                "prefer_websockets": True,
                "context_window": 128000,
                "tool_mode": "code_mode_only",
            },
            {
                "slug": THIRD_MODEL,
                "supported_reasoning_levels": [
                    {"effort": "high", "description": "Deep"},
                ],
                "default_reasoning_level": "high",
                "prefer_websockets": False,
                "context_window": 200000,
                "tool_mode": "full_mode",
            },
        ]
    }


def _agent(tmp_path: Path, **kwargs: object) -> Codex:
    return Codex(logs_dir=tmp_path, model_name=MODEL, **kwargs)


@pytest.mark.parametrize(
    ("restrict_model_catalog", "use_catalog_file"),
    [(False, False), (True, False), (True, True)],
)
def test_constructor_accepts_valid_catalog_options(
    tmp_path: Path, restrict_model_catalog: bool, use_catalog_file: bool
) -> None:
    catalog_file = None
    if use_catalog_file:
        path = tmp_path / "model-catalog.json"
        path.write_text('{"models": []}')
        catalog_file = str(path)

    agent = _agent(
        tmp_path,
        restrict_model_catalog=restrict_model_catalog,
        model_catalog_file=catalog_file,
    )

    assert agent._restrict_model_catalog is restrict_model_catalog
    assert agent._model_catalog_file == catalog_file


def test_constructor_rejects_catalog_file_without_restriction(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="restrict_model_catalog"):
        _agent(
            tmp_path,
            model_catalog_file=str(tmp_path / "model-catalog.json"),
        )


def test_narrow_model_catalog_keeps_selected_metadata_and_requested_effort(
    mini_catalog: dict[str, object],
) -> None:
    original = copy.deepcopy(mini_catalog)
    selected = mini_catalog["models"][0]  # type: ignore[index]

    narrowed = Codex.narrow_model_catalog(mini_catalog, MODEL, "low")

    expected = copy.deepcopy(selected)
    expected["supported_reasoning_levels"] = [  # type: ignore[index]
        {"effort": "low", "description": "Fast"}
    ]
    expected["default_reasoning_level"] = "low"  # type: ignore[index]
    # Narrowing constrains model and effort only; transport preference is the
    # supplied entry's own metadata and must survive untouched.
    assert narrowed == {"models": [expected]}
    assert mini_catalog == original


def test_narrow_model_catalog_leaves_reasoning_levels_untouched_without_effort(
    mini_catalog: dict[str, object],
) -> None:
    original = copy.deepcopy(mini_catalog)

    narrowed = Codex.narrow_model_catalog(mini_catalog, MODEL, None)

    expected = copy.deepcopy(mini_catalog["models"][0])  # type: ignore[index]
    # Narrowing constrains model and effort only; transport preference is the
    # supplied entry's own metadata and must survive untouched.
    assert narrowed == {"models": [expected]}
    assert mini_catalog == original


def test_narrow_model_catalog_reports_unknown_model_and_known_slugs(
    mini_catalog: dict[str, object],
) -> None:
    with pytest.raises(ValueError) as exc_info:
        Codex.narrow_model_catalog(mini_catalog, "gpt-5.6-unknown", "low")

    message = str(exc_info.value)
    assert "model_catalog_file" in message
    assert MODEL in message
    assert OTHER_MODEL in message
    assert THIRD_MODEL in message


def test_narrow_model_catalog_rejects_unsupported_effort(
    mini_catalog: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="does not support reasoning effort 'medium'"):
        Codex.narrow_model_catalog(mini_catalog, MODEL, "medium")


def test_benchmark_config_overrides_are_empty_when_restriction_is_off(
    tmp_path: Path,
) -> None:
    assert (
        _agent(
            tmp_path,
            restrict_model_catalog=False,
            reasoning_effort="xhigh",
        )._benchmark_config_overrides(MODEL)
        == []
    )


def test_benchmark_config_overrides_pin_model_catalog_and_effort(
    tmp_path: Path,
) -> None:
    overrides = _agent(
        tmp_path,
        restrict_model_catalog=True,
        reasoning_effort="xhigh",
    )._benchmark_config_overrides(MODEL)

    assert {
        "model_catalog_json=/tmp/codex-home/model-catalog.json",
        f"agents.default_subagent_model={MODEL}",
        "features.multi_agent_v2.expose_spawn_agent_model_overrides=false",
        "agents.default_subagent_reasoning_effort=xhigh",
    } == set(overrides)


def test_benchmark_config_overrides_omit_unconfigured_effort(tmp_path: Path) -> None:
    overrides = _agent(
        tmp_path,
        restrict_model_catalog=True,
        reasoning_effort=None,
    )._benchmark_config_overrides(MODEL)

    assert {
        "model_catalog_json=/tmp/codex-home/model-catalog.json",
        f"agents.default_subagent_model={MODEL}",
        "features.multi_agent_v2.expose_spawn_agent_model_overrides=false",
    } == set(overrides)
    assert not any(
        override.startswith("agents.default_subagent_reasoning_effort=")
        for override in overrides
    )


@pytest.mark.asyncio
async def test_install_model_catalog_narrows_bundled_catalog(
    tmp_path: Path, mini_catalog: dict[str, object]
) -> None:
    agent = _agent(
        tmp_path,
        restrict_model_catalog=True,
        reasoning_effort="low",
    )
    agent.exec_as_agent = AsyncMock(
        # The sandbox merges stderr into stdout and Codex prints warnings there,
        # so the selector delimits its payload; mimic that here.
        side_effect=[
            SimpleNamespace(
                stdout=(
                    "WARNING: proceeding, even though we could not create PATH "
                    "aliases\n<<CATALOG>>" + json.dumps(mini_catalog) + "<<END>>"
                )
            ),
            None,
        ]
    )

    await agent._install_model_catalog(SimpleNamespace(), MODEL, {})

    calls = agent.exec_as_agent.await_args_list
    assert len(calls) == 2
    assert "codex debug models --bundled" in calls[0].kwargs["command"]

    command = calls[1].kwargs["command"]
    payload = command.split("<<'CATALOG'\n", 1)[1].rsplit("\nCATALOG", 1)[0]
    assert json.loads(payload) == Codex.narrow_model_catalog(mini_catalog, MODEL, "low")
    assert calls[1].kwargs["env"] == {}
