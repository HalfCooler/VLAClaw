from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from vlaclaw.agent import AgentResult
from vlaclaw.cli import CliConfig, ProviderConfig, _execute_agent, load_config


class _Backend:
    platform = "android"


def _provider(model: str) -> ProviderConfig:
    return ProviderConfig(base_url="https://example.test/v1", model=model)


def _args() -> SimpleNamespace:
    return SimpleNamespace(agent_profile=None, json_output=False)


def test_load_config_defaults_large_model_test_to_false(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "provider:\n  base_url: https://example.test/v1\n  model: small\n",
        encoding="utf-8",
    )

    assert load_config(config_path).large_model_test is False


def test_load_config_reads_large_model_test(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            (
                "provider:",
                "  base_url: https://example.test/v1",
                "  model: small",
                "postprocess_provider:",
                "  base_url: https://example.test/v1",
                "  model: large",
                "large_model_test: true",
            )
        ),
        encoding="utf-8",
    )

    assert load_config(config_path).large_model_test is True


@pytest.mark.asyncio
async def test_large_model_test_uses_large_model_as_the_actor(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    large_llm = object()

    def build_provider(config: ProviderConfig) -> object:
        assert config.model == "large"
        return large_llm

    class CapturingAgent:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        async def run(self, task: str) -> AgentResult:
            assert task == "Complete the task"
            return AgentResult(True, "done", None, None, 0, None)

    monkeypatch.setattr("vlaclaw.cli.build_llm_provider", build_provider)
    monkeypatch.setattr("vlaclaw.cli.GuiAgent", CapturingAgent)
    config = CliConfig(
        provider=_provider("small"),
        postprocess_provider=_provider("large"),
        large_model_test=True,
        enable_repeat_escalation=False,
        enable_difficulty_routing=False,
    )

    result = await _execute_agent(_args(), config, _Backend(), object(), "Complete the task")

    assert result.success is True
    assert captured["llm"] is large_llm
    assert captured["model"] == "large"
    assert captured["actor_role"] == "large"


@pytest.mark.asyncio
async def test_large_model_test_requires_a_large_model_provider() -> None:
    config = CliConfig(provider=_provider("small"), large_model_test=True)

    with pytest.raises(ValueError, match="large_model_test requires postprocess_provider"):
        await _execute_agent(_args(), config, _Backend(), object(), "Complete the task")
