"""Standalone CLI: ``vlaclaw --backend adb "<task>"``."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import json_repair
import yaml
from openai import AsyncOpenAI

from vlaclaw.agent import AgentResult, GuiAgent
from vlaclaw.agent_profiles import SUPPORTED_AGENT_PROFILES
from vlaclaw.backends.adb import AdbBackend
from vlaclaw.difficulty import resolve_difficulty_route
from vlaclaw.interfaces import LLMResponse, ToolCall
from vlaclaw.paths import DEFAULT_GUI_RUNS_DIR
from vlaclaw.planner_escalation import canonicalize_repeat_judge_model
from vlaclaw.trajectory.recorder import TrajectoryRecorder

logger = logging.getLogger(__name__)
DEFAULT_CONFIG_PATH = Path.home() / ".vlaclaw" / "config.yaml"


@dataclass(slots=True)
class ProviderConfig:
    base_url: str
    model: str
    api_key: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    vl_high_resolution_images: bool | None = None
    reasoning_effort: str | None = None
    extra_body: dict[str, Any] | None = None


@dataclass(slots=True)
class AdbConfig:
    serial: str | None = None
    adb_path: str = "adb"
    app_aliases: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class CliConfig:
    provider: ProviderConfig
    postprocess_provider: ProviderConfig | None = None
    large_model_test: bool = False
    adb: AdbConfig = field(default_factory=AdbConfig)
    max_steps: int = 15
    stagnation_limit: int = 0
    image_scale_ratio: float = 0.5
    history_image_window: int | None = None
    enable_repeat_escalation: bool = True
    repeat_judge_model: str = "small"
    enable_difficulty_routing: bool = False
    agent_profile: str | None = None


class OpenAICompatibleLLMProvider:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        vl_high_resolution_images: bool | None = None,
        reasoning_effort: str | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self._base_url = base_url
        self._model = model
        self._temperature = temperature
        self._top_p = top_p
        self._vl_high_resolution_images = vl_high_resolution_images
        self._reasoning_effort = reasoning_effort
        self._extra_body = dict(extra_body or {})
        self._client = AsyncOpenAI(api_key=api_key or "no-key", base_url=base_url)

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> LLMResponse:
        effective_model = model or self._model
        kwargs: dict[str, Any] = {
            "model": effective_model,
            "messages": _sanitize_messages(messages),
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice or "auto"
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if self._temperature is not None:
            kwargs["temperature"] = self._temperature
        if self._top_p is not None:
            kwargs["top_p"] = self._top_p
        extra_body: dict[str, Any] = {}
        high_resolution = self._vl_high_resolution_images
        if high_resolution is None:
            high_resolution = _is_dashscope_gui_plus(self._base_url, effective_model)
        if high_resolution:
            extra_body["vl_high_resolution_images"] = True
        effective_reasoning_effort = reasoning_effort or self._reasoning_effort
        if effective_reasoning_effort:
            thinking_enabled = effective_reasoning_effort.strip().lower() not in {
                "none",
                "minimal",
                "minimum",
            }
            if _is_dashscope_endpoint(self._base_url):
                extra_body["enable_thinking"] = thinking_enabled
            else:
                extra_body["chat_template_kwargs"] = {"enable_thinking": thinking_enabled}
        extra_body = _deep_merge(extra_body, self._extra_body)
        if extra_body:
            kwargs["extra_body"] = extra_body
        inference_started_at = time.perf_counter()
        response = await self._client.chat.completions.create(**kwargs)
        latency_s = time.perf_counter() - inference_started_at
        if not response.choices:
            raise RuntimeError("OpenAI-compatible API returned no choices")
        choice = response.choices[0]
        message = choice.message
        parsed_tool_calls: list[ToolCall] = []
        for index, tool_call in enumerate(message.tool_calls or []):
            parsed_tool_calls.append(
                ToolCall(
                    id=tool_call.id or f"tool-call-{index}",
                    name=tool_call.function.name,
                    arguments=_parse_tool_arguments(tool_call.function.arguments),
                )
            )
        usage_obj = getattr(response, "usage", None)
        usage: dict[str, int] = (
            {
                "prompt_tokens": getattr(usage_obj, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(usage_obj, "completion_tokens", 0) or 0,
                "total_tokens": getattr(usage_obj, "total_tokens", 0) or 0,
            }
            if usage_obj is not None
            else {}
        )
        return LLMResponse(
            content=_coerce_message_content(message.content),
            tool_calls=parsed_tool_calls or None,
            raw=response,
            usage=usage,
            latency_s=latency_s,
        )


def build_llm_provider(config: ProviderConfig) -> OpenAICompatibleLLMProvider:
    return OpenAICompatibleLLMProvider(
        base_url=config.base_url,
        model=config.model,
        api_key=config.api_key,
        temperature=config.temperature,
        top_p=config.top_p,
        vl_high_resolution_images=config.vl_high_resolution_images,
        reasoning_effort=config.reasoning_effort,
        extra_body=config.extra_body,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="vlaclaw")
    parser.add_argument("task_input", nargs="?", help="Task description")
    parser.add_argument("--task", dest="task_flag", help="Task description")
    parser.add_argument(
        "--backend",
        choices=("adb",),
        default="adb",
        help="Execution backend (VLAClaw currently supports adb only)",
    )
    parser.add_argument(
        "--agent-profile",
        choices=SUPPORTED_AGENT_PROFILES,
        default=None,
        help="Skip difficulty routing and force this profile.",
    )
    parser.add_argument("--json", dest="json_output", action="store_true", help="Emit JSON output")
    parser.add_argument("--config", type=Path, help="Config file path")
    args = parser.parse_args(argv)
    if not args.task_input and not args.task_flag:
        parser.error("task is required via positional input or --task")
    return args


def resolve_task(args: argparse.Namespace) -> str:
    task_flag = (args.task_flag or "").strip()
    task_input = (args.task_input or "").strip()
    if task_flag and task_input and task_flag != task_input:
        raise ValueError("Positional task and --task disagree")
    task = task_flag or task_input
    if not task:
        raise ValueError("Task is required")
    return task


def load_config(path: Path | None = None) -> CliConfig:
    config_path = path or DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("Config root must be a mapping")
    provider_raw = _require_mapping(raw, "provider")
    provider = _load_provider(provider_raw, fallback_api_key=os.getenv("OPENAI_API_KEY"))
    postprocess_raw = raw.get("postprocess_provider")
    postprocess_provider = None
    if postprocess_raw is not None:
        if not isinstance(postprocess_raw, dict):
            raise ValueError("postprocess_provider config must be a mapping")
        postprocess_provider = _load_provider(
            postprocess_raw,
            fallback_api_key=os.getenv("OPENAI_API_KEY") or provider.api_key,
        )
    adb_raw = raw.get("adb") or {}
    if not isinstance(adb_raw, dict):
        raise ValueError("adb config must be a mapping")
    return CliConfig(
        provider=provider,
        postprocess_provider=postprocess_provider,
        large_model_test=_coerce_bool(raw.get("large_model_test"), default=False),
        adb=AdbConfig(
            serial=_optional_string(adb_raw, "serial"),
            adb_path=_optional_string(adb_raw, "adb_path") or "adb",
            app_aliases=_coerce_app_aliases(adb_raw.get("app_aliases")),
        ),
        max_steps=_coerce_positive_int(raw.get("max_steps"), default=15),
        stagnation_limit=_coerce_non_negative_int(raw.get("stagnation_limit"), default=0),
        image_scale_ratio=_coerce_image_scale_ratio(raw.get("image_scale_ratio"), default=0.5),
        history_image_window=_coerce_optional_positive_int(raw.get("history_image_window")),
        enable_repeat_escalation=_coerce_bool(raw.get("enable_repeat_escalation"), default=True),
        repeat_judge_model=_coerce_repeat_judge_model(raw.get("repeat_judge_model")),
        enable_difficulty_routing=_coerce_bool(raw.get("enable_difficulty_routing"), default=False),
        agent_profile=_optional_string(raw, "agent_profile"),
    )


def _load_provider(raw: dict[str, Any], *, fallback_api_key: str | None) -> ProviderConfig:
    extra_body = raw.get("extra_body")
    if extra_body is not None and not isinstance(extra_body, dict):
        raise ValueError("provider.extra_body config must be a mapping")
    high_resolution_raw = raw.get("vl_high_resolution_images")
    return ProviderConfig(
        base_url=_require_string(raw, "base_url"),
        model=_require_string(raw, "model"),
        api_key=_optional_string(raw, "api_key") or fallback_api_key,
        temperature=_coerce_optional_float(raw.get("temperature"), name="temperature", minimum=0),
        top_p=_coerce_optional_top_p(raw.get("top_p")),
        vl_high_resolution_images=(
            None if high_resolution_raw is None else _coerce_bool(high_resolution_raw, default=False)
        ),
        reasoning_effort=_optional_string(raw, "reasoning_effort"),
        extra_body=extra_body,
    )


def build_backend(config: CliConfig) -> AdbBackend:
    return AdbBackend(
        serial=config.adb.serial,
        adb_path=config.adb.adb_path or "adb",
        use_scrcpy=False,
        collect_ui_tree=True,
        collect_ui_tree_nodes=True,
        app_aliases=config.adb.app_aliases,
    )


async def _execute_agent(
    args: argparse.Namespace,
    config: CliConfig,
    backend: AdbBackend,
    provider: OpenAICompatibleLLMProvider,
    task: str,
) -> AgentResult:
    run_root = DEFAULT_GUI_RUNS_DIR / datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S_%f")
    if config.large_model_test and config.postprocess_provider is None:
        raise ValueError(
            "large_model_test requires postprocess_provider to configure the large model"
        )
    large_llm = (
        build_llm_provider(config.postprocess_provider)
        if config.postprocess_provider is not None
        and (
            config.large_model_test
            or config.enable_repeat_escalation
            or config.enable_difficulty_routing
        )
        else None
    )
    explicit_profile = args.agent_profile or config.agent_profile
    difficulty_route = await resolve_difficulty_route(
        task=task,
        enabled=config.enable_difficulty_routing,
        large_llm=large_llm,
        explicit_profile=explicit_profile,
    )
    if config.large_model_test:
        # Keep the existing agent loop, profiles, and recovery behavior, but
        # replace every normal actor step with the configured large model.
        assert large_llm is not None
        assert config.postprocess_provider is not None
        actor_llm = large_llm
        actor_model = config.postprocess_provider.model
        agent_profile = (
            difficulty_route.agent_profile
            if difficulty_route is not None
            else explicit_profile or config.agent_profile
        )
        difficulty_snapshot = difficulty_route.snapshot() if difficulty_route is not None else None
        if difficulty_snapshot is not None:
            difficulty_snapshot["actor"] = "large"
            difficulty_snapshot["large_model_test"] = True
        actor_reasoning_effort = config.postprocess_provider.reasoning_effort
    elif difficulty_route is not None:
        actor_llm = (large_llm or provider) if difficulty_route.use_large_model else provider
        actor_model = (
            config.postprocess_provider.model
            if difficulty_route.use_large_model and config.postprocess_provider is not None
            else config.provider.model
        )
        agent_profile = difficulty_route.agent_profile
        difficulty_snapshot = difficulty_route.snapshot()
        actor_reasoning_effort = (
            config.postprocess_provider.reasoning_effort
            if difficulty_route.use_large_model and config.postprocess_provider is not None
            else config.provider.reasoning_effort
        )
    else:
        actor_llm = provider
        actor_model = config.provider.model
        agent_profile = explicit_profile or config.agent_profile
        difficulty_snapshot = None
        actor_reasoning_effort = config.provider.reasoning_effort

    recorder = TrajectoryRecorder(output_dir=run_root, task=task, platform=backend.platform)
    agent = GuiAgent(
        llm=actor_llm,
        backend=backend,
        trajectory_recorder=recorder,
        model=actor_model,
        artifacts_root=run_root,
        max_steps=config.max_steps or 15,
        progress_callback=_make_progress_printer(json_output=args.json_output),
        agent_profile=agent_profile,
        planner_llm=large_llm if config.enable_repeat_escalation else None,
        enable_repeat_escalation=config.enable_repeat_escalation,
        repeat_judge_model=config.repeat_judge_model,
        difficulty_snapshot=difficulty_snapshot,
        planner_model=(
            config.postprocess_provider.model if config.postprocess_provider is not None else ""
        ),
        image_scale_ratio=config.image_scale_ratio,
        history_image_window=config.history_image_window,
        stagnation_limit=config.stagnation_limit,
        reasoning_effort=actor_reasoning_effort,
        actor_role="large" if config.large_model_test else "small",
    )
    return await agent.run(task)


async def run_cli(args: argparse.Namespace) -> AgentResult:
    task = resolve_task(args)
    config = load_config(args.config)
    backend = build_backend(config)
    provider = build_llm_provider(config.provider)
    return await _execute_agent(args, config, backend, provider, task)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(list(sys.argv[1:] if argv is None else argv))
    try:
        result = asyncio.run(run_cli(args))
    except Exception as exc:
        result = AgentResult(
            success=False,
            summary="CLI execution failed.",
            model_summary=None,
            trace_path=None,
            steps_taken=0,
            error=f"{type(exc).__name__}: {exc}",
        )
    if args.json_output:
        print(
            json.dumps(
                {
                    "success": result.success,
                    "summary": result.summary,
                    "model_summary": result.model_summary,
                    "trace_path": result.trace_path,
                    "steps_taken": result.steps_taken,
                    "error": result.error,
                }
            )
        )
    else:
        print(f"status: {'success' if result.success else 'failure'}")
        print(f"success: {'true' if result.success else 'false'}")
        print(f"summary: {result.summary}")
        if result.model_summary is not None:
            print(f"model_summary: {result.model_summary}")
        print(f"trace_path: {result.trace_path}")
        print(f"steps_taken: {result.steps_taken}")
        if result.error is not None:
            print(f"error: {result.error}")
    return 0 if result.success else 1


def _make_progress_printer(*, json_output: bool) -> Any:
    from vlaclaw.hf_cli import HfCliProgressPrinter

    return HfCliProgressPrinter(json_output=json_output)


def _sanitize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sanitized: list[dict[str, Any]] = []
    for message in messages:
        normalized = dict(message)
        if normalized.get("content") is None:
            normalized["content"] = ""
        sanitized.append(normalized)
    return sanitized

def _coerce_message_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif hasattr(item, "text"):
                parts.append(str(item.text))
        return "".join(parts)
    return str(content)


def _parse_tool_arguments(arguments: Any) -> dict[str, Any]:
    if arguments is None:
        return {}
    if isinstance(arguments, str):
        if not arguments.strip():
            return {}
        parsed = json_repair.loads(arguments)
    else:
        parsed = arguments
    if isinstance(parsed, dict):
        return parsed
    return {"value": parsed}


def _require_mapping(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} config must be a mapping")
    return value


def _require_string(raw: dict[str, Any], key: str) -> str:
    value = _optional_string(raw, key)
    if not value:
        raise ValueError(f"Missing required config key: {key}")
    return value


def _coerce_app_aliases(value: Any) -> dict[str, str]:
    if value in (None, ""):
        return {}
    if not isinstance(value, dict):
        raise ValueError("adb.app_aliases must be a mapping of name to package")
    aliases: dict[str, str] = {}
    for raw_name, raw_package in value.items():
        name = str(raw_name).strip()
        package = "" if raw_package is None else str(raw_package).strip()
        if not name or not package:
            raise ValueError(f"Invalid adb.app_aliases entry: {raw_name!r} -> {raw_package!r}")
        aliases[name] = package
    return aliases


def _optional_string(raw: dict[str, Any], key: str) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_positive_int(value: Any, *, default: int) -> int:
    if value in (None, ""):
        return default
    parsed = int(value)
    return parsed if parsed > 0 else default


def _coerce_optional_positive_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    parsed = int(value)
    if parsed < 1:
        raise ValueError(f"Expected positive integer, got {value!r}")
    return parsed


def _coerce_non_negative_int(value: Any, *, default: int) -> int:
    if value in (None, ""):
        return default
    parsed = int(value)
    return parsed if parsed >= 0 else default


def _coerce_image_scale_ratio(value: Any, *, default: float) -> float:
    if value in (None, ""):
        return default
    parsed = float(value)
    if not (0 < parsed <= 1):
        raise ValueError(f"Expected image_scale_ratio in (0, 1], got {value!r}")
    return parsed


def _coerce_optional_float(value: Any, *, name: str, minimum: float) -> float | None:
    if value in (None, ""):
        return None
    parsed = float(value)
    if parsed < minimum:
        raise ValueError(f"Expected {name} >= {minimum}, got {value!r}")
    return parsed


def _coerce_optional_top_p(value: Any) -> float | None:
    if value in (None, ""):
        return None
    parsed = float(value)
    if not 0 < parsed <= 1:
        raise ValueError(f"Expected top_p in (0, 1], got {value!r}")
    return parsed


def _coerce_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"Expected boolean, got {value!r}")
    return value


def _coerce_repeat_judge_model(value: Any) -> str:
    if value is None or value == "":
        return "small"
    return canonicalize_repeat_judge_model(str(value))


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _is_dashscope_endpoint(base_url: str) -> bool:
    host = (urlparse(base_url).hostname or "").lower()
    return host.endswith(".aliyuncs.com") and any(
        label.startswith("dashscope") or label == "maas" for label in host.split(".")
    )


def _is_dashscope_gui_plus(base_url: str, model: str) -> bool:
    model_name = model.rsplit("/", 1)[-1].strip().lower()
    return _is_dashscope_endpoint(base_url) and (
        model_name == "gui-plus" or model_name.startswith("gui-plus-")
    )


if __name__ == "__main__":
    raise SystemExit(main())
