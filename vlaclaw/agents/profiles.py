"""Compact and e2e prompt construction plus response normalization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from vlaclaw.agents.implementations import general_e2e_agent
from vlaclaw.agents.runtime.models import (
    ANSWER,
    ASK_USER,
    CLICK,
    DOUBLE_TAP,
    DRAG,
    ENV_FAIL,
    FINISHED,
    INPUT_TEXT,
    KEYBOARD_ENTER,
    LONG_PRESS,
    MCP,
    NAVIGATE_BACK,
    NAVIGATE_HOME,
    OPEN_APP,
    SCROLL,
    UNKNOWN,
    WAIT,
)
from vlaclaw.agents.utils.helpers import pil_adaptive_resize, pil_to_base64
from vlaclaw.agents.utils.prompts import (
    GENERAL_COMPACT_PROMPT_TEMPLATE,
    GENERAL_E2E_PROMPT_TEMPLATE,
)
from vlaclaw.image_utils import normalize_image_scale_ratio
from vlaclaw.interfaces import LLMResponse, ToolCall
from vlaclaw.observation import Observation
from vlaclaw.social_state import like_task_policy

SUPPORTED_AGENT_PROFILES: tuple[str, ...] = ("general_compact", "general_e2e")
_CLAUDE_IMAGE_SIZE = (1280, 720)
_CLAUDE_OPUS_MAX_DIMENSION = 1280
DEFAULT_SCROLL_PIXELS = 400
_PROFILE_LLM_DEFAULTS: dict[str, dict[str, Any]] = {
    "general_compact": {"reasoning_effort": "none", "max_tokens": 48},
}


def canonicalize_agent_profile(profile_name: str | None) -> str:
    key = "general_compact" if profile_name in (None, "") else profile_name
    if key not in SUPPORTED_AGENT_PROFILES:
        raise ValueError(
            f"Unsupported agent profile {profile_name!r}. "
            f"Expected one of: {', '.join(SUPPORTED_AGENT_PROFILES)}."
        )
    return key


def profile_uses_native_tools(profile_name: str | None) -> bool:
    del profile_name
    return False


def coordinate_mode_for_profile(profile_name: str | None, model_name: str = "") -> str:
    del profile_name, model_name
    return "absolute"


def profile_llm_defaults(profile_name: str | None) -> dict[str, Any]:
    return dict(_PROFILE_LLM_DEFAULTS.get(canonicalize_agent_profile(profile_name), {}))


def general_e2e_scale_factor(
    model_name: str,
    screen_width: int,
    screen_height: int,
) -> int | tuple[int, int]:
    return _general_e2e_scale_factor(model_name, screen_width, screen_height)


def build_profile_messages(
    profile_name: str | None,
    *,
    task: str,
    current_observation: Observation,
    history: list[Any],
    model_name: str,
    history_image_window: int | None,
    compact_prompt_parts: Any | None = None,
    available_apps: tuple[str, ...] | list[str] = (),
    image_scale_ratio: float = 1.0,
) -> list[dict[str, Any]]:
    del available_apps
    profile = canonicalize_agent_profile(profile_name)
    effective_history_image_window = (
        1 if history_image_window is None else max(1, int(history_image_window))
    )
    prompt_template = (
        GENERAL_COMPACT_PROMPT_TEMPLATE
        if profile == "general_compact"
        else GENERAL_E2E_PROMPT_TEMPLATE
    )
    return _build_general_e2e_messages(
        task=task,
        current_observation=current_observation,
        history=history,
        model_name=model_name,
        history_image_window=effective_history_image_window,
        prompt_template=prompt_template,
        compact_prompt_parts=compact_prompt_parts,
        image_scale_ratio=(image_scale_ratio if profile == "general_compact" else 1.0),
        rolling_memory_history=(profile == "general_compact"),
    )


def normalize_profile_response_for_observation(
    profile_name: str | None,
    response: LLMResponse,
    observation: Observation,
    *,
    model_name: str = "",
    image_scale_ratio: float = 1.0,
) -> LLMResponse:
    return normalize_profile_response_for_screen(
        profile_name,
        response,
        screen_width=int(observation.screen_width or 999),
        screen_height=int(observation.screen_height or 999),
        model_name=model_name,
        image_scale_ratio=image_scale_ratio,
    )


def normalize_profile_response_for_screen(
    profile_name: str | None,
    response: LLMResponse,
    *,
    screen_width: int,
    screen_height: int,
    model_name: str = "",
    image_scale_ratio: float = 1.0,
    fallback_relative: bool = False,
) -> LLMResponse:
    del image_scale_ratio
    profile = canonicalize_agent_profile(profile_name)
    content = response.content or ""
    if not content.strip() and response.tool_calls:
        return response
    try:
        payload = parse_profile_action(
            profile,
            content,
            screen_width=screen_width,
            screen_height=screen_height,
            model_name=model_name,
        )
    except Exception as exc:
        raise ValueError(f"Failed to parse {profile} response: {exc}") from exc
    if fallback_relative and payload.get("action_type") in {
        "tap",
        "long_press",
        "double_tap",
        "drag",
        "swipe",
        "scroll",
    }:
        payload.setdefault("relative", True)
    return LLMResponse(
        content=response.content,
        tool_calls=[ToolCall(id="content-tool-call-0", name="computer_use", arguments=payload)],
        raw=response.raw,
        usage=response.usage,
        ttft_s=response.ttft_s,
        latency_s=response.latency_s,
    )


def parse_profile_action(
    profile_name: str | None,
    content: str,
    *,
    screen_width: int,
    screen_height: int,
    model_name: str = "",
    image_scale_ratio: float = 1.0,
) -> dict[str, Any]:
    del image_scale_ratio
    profile = canonicalize_agent_profile(profile_name)
    action_str = _general_e2e_action_text(content)
    summary = content
    intent = content
    if profile == "general_compact":
        parsed_action = general_e2e_agent.parse_json_markdown(action_str)
        if isinstance(parsed_action, list) and len(parsed_action) == 1:
            parsed_action = parsed_action[0]
        if isinstance(parsed_action, dict):
            parsed_intent = parsed_action.get("intent")
            parsed_memory = parsed_action.get("memory")
            structured_memory = _general_compact_structured_memory(parsed_action)
            parsed_result = parsed_action.get("result")
            if isinstance(parsed_intent, str) and parsed_intent.strip():
                intent = parsed_intent.strip()
            if isinstance(parsed_memory, str) and parsed_memory.strip():
                summary = parsed_memory.strip()
            elif structured_memory is not None:
                summary = _general_compact_memory_update_summary(structured_memory)
            elif isinstance(parsed_result, str) and parsed_result.strip():
                summary = parsed_result.strip()
            else:
                summary = intent
            if not isinstance(parsed_intent, str) or not parsed_intent.strip():
                intent = summary
            _reject_compact_scroll_without_direction(parsed_action)
    action = general_e2e_agent.parse_response_to_action(
        action_str,
        screen_width,
        screen_height,
        scale_factor=_general_e2e_scale_factor(model_name, screen_width, screen_height),
    )
    payload = _to_guiclaw_payload(action, summary=summary)
    if profile == "general_compact":
        payload["intent"] = intent
    return payload


def _build_general_e2e_messages(
    *,
    task: str,
    current_observation: Observation,
    history: list[Any],
    model_name: str,
    history_image_window: int,
    prompt_template: Any,
    compact_prompt_parts: Any | None = None,
    image_scale_ratio: float = 1.0,
    rolling_memory_history: bool = False,
) -> list[dict[str, Any]]:
    task_policy = like_task_policy(task)
    task_instruction = f"{task}\n\n{task_policy}" if task_policy else task
    observations = [turn.observation for turn in history] + [current_observation]
    tool_results = [turn.tool_result_message.get("content") for turn in history]
    scale_factor = _general_e2e_scale_factor(
        model_name,
        int(current_observation.screen_width or 999),
        int(current_observation.screen_height or 999),
    )
    system_message = {
        "role": "system",
        "content": prompt_template.render(
            tools="",
            scale_factor=scale_factor,
            extra_action_rows=getattr(compact_prompt_parts, "action_rows", "") or "",
            decision_rules=getattr(compact_prompt_parts, "decision_rules", "") or "",
            compact_skill_instructions=getattr(
                compact_prompt_parts,
                "compact_skill_instructions",
                "",
            )
            or "",
        ),
    }
    if rolling_memory_history:
        latest_memory = _general_compact_history_memory(history)
        instruction_parts = [f"Instruction: {task_instruction}"]
        if latest_memory:
            instruction_parts.extend(["", f"Memory state:\n{latest_memory}"])
        return [
            system_message,
            _general_user_message(
                current_observation,
                tool_result=None,
                instruction="\n".join(instruction_parts),
                model_name=model_name,
                image_scale_ratio=image_scale_ratio,
            ),
        ]

    responses = [_history_raw_response(turn) for turn in history]
    messages = [
        system_message,
        _general_user_message(
            observations[0],
            tool_result=None,
            instruction=task_instruction,
            model_name=model_name,
            image_scale_ratio=image_scale_ratio,
        ),
    ]
    for index, response in enumerate(responses):
        messages.append({"role": "assistant", "content": [{"type": "text", "text": response}]})
        messages.append(
            _general_user_message(
                observations[index + 1],
                tool_result=tool_results[index],
                instruction=None,
                model_name=model_name,
                image_scale_ratio=image_scale_ratio,
            )
        )
    return _hide_history_images_like_general(messages, history_image_window)


def _history_raw_response(turn: Any) -> str:
    raw = getattr(turn, "raw_response_content", None)
    if isinstance(raw, str) and raw.strip():
        return raw
    content = (
        turn.assistant_message.get("content") if isinstance(turn.assistant_message, dict) else None
    )
    return str(content or turn.action_summary or "")


_COMPACT_SCROLL_ACTIONS = frozenset({"scroll", "swipe", "fling"})
_COMPACT_SCROLL_DIRECTIONS = frozenset({"up", "down", "left", "right"})


def _reject_compact_scroll_without_direction(parsed_action: dict[str, Any]) -> None:
    action_type = str(
        parsed_action.get("action_type") or parsed_action.get("action") or ""
    ).strip().lower()
    if action_type not in _COMPACT_SCROLL_ACTIONS:
        return
    raw_direction = parsed_action.get("direction")
    direction = str(raw_direction).strip().lower() if raw_direction is not None else ""
    if direction not in _COMPACT_SCROLL_DIRECTIONS:
        raise ValueError("scroll requires direction: up, down, left, or right")


def _general_compact_memory_items(value: Any) -> list[str]:
    values = value if isinstance(value, list) else [value]
    return [str(item).strip() for item in values if isinstance(item, str) and item.strip()]


def _general_compact_structured_memory(parsed: Any) -> dict[str, Any] | None:
    if not isinstance(parsed, dict):
        return None
    nested = parsed.get("memory")
    if isinstance(nested, dict):
        return nested
    flat = {key: parsed[key] for key in ("add", "drop", "current", "remaining") if key in parsed}
    return flat or None


def _general_compact_memory_update_summary(memory: dict[str, Any]) -> str:
    parts: list[str] = []
    additions = _general_compact_memory_items(memory.get("add"))
    removals = _general_compact_memory_items(memory.get("drop"))
    current = memory.get("current")
    remaining = memory.get("remaining")
    if additions:
        parts.append(f"新增锁定：{'；'.join(additions)}")
    if removals and additions:
        parts.append(f"修正锁定：{'；'.join(removals)}")
    if isinstance(current, str) and current.strip():
        parts.append(f"当前：{current.strip()}")
    if isinstance(remaining, str) and remaining.strip():
        parts.append(f"剩余/约束：{remaining.strip()}")
    return "；".join(parts) or "Memory 未提供有效更新"


def _general_compact_legacy_fields(memory: str) -> tuple[list[str], str, str]:
    text = memory.strip()
    completed_prefix = "已完成/事实："
    current_marker = "；当前："
    remaining_marker = "；剩余/约束："
    if text.startswith(completed_prefix) and current_marker in text:
        completed, _, tail = text[len(completed_prefix) :].partition(current_marker)
        current, separator, remaining = tail.partition(remaining_marker)
        locked = [] if completed.strip() in {"", "无", "暂无"} else [completed.strip()]
        return locked, current.strip(), remaining.strip() if separator else ""
    return ([text] if text else []), "", ""


def _general_compact_history_memory(history: list[Any]) -> str:
    locked: list[str] = []
    current = ""
    remaining = ""

    def append_locked(items: list[str]) -> None:
        known = {" ".join(item.split()) for item in locked}
        for item in items:
            normalized = " ".join(item.split())
            if normalized and normalized not in known:
                locked.append(item)
                known.add(normalized)

    def merge_legacy(text: str) -> None:
        nonlocal current, remaining
        additions, legacy_current, legacy_remaining = _general_compact_legacy_fields(text)
        append_locked(additions)
        if legacy_current:
            current = legacy_current
        if legacy_remaining:
            remaining = legacy_remaining

    for turn in history:
        raw = _history_raw_response(turn).strip()
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            parsed = None
        if isinstance(parsed, list) and len(parsed) == 1:
            parsed = parsed[0]

        parsed_memory = parsed.get("memory") if isinstance(parsed, dict) else None
        structured_memory = _general_compact_structured_memory(parsed)
        if structured_memory is not None:
            additions = _general_compact_memory_items(structured_memory.get("add"))
            removals = _general_compact_memory_items(structured_memory.get("drop"))
            if additions and removals:
                removal_keys = {" ".join(item.split()) for item in removals}
                locked[:] = [
                    item for item in locked if " ".join(item.split()) not in removal_keys
                ]
            append_locked(additions)
            new_current = structured_memory.get("current")
            new_remaining = structured_memory.get("remaining")
            if isinstance(new_current, str) and new_current.strip():
                current = new_current.strip()
            if isinstance(new_remaining, str) and new_remaining.strip():
                remaining = new_remaining.strip()
            continue

        legacy = ""
        if isinstance(parsed_memory, str) and parsed_memory.strip():
            legacy = parsed_memory.strip()
        elif isinstance(parsed, dict):
            parsed_result = parsed.get("result")
            if isinstance(parsed_result, str) and parsed_result.strip():
                legacy = parsed_result.strip()
        if not legacy:
            stored_result = str(getattr(turn, "state_summary", "") or "").strip()
            if stored_result and not stored_result.startswith(("{", "[")):
                legacy = stored_result
        if not legacy and isinstance(parsed, dict):
            parsed_intent = parsed.get("intent")
            if isinstance(parsed_intent, str) and parsed_intent.strip():
                legacy = parsed_intent.strip()
        if not legacy:
            stored_intent = str(getattr(turn, "action_intent", "") or "").strip()
            if stored_intent and not stored_intent.startswith(("{", "[")):
                legacy = stored_intent
        if legacy:
            merge_legacy(legacy)

    if not locked and not current and not remaining:
        return ""
    locked_text = "\n".join(f"- {item}" for item in locked) if locked else "- 无"
    return (
        f"已确认/锁定：\n{locked_text}\n"
        f"当前：{current or '未知'}\n"
        f"剩余：{remaining or '未知'}"
    )


def _general_user_message(
    observation: Observation,
    *,
    tool_result: Any,
    instruction: str | None,
    model_name: str,
    image_scale_ratio: float,
) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    if instruction is not None:
        content.append({"type": "text", "text": instruction})
    if tool_result is not None:
        content.append({"type": "text", "text": f"Tool call result: {tool_result}"})
    content.append(
        _general_e2e_image_content(
            observation,
            model_name=model_name,
            image_scale_ratio=image_scale_ratio,
        )
    )
    return {"role": "user", "content": content}


def _general_e2e_image_content(
    observation: Observation,
    *,
    model_name: str,
    image_scale_ratio: float,
) -> dict[str, Any]:
    return {
        "type": "image_url",
        "image_url": {
            "url": (
                "data:image/png;base64,"
                f"{_general_e2e_observation_base64(observation, model_name=model_name, image_scale_ratio=image_scale_ratio)}"
            )
        },
    }


def _general_e2e_observation_base64(
    observation: Observation,
    *,
    model_name: str,
    image_scale_ratio: float,
) -> str:
    if not observation.screenshot_path:
        raise ValueError("VLAClaw profiles require screenshots.")
    path = Path(observation.screenshot_path)
    with Image.open(path) as image:
        image = image.convert("RGB")
        ratio = normalize_image_scale_ratio(image_scale_ratio)
        scaled_size = (
            max(1, int(image.width * ratio)),
            max(1, int(image.height * ratio)),
        )
        if image.size != scaled_size:
            image = image.resize(scaled_size, Image.Resampling.LANCZOS)
        model = model_name.lower()
        if "opus-4" in model or "opus_4" in model:
            image, _, _ = pil_adaptive_resize(image, _CLAUDE_OPUS_MAX_DIMENSION)
        elif "claude" in model:
            image = image.resize(_CLAUDE_IMAGE_SIZE)
        return pil_to_base64(image)


def _general_e2e_scale_factor(
    model_name: str,
    screen_width: int,
    screen_height: int,
) -> int | tuple[int, int]:
    model = model_name.lower()
    if "opus-4" in model or "opus_4" in model:
        largest = max(screen_width, screen_height)
        if largest <= _CLAUDE_OPUS_MAX_DIMENSION:
            return (screen_width, screen_height)
        scale = _CLAUDE_OPUS_MAX_DIMENSION / largest
        return (max(1, round(screen_width * scale)), max(1, round(screen_height * scale)))
    if "claude" in model:
        return _CLAUDE_IMAGE_SIZE
    if "kimi-k" in model:
        return 1
    return 1000


def _general_e2e_action_text(content: str) -> str:
    if "Action:" not in content:
        action_str = content.strip()
    else:
        try:
            _thought, action_str = general_e2e_agent.parse_action(content)
        except ValueError:
            action_str = content.strip()
    parsed = general_e2e_agent.parse_json_markdown(action_str)
    if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], dict):
        parsed = parsed[0]
    if isinstance(parsed, dict):
        action = dict(parsed)
        if "action_type" not in action and "action" in action:
            action["action_type"] = action.pop("action")
        if "coordinate" not in action and "target" in action:
            action["coordinate"] = action.pop("target")
        return json.dumps(action, ensure_ascii=False)
    return action_str


def _hide_history_images_like_general(
    messages: list[dict[str, Any]], history_image_window: int
) -> list[dict[str, Any]]:
    used = 0
    for idx in range(len(messages) - 1, -1, -1):
        message = messages[idx]
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        image_idx = next(
            (i for i, item in enumerate(content) if item.get("type") == "image_url"), None
        )
        if image_idx is None:
            continue
        if used < history_image_window:
            used += 1
        else:
            content[image_idx] = {"type": "text", "text": "(Previous turn, screen not shown)"}
    return messages


def _to_guiclaw_payload(action: dict[str, Any], *, summary: str) -> dict[str, Any]:
    action_type = action.get("action_type")
    payload: dict[str, Any] = {"summary": summary, "intent": summary}
    if action_type in {CLICK, "click"}:
        payload.update({"action_type": "tap", "x": action.get("x"), "y": action.get("y")})
    elif action_type in {LONG_PRESS, "long_press"}:
        payload.update({"action_type": "long_press", "x": action.get("x"), "y": action.get("y")})
    elif action_type in {DOUBLE_TAP, "double_tap"}:
        payload.update({"action_type": "double_tap", "x": action.get("x"), "y": action.get("y")})
    elif action_type in {DRAG, "drag"}:
        payload.update(
            {
                "action_type": "drag",
                "x": action.get("start_x", action.get("x")),
                "y": action.get("start_y", action.get("y")),
                "x2": action.get("end_x", action.get("x2")),
                "y2": action.get("end_y", action.get("y2")),
            }
        )
    elif action_type in {SCROLL, "scroll"}:
        payload.update(
            {
                "action_type": "scroll",
                "direction": action.get("direction", action.get("text", "down")),
                "pixels": action.get("pixels", DEFAULT_SCROLL_PIXELS),
                "x": action.get("x"),
                "y": action.get("y"),
            }
        )
    elif action_type in {INPUT_TEXT, "input_text"}:
        payload.update({"action_type": "input_text", "text": action.get("text", "")})
    elif action_type in {OPEN_APP, "open_app"}:
        payload.update(
            {"action_type": "open_app", "text": action.get("app_name") or action.get("text", "")}
        )
    elif action_type in {NAVIGATE_BACK, "navigate_back"}:
        payload.update({"action_type": "back"})
    elif action_type in {NAVIGATE_HOME, "navigate_home"}:
        payload.update({"action_type": "home"})
    elif action_type in {KEYBOARD_ENTER, "keyboard_enter"}:
        payload.update({"action_type": "enter"})
    elif action_type in {WAIT, "wait"}:
        payload.update({"action_type": "wait"})
        if action.get("duration_ms") is not None:
            payload["duration_ms"] = action["duration_ms"]
    elif action_type == "inspect":
        payload.update({"action_type": "inspect", "x": action.get("x"), "y": action.get("y")})
    elif action_type in {ANSWER, FINISHED, "answer", "finished"}:
        payload.update(
            {"action_type": "done", "status": _done_status(action), "text": action.get("text", "")}
        )
    elif action_type in {
        ASK_USER,
        MCP,
        UNKNOWN,
        ENV_FAIL,
        "ask_user",
        "mcp",
        "unknown",
        "env_fail",
    }:
        payload.update(
            {
                "action_type": "request_intervention",
                "text": action.get("text") or f"Unsupported action: {action_type}",
            }
        )
    else:
        payload.update(
            {
                "action_type": str(action_type or "request_intervention"),
                **{k: v for k, v in action.items() if k != "action_type"},
            }
        )
    return {key: value for key, value in payload.items() if value is not None}


def _done_status(action: dict[str, Any]) -> str:
    text = str(
        action.get("text") or action.get("status") or action.get("goal_status") or "success"
    ).lower()
    return "failure" if "fail" in text or "infeasible" in text or "abort" in text else "success"
