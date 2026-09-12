"""JSON action parsing shared by general_compact and general_e2e."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from vlaclaw.agents.utils.parsers import parse_json_markdown

logger = logging.getLogger(__name__)

ACTION_ALIASES = {
    "click": ["tap", "press", "touch"],
    "long_press": ["long tap", "long press", "hold"],
    "input_text": ["type", "enter_text", "write", "enter"],
    "scroll": ["swipe", "fling"],
    "keyboard_enter": ["enter"],
}
NORMALIZED_ACTION_MAP: dict[str, str] = {}
for standard_action, aliases in ACTION_ALIASES.items():
    NORMALIZED_ACTION_MAP[standard_action] = standard_action
    for alias in aliases:
        NORMALIZED_ACTION_MAP[alias.replace(" ", "_")] = standard_action
        NORMALIZED_ACTION_MAP[alias] = standard_action


def normalize_action_type(action_type: str) -> str | None:
    if not action_type:
        return None
    processed_type = action_type.lower().strip().replace(" ", "_")
    return NORMALIZED_ACTION_MAP.get(processed_type, action_type)


def parse_action(plan_output: str) -> tuple[str, str]:
    match = re.search(r"Action:", plan_output)
    if match is None:
        raise ValueError("Expected at least one 'Action:' in the output")
    thought_part = plan_output[: match.start()].strip()
    thought = thought_part[8:].strip() if thought_part.startswith("Thought:") else thought_part
    action = plan_output[match.end() :].strip()
    brace = action.find("{")
    if brace != -1:
        try:
            _obj, end = json.JSONDecoder().raw_decode(action[brace:])
            action = action[brace : brace + end]
        except json.JSONDecodeError:
            pass
    return thought, action


def parse_response_to_action(
    action_str: str,
    image_width: int,
    image_height: int,
    scale_factor: int | tuple[int, int] = 1000,
) -> dict[str, Any]:
    try:
        action_data = parse_json_markdown(action_str)
        original_action_type = action_data.get("action_type")
        normalized_action_type = normalize_action_type(original_action_type)
        if not normalized_action_type:
            raise ValueError("Action type is missing or empty.")
        action_data["action_type"] = normalized_action_type
        action_type = normalized_action_type
        scale_factor_x, scale_factor_y = (
            [scale_factor, scale_factor] if isinstance(scale_factor, int) else scale_factor
        )

        if action_type in ["click", "double_tap", "long_press"]:
            if "coordinate" not in action_data:
                raise ValueError(f"Missing coordinate for action type: {action_type}")
            coord = action_data["coordinate"]
            if not (isinstance(coord, list) and len(coord) == 2):
                raise ValueError(f"Invalid coordinate format: {coord}")
            relative_x, relative_y = coord[0], coord[1]
            return {
                "action_type": action_type,
                "x": int(relative_x * image_width / scale_factor_x),
                "y": int(relative_y * image_height / scale_factor_y),
            }

        if action_type == "drag":
            start_coord = action_data.get("start_coordinate")
            end_coord = action_data.get("end_coordinate")
            if not (
                isinstance(start_coord, list)
                and len(start_coord) == 2
                and isinstance(end_coord, list)
                and len(end_coord) == 2
            ):
                raise ValueError(f"Invalid drag coordinates: {start_coord}, {end_coord}")
            return {
                "action_type": "drag",
                "start_x": int(start_coord[0] * image_width / scale_factor_x),
                "start_y": int(start_coord[1] * image_height / scale_factor_y),
                "end_x": int(end_coord[0] * image_width / scale_factor_x),
                "end_y": int(end_coord[1] * image_height / scale_factor_y),
            }

        if action_type == "scroll":
            start_coord = action_data.get(
                "start_coordinate",
                [scale_factor_x / 2, scale_factor_y / 2],
            )
            if not isinstance(start_coord, list) or len(start_coord) != 2:
                raise ValueError(f"Invalid scroll start coordinate: {start_coord}")
            return {
                "action_type": "scroll",
                "direction": action_data.get("direction", "down"),
                "x": int(start_coord[0] * image_width / scale_factor_x),
                "y": int(start_coord[1] * image_height / scale_factor_y),
            }

        if action_type in [
            "open_app",
            "answer",
            "navigate_home",
            "navigate_back",
            "wait",
            "ask_user",
            "keyboard_enter",
        ]:
            return action_data
        if action_type == "input_text":
            return {"action_type": "input_text", "text": action_data.get("text", "")}
        if action_type == "status":
            return {
                "action_type": "answer",
                "text": "task finished"
                if action_data.get("goal_status") == "complete"
                else "task failed",
            }
        return action_data
    except json.JSONDecodeError as exc:
        logger.error("Error parsing JSON action: %s", exc)
        raise ValueError(f"Invalid JSON format in action: {action_str}") from exc
    except Exception as exc:
        logger.error("Error parsing action: %s", exc)
        raise ValueError(f"Error parsing action: {action_str}") from exc
