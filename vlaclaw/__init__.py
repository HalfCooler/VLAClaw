"""VLAClaw — slim ADB GUI agent skeleton extracted from GUIClaw."""

from __future__ import annotations

from vlaclaw.action import Action, ActionError, describe_action, parse_action
from vlaclaw.interfaces import DeviceBackend, LLMProvider, LLMResponse, ProgressCallback, ToolCall
from vlaclaw.observation import Observation


def __getattr__(name: str) -> object:
    if name == "GuiAgent":
        from vlaclaw.agent import GuiAgent

        return GuiAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Action",
    "ActionError",
    "parse_action",
    "describe_action",
    "Observation",
    "LLMProvider",
    "DeviceBackend",
    "LLMResponse",
    "ToolCall",
    "ProgressCallback",
]
