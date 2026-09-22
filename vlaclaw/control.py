"""Post-action progress control for GUI actions."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from vlaclaw.action import Action
from vlaclaw.observation import Observation

_PAYMENT_APP_MARKERS = (
    "alipay",
    "com.eg.android.alipaygphone",
    "tenpay",
    "unionpay",
    "云闪付",
    "支付宝",
)
_PAYMENT_APP_TASK_NAMES = ("alipay", "支付宝", "云闪付", "unionpay")
_PLAY_TASK_RE = re.compile(r"继续观看|继续播放|播放|观看|看.+视频|听.+(?:歌|音乐|音频)")
_PLAYBACK_EVIDENCE_RE = re.compile(
    r"试看\s*\d*\s*分钟|正在播放|播放中|暂停(?:播放)?|\bplaying\b|\bpaused?\b",
    re.IGNORECASE,
)

_COORDINATE_ACTIONS = frozenset(
    {
        "tap",
        "double_tap",
        "long_press",
        "drag",
        "swipe",
        "scroll",
        "click_multi",
        "click_then_type",
        "inspect",
    }
)
_MUTATING_ACTIONS = frozenset(
    {"tap", "double_tap", "long_press", "input_text", "enter", "click_multi", "click_then_type"}
)


@dataclass(frozen=True)
class TransitionVerdict:
    status: str
    reason: str
    before_state: str
    after_state: str
    loop_detected: bool = False

    def snapshot(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "before_state": self.before_state,
            "after_state": self.after_state,
            "loop_detected": self.loop_detected,
        }


class TransitionMonitor:
    """Detect no-op transitions and short cycles regardless of action type."""

    def __init__(
        self,
        initial_observation: Observation,
        task: str,
        *,
        no_progress_limit: int = 3,
        history_size: int = 8,
    ) -> None:
        self._task = str(task or "")
        self._no_progress_limit = max(2, int(no_progress_limit))
        self._states: deque[str] = deque(
            [observation_state_id(initial_observation)], maxlen=max(6, int(history_size))
        )
        self._no_progress_streak = 0

    def evaluate(
        self,
        before: Observation,
        action: Action,
        after: Observation,
    ) -> TransitionVerdict:
        before_state = observation_state_id(before)
        after_state = observation_state_id(after)
        before_app = str(before.foreground_app or "")
        after_app = str(after.foreground_app or "")

        if (
            not _is_payment_app(before_app)
            and _is_payment_app(after_app)
            and not _task_authorizes_payment(self._task)
        ):
            self._states.append(after_state)
            return TransitionVerdict(
                "off_task_risk",
                "entered_payment_app_outside_task_scope",
                before_state,
                after_state,
            )

        changed = before_state != after_state
        if changed:
            self._no_progress_streak = 0
        elif action.action_type in (
            (_COORDINATE_ACTIONS - {"inspect"}) | _MUTATING_ACTIONS | {"wait", "back", "home"}
        ):
            self._no_progress_streak += 1

        self._states.append(after_state)
        states = list(self._states)
        loop_reason = ""
        if self._no_progress_streak >= self._no_progress_limit:
            loop_reason = f"same_state_after_{self._no_progress_streak}_actions"
        elif len(states) >= 4 and states[-4] == states[-2] and states[-3] == states[-1]:
            loop_reason = "two_state_cycle"
        elif (
            Counter(states[-6:])[after_state] >= self._no_progress_limit + 1
            and len(set(states[-4:])) <= 2
        ):
            loop_reason = "revisited_state_without_new_progress"

        if loop_reason:
            return TransitionVerdict(
                "loop",
                loop_reason,
                before_state,
                after_state,
                loop_detected=True,
            )
        return TransitionVerdict(
            "progress" if changed else "no_effect",
            "state_changed" if changed else "state_unchanged",
            before_state,
            after_state,
        )


def observation_state_id(observation: Observation) -> str:
    """Build a stable, compact state identifier from app and UI semantics."""
    nodes = observation.extra.get("ui_tree")
    semantic: Any
    if isinstance(nodes, list) and nodes:
        semantic = [
            {
                key: node.get(key)
                for key in (
                    "text",
                    "content_desc",
                    "resource_id",
                    "class",
                    "clickable",
                    "checkable",
                    "checked",
                    "selected",
                    "bounds",
                )
                if node.get(key) not in (None, "", False)
            }
            for node in nodes
            if isinstance(node, dict)
        ]
    else:
        semantic = {
            key: observation.extra.get(key)
            for key in ("visible_text", "content_desc", "resource_ids")
            if observation.extra.get(key)
        }
    payload = {
        "app": observation.foreground_app or "unknown",
        "resolution": [observation.screen_width, observation.screen_height],
        "semantic": semantic,
        "media_playback": observation.extra.get("media_playback"),
        "visual": _visual_average_hash(observation.screenshot_path),
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    return f"{observation.foreground_app or 'unknown'}:{digest}"


def observation_text(observation: Observation) -> str:
    values: list[str] = []
    for key in ("visible_text", "content_desc", "clickable_text", "focused_text"):
        value = observation.extra.get(key)
        if isinstance(value, list):
            values.extend(str(item) for item in value if item)
        elif value:
            values.append(str(value))
    for node in observation.extra.get("ui_tree") or []:
        if not isinstance(node, dict):
            continue
        values.extend(
            str(node.get(key))
            for key in ("text", "content_desc", "resource_id")
            if node.get(key)
        )
    return " ".join(values)


def _is_payment_app(app: str) -> bool:
    normalized = str(app or "").lower()
    return any(marker in normalized for marker in _PAYMENT_APP_MARKERS)


def _task_authorizes_payment(task: str) -> bool:
    normalized = str(task or "").lower()
    return any(marker in normalized for marker in _PAYMENT_APP_TASK_NAMES) or bool(
        re.search(r"付款|支付|购买|下单|充值|转账|pay\b|purchase|buy\b|checkout", normalized)
    )


def playback_goal_is_active(task: str, observation: Observation) -> bool:
    """Return true only for a foreground app's authoritative active media session."""
    if not _PLAY_TASK_RE.search(task or ""):
        return False
    media = observation.extra.get("media_playback")
    if not isinstance(media, dict) or str(media.get("state") or "").casefold() != "playing":
        return False
    package = str(media.get("package") or "").casefold()
    foreground = str(observation.foreground_app or "").casefold()
    return bool(package and foreground and package == foreground)


def playback_goal_satisfied(task: str, observation: Observation) -> bool:
    """Combine strong system playback state with compatible on-screen evidence."""
    return playback_goal_is_active(task, observation) or bool(
        _PLAY_TASK_RE.search(task or "")
        and _PLAYBACK_EVIDENCE_RE.search(observation_text(observation))
    )


def _visual_average_hash(screenshot_path: str | None) -> str:
    path = Path(screenshot_path or "")
    if not screenshot_path or not path.is_file():
        return ""
    try:
        with Image.open(path) as image:
            small = image.convert("L").resize((8, 8), Image.Resampling.BILINEAR)
            pixels = list(
                small.get_flattened_data() if hasattr(small, "get_flattened_data") else small.getdata()
            )
    except Exception:
        return ""
    average = sum(pixels) / max(1, len(pixels))
    bits = "".join("1" if pixel >= average else "0" for pixel in pixels)
    return f"{int(bits, 2):016x}"
