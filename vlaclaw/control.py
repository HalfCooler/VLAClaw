"""Deterministic safety and progress control for GUI actions.

The model proposes an action; this module decides whether the action may be
executed and whether the resulting transition made progress.  These checks are
deliberately model-free so that recovery does not depend on the same visual
reasoning failure that produced a bad action.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from vlaclaw.action import Action, resolve_coordinate
from vlaclaw.observation import Observation
from vlaclaw.social_state import task_requests_like_off, task_requests_like_on

_BOUNDS_RE = re.compile(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]")

_RISK_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "authentication",
        re.compile(
            r"输入密码|支付密码|人脸|面容|指纹|验证码|身份验证|"
            r"password|face\s*(?:id|verify)|fingerprint|verification code",
            re.IGNORECASE,
        ),
    ),
    (
        "financial",
        re.compile(
            r"付款|支付|确认付款|立即购买|购买|下单|提交订单|充值|转账|红包|"
            r"开通会员|续费|订阅|内购|pay\b|purchase|subscribe|checkout",
            re.IGNORECASE,
        ),
    ),
    (
        "destructive",
        re.compile(
            r"永久删除|删除账户|注销账户|清空|恢复出厂|格式化|卸载|"
            r"permanently delete|delete account|factory reset|format|uninstall",
            re.IGNORECASE,
        ),
    ),
    (
        "external_write",
        re.compile(
            r"发送|发表|发布|评论|回复|提交表单|send\b|post\b|publish|comment|reply",
            re.IGNORECASE,
        ),
    ),
    (
        "not_interested",
        re.compile(r"不感兴趣|减少推荐|少推荐|not[\s_-]*interested|dislike", re.I),
    ),
    (
        "like_off",
        re.compile(r"已点赞|取消点赞|撤销点赞|\bunlike\b|like[\s_-]*(?:on|selected|active)", re.I),
    ),
    (
        "like_on",
        re.compile(r"点赞|点个赞|赞一下|\blike\b|(?:^|[_./-])like(?:[_./-]|$)", re.I),
    ),
    (
        "social_state",
        re.compile(r"收藏|关注|预约|加入购物车|favorite|follow|book\b", re.I),
    ),
)

_TASK_AUTH_PATTERNS: dict[str, re.Pattern[str]] = {
    "financial": re.compile(
        r"付款|支付|购买|买(?:一|个|件|张|份)?|下单|充值|转账|开通会员|续费|订阅|"
        r"pay\b|purchase|buy\b|checkout|subscribe",
        re.IGNORECASE,
    ),
    "authentication": re.compile(
        r"登录|输入密码|验证码|身份验证|人脸|指纹|log\s*in|sign\s*in|verify",
        re.IGNORECASE,
    ),
    "destructive": re.compile(
        r"删除|清空|注销|卸载|格式化|恢复出厂|delete|clear|remove|uninstall|format",
        re.IGNORECASE,
    ),
    "external_write": re.compile(
        r"发送|发表|发布|评论|回复|留言|提交表单|send|post|publish|comment|reply",
        re.IGNORECASE,
    ),
    "social_state": re.compile(
        r"收藏|关注|预约|加入购物车|favorite|follow|book",
        re.IGNORECASE,
    ),
    "not_interested": re.compile(
        r"不感兴趣|减少推荐|少推荐|not[\s_-]*interested|dislike",
        re.IGNORECASE,
    ),
}

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
_PLAYBACK_EVIDENCE_RE = re.compile(r"试看\s*\d*\s*分钟|正在播放|播放中|暂停播放")

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
class TaskContract:
    task: str
    authorized_effects: frozenset[str]

    def authorizes(self, effect: str) -> bool:
        return effect in self.authorized_effects

    def mentions_payment_app(self) -> bool:
        task_lower = self.task.lower()
        return any(marker in task_lower for marker in _PAYMENT_APP_TASK_NAMES)


@dataclass(frozen=True)
class GuardVerdict:
    allowed: bool
    reason: str
    effect: str | None = None
    target_text: str = ""
    target_node: dict[str, Any] | None = None
    goal_already_satisfied: bool = False

    def snapshot(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "effect": self.effect,
            "target_text": self.target_text,
            "target_node": self.target_node,
            "goal_already_satisfied": self.goal_already_satisfied,
        }


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


def build_task_contract(task: str) -> TaskContract:
    normalized = str(task or "").strip()
    authorized = {
        effect for effect, pattern in _TASK_AUTH_PATTERNS.items() if pattern.search(normalized)
    }
    if task_requests_like_off(normalized):
        authorized.add("like_off")
    elif task_requests_like_on(normalized):
        authorized.add("like_on")
    return TaskContract(task=normalized, authorized_effects=frozenset(authorized))


def guard_action(
    action: Action,
    observation: Observation,
    contract: TaskContract,
) -> GuardVerdict:
    """Check a proposed action immediately before device execution."""
    target_text, target_node = _action_target(action, observation)
    page_text = observation_text(observation)
    app = str(observation.foreground_app or "")

    if action.action_type in {
        "back",
        "home",
        "app_switch",
        "wait",
        "screenshot",
        "inspect",
        "done",
    }:
        return GuardVerdict(True, "non_mutating_navigation", target_text=target_text)

    if _is_payment_app(app) and not (
        contract.authorizes("financial") or contract.mentions_payment_app()
    ):
        goal_satisfied = _playback_goal_evidence(contract.task, page_text)
        return GuardVerdict(
            False,
            "payment_app_outside_task_scope",
            effect="financial",
            target_text=target_text,
            target_node=target_node,
            goal_already_satisfied=goal_satisfied,
        )

    if action.action_type == "open_app" and _is_payment_app(action.text or "") and not (
        contract.authorizes("financial") or contract.mentions_payment_app()
    ):
        return GuardVerdict(
            False,
            "payment_app_outside_task_scope",
            effect="financial",
            target_text=action.text or "",
        )

    candidate_text = target_text
    if action.action_type in {"open_app", "open_deeplink", "open_intent"}:
        candidate_text = " ".join(
            part for part in (target_text, action.text or "") if str(part).strip()
        )
    effect = _detect_risk_effect(candidate_text)
    if effect is None and action.action_type == "input_text":
        effect = _detect_risk_effect(_focused_text(observation))

    if effect is not None and not contract.authorizes(effect):
        return GuardVerdict(
            False,
            f"{effect}_effect_not_authorized_by_task",
            effect=effect,
            target_text=target_text,
            target_node=target_node,
            goal_already_satisfied=(
                (effect == "financial" and _playback_goal_evidence(contract.task, page_text))
                or (effect == "like_off" and contract.authorizes("like_on"))
            ),
        )

    return GuardVerdict(
        True,
        "authorized_or_low_risk",
        effect=effect,
        target_text=target_text,
        target_node=target_node,
    )


class TransitionMonitor:
    """Detect no-op transitions and short cycles regardless of action type."""

    def __init__(
        self,
        initial_observation: Observation,
        contract: TaskContract,
        *,
        no_progress_limit: int = 3,
        history_size: int = 8,
    ) -> None:
        self._contract = contract
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
            and not (self._contract.authorizes("financial") or self._contract.mentions_payment_app())
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


def _action_target(
    action: Action,
    observation: Observation,
) -> tuple[str, dict[str, Any] | None]:
    if action.action_type not in _COORDINATE_ACTIONS:
        return "", None
    point = _first_action_point(action, observation)
    if point is None:
        return "", None
    x, y = point
    nearby: list[tuple[float, int, dict[str, Any], str]] = []
    threshold = max(72.0, math.hypot(observation.screen_width, observation.screen_height) * 0.04)
    for node in observation.extra.get("ui_tree") or []:
        if not isinstance(node, dict):
            continue
        bounds = _parse_bounds(node.get("bounds"))
        if bounds is None:
            continue
        label = " ".join(
            str(node.get(key)).strip()
            for key in ("text", "content_desc", "resource_id")
            if str(node.get(key) or "").strip()
        )
        if not label:
            continue
        distance = _distance_to_rect(x, y, bounds)
        if distance <= threshold:
            area = max(1, (bounds[2] - bounds[0]) * (bounds[3] - bounds[1]))
            nearby.append((distance, area, node, label))
    if not nearby:
        return "", None
    nearby.sort(key=lambda item: (item[0], item[1]))
    primary = nearby[0][2]
    labels: list[str] = []
    for _distance, _area, _node, label in nearby[:4]:
        if label not in labels:
            labels.append(label)
    return " | ".join(labels), dict(primary)


def _first_action_point(action: Action, observation: Observation) -> tuple[int, int] | None:
    if action.x is not None and action.y is not None:
        return (
            resolve_coordinate(action.x, observation.screen_width, relative=action.relative),
            resolve_coordinate(action.y, observation.screen_height, relative=action.relative),
        )
    if action.points:
        x, y = action.points[0]
        return (
            resolve_coordinate(x, observation.screen_width, relative=action.relative),
            resolve_coordinate(y, observation.screen_height, relative=action.relative),
        )
    return None


def _parse_bounds(value: Any) -> tuple[int, int, int, int] | None:
    match = _BOUNDS_RE.fullmatch(str(value or "").strip())
    if match is None:
        return None
    return tuple(int(part) for part in match.groups())  # type: ignore[return-value]


def _distance_to_rect(x: int, y: int, bounds: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = bounds
    dx = max(x1 - x, 0, x - x2)
    dy = max(y1 - y, 0, y - y2)
    return math.hypot(dx, dy)


def _detect_risk_effect(text: str) -> str | None:
    for effect, pattern in _RISK_PATTERNS:
        if pattern.search(text or ""):
            return effect
    return None


def _focused_text(observation: Observation) -> str:
    values = observation.extra.get("focused_text")
    if isinstance(values, list):
        return " ".join(str(value) for value in values)
    return str(values or "")


def _is_payment_app(app: str) -> bool:
    normalized = str(app or "").lower()
    return any(marker in normalized for marker in _PAYMENT_APP_MARKERS)


def _playback_goal_evidence(task: str, page_text: str) -> bool:
    return bool(_PLAY_TASK_RE.search(task or "") and _PLAYBACK_EVIDENCE_RE.search(page_text or ""))


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
