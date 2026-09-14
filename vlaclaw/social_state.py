"""State-aware safeguards for generic social controls.

The full-screen GUI actor is intentionally allowed to use a compressed image.
For like operations, this module supports a second, low-token inspection of a
small crop around the proposed coordinate so that outline, filled, and slashed
heart icons are not treated as interchangeable click targets.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from vlaclaw.action import Action, resolve_coordinate
from vlaclaw.observation import Observation

LIKE_UNLIKED = "unliked"
LIKE_LIKED = "liked"
LIKE_NOT_INTERESTED = "not_interested"
LIKE_OTHER = "other"
LIKE_UNKNOWN = "unknown"

LIKE_STATES = frozenset(
    {LIKE_UNLIKED, LIKE_LIKED, LIKE_NOT_INTERESTED, LIKE_OTHER, LIKE_UNKNOWN}
)
LIKE_CONFIDENCE_THRESHOLD = 0.80
LIKE_CROP_OUTPUT_SIZE = 384

_LIKE_OFF_TASK_RE = re.compile(
    r"取消点赞|撤销点赞|取消赞|\bunlike\b|remove\s+(?:the\s+)?like",
    re.IGNORECASE,
)
_LIKE_ON_TASK_RE = re.compile(
    r"点赞|点个赞|赞一下|\blike\b",
    re.IGNORECASE,
)
_LIKE_NEGATED_TASK_RE = re.compile(
    r"(?:不要|别|禁止|无需|不用).{0,6}(?:点赞|点个赞)|"
    r"(?:do\s+not|don't|never)\s+like",
    re.IGNORECASE,
)
_LIKE_READ_ONLY_TASK_RE = re.compile(
    r"(?:是否|有没有|是不是|检查|确认).{0,8}(?:已)?点赞|"
    r"点赞(?:数|数量|列表|记录)|获赞|多少(?:个)?赞|"
    r"like[\s_-]*count|who\s+liked",
    re.IGNORECASE,
)
_NOT_INTERESTED_RE = re.compile(
    r"不感兴趣|减少推荐|少推荐|屏蔽.{0,4}(?:内容|视频|帖子)|"
    r"not[\s_-]*interested|dislike|heart[\s_-]*slash|slash[\s_-]*heart",
    re.IGNORECASE,
)
_LIKED_RE = re.compile(
    r"已点赞|取消点赞|撤销点赞|\bliked\b|\bunlike\b|like[\s_-]*(?:on|selected|active)",
    re.IGNORECASE,
)
_UNLIKED_RE = re.compile(r"未点赞|尚未点赞|\bunliked\b|\blike\b", re.IGNORECASE)
_LIKE_LABEL_RE = re.compile(
    r"点赞|点个赞|赞一下|like|heart",
    re.IGNORECASE,
)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class LikeStateVerdict:
    state: str
    confidence: float
    reason: str = ""
    source: str = "visual"
    raw_content: str = ""
    crop_path: str | None = None

    @property
    def decisive(self) -> bool:
        return self.state != LIKE_UNKNOWN and self.confidence >= LIKE_CONFIDENCE_THRESHOLD

    def snapshot(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "confidence": round(self.confidence, 3),
            "reason": self.reason,
            "source": self.source,
            "crop_path": self.crop_path,
        }


def task_requests_like_on(task: str) -> bool:
    """Return whether *task* asks to establish (rather than remove) a like."""
    text = str(task or "")
    return bool(
        _LIKE_ON_TASK_RE.search(text)
        and not _LIKE_OFF_TASK_RE.search(text)
        and not _LIKE_NEGATED_TASK_RE.search(text)
        and not _LIKE_READ_ONLY_TASK_RE.search(text)
    )


def task_requests_like_off(task: str) -> bool:
    return bool(_LIKE_OFF_TASK_RE.search(str(task or "")))


def like_task_policy(task: str) -> str:
    """Return a compact actor policy only for tasks that establish a like."""
    if not task_requests_like_on(task):
        return ""
    return (
        "点赞状态规则：空心且无斜杠的爱心=未点赞，可点击；实心爱心=已点赞，禁止再次点击；"
        "带斜杠爱心=不感兴趣，绝对禁止点击。看不清时禁止猜测，使用 inspect 并给出该图标中心坐标；"
        "即使看起来已点赞，也必须先 inspect，不能直接声明完成；控制器仅在局部检查确认为空心后执行点赞。"
        "inspect 是本任务额外允许的动作，格式："
        '{"action_type":"inspect","coordinate":[500,500],"intent":"检查爱心状态",'
        '"memory":{"current":"检查候选爱心","remaining":"点赞"}}。'
    )


def action_likely_targets_like(
    action: Action,
    *,
    explicit_intent: str = "",
    target_text: str = "",
    target_node: dict[str, Any] | None = None,
) -> bool:
    if action.action_type not in {"tap", "inspect"}:
        return False
    node_text = _node_text(target_node)
    evidence = " ".join((str(explicit_intent or ""), str(target_text or ""), node_text))
    return bool(_LIKE_LABEL_RE.search(evidence) or _NOT_INTERESTED_RE.search(evidence))


def infer_like_state_from_semantics(
    target_text: str,
    target_node: dict[str, Any] | None,
) -> LikeStateVerdict | None:
    """Use accessibility state when it unambiguously describes a like control."""
    node = target_node or {}
    node_text = _node_text(node)
    evidence = " ".join((str(target_text or ""), node_text)).strip()
    if _NOT_INTERESTED_RE.search(evidence):
        return LikeStateVerdict(
            LIKE_NOT_INTERESTED,
            1.0,
            "accessibility label identifies a not-interested control",
            source="semantics",
        )
    if not _LIKE_LABEL_RE.search(evidence):
        return None
    if _LIKED_RE.search(evidence) or node.get("checked") is True or node.get("selected") is True:
        return LikeStateVerdict(
            LIKE_LIKED,
            1.0,
            "accessibility state identifies an active like",
            source="semantics",
        )
    if node.get("checkable") is True and node.get("checked") is False:
        return LikeStateVerdict(
            LIKE_UNLIKED,
            1.0,
            "checkable like control is not checked",
            source="semantics",
        )
    if re.search(r"(?:^|\s)点赞(?:\s|$)", evidence) or _UNLIKED_RE.search(evidence):
        return LikeStateVerdict(
            LIKE_UNLIKED,
            0.95,
            "accessibility label identifies an inactive like action",
            source="semantics",
        )
    return None


def build_like_verifier_prompt(*, phase: str, target_text: str = "") -> str:
    nearby = " ".join(str(target_text or "").split())[:300] or "(none)"
    expected = (
        "This image was captured after a like tap; the required result is liked."
        if phase == "post"
        else "This image is a pre-action inspection centered on a proposed tap."
    )
    return (
        "You are a safety classifier for a generic mobile GUI. The only image is a high-resolution "
        "local crop centered on the proposed control; it is not the compressed full screen. "
        f"{expected}\n"
        "Classify the control at the center as exactly one state:\n"
        "- unliked: outline/hollow heart with no diagonal slash\n"
        "- liked: solid/filled/selected heart\n"
        "- not_interested: any heart containing or crossed by a diagonal slash\n"
        "- other: clearly a different control, not a heart like control\n"
        "- unknown: obscured, ambiguous, loading, or insufficient evidence\n"
        "A diagonal slash always wins over outline/fill. Do not infer state from the task wording. "
        f"Nearby accessibility text: {nearby}\n"
        'Return JSON only: {"state":"unliked|liked|not_interested|other|unknown",'
        '"confidence":0.0,"reason":"brief visual evidence"}'
    )


def parse_like_state_verdict(content: str) -> LikeStateVerdict:
    raw = str(content or "").strip()
    payload = _extract_json_object(raw)
    if not isinstance(payload, dict):
        return LikeStateVerdict(
            LIKE_UNKNOWN,
            0.0,
            "unparseable verifier response",
            raw_content=raw,
        )
    state = _normalize_state(payload.get("state"))
    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    reason = " ".join(str(payload.get("reason") or "").split())[:300]
    if state not in LIKE_STATES or confidence < LIKE_CONFIDENCE_THRESHOLD:
        low_state = state if state in LIKE_STATES else LIKE_UNKNOWN
        return LikeStateVerdict(
            LIKE_UNKNOWN,
            confidence,
            reason or f"non-decisive state: {low_state}",
            raw_content=raw,
        )
    return LikeStateVerdict(state, confidence, reason, raw_content=raw)


def crop_action_target(
    observation: Observation,
    action: Action,
    *,
    output_path: Path,
    output_size: int = LIKE_CROP_OUTPUT_SIZE,
) -> Path:
    """Save a square, full-source-resolution crop around an action coordinate."""
    if not observation.screenshot_path or action.x is None or action.y is None:
        raise ValueError("Like inspection requires a screenshot and one action coordinate.")
    screenshot_path = Path(observation.screenshot_path)
    with Image.open(screenshot_path) as source:
        source = source.convert("RGB")
        screen_width = max(1, int(observation.screen_width or source.width))
        screen_height = max(1, int(observation.screen_height or source.height))
        action_x = resolve_coordinate(action.x, screen_width, relative=action.relative)
        action_y = resolve_coordinate(action.y, screen_height, relative=action.relative)
        center_x = round(action_x / screen_width * source.width)
        center_y = round(action_y / screen_height * source.height)
        crop_extent = max(96, round(min(source.width, source.height) * 0.18))
        crop_extent = min(crop_extent, max(source.width, source.height))
        half = crop_extent // 2
        left = center_x - half
        top = center_y - half
        right = left + crop_extent
        bottom = top + crop_extent
        if left < 0:
            right -= left
            left = 0
        if top < 0:
            bottom -= top
            top = 0
        if right > source.width:
            left -= right - source.width
            right = source.width
        if bottom > source.height:
            top -= bottom - source.height
            bottom = source.height
        left = max(0, left)
        top = max(0, top)
        crop = source.crop((left, top, right, bottom))
        crop.thumbnail((output_size, output_size), Image.Resampling.LANCZOS)
        if crop.size != (output_size, output_size):
            crop = crop.resize((output_size, output_size), Image.Resampling.LANCZOS)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(output_path, format="PNG")
    return output_path


def image_path_to_data_url(path: Path) -> str:
    import base64

    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _node_text(node: dict[str, Any] | None) -> str:
    if not isinstance(node, dict):
        return ""
    return " ".join(
        str(node.get(key) or "") for key in ("text", "content_desc", "resource_id")
    )


def _extract_json_object(text: str) -> dict[str, Any] | None:
    candidates = [text]
    match = _JSON_OBJECT_RE.search(text)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(value, dict):
            return value
    return None


def _normalize_state(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "outline": LIKE_UNLIKED,
        "hollow": LIKE_UNLIKED,
        "empty": LIKE_UNLIKED,
        "filled": LIKE_LIKED,
        "selected": LIKE_LIKED,
        "active": LIKE_LIKED,
        "dislike": LIKE_NOT_INTERESTED,
        "heart_slash": LIKE_NOT_INTERESTED,
        "slash_heart": LIKE_NOT_INTERESTED,
        "not_like_control": LIKE_OTHER,
    }
    return aliases.get(text, text)
