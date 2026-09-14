from __future__ import annotations

import asyncio
import base64
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw

from vlaclaw.action import Action
from vlaclaw.agent import GuiAgent
from vlaclaw.agents.profiles import parse_profile_action
from vlaclaw.backends.adb import _parse_ui_tree_xml
from vlaclaw.control import build_task_contract, guard_action
from vlaclaw.interfaces import LLMResponse
from vlaclaw.observation import Observation
from vlaclaw.social_state import (
    LIKE_LIKED,
    LIKE_NOT_INTERESTED,
    LIKE_UNLIKED,
    crop_action_target,
    infer_like_state_from_semantics,
    parse_like_state_verdict,
    task_requests_like_on,
)
from vlaclaw.trajectory.recorder import TrajectoryRecorder


def test_like_task_contract_is_directional() -> None:
    contract = build_task_contract("给当前视频点赞")
    assert contract.authorizes("like_on")
    assert not contract.authorizes("like_off")
    assert not contract.authorizes("not_interested")

    observation = Observation(
        screenshot_path=None,
        screen_width=100,
        screen_height=200,
        extra={
            "ui_tree": [
                {
                    "content_desc": "不感兴趣",
                    "clickable": True,
                    "bounds": "[40,80][60,120]",
                }
            ]
        },
    )
    verdict = guard_action(Action("tap", x=50, y=100), observation, contract)
    assert verdict.allowed is False
    assert verdict.effect == "not_interested"
    assert not task_requests_like_on("查看当前视频的点赞数")
    assert not task_requests_like_on("打开视频但不要点赞")


def test_already_liked_semantics_satisfy_like_on_without_tap() -> None:
    contract = build_task_contract("点赞")
    observation = Observation(
        screenshot_path=None,
        screen_width=100,
        screen_height=200,
        extra={
            "ui_tree": [
                {
                    "content_desc": "已点赞",
                    "selected": True,
                    "bounds": "[40,80][60,120]",
                }
            ]
        },
    )
    verdict = guard_action(Action("tap", x=50, y=100), observation, contract)
    assert verdict.allowed is False
    assert verdict.goal_already_satisfied is True


def test_accessibility_parser_keeps_checked_and_selected_state() -> None:
    parsed = _parse_ui_tree_xml(
        '<hierarchy><node text="" content-desc="点赞" resource-id="like" '
        'class="android.widget.ImageButton" clickable="true" enabled="true" '
        'checkable="true" checked="false" selected="true" scrollable="false" '
        'focused="false" bounds="[1,2][11,12]" /></hierarchy>',
        include_nodes=True,
    )
    node = parsed["ui_tree"][0]
    assert node["checkable"] is True
    assert node["checked"] is False
    assert node["selected"] is True


def test_like_semantics_and_visual_response_parser() -> None:
    semantic = infer_like_state_from_semantics(
        "点赞",
        {"resource_id": "feed_like", "checkable": True, "checked": False},
    )
    assert semantic is not None and semantic.state == LIKE_UNLIKED
    assert parse_like_state_verdict(
        '{"state":"not_interested","confidence":0.98,"reason":"diagonal slash"}'
    ).state == LIKE_NOT_INTERESTED
    low_confidence = parse_like_state_verdict(
        '{"state":"liked","confidence":0.50,"reason":"blurred"}'
    )
    assert low_confidence.state == "unknown"


def test_inspect_action_is_parsed_as_a_non_tap_coordinate_action() -> None:
    payload = parse_profile_action(
        "general_compact",
        "Thought: 我要检查爱心状态。\n"
        'Action: {"action_type":"inspect","coordinate":[500,250]}',
        screen_width=200,
        screen_height=400,
    )
    assert payload["action_type"] == "inspect"
    assert payload["x"] == 100
    assert payload["y"] == 100
    assert payload["intent"] == "我要检查爱心状态。"


def test_target_crop_uses_full_source_and_has_bounded_size(tmp_path: Path) -> None:
    source = tmp_path / "screen.png"
    image = Image.new("RGB", (1080, 1920), "white")
    ImageDraw.Draw(image).rectangle((500, 900, 580, 980), fill="red")
    image.save(source)
    observation = Observation(str(source), 1080, 1920)
    output = crop_action_target(
        observation,
        Action("tap", x=540, y=940),
        output_path=tmp_path / "crop.png",
    )
    with Image.open(output) as crop:
        assert crop.size == (384, 384)
        assert crop.getpixel((192, 192))[0] > 200


class _LikeBackend:
    platform = "android"

    def __init__(self, state: str) -> None:
        self.state = state
        self.executed: list[Action] = []

    async def preflight(self) -> None:
        return None

    async def list_apps(self) -> list[str]:
        return []

    async def observe(self, screenshot_path: Path, timeout: float = 5.0) -> Observation:
        del timeout
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new("RGB", (200, 400), "white")
        draw = ImageDraw.Draw(image)
        if self.state == LIKE_LIKED:
            draw.polygon([(100, 220), (75, 190), (55, 215), (100, 270), (145, 215), (125, 190)], fill="red")
        elif self.state == LIKE_NOT_INTERESTED:
            draw.ellipse((65, 185, 135, 255), outline="black", width=5)
            draw.line((60, 180, 140, 260), fill="black", width=8)
        else:
            draw.ellipse((65, 185, 135, 255), outline="black", width=5)
        image.save(screenshot_path)
        return Observation(
            screenshot_path=str(screenshot_path),
            screen_width=200,
            screen_height=400,
            foreground_app="generic.feed",
            platform="android",
            extra={},
        )

    async def execute(self, action: Action, timeout: float = 5.0) -> str:
        del timeout
        self.executed.append(action)
        self.state = LIKE_LIKED
        return "executed"


class _LikeLLM:
    def __init__(self, backend: _LikeBackend, *, actor_action: str = "click") -> None:
        self.backend = backend
        self.actor_action = actor_action
        self.crop_sizes: list[tuple[int, int]] = []
        self.actor_image_sizes: list[tuple[int, int]] = []

    async def chat(self, **kwargs: object) -> LLMResponse:
        messages = kwargs["messages"]
        assert isinstance(messages, list)
        first = messages[0]
        content = first.get("content")
        if isinstance(content, list) and any(
            "safety classifier" in str(item.get("text") or "")
            for item in content
            if isinstance(item, dict)
        ):
            image_item = next(item for item in content if item.get("type") == "image_url")
            encoded = image_item["image_url"]["url"].split(",", 1)[1]
            with Image.open(BytesIO(base64.b64decode(encoded))) as crop:
                self.crop_sizes.append(crop.size)
            return LLMResponse(
                content=(
                    '{"state":"'
                    + self.backend.state
                    + '","confidence":0.99,"reason":"fixture"}'
                ),
                usage={"total_tokens": 3},
            )
        actor_images = [
            item
            for message in messages
            for item in (message.get("content") if isinstance(message.get("content"), list) else [])
            if isinstance(item, dict) and item.get("type") == "image_url"
        ]
        if actor_images:
            encoded = actor_images[-1]["image_url"]["url"].split(",", 1)[1]
            with Image.open(BytesIO(base64.b64decode(encoded))) as actor_image:
                self.actor_image_sizes.append(actor_image.size)
        return LLMResponse(
            content=(
                "Thought: 我要检查并点击点赞爱心。\n"
                'Action: {"action_type":"'
                + self.actor_action
                + '","coordinate":[500,550]}'
            ),
            usage={"total_tokens": 5},
        )


def _run_like_agent(tmp_path: Path, state: str, *, actor_action: str = "click"):
    backend = _LikeBackend(state)
    llm = _LikeLLM(backend, actor_action=actor_action)
    run_root = tmp_path / "run"
    agent = GuiAgent(
        llm=llm,
        backend=backend,
        trajectory_recorder=TrajectoryRecorder(output_dir=run_root, task="给当前内容点赞"),
        model="small-gui",
        artifacts_root=run_root,
        max_steps=3,
        agent_profile="general_compact",
        image_scale_ratio=0.5,
    )
    result = asyncio.run(agent.run("给当前内容点赞", max_retries=1))
    return result, backend, llm


def test_like_tap_requires_unliked_then_verifies_liked(tmp_path: Path) -> None:
    assert task_requests_like_on("给当前内容点赞")
    result, backend, llm = _run_like_agent(tmp_path, LIKE_UNLIKED)
    assert result.success is True
    assert len(backend.executed) == 1
    assert llm.actor_image_sizes == [(100, 200)]
    assert llm.crop_sizes == [(384, 384), (384, 384)]


def test_slash_heart_is_blocked_without_execution(tmp_path: Path) -> None:
    result, backend, llm = _run_like_agent(tmp_path, LIKE_NOT_INTERESTED)
    assert result.success is False
    assert backend.executed == []
    assert llm.crop_sizes == [(384, 384)]
    assert result.error and "not_interested_is_not_like" in result.error


def test_inspect_of_filled_heart_completes_without_toggling(tmp_path: Path) -> None:
    result, backend, llm = _run_like_agent(tmp_path, LIKE_LIKED, actor_action="inspect")
    assert result.success is True
    assert backend.executed == []
    assert llm.crop_sizes == [(384, 384)]


def test_inspect_of_outline_heart_is_promoted_to_verified_tap(tmp_path: Path) -> None:
    result, backend, llm = _run_like_agent(tmp_path, LIKE_UNLIKED, actor_action="inspect")
    assert result.success is True
    assert [action.action_type for action in backend.executed] == ["tap"]
    assert llm.crop_sizes == [(384, 384), (384, 384)]
