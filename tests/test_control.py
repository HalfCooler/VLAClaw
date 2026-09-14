from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from vlaclaw.action import Action
from vlaclaw.agent import GuiAgent
from vlaclaw.backends.adb import _media_playback_extra, _parse_media_session_states
from vlaclaw.cli import OpenAICompatibleLLMProvider
from vlaclaw.control import (
    TransitionMonitor,
    build_task_contract,
    guard_action,
    playback_goal_is_active,
)
from vlaclaw.difficulty import route_for_difficulty
from vlaclaw.interfaces import LLMResponse
from vlaclaw.observation import Observation
from vlaclaw.trajectory.recorder import TrajectoryRecorder, load_trajectory_events


def _youku_playback_observation(screenshot_path: str | None = None) -> Observation:
    return Observation(
        screenshot_path=screenshot_path,
        screen_width=1200,
        screen_height=2608,
        foreground_app="com.youku.phone",
        platform="android",
        extra={
            "visible_text": ["试看6分钟", "开通会员看完整视频"],
            "ui_tree": [
                {
                    "text": "试看6分钟 开通会员看完整视频",
                    "class": "android.widget.TextView",
                    "enabled": True,
                    "bounds": "[0,574][754,711]",
                }
            ],
        },
    )


def _observation(label: str, app: str = "example.app") -> Observation:
    return Observation(
        screenshot_path=None,
        screen_width=1080,
        screen_height=1920,
        foreground_app=app,
        platform="android",
        extra={"visible_text": [label]},
    )


def test_youku_payment_diversion_replay_is_blocked_and_goal_is_complete() -> None:
    observation = _youku_playback_observation()
    action = Action(action_type="tap", x=296, y=738)

    verdict = guard_action(
        action,
        observation,
        build_task_contract("继续观看优酷视频历史记录里的第一个视频。"),
    )

    assert verdict.allowed is False
    assert verdict.effect == "financial"
    assert verdict.goal_already_satisfied is True
    assert "开通会员" in verdict.target_text


def test_android_media_session_exposes_foreground_playback_as_authoritative_context() -> None:
    output = """
Sessions Stack - have 2 sessions:
  Session #0:
    com.youku.phone/player (userId=0)
      state=PlaybackState {state=3, position=9812, speed=1.0}
  Session #1:
    package=com.spotify.music
      state=PlaybackState {state=2, position=42, speed=0.0}
"""

    sessions = _parse_media_session_states(output)
    extra = _media_playback_extra(sessions, "com.youku.phone")

    assert sessions == [
        {"package": "com.youku.phone", "state": "playing"},
        {"package": "com.spotify.music", "state": "paused"},
    ]
    assert extra["media_playback"] == {
        "package": "com.youku.phone",
        "state": "playing",
        "source": "android_media_session",
    }


def test_playback_goal_requires_playing_session_to_match_foreground_app() -> None:
    observation = _observation("已选中第一集", app="com.youku.phone")
    observation.extra["media_playback"] = {
        "package": "com.spotify.music",
        "state": "playing",
        "source": "android_media_session",
    }

    assert playback_goal_is_active("播放第一集", observation) is False

    observation.extra["media_playback"]["package"] = "com.youku.phone"
    assert playback_goal_is_active("播放第一集", observation) is True


def test_explicit_purchase_task_authorizes_payment_action() -> None:
    observation = _youku_playback_observation()
    verdict = guard_action(
        Action(action_type="tap", x=296, y=738),
        observation,
        build_task_contract("购买优酷会员并付款"),
    )
    assert verdict.allowed is True
    assert verdict.effect == "financial"


def test_payment_authorization_does_not_authorize_password_entry() -> None:
    observation = Observation(
        screenshot_path=None,
        screen_width=1080,
        screen_height=1920,
        foreground_app="com.youku.phone",
        extra={
            "ui_tree": [
                {"text": "输入支付密码", "bounds": "[100,100][900,300]", "enabled": True}
            ]
        },
    )
    verdict = guard_action(
        Action(action_type="tap", x=500, y=200),
        observation,
        build_task_contract("购买优酷会员并付款"),
    )
    assert verdict.allowed is False
    assert verdict.effect == "authentication"


def test_transition_monitor_detects_alternating_action_cycle() -> None:
    first = _observation("A")
    second = _observation("B")
    monitor = TransitionMonitor(first, build_task_contract("打开示例页面"))

    assert monitor.evaluate(first, Action(action_type="tap", x=1, y=1), second).status == "progress"
    assert monitor.evaluate(second, Action(action_type="wait"), first).status == "progress"
    verdict = monitor.evaluate(first, Action(action_type="back"), second)

    assert verdict.status == "loop"
    assert verdict.loop_detected is True
    assert verdict.reason == "two_state_cycle"


def test_transition_monitor_rejects_off_task_payment_app() -> None:
    before = _observation("video", app="com.youku.phone")
    after = _observation("输入密码", app="com.eg.android.AlipayGphone")
    monitor = TransitionMonitor(before, build_task_contract("继续观看视频"))

    verdict = monitor.evaluate(before, Action(action_type="tap", x=10, y=10), after)

    assert verdict.status == "off_task_risk"
    assert verdict.reason == "entered_payment_app_outside_task_scope"


def test_difficulty_never_hands_whole_run_to_large_model() -> None:
    for difficulty in ("easy", "medium", "hard"):
        route = route_for_difficulty(difficulty)
        assert route.use_large_model is False
        assert route.agent_profile == "general_compact"


class _FakeBackend:
    platform = "android"

    def __init__(self) -> None:
        self.executed: list[Action] = []

    async def preflight(self) -> None:
        return None

    async def list_apps(self) -> list[str]:
        return []

    async def observe(self, screenshot_path: Path, timeout: float = 5.0) -> Observation:
        del timeout
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (120, 260), "black").save(screenshot_path)
        return _youku_playback_observation(str(screenshot_path))

    async def execute(self, action: Action, timeout: float = 5.0) -> str:
        del timeout
        self.executed.append(action)
        return "executed"


class _FakeLLM:
    async def chat(self, **kwargs: object) -> LLMResponse:
        assert "max_tokens" not in kwargs
        return LLMResponse(
            content=(
                "Thought: 我要点击视频继续观看。\n"
                'Action: {"action_type":"click","coordinate":[247,283]}'
            ),
            usage={"completion_tokens": 35, "total_tokens": 35},
        )


class _PlaybackBackend(_FakeBackend):
    def __init__(self, *, initially_playing: bool = False) -> None:
        super().__init__()
        self.initially_playing = initially_playing

    async def observe(self, screenshot_path: Path, timeout: float = 5.0) -> Observation:
        del timeout
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (120, 260), "navy").save(screenshot_path)
        extra: dict[str, object] = {"visible_text": ["第一集", "已选中"]}
        if self.initially_playing or self.executed:
            extra["media_playback"] = {
                "package": "com.youku.phone",
                "state": "playing",
                "source": "android_media_session",
            }
        return Observation(
            screenshot_path=str(screenshot_path),
            screen_width=1200,
            screen_height=2608,
            foreground_app="com.youku.phone",
            platform="android",
            extra=extra,
        )


class _CountingPlaybackLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def chat(self, **kwargs: object) -> LLMResponse:
        del kwargs
        self.calls += 1
        return LLMResponse(
            content=(
                "Thought: 我要点击第一集开始播放。\n"
                'Action: {"action_type":"click","coordinate":[300,300]}'
            )
        )


def test_agent_does_not_click_when_foreground_media_is_already_playing(tmp_path: Path) -> None:
    backend = _PlaybackBackend(initially_playing=True)
    llm = _CountingPlaybackLLM()
    run_root = tmp_path / "already-playing"
    agent = GuiAgent(
        llm=llm,
        backend=backend,
        trajectory_recorder=TrajectoryRecorder(output_dir=run_root, task="播放第一集"),
        model="qwen3.5-4b",
        artifacts_root=run_root,
        max_steps=4,
        agent_profile="general_compact",
    )

    result = asyncio.run(agent.run("播放第一集", max_retries=1))

    assert result.success is True
    assert result.steps_taken == 0
    assert llm.calls == 0
    assert backend.executed == []


def test_agent_stops_immediately_after_click_starts_playback(tmp_path: Path) -> None:
    backend = _PlaybackBackend()
    llm = _CountingPlaybackLLM()
    run_root = tmp_path / "starts-playing"
    agent = GuiAgent(
        llm=llm,
        backend=backend,
        trajectory_recorder=TrajectoryRecorder(output_dir=run_root, task="播放第一集"),
        model="qwen3.5-4b",
        artifacts_root=run_root,
        max_steps=4,
        agent_profile="general_compact",
    )

    result = asyncio.run(agent.run("播放第一集", max_retries=1))

    assert result.success is True
    assert result.steps_taken == 1
    assert llm.calls == 1
    assert len(backend.executed) == 1
    trajectory = json.loads((run_root / "traj.json").read_text(encoding="utf-8"))
    assert trajectory["steps"][0]["model_output"]["playback_verification"]["state"] == "playing"


def test_agent_replay_stops_before_executing_membership_tap(tmp_path: Path) -> None:
    backend = _FakeBackend()
    run_root = tmp_path / "run"
    recorder = TrajectoryRecorder(output_dir=run_root, task="继续观看优酷视频历史记录里的第一个视频")
    agent = GuiAgent(
        llm=_FakeLLM(),
        backend=backend,
        trajectory_recorder=recorder,
        model="qwen3.5-4b",
        artifacts_root=run_root,
        max_steps=4,
        agent_profile="general_compact",
    )

    result = asyncio.run(
        agent.run("继续观看优酷视频历史记录里的第一个视频", max_retries=1)
    )

    assert result.success is True
    assert backend.executed == []
    trajectory = json.loads((run_root / "traj.json").read_text(encoding="utf-8"))
    assert trajectory["steps"][0]["model_output"]["action_executed"] is False
    assert trajectory["steps"][0]["model_output"]["actor"] == "small"
    assert trajectory["steps"][0]["model_output"]["guard"]["effect"] == "financial"
    assert any(event["type"] == "action_guard" for event in trajectory["events"])
    normalized_events = load_trajectory_events(run_root / "traj.json")
    assert any(event["type"] == "action_guard" for event in normalized_events)


class _GuardRecoveryBackend(_FakeBackend):
    async def observe(self, screenshot_path: Path, timeout: float = 5.0) -> Observation:
        del timeout
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (100, 180), "white").save(screenshot_path)
        return Observation(
            screenshot_path=str(screenshot_path),
            screen_width=1000,
            screen_height=1800,
            foreground_app="com.example.settings",
            platform="android",
            extra={
                "ui_tree": [
                    {"text": "删除账户", "bounds": "[100,100][900,300]", "enabled": True}
                ]
            },
        )


class _SmallUnsafeLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def chat(self, **kwargs: object) -> LLMResponse:
        self.calls += 1
        assert "max_tokens" not in kwargs
        return LLMResponse(
            content=(
                "Thought: 我要点击账户设置。\n"
                'Action: {"action_type":"click","coordinate":[500,111]}'
            )
        )


class _LargeRecoveryLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def chat(self, **kwargs: object) -> LLMResponse:
        self.calls += 1
        assert "max_tokens" not in kwargs
        return LLMResponse(
            content=(
                "Thought: 设置已查看，任务已经完成。\n"
                'Action: {"action_type":"status","goal_status":"complete"}'
            )
        )


def test_guard_failure_gets_one_bounded_large_model_recovery(tmp_path: Path) -> None:
    small = _SmallUnsafeLLM()
    large = _LargeRecoveryLLM()
    backend = _GuardRecoveryBackend()
    run_root = tmp_path / "guard-recovery"
    agent = GuiAgent(
        llm=small,
        planner_llm=large,
        planner_model="qwen3.8-flash",
        backend=backend,
        trajectory_recorder=TrajectoryRecorder(output_dir=run_root, task="查看账户设置"),
        model="qwen3.5-4b",
        artifacts_root=run_root,
        max_steps=2,
        agent_profile="general_compact",
    )

    result = asyncio.run(agent.run("查看账户设置", max_retries=1))

    assert result.success is True
    assert small.calls == 1
    assert large.calls == 1
    assert backend.executed == []
    trajectory = json.loads((run_root / "traj.json").read_text(encoding="utf-8"))
    model_output = trajectory["steps"][0]["model_output"]
    assert model_output["actor"] == "large"
    assert model_output["model"] == "qwen3.8-flash"
    assert model_output["trigger"] == "action_guard"
    assert [call["actor"] for call in model_output["model_calls"]] == ["small", "large"]
    escalations = [event for event in trajectory["events"] if event["type"] == "planner_escalation"]
    assert escalations[-1]["trigger"] == "action_guard"


class _FakeCompletions:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] = {}

    async def create(self, **kwargs: object) -> SimpleNamespace:
        self.kwargs = kwargs
        message = SimpleNamespace(content="{}", tool_calls=None)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )


def test_provider_does_not_force_output_token_cap() -> None:
    provider = OpenAICompatibleLLMProvider(
        base_url="https://example.invalid/v1",
        model="large",
    )
    completions = _FakeCompletions()
    provider._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))  # type: ignore[assignment]

    asyncio.run(provider.chat(messages=[{"role": "user", "content": "x"}]))

    assert "max_tokens" not in completions.kwargs
