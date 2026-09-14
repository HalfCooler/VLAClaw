from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from vlaclaw.action import Action
from vlaclaw.agent import GuiAgent
from vlaclaw.cli import OpenAICompatibleLLMProvider
from vlaclaw.control import TransitionMonitor, build_task_contract, guard_action
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
        assert kwargs.get("max_tokens") == 48
        return LLMResponse(
            content=(
                '{"action_type":"click","coordinate":[247,283],'
                '"memory":{"current":"视频正在试看","remaining":"继续观看"}}'
            ),
            usage={"completion_tokens": 35, "total_tokens": 35},
        )


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
        assert kwargs.get("max_tokens") == 48
        return LLMResponse(
            content=(
                '{"action_type":"click","coordinate":[500,111],'
                '"memory":{"current":"账户设置","remaining":"查看设置"}}'
            )
        )


class _LargeRecoveryLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def chat(self, **kwargs: object) -> LLMResponse:
        self.calls += 1
        assert kwargs.get("max_tokens") == 96
        return LLMResponse(
            content=(
                '{"action_type":"status","goal_status":"complete",'
                '"memory":{"current":"设置已查看","remaining":"无"}}'
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


def test_provider_hard_output_cap_cannot_be_overridden() -> None:
    provider = OpenAICompatibleLLMProvider(
        base_url="https://example.invalid/v1",
        model="large",
        hard_max_tokens=96,
    )
    completions = _FakeCompletions()
    provider._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))  # type: ignore[assignment]

    asyncio.run(provider.chat(messages=[{"role": "user", "content": "x"}], max_tokens=256))

    assert completions.kwargs["max_tokens"] == 96
