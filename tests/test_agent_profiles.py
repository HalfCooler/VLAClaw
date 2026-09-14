from __future__ import annotations

from types import SimpleNamespace

from PIL import Image

from vlaclaw.agents.profiles import build_profile_messages, parse_profile_action
from vlaclaw.agents.utils.prompts import GENERAL_COMPACT_PROMPT_TEMPLATE
from vlaclaw.observation import Observation


def _observation(path: str) -> Observation:
    return Observation(
        screenshot_path=path,
        screen_width=200,
        screen_height=400,
        foreground_app="com.example.app",
        platform="android",
    )


def test_general_compact_prompt_requires_thought_and_action_without_memory() -> None:
    prompt = GENERAL_COMPACT_PROMPT_TEMPLATE.render(
        tools="",
        scale_factor=1000,
        extra_action_rows="",
        decision_rules="",
        compact_skill_instructions="",
    )

    assert "Thought: [Analysis including reference to key steps/points when applicable]" in prompt
    assert "Action: [Single JSON action]" in prompt
    assert "Thought: 我要点击目标控件来完成任务。" in prompt
    assert "顶层必须同时包含 action_type 和 memory" not in prompt
    assert "memory 只提交本轮增量" not in prompt


def test_general_compact_parser_uses_thought_as_action_semantics() -> None:
    payload = parse_profile_action(
        "general_compact",
        'Thought: 我要点击搜索框。\nAction: {"action_type":"click","coordinate":[500,250]}',
        screen_width=200,
        screen_height=400,
    )

    assert payload == {
        "summary": "我要点击搜索框。",
        "intent": "我要点击搜索框。",
        "action_type": "tap",
        "x": 100,
        "y": 100,
    }


def test_general_compact_history_keeps_previous_thought_action_pair(tmp_path) -> None:
    first_path = tmp_path / "first.png"
    current_path = tmp_path / "current.png"
    Image.new("RGB", (200, 400), "black").save(first_path)
    Image.new("RGB", (200, 400), "white").save(current_path)
    previous_response = (
        'Thought: 我要打开搜索。\n'
        'Action: {"action_type":"click","coordinate":[500,250]}'
    )
    history = [
        SimpleNamespace(
            observation=_observation(str(first_path)),
            tool_result_message={"content": "executed"},
            assistant_message={"content": previous_response},
            action_summary="我要打开搜索。",
            raw_response_content=previous_response,
        )
    ]

    messages = build_profile_messages(
        "general_compact",
        task="搜索天气",
        current_observation=_observation(str(current_path)),
        history=history,
        model_name="small-gui",
        history_image_window=1,
    )

    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert messages[2]["content"] == [{"type": "text", "text": previous_response}]
    assert messages[3]["content"][0] == {"type": "text", "text": "Tool call result: executed"}
    assert "Memory state:" not in str(messages)
