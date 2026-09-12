"""VLAClaw vision-action loop: observe, act, escalate repeats, stop on stagnation."""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from vlaclaw.action import Action, ActionError, describe_action, parse_action, resolve_coordinate
from vlaclaw.agent_profiles import (
    build_profile_messages,
    canonicalize_agent_profile,
    coordinate_mode_for_profile,
    normalize_profile_response_for_observation,
    profile_llm_defaults,
    profile_uses_native_tools,
)
from vlaclaw.image_utils import scale_image
from vlaclaw.interfaces import DeviceBackend, LLMProvider, LLMResponse, ProgressCallback, ToolCall
from vlaclaw.observation import Observation
from vlaclaw.paths import DEFAULT_GUI_RUNS_DIR
from vlaclaw.planner_escalation import (
    RepeatVerdict,
    build_repeat_escalation_text,
    build_repeat_judge_text,
    canonicalize_repeat_judge_model,
    inject_repeat_escalation_hint,
    observation_ui_tree,
    parse_repeat_verdict,
    should_judge_repeat,
    ui_tree_difference_ratio,
)
from vlaclaw.trajectory.recorder import ExecutionPhase, TrajectoryRecorder
from vlaclaw.trajectory.summarizer import build_state_note

logger = logging.getLogger(__name__)

_DONE_FAILURE_HINTS: tuple[str, ...] = (
    "fail",
    "failed",
    "failure",
    "unable",
    "cannot",
    "can't",
    "error",
    "not completed",
    "incomplete",
    "失败",
    "无法",
    "不能",
    "错误",
    "未完成",
)


@dataclass(frozen=True)
class StepResult:
    action: Action
    tool_call_id: str
    tool_result: str
    assistant_message: dict[str, Any]
    action_summary: str
    action_intent: str | None = None
    state_summary: str | None = None
    next_observation: Observation | None = None
    prompt_snapshot: dict[str, Any] | None = None
    model_snapshot: dict[str, Any] | None = None
    done: bool = False
    intervention_requested: bool = False
    step_usage: dict[str, int] = dataclasses.field(default_factory=dict)
    event_usage: dict[str, int] = dataclasses.field(default_factory=dict)
    duration_s: float = 0.0
    chat_latency_s: float | None = None
    ttft_s: float | None = None


@dataclass(frozen=True)
class HistoryTurn:
    step_index: int
    observation: Observation
    assistant_message: dict[str, Any]
    tool_result_message: dict[str, Any]
    action_summary: str
    action_intent: str | None = None
    state_summary: str | None = None
    raw_response_content: str | None = None


@dataclass(frozen=True)
class AgentResult:
    success: bool
    summary: str
    model_summary: str | None = None
    trace_path: str | None = None
    steps_taken: int = 0
    error: str | None = None
    token_usage: dict[str, int] = dataclasses.field(default_factory=dict)


@dataclass(frozen=True)
class _ScreenFingerprint:
    app: str | None
    method: str
    digest: str


class _StepExecutionError(RuntimeError):
    def __init__(self, message: str, *, model_snapshot: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.model_snapshot = model_snapshot


class GuiAgent:
    _MAX_TOOL_RETRIES = 3
    _COORDINATE_ACTIONS = frozenset({"tap", "double_tap", "long_press", "swipe", "drag", "scroll"})
    _POST_ACTION_SETTLE_SECONDS = 0.50
    _OPEN_APP_SETTLE_SECONDS = 5.00
    _POST_ACTION_STABILITY_WINDOW_SECONDS = 2.0
    _POST_ACTION_STABILITY_POLL_SECONDS = 0.15
    _POST_ACTION_STABILITY_MAX_ATTEMPTS = 4
    _POST_ACTION_STABILITY_FRAMES_REQUIRED = 2
    _POST_ACTION_OBSERVE_TIMEOUT_SECONDS = 8.0
    _NO_SETTLE_ACTIONS = frozenset({"wait", "done", "request_intervention"})
    _STAGNATION_SSIM_SIZE = 64
    _STAGNATION_SSIM_THRESHOLD = 0.985
    _STAGNATION_CLICK_DISTANCE = 0.03
    _PROGRESS_TEXT_LIMIT = 4000

    def __init__(
        self,
        llm: LLMProvider,
        backend: DeviceBackend,
        trajectory_recorder: TrajectoryRecorder,
        model: str = "",
        artifacts_root: Path | str = DEFAULT_GUI_RUNS_DIR,
        max_steps: int = 15,
        step_timeout: float = 90.0,
        history_image_window: int | None = None,
        progress_callback: ProgressCallback | None = None,
        agent_profile: str | None = None,
        image_scale_ratio: float = 0.5,
        stagnation_limit: int = 0,
        reasoning_effort: str | None = None,
        planner_llm: LLMProvider | None = None,
        enable_repeat_escalation: bool = True,
        repeat_judge_model: str = "small",
        difficulty_snapshot: dict[str, Any] | None = None,
    ) -> None:
        self.llm = llm
        self.backend = backend
        self.model = model
        self.agent_profile = canonicalize_agent_profile(agent_profile)
        profile_llm = profile_llm_defaults(self.agent_profile)
        self._reasoning_effort = (
            reasoning_effort
            if reasoning_effort not in (None, "", "auto")
            else profile_llm.get("reasoning_effort")
        )
        self._step_max_tokens = profile_llm.get("max_tokens")
        self.artifacts_root = Path(artifacts_root)
        self.max_steps = max_steps
        self.step_timeout = step_timeout
        self.history_image_window = (
            None if history_image_window is None else max(1, history_image_window)
        )
        self.progress_callback = progress_callback
        self._trajectory_recorder = trajectory_recorder
        self._planner_llm = planner_llm
        self._enable_repeat_escalation = bool(enable_repeat_escalation) and planner_llm is not None
        self._repeat_judge_model = canonicalize_repeat_judge_model(repeat_judge_model)
        self._difficulty_snapshot = (
            dict(difficulty_snapshot) if isinstance(difficulty_snapshot, dict) else None
        )
        self._image_scale_ratio = image_scale_ratio
        try:
            parsed_stagnation_limit = int(stagnation_limit)
        except (TypeError, ValueError):
            parsed_stagnation_limit = 0
        self.stagnation_limit = max(0, parsed_stagnation_limit)

    async def run(self, task: str, *, max_retries: int = 3) -> AgentResult:
        self._trajectory_recorder.start(phase=ExecutionPhase.AGENT)
        if self._difficulty_snapshot:
            self._trajectory_recorder.record_event("difficulty_route", **self._difficulty_snapshot)
            await self._emit_difficulty_progress(self._difficulty_snapshot.get("difficulty"))

        last_error: str | None = None
        last_model_summary: str | None = None
        last_trace_path: str | None = None
        last_steps_taken = 0
        result: AgentResult | None = None
        total_usage: dict[str, int] = {}

        for attempt in range(max_retries):
            run_dir = self._make_run_dir(attempt)
            last_trace_path = str(run_dir)
            self._trajectory_recorder.set_attempt(attempt + 1)
            await self._log_attempt_event(run_dir, "attempt_start", attempt=attempt, task=task)
            try:
                result = await self._run_once(task, run_dir=run_dir)
                for key, value in result.token_usage.items():
                    total_usage[key] = total_usage.get(key, 0) + value
                await self._log_attempt_event(
                    run_dir,
                    "attempt_result",
                    attempt=attempt,
                    success=result.success,
                    summary=result.summary,
                    error=result.error,
                    steps_taken=result.steps_taken,
                )
                if result.success:
                    last_error = None
                    break
                last_error = result.error
                last_model_summary = result.model_summary
                last_trace_path = result.trace_path or last_trace_path
                last_steps_taken = result.steps_taken
                if result.error and (
                    result.error.startswith("intervention_cancelled")
                    or result.error == "stagnation_detected"
                ):
                    break
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                await self._log_attempt_event(
                    run_dir,
                    "attempt_exception",
                    attempt=attempt,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    model_response=getattr(exc, "model_snapshot", None),
                )

        if result is None:
            result = AgentResult(
                success=False,
                summary=self._build_state_note(
                    status="blocked" if last_error and "stagnation" in last_error.lower() else "partial",
                    history=[],
                    current_observation=None,
                    error=last_error or f"Failed after {max_retries} attempt(s).",
                ),
                model_summary=last_model_summary,
                trace_path=last_trace_path,
                steps_taken=last_steps_taken,
                error=last_error,
                token_usage=total_usage,
            )
        else:
            result = dataclasses.replace(
                result,
                token_usage=total_usage,
                trace_path=last_trace_path or result.trace_path,
                error=result.error if result.success else last_error or result.error,
            )
        self._trajectory_recorder.finish(
            success=result.success,
            error=result.error,
            summary=result.summary,
            model_summary=result.model_summary,
        )
        return result

    async def _run_once(self, task: str, *, run_dir: Path) -> AgentResult:
        try:
            await self.backend.preflight()
        except Exception as exc:
            return AgentResult(
                success=False,
                summary=self._build_state_note(
                    status="blocked",
                    history=[],
                    current_observation=None,
                    error=f"Preflight failed: {exc}",
                ),
                trace_path=str(run_dir),
                error=str(exc),
            )

        initial_screenshot = run_dir / "screenshots" / "000_initial.png"
        obs = await self.backend.observe(initial_screenshot, timeout=self.step_timeout)
        self._trajectory_recorder.record_screenshot(initial_screenshot, kind="initial")

        history: list[HistoryTurn] = []
        total_usage: dict[str, int] = {}
        previous_fingerprint: _ScreenFingerprint | None = None
        previous_action_type: str | None = None
        previous_action: Action | None = None
        previous_observation: Observation | None = None
        stagnation_streak = 0
        if self.stagnation_limit > 0:
            previous_fingerprint = self._build_screen_fingerprint(obs)

        steps_taken = 0
        for step in range(self.max_steps):
            step_index = step + 1
            messages = self._build_messages(task=task, current_observation=obs, history=history)
            try:
                result = await asyncio.wait_for(
                    self._run_step(
                        messages=messages,
                        step_index=step_index,
                        total_steps=self.max_steps,
                        current_observation=obs,
                        previous_action=previous_action,
                        previous_observation=previous_observation,
                        task=task,
                    ),
                    timeout=self.step_timeout * 3,
                )
            except asyncio.TimeoutError:
                await self._log_attempt_event(run_dir, "timeout", step_index=step_index)
                return AgentResult(
                    success=False,
                    summary=self._build_state_note(
                        status="partial",
                        history=history,
                        current_observation=obs,
                        error="step_timeout",
                    ),
                    trace_path=str(run_dir),
                    steps_taken=step_index,
                    error="step_timeout",
                    token_usage=total_usage,
                )

            steps_taken = step_index
            for key, value in result.step_usage.items():
                total_usage[key] = total_usage.get(key, 0) + value

            if result.intervention_requested:
                await self._record_completed_step(
                    run_dir=run_dir,
                    step_index=step_index,
                    current_observation=obs,
                    result=result,
                )
                return AgentResult(
                    success=False,
                    summary=self._build_state_note(
                        status="blocked",
                        history=history + [self._history_turn_from_step(step_index, obs, result)],
                        current_observation=obs,
                        error="intervention_cancelled",
                    ),
                    model_summary=result.state_summary or result.action_summary,
                    trace_path=str(run_dir),
                    steps_taken=steps_taken,
                    error="intervention_cancelled: no_handler",
                    token_usage=total_usage,
                )

            await self._record_completed_step(
                run_dir=run_dir,
                step_index=step_index,
                current_observation=obs,
                result=result,
            )

            if result.done:
                success = self._resolve_done_status(result.action) == "success"
                return AgentResult(
                    success=success,
                    summary=self._build_state_note(
                        status="completed" if success else "blocked",
                        history=history,
                        current_observation=obs,
                        current_action_summary=result.state_summary or result.action_summary,
                        error=None if success else result.tool_result,
                    ),
                    model_summary=result.state_summary or result.action_summary,
                    trace_path=str(run_dir),
                    steps_taken=steps_taken,
                    error=None if success else result.tool_result,
                    token_usage=total_usage,
                )

            if self.stagnation_limit > 0 and result.next_observation is not None:
                current_fingerprint = self._build_screen_fingerprint(result.next_observation)
                similar_plan = self._is_similar_planned_action(previous_action, result.action, obs)
                if (
                    previous_fingerprint is not None
                    and current_fingerprint is not None
                    and (
                        previous_action_type is None
                        or previous_action_type == result.action.action_type
                    )
                    and similar_plan
                    and self._is_same_screen(previous_fingerprint, current_fingerprint)
                ):
                    stagnation_streak += 1
                else:
                    stagnation_streak = 0
                previous_fingerprint = current_fingerprint
                previous_action_type = result.action.action_type

                if stagnation_streak >= self.stagnation_limit:
                    app_label = (
                        result.next_observation.foreground_app or obs.foreground_app or "unknown"
                    )
                    history_with_current_step = history + [
                        self._history_turn_from_step(step_index, obs, result)
                    ]
                    await self._log_attempt_event(
                        run_dir,
                        "stagnation_detected",
                        step_index=step_index,
                        stagnation_streak=stagnation_streak,
                        stagnation_limit=self.stagnation_limit,
                        foreground_app=app_label,
                        previous_action=(
                            describe_action(previous_action) if previous_action is not None else None
                        ),
                        current_action=describe_action(result.action),
                    )
                    return AgentResult(
                        success=False,
                        summary=self._build_state_note(
                            status="blocked",
                            history=history_with_current_step,
                            current_observation=result.next_observation or obs,
                            error="stagnation_detected",
                        ),
                        model_summary=result.state_summary or result.action_summary,
                        trace_path=str(run_dir),
                        steps_taken=steps_taken,
                        error="stagnation_detected",
                        token_usage=total_usage,
                    )

            history.append(self._history_turn_from_step(step_index, obs, result))
            previous_action = result.action
            previous_observation = obs
            if result.next_observation is not None:
                obs = result.next_observation

        return AgentResult(
            success=False,
            summary=self._build_state_note(
                status="partial",
                history=history,
                current_observation=obs,
                error="max_steps_exceeded",
            ),
            trace_path=str(run_dir),
            steps_taken=steps_taken,
            error="max_steps_exceeded",
            token_usage=total_usage,
        )

    def _history_turn_from_step(
        self,
        step_index: int,
        observation: Observation,
        result: StepResult,
    ) -> HistoryTurn:
        return HistoryTurn(
            step_index=step_index,
            observation=observation,
            assistant_message=result.assistant_message,
            tool_result_message={
                "role": "tool",
                "tool_call_id": result.tool_call_id,
                "content": result.tool_result,
            },
            action_summary=result.action_summary,
            action_intent=result.action_intent,
            state_summary=result.state_summary,
            raw_response_content=(
                result.model_snapshot.get("raw_content")
                if isinstance(result.model_snapshot, dict)
                else None
            ),
        )

    async def _record_completed_step(
        self,
        *,
        run_dir: Path,
        step_index: int,
        current_observation: Observation,
        result: StepResult,
    ) -> None:
        del run_dir, step_index
        recorded_observation = result.next_observation or current_observation
        screenshot_path = (
            result.next_observation.screenshot_path
            if result.next_observation and result.next_observation.screenshot_path
            else current_observation.screenshot_path
        )
        model_snapshot = result.model_snapshot if isinstance(result.model_snapshot, dict) else {}
        if model_snapshot:
            model_output: Any = {"content": model_snapshot.get("raw_content") or ""}
            model_output["tool_calls"] = model_snapshot.get("tool_calls") or []
        else:
            model_output = result.action_intent or result.action_summary or ""
        self._trajectory_recorder.record_step(
            action=self._serialize_action(result.action),
            model_output=model_output,
            screenshot_path=str(screenshot_path) if screenshot_path else None,
            foreground_app=recorded_observation.foreground_app,
            token_usage=(result.event_usage or result.step_usage) or None,
            inference_time_s=result.chat_latency_s,
        )

    @staticmethod
    def _finalize_step_result(
        result: StepResult,
        *,
        step_usage: dict[str, int],
        step_start: float,
        step_chat_latency_s: float,
        step_ttft_s: float | None,
    ) -> StepResult:
        return replace(
            result,
            step_usage=step_usage,
            duration_s=time.monotonic() - step_start,
            chat_latency_s=step_chat_latency_s or None,
            ttft_s=step_ttft_s,
        )

    def _skipped_step_result(
        self,
        *,
        reason: str,
        step_usage: dict[str, int],
        step_start: float,
        step_chat_latency_s: float,
        step_ttft_s: float | None,
        current_observation: Observation,
        model_snapshot: dict[str, Any] | None = None,
    ) -> StepResult:
        note = f"Step skipped: {reason}"
        return self._finalize_step_result(
            StepResult(
                action=Action(action_type="wait"),
                tool_call_id="skipped",
                tool_result=note,
                assistant_message={"role": "assistant", "content": ""},
                action_summary=note,
                next_observation=current_observation,
                model_snapshot=model_snapshot,
            ),
            step_usage=step_usage,
            step_start=step_start,
            step_chat_latency_s=step_chat_latency_s,
            step_ttft_s=step_ttft_s,
        )

    def _repeat_judge_llm(self) -> LLMProvider:
        if self._repeat_judge_model == "large" and self._planner_llm is not None:
            return self._planner_llm
        return self.llm

    async def _repeat_replan_decision(
        self,
        *,
        previous_action: Action | None,
        proposed_action: Action,
        current_observation: Observation,
        previous_observation: Observation | None,
        task: str,
        messages: list[dict[str, Any]],
        original_messages: list[dict[str, Any]],
        step_usage: dict[str, int],
        escalated: bool,
        step_index: int,
    ) -> tuple[bool, dict[str, Any] | None]:
        previous_tree = observation_ui_tree(previous_observation)
        current_tree = observation_ui_tree(current_observation)
        if (
            escalated
            or not self._enable_repeat_escalation
            or self._planner_llm is None
            or not should_judge_repeat(
                previous_action,
                proposed_action,
                previous_tree=previous_tree,
                current_tree=current_tree,
            )
        ):
            return False, None
        assert previous_action is not None

        verdict, judge_usage = await self._judge_repeated_action(
            task=task,
            previous_action=previous_action,
            proposed_action=proposed_action,
            current_observation=current_observation,
        )
        for key, value in judge_usage.items():
            step_usage[key] = step_usage.get(key, 0) + value
        snapshot = {
            "judge_model": self._repeat_judge_model,
            "previous_action_type": previous_action.action_type,
            "proposed_action_type": proposed_action.action_type,
            "ui_tree_difference": ui_tree_difference_ratio(previous_tree, current_tree),
            "repeated": verdict.repeated,
            "reason": verdict.reason,
        }
        self._trajectory_recorder.record_event("repeat_judge", step_index=step_index, **snapshot)
        if not verdict.repeated:
            return False, snapshot

        messages[:] = inject_repeat_escalation_hint(
            original_messages,
            build_repeat_escalation_text(
                previous=previous_action,
                proposed=proposed_action,
                reason=verdict.reason,
                ui_tree_difference=snapshot["ui_tree_difference"],
            ),
        )
        self._trajectory_recorder.record_event(
            "planner_escalation",
            step_index=step_index,
            reason=verdict.reason or "repeat_confirmed",
            previous_action_type=previous_action.action_type,
            proposed_action_type=proposed_action.action_type,
        )
        return True, snapshot

    async def _judge_repeated_action(
        self,
        *,
        task: str,
        previous_action: Action,
        proposed_action: Action,
        current_observation: Observation,
    ) -> tuple[RepeatVerdict, dict[str, int]]:
        prompt = build_repeat_judge_text(
            task=task,
            previous=previous_action,
            proposed=proposed_action,
        )
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        screenshot = Path(current_observation.screenshot_path or "")
        if screenshot.is_file():
            try:
                content.append(self._image_block(screenshot))
            except Exception:
                logger.debug("Repeat judge skipped screenshot for %s", screenshot, exc_info=True)
        try:
            response = await self._repeat_judge_llm().chat(
                messages=[{"role": "user", "content": content}],
                tools=None,
            )
        except Exception as exc:
            logger.warning("Repeat judge failed; treating as not-repeat: %s", exc)
            return RepeatVerdict(repeated=False, reason=f"judge_error: {exc}"), {}
        usage = {
            str(key): int(value)
            for key, value in (response.usage or {}).items()
            if isinstance(value, int)
        }
        return parse_repeat_verdict(response.content), usage

    async def _run_step(
        self,
        messages: list[dict[str, Any]],
        step_index: int,
        total_steps: int,
        current_observation: Observation,
        previous_action: Action | None = None,
        previous_observation: Observation | None = None,
        task: str = "",
    ) -> StepResult:
        step_start = time.monotonic()
        retries_left = self._MAX_TOOL_RETRIES + 1
        step_usage: dict[str, int] = {}
        step_chat_latency_s = 0.0
        step_ttft_s: float | None = None
        actor: LLMProvider = self.llm
        escalated = False
        repeat_judge_snapshot: dict[str, Any] | None = None
        original_messages = list(messages)

        while retries_left > 0:
            retries_left -= 1
            native_tools_enabled = profile_uses_native_tools(self.agent_profile)
            chat_kwargs: dict[str, Any] = {
                "messages": messages,
                "tools": None,
                "tool_choice": None,
            }
            if self._reasoning_effort is not None:
                chat_kwargs["reasoning_effort"] = self._reasoning_effort
            if self._step_max_tokens is not None:
                chat_kwargs["max_tokens"] = self._step_max_tokens
            inference_started_at = time.time()
            try:
                response: LLMResponse = await actor.chat(**chat_kwargs)
            finally:
                step_chat_latency_s += time.time() - inference_started_at
            for key, value in (response.usage or {}).items():
                step_usage[key] = step_usage.get(key, 0) + value
            if step_ttft_s is None and response.ttft_s is not None:
                step_ttft_s = response.ttft_s
            raw_response_snapshot = self._snapshot_failed_model_response(response)

            try:
                response = normalize_profile_response_for_observation(
                    self.agent_profile,
                    response,
                    current_observation,
                    model_name=self.model,
                    image_scale_ratio=self._image_scale_ratio,
                )
            except ValueError as exc:
                if retries_left > 0:
                    continue
                return self._skipped_step_result(
                    reason=f"unparsable response: {exc}",
                    step_usage=step_usage,
                    step_start=step_start,
                    step_chat_latency_s=step_chat_latency_s,
                    step_ttft_s=step_ttft_s,
                    current_observation=current_observation,
                    model_snapshot=raw_response_snapshot,
                )

            assistant_msg = self._build_assistant_message(
                response,
                include_tool_calls=native_tools_enabled,
            )
            messages.append(assistant_msg)
            assistant_snapshot = self._snapshot_failed_model_response(
                response,
                assistant_message=assistant_msg,
            )
            if not response.tool_calls:
                return self._skipped_step_result(
                    reason="no action payload",
                    step_usage=step_usage,
                    step_start=step_start,
                    step_chat_latency_s=step_chat_latency_s,
                    step_ttft_s=step_ttft_s,
                    current_observation=current_observation,
                    model_snapshot=assistant_snapshot,
                )

            tool_call = response.tool_calls[0]
            action_intent, state_summary = self._tool_call_semantics(tool_call)
            try:
                action = parse_action(tool_call.arguments)
                action = self._normalize_relative_coordinates(action)
            except ActionError as exc:
                return self._skipped_step_result(
                    reason=f"unparsable action: {exc}",
                    step_usage=step_usage,
                    step_start=step_start,
                    step_chat_latency_s=step_chat_latency_s,
                    step_ttft_s=step_ttft_s,
                    current_observation=current_observation,
                    model_snapshot=assistant_snapshot,
                )

            replan, judge_snapshot = await self._repeat_replan_decision(
                previous_action=previous_action,
                proposed_action=action,
                current_observation=current_observation,
                previous_observation=previous_observation,
                task=task,
                messages=messages,
                original_messages=original_messages,
                step_usage=step_usage,
                escalated=escalated,
                step_index=step_index,
            )
            if judge_snapshot is not None:
                repeat_judge_snapshot = judge_snapshot
            if replan:
                actor = self._planner_llm or self.llm
                escalated = True
                retries_left = self._MAX_TOOL_RETRIES + 1
                continue

            await self._report_step_progress(
                step_index=step_index,
                total_steps=total_steps,
                action=action,
                response=response,
                escalated=escalated,
            )
            action_text = self._normalize_action_text(
                response.content,
                action,
                tool_summary=action_intent or state_summary,
            )
            action_summary = action_intent or self._action_summary(action_text)
            assistant_message = self._build_assistant_message(
                response,
                content_override=action_text,
                include_tool_calls=native_tools_enabled,
            )
            model_snapshot = {
                "raw_content": response.content,
                "tool_calls": [
                    {"id": call.id, "name": call.name, "arguments": call.arguments}
                    for call in (response.tool_calls or [])
                ],
                "parsed_action": self._serialize_action(action),
                "action_text": action_text,
                "actor": "planner" if escalated else "gui",
            }
            if repeat_judge_snapshot is not None:
                model_snapshot["repeat_judge"] = repeat_judge_snapshot

            if action.action_type == "done":
                done_status = self._resolve_done_status(action)
                if action.status != done_status:
                    action = replace(action, status=done_status)
                tool_result = f"Task terminated with status: {done_status}"
                return self._finalize_step_result(
                    StepResult(
                        action=action,
                        tool_call_id=tool_call.id,
                        tool_result=tool_result,
                        assistant_message=assistant_message,
                        action_summary=action_summary,
                        action_intent=action_summary,
                        state_summary=state_summary,
                        model_snapshot=model_snapshot,
                        done=True,
                    ),
                    step_usage=step_usage,
                    step_start=step_start,
                    step_chat_latency_s=step_chat_latency_s,
                    step_ttft_s=step_ttft_s,
                )

            if action.action_type == "request_intervention":
                return self._finalize_step_result(
                    StepResult(
                        action=action,
                        tool_call_id=tool_call.id,
                        tool_result="intervention_requested",
                        assistant_message=assistant_message,
                        action_summary=action_summary,
                        action_intent=action_summary,
                        state_summary=state_summary,
                        model_snapshot=model_snapshot,
                        intervention_requested=True,
                    ),
                    step_usage=step_usage,
                    step_start=step_start,
                    step_chat_latency_s=step_chat_latency_s,
                    step_ttft_s=step_ttft_s,
                )

            try:
                result_text = await self.backend.execute(action, timeout=self.step_timeout)
            except Exception as exc:
                result_text = f"Action failed: {exc}"

            settle_seconds = self._post_action_settle_seconds(action)
            if settle_seconds > 0:
                await asyncio.sleep(settle_seconds)

            run_dir = Path(current_observation.screenshot_path or ".").parent.parent
            next_screenshot = self._step_screenshot_path(run_dir, step_index, action.action_type)
            next_observation = await self._observe_after_action(
                next_screenshot,
                previous_observation=current_observation,
                action=action,
                timeout=self.step_timeout,
            )
            return self._finalize_step_result(
                StepResult(
                    action=action,
                    tool_call_id=tool_call.id,
                    tool_result=result_text,
                    assistant_message=assistant_message,
                    action_summary=action_summary,
                    action_intent=action_summary,
                    state_summary=state_summary,
                    next_observation=next_observation,
                    model_snapshot=model_snapshot,
                ),
                step_usage=step_usage,
                step_start=step_start,
                step_chat_latency_s=step_chat_latency_s,
                step_ttft_s=step_ttft_s,
            )

        raise RuntimeError("GUI model did not return a valid action after retries.")

    def _coordinate_mode(self) -> str:
        return coordinate_mode_for_profile(self.agent_profile, self.model)

    def _normalize_relative_coordinates(self, action: Action) -> Action:
        if action.relative or action.action_type not in self._COORDINATE_ACTIONS:
            return action
        if self._coordinate_mode() != "relative_999":
            return action
        coords = [
            value for value in (action.x, action.y, action.x2, action.y2) if value is not None
        ]
        if coords and all(0 <= value <= 999 for value in coords):
            return replace(action, relative=True)
        return action

    def _post_action_settle_seconds(self, action: Action) -> float:
        if action.action_type in self._NO_SETTLE_ACTIONS:
            return 0.0
        if action.action_type == "open_app":
            return self._OPEN_APP_SETTLE_SECONDS
        return self._POST_ACTION_SETTLE_SECONDS

    async def _observe_after_action(
        self,
        screenshot_path: Path,
        *,
        previous_observation: Observation | None = None,
        action: Action | None = None,
        timeout: float,
    ) -> Observation | None:
        max_attempts = self._POST_ACTION_STABILITY_MAX_ATTEMPTS
        if action is not None and action.action_type in self._NO_SETTLE_ACTIONS:
            max_attempts = 1
        if action is None or self._post_action_settle_seconds(action) <= 0:
            max_attempts = min(max_attempts, 1)
        window_seconds = min(timeout, self._POST_ACTION_STABILITY_WINDOW_SECONDS)
        poll_interval = self._POST_ACTION_STABILITY_POLL_SECONDS
        observe_timeout = min(timeout, self._POST_ACTION_OBSERVE_TIMEOUT_SECONDS)
        attempts_within_window = max(1, int(window_seconds / max(poll_interval, 1e-6)))
        max_attempts = max(1, min(max_attempts, attempts_within_window))
        previous_fingerprint = (
            self._build_screen_fingerprint(previous_observation)
            if previous_observation is not None
            else None
        )
        stable_count = 0
        last_observation: Observation | None = None
        deadline = time.monotonic() + window_seconds
        for _attempt in range(max_attempts):
            try:
                observation = await self.backend.observe(screenshot_path, timeout=observe_timeout)
                last_observation = observation
                current_fingerprint = self._build_screen_fingerprint(observation)
                if previous_fingerprint is not None and current_fingerprint is not None:
                    if self._is_same_screen(previous_fingerprint, current_fingerprint):
                        stable_count += 1
                    else:
                        stable_count = 1
                        previous_fingerprint = current_fingerprint
                else:
                    stable_count = min(stable_count + 1, self._POST_ACTION_STABILITY_FRAMES_REQUIRED)
                if stable_count >= self._POST_ACTION_STABILITY_FRAMES_REQUIRED:
                    return observation
            except Exception:
                pass
            if max_attempts > 1 and time.monotonic() < deadline:
                await asyncio.sleep(poll_interval)
        return last_observation

    def _build_screen_fingerprint(self, observation: Observation | None) -> _ScreenFingerprint | None:
        if observation is None or not observation.screenshot_path:
            return None
        screenshot_path = Path(observation.screenshot_path)
        if not screenshot_path.exists():
            return None
        app_name = observation.foreground_app or None
        try:
            data = screenshot_path.read_bytes()
        except OSError:
            return None
        try:
            from PIL import Image

            with Image.open(screenshot_path) as img:
                resampling = getattr(Image, "Resampling", Image)
                grayscale = img.convert("L").resize(
                    (self._STAGNATION_SSIM_SIZE, self._STAGNATION_SSIM_SIZE),
                    resampling.BILINEAR,
                )
                pixels = list(grayscale.tobytes())
                if pixels:
                    return _ScreenFingerprint(
                        app=app_name,
                        method="ssim",
                        digest=base64.b64encode(bytes(pixels)).decode("ascii"),
                    )
        except Exception:
            pass
        return _ScreenFingerprint(app=app_name, method="sha256", digest=hashlib.sha256(data).hexdigest())

    @classmethod
    def _is_same_screen(cls, previous: _ScreenFingerprint, current: _ScreenFingerprint) -> bool:
        if previous.app and current.app and previous.app != current.app:
            return False
        if previous.method == "ssim" and current.method == "ssim":
            try:
                return cls._ssim_is_similar(previous.digest, current.digest)
            except Exception:
                return previous.digest == current.digest
        if previous.method != current.method:
            return False
        return previous.digest == current.digest

    @classmethod
    def _is_similar_planned_action(
        cls,
        previous: Action | None,
        current: Action,
        observation: Observation,
    ) -> bool:
        if previous is None:
            return True
        previous_type = str(previous.action_type or "").strip().lower()
        current_type = str(current.action_type or "").strip().lower()
        if previous_type != current_type:
            return False
        previous_points = cls._normalized_action_points(previous, observation)
        current_points = cls._normalized_action_points(current, observation)
        if previous_points or current_points:
            if not cls._points_are_close(previous_points, current_points):
                return False
        if current_type == "scroll":
            return (previous.text or "").strip().lower() == (current.text or "").strip().lower()
        return True

    @classmethod
    def _normalized_action_points(
        cls,
        action: Action,
        observation: Observation,
    ) -> tuple[tuple[float, float], ...]:
        del cls
        width = max(int(observation.screen_width or 0), 1)
        height = max(int(observation.screen_height or 0), 1)
        raw: list[tuple[float, float]] = []
        if action.points:
            raw.extend(action.points)
        elif action.x is not None and action.y is not None:
            raw.append((action.x, action.y))
            if action.x2 is not None and action.y2 is not None:
                raw.append((action.x2, action.y2))
        return tuple(
            (
                resolve_coordinate(x, width, relative=action.relative) / width,
                resolve_coordinate(y, height, relative=action.relative) / height,
            )
            for x, y in raw
        )

    @classmethod
    def _points_are_close(
        cls,
        previous_points: tuple[tuple[float, float], ...],
        current_points: tuple[tuple[float, float], ...],
        *,
        max_distance: float | None = None,
    ) -> bool:
        if len(previous_points) != len(current_points) or not previous_points:
            return False
        threshold = cls._STAGNATION_CLICK_DISTANCE if max_distance is None else max_distance
        return all(
            ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5 <= threshold
            for (px, py), (cx, cy) in zip(previous_points, current_points)
        )

    @classmethod
    def _ssim_is_similar(cls, previous_digest: str, current_digest: str) -> bool:
        previous_pixels = base64.b64decode(previous_digest)
        current_pixels = base64.b64decode(current_digest)
        if len(previous_pixels) != len(current_pixels) or len(previous_pixels) == 0:
            return False
        return cls._ssim_score(previous_pixels, current_pixels) >= cls._STAGNATION_SSIM_THRESHOLD

    @classmethod
    def _ssim_score(cls, previous_pixels: bytes, current_pixels: bytes) -> float:
        del cls
        n = len(previous_pixels)
        previous_values = list(previous_pixels)
        current_values = list(current_pixels)
        previous_mean = sum(previous_values) / n
        current_mean = sum(current_values) / n
        previous_variance = sum((value - previous_mean) ** 2 for value in previous_values) / n
        current_variance = sum((value - current_mean) ** 2 for value in current_values) / n
        covariance = (
            sum(
                (previous_value - previous_mean) * (current_value - current_mean)
                for previous_value, current_value in zip(previous_values, current_values)
            )
            / n
        )
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2
        denominator = (previous_mean * previous_mean + current_mean * current_mean + c1) * (
            previous_variance + current_variance + c2
        )
        if denominator == 0:
            return 1.0 if previous_mean == current_mean else 0.0
        numerator = (2 * previous_mean * current_mean + c1) * (2 * covariance + c2)
        return numerator / denominator

    def _build_messages(
        self,
        *,
        task: str,
        current_observation: Observation,
        history: list[HistoryTurn],
    ) -> list[dict[str, Any]]:
        return build_profile_messages(
            self.agent_profile,
            task=task,
            current_observation=current_observation,
            history=history,
            model_name=self.model,
            history_image_window=self.history_image_window,
            image_scale_ratio=self._image_scale_ratio,
        )

    def _build_state_note(
        self,
        *,
        status: str,
        history: list[HistoryTurn],
        current_observation: Observation | None,
        current_action_summary: str | None = None,
        error: str | None = None,
    ) -> str:
        done = "; ".join(turn.action_summary for turn in history if turn.action_summary) or "none"
        current = (
            current_observation.foreground_app
            if current_observation and current_observation.foreground_app
            else "unknown"
        )
        remaining = error or ("none" if status == "completed" else "continue from current screen")
        resume = (
            "none"
            if status == "completed"
            else "Resume from the current screen and finish the remaining steps."
        )
        if current_action_summary:
            done = current_action_summary if not history else f"{done}; {current_action_summary}"
        return build_state_note(
            status=status,
            done=done,
            remaining=remaining,
            current=current,
            resume=resume,
        )

    @staticmethod
    def _build_assistant_message(
        response: LLMResponse,
        *,
        content_override: str | None = None,
        include_tool_calls: bool = True,
    ) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": "assistant"}
        content = content_override if content_override is not None else response.content
        if content:
            msg["content"] = content
        if include_tool_calls and response.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments)
                        if isinstance(call.arguments, dict)
                        else str(call.arguments),
                    },
                }
                for call in response.tool_calls
            ]
        return msg

    async def _emit_difficulty_progress(self, task_difficulty: Any) -> None:
        emit_difficulty = getattr(self.progress_callback, "emit_difficulty", None)
        if callable(emit_difficulty):
            await emit_difficulty(task_difficulty)

    async def _report_step_progress(
        self,
        *,
        step_index: int,
        total_steps: int,
        action: Action,
        response: LLMResponse,
        escalated: bool = False,
    ) -> None:
        if self.progress_callback is None:
            return
        action_text = describe_action(action)
        emit_step = getattr(self.progress_callback, "emit_step", None)
        if callable(emit_step):
            await emit_step(
                step_index=step_index,
                total_steps=total_steps,
                action=action_text,
                model_output=str(response.content or "")[: self._PROGRESS_TEXT_LIMIT],
                switched=escalated,
            )
            return
        await self.progress_callback(f"GUI step {step_index}/{total_steps}: {action_text}")

    def _snapshot_failed_model_response(
        self,
        response: LLMResponse,
        *,
        assistant_message: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        snapshot: dict[str, Any] = {
            "raw_content": response.content,
            "tool_calls": [
                {"id": call.id, "name": call.name, "arguments": call.arguments}
                for call in (response.tool_calls or [])
            ],
        }
        if assistant_message is not None:
            snapshot["assistant_message"] = assistant_message
        return snapshot

    @staticmethod
    def _normalize_action_text(
        content: str,
        action: Action,
        *,
        tool_summary: str | None = None,
    ) -> str:
        summary = GuiAgent._clean_action_summary(tool_summary)
        if summary:
            return f"Action: {summary}"
        text = content.strip() if content else ""
        if text:
            first_line = text.splitlines()[0].strip()
            if first_line.lower().startswith("action:"):
                return first_line
            return f"Action: {first_line}"
        return f"Action: {describe_action(action)}"

    @staticmethod
    def _tool_call_semantics(tool_call: ToolCall) -> tuple[str | None, str | None]:
        arguments = tool_call.arguments or {}
        intent = GuiAgent._clean_action_summary(arguments.get("intent"))
        summary = GuiAgent._clean_action_summary(arguments.get("summary"))
        return intent, summary

    @staticmethod
    def _clean_action_summary(value: Any) -> str | None:
        if value is None:
            return None
        text = " ".join(str(value).split()).strip()
        if not text:
            return None
        if text.casefold().startswith("action:"):
            text = text.split(":", 1)[1].strip()
        return text.strip("`\"'") or None

    @staticmethod
    def _action_summary(action_text: str) -> str:
        if action_text.lower().startswith("action:"):
            return action_text.split(":", 1)[1].strip()
        return action_text.strip()

    @staticmethod
    def _resolve_done_status(action: Action) -> str:
        if action.status in {"success", "failure"}:
            return action.status
        text = (action.text or "").strip().lower()
        if any(hint in text for hint in _DONE_FAILURE_HINTS):
            return "failure"
        return "success"

    def _image_block(self, path: Path) -> dict[str, Any]:
        b64 = base64.b64encode(
            scale_image(path.read_bytes(), scale_ratio=self._image_scale_ratio)
        ).decode()
        return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}

    @staticmethod
    def _serialize_action(action: Action) -> dict[str, Any]:
        payload = dataclasses.asdict(action)
        return {
            key: value
            for key, value in payload.items()
            if value is not None and not (key == "relative" and value is False)
        }

    def _make_run_dir(self, attempt: int) -> Path:
        run_dir = self.artifacts_root / f"attempt_{attempt + 1:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "screenshots").mkdir(exist_ok=True)
        return run_dir

    @staticmethod
    def _step_screenshot_path(run_dir: Path, step_index: int, action_type: str) -> Path:
        kind = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(action_type or "action")).strip("_")
        return run_dir / "screenshots" / f"{step_index:03d}_{kind or 'action'}.png"

    async def _log_attempt_event(self, run_dir: Path, event: str, **payload: Any) -> None:
        del run_dir
        self._trajectory_recorder.record_event(event, **payload)
