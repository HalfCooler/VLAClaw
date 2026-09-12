"""Compact 5-line GUI state notes used when a run stops."""

from __future__ import annotations

from typing import Any

_STATE_NOTE_LABELS: tuple[str, ...] = ("Status", "Done", "Remaining", "Current", "Resume")


def build_state_note(*, status: str, done: str, remaining: str, current: str, resume: str) -> str:
    values = {
        "Status": status,
        "Done": done,
        "Remaining": remaining,
        "Current": current,
        "Resume": resume,
    }
    lines = []
    for label in _STATE_NOTE_LABELS:
        value = _normalize_note_value(
            values[label],
            default="none" if label != "Status" else "blocked",
        )
        lines.append(f"{label}: {value}")
    return "\n".join(lines)


def is_state_note(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) != len(_STATE_NOTE_LABELS):
        return False
    for line, label in zip(lines, _STATE_NOTE_LABELS):
        prefix, separator, _ = line.partition(":")
        if separator != ":" or prefix.strip() != label:
            return False
    return True


def _normalize_note_value(value: Any, *, default: str) -> str:
    text = " ".join(str(value or "").split()).strip()
    return text or default
