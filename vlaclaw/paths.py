"""Default filesystem locations owned by VLAClaw."""

from pathlib import Path

DEFAULT_VLACLAW_HOME = Path.home() / ".vlaclaw"
DEFAULT_GUI_RUNS_DIR = DEFAULT_VLACLAW_HOME / "gui_runs"


def resolve_vlaclaw_data_dir(path: Path | str) -> Path:
    resolved = Path(path).expanduser()
    if resolved.is_absolute():
        return resolved
    return DEFAULT_VLACLAW_HOME / resolved
