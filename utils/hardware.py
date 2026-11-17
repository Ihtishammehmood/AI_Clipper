"""Hardware / dependency detection."""
from __future__ import annotations

import os
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path


class MissingDependencyError(RuntimeError):
    pass


def _configured_bin_dir() -> Path | None:
    configured = os.environ.get("FFMPEG_PATH", "").strip().strip('"')
    if not configured:
        return None
    path = Path(configured).expanduser()
    if path.is_dir():
        return path
    if path.is_file():
        return path.parent
    return None


def executable_path(tool: str) -> str:
    """Resolve FFmpeg tools from PATH or an explicit FFMPEG_PATH override."""
    if tool not in {"ffmpeg", "ffprobe"}:
        raise ValueError(f"Unsupported media executable: {tool}")
    extension = ".exe" if os.name == "nt" else ""
    executable = tool + extension

    configured = os.environ.get("FFMPEG_PATH", "").strip().strip('"')
    if configured:
        configured_path = Path(configured).expanduser()
        if configured_path.is_file() and configured_path.name.lower().startswith(tool):
            return str(configured_path)
        candidate = configured_path / executable
        if candidate.is_file():
            return str(candidate)
        bin_dir = _configured_bin_dir()
        if bin_dir:
            candidate = bin_dir / executable
            if candidate.is_file():
                return str(candidate)

    found = shutil.which(tool)
    if found:
        return found
    raise MissingDependencyError(
        "FFmpeg was not found. On Windows, run `winget install --id Gyan.FFmpeg --exact`, "
        "then fully restart VS Code/your terminal and launch the app again. "
        "If it is already installed, set FFMPEG_PATH to its bin folder "
        "(for example `C:\\ffmpeg\\bin`) or to `ffmpeg.exe`."
    )


def require_ffmpeg() -> None:
    executable_path("ffmpeg")
    try:
        executable_path("ffprobe")
    except MissingDependencyError as exc:
        raise MissingDependencyError(
            "ffprobe was not found beside FFmpeg or on PATH. Install a full FFmpeg build, "
            "then restart the app. FFMPEG_PATH should point to the folder containing both ffmpeg and ffprobe."
        ) from exc


@lru_cache(maxsize=1)
def cuda_available() -> bool:
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


@lru_cache(maxsize=1)
def ffmpeg_encoders() -> str:
    try:
        return subprocess.run([executable_path("ffmpeg"), "-hide_banner", "-encoders"],
                              capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return ""


def resolve_device(pref: str) -> str:
    if pref == "cpu":
        return "cpu"
    if pref == "cuda":
        return "cuda" if cuda_available() else "cpu"
    return "cuda" if cuda_available() else "cpu"


def default_whisper_model(device: str) -> str:
    return "small" if device == "cuda" else "base"


def video_encoder(device: str) -> str:
    return "h264_nvenc" if device == "cuda" and "h264_nvenc" in ffmpeg_encoders() else "libx264"
