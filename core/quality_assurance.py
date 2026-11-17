"""Technical QA of rendered clips (ffprobe based)."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from core.models import QAReport, VideoInfo
from utils.hardware import executable_path


def check_output(path: Path, expected_duration: float, src: VideoInfo,
                 width: int = 1080, height: int = 1920) -> QAReport:
    checks: dict[str, bool] = {}
    issues: list[str] = []
    try:
        r = subprocess.run([executable_path("ffprobe"), "-v", "error", "-print_format", "json", "-show_format",
                            "-show_streams", str(path)], capture_output=True, text=True, timeout=60)
        data = json.loads(r.stdout or "{}")
    except Exception as e:
        return QAReport(passed=False, checks={"readable": False}, issues=[f"ffprobe failed: {e}"])
    streams = data.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    checks["readable"] = v is not None
    checks["resolution"] = bool(v) and (v["width"], v["height"]) == (width, height)
    checks["codec_h264"] = bool(v) and v.get("codec_name") == "h264"
    checks["audio_present"] = (a is not None) or not src.has_audio
    dur = float(data.get("format", {}).get("duration") or 0)
    checks["duration"] = abs(dur - expected_duration) <= 1.5
    checks["file_size"] = path.exists() and path.stat().st_size > 10_000
    msgs = {"readable": "no video stream", "resolution": f"resolution is not {width}x{height}",
            "codec_h264": "video codec is not H.264", "audio_present": "audio track missing",
            "duration": f"duration {dur:.1f}s differs from expected {expected_duration:.1f}s",
            "file_size": "output file suspiciously small"}
    issues = [msgs[k] for k, ok in checks.items() if not ok]
    return QAReport(passed=not issues, checks=checks, issues=issues)
