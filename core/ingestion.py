"""Video ingestion: validation, probing, YouTube download, audio extraction."""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from core.models import VideoInfo
from utils.hardware import executable_path, require_ffmpeg
from utils.logger import get_logger

log = get_logger("ingestion")

YOUTUBE_RE = re.compile(r"^(https?://)?(www\.|m\.|music\.)?(youtube\.com|youtu\.be)/\S+", re.I)


class IngestionError(RuntimeError):
    pass


def is_youtube_url(s: str) -> bool:
    return bool(YOUTUBE_RE.match(s.strip()))


def _run(cmd: list[str], timeout: int | None = None) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise IngestionError(f"Command timed out: {cmd[0]}") from e
    if r.returncode != 0:
        raise IngestionError(f"{cmd[0]} failed: {r.stderr.strip()[-800:]}")
    return r.stdout


def probe_video(path: str | Path) -> VideoInfo:
    """Validate a media file and extract its properties with ffprobe."""
    require_ffmpeg()
    path = Path(path)
    if not path.is_file():
        raise IngestionError(f"File not found: {path}")
    data = json.loads(_run([executable_path("ffprobe"), "-v", "error", "-print_format", "json",
                            "-show_format", "-show_streams", str(path)], timeout=60))
    streams = data.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    if v is None:
        raise IngestionError("No video stream found (corrupted or unsupported file).")
    try:
        num, den = v.get("avg_frame_rate", "0/1").split("/")
        fps = float(num) / float(den) if float(den) else 0.0
    except ValueError:
        fps = 0.0
    if fps <= 0:
        fps = 30.0
    duration = float(data.get("format", {}).get("duration") or v.get("duration") or 0)
    if duration <= 0:
        raise IngestionError("Could not determine video duration.")
    return VideoInfo(
        path=str(path), duration=duration, width=int(v["width"]), height=int(v["height"]),
        fps=fps, has_audio=any(s.get("codec_type") == "audio" for s in streams),
        video_codec=v.get("codec_name", ""),
    )


def download_youtube(url: str, dest_dir: Path) -> Path:
    """Download a public YouTube video with yt-dlp (no DRM/private bypass)."""
    if not is_youtube_url(url):
        raise IngestionError("Not a valid YouTube URL.")
    try:
        import yt_dlp
    except ImportError as e:
        raise IngestionError("yt-dlp is not installed (pip install yt-dlp).") from e
    dest_dir.mkdir(parents=True, exist_ok=True)
    opts = {
        "format": "bv*[height<=1080]+ba/b[height<=1080]/b",
        "merge_output_format": "mp4",
        "outtmpl": str(dest_dir / "source.%(ext)s"),
        "noplaylist": True, "quiet": True, "no_warnings": True,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
    except Exception as e:  # yt_dlp raises several error types
        raise IngestionError(f"YouTube download failed: {e}") from e
    files = sorted(dest_dir.glob("source.*"))
    if not files:
        raise IngestionError("Download finished but no file was produced.")
    return files[0]


def extract_audio(video: Path, wav: Path) -> Path:
    """Extract 16 kHz mono WAV (the format Whisper wants)."""
    wav.parent.mkdir(parents=True, exist_ok=True)
    _run([executable_path("ffmpeg"), "-y", "-v", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
          "-c:a", "pcm_s16le", str(wav)])
    return wav
