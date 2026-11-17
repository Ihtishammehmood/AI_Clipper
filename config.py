"""Central, environment-driven configuration (Pydantic)."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")

    # LLM (Gemini)
    gemini_api_key: str = ""
    gemini_model: str = "models/gemini-3.8-flash"
    gemini_thinking_level: str = "low"
    gemini_max_output_tokens: int = 10000
    llm_retries: int = 2

    # Hardware
    device: Literal["auto", "cpu", "cuda"] = "auto"
    whisper_model: str = ""  # empty => hardware-aware default
    threads: int = 0  # 0 => auto

    # Output
    out_width: int = 1080
    out_height: int = 1920
    output_fps: int = 30
    crf: int = 20
    x264_preset: str = "veryfast"
    audio_sample_rate: int = 48000
    loudness_target_lufs: float = -14.0
    true_peak_db: float = -1.0
    denoise: bool = False
    huggingface_token: str = ""

    # Clip discovery
    window_seconds: float = 600.0
    window_overlap_seconds: float = 90.0
    min_clip_gap_seconds: float = 10.0
    snap_tolerance_seconds: float = 3.0
    pad_start: float = 0.15
    pad_end: float = 0.30
    score_weights: dict[str, float] = {
        "hook": 0.20, "value": 0.20, "story": 0.15, "emotion": 0.10,
        "clarity": 0.15, "retention": 0.10, "payoff": 0.10,
    }

    # Subtitles
    subtitle_y_fraction: float = 0.72  # 65-78% safe region
    fonts_dir: Path = BASE_DIR / "assets" / "fonts"

    # Paths
    temp_dir: Path = BASE_DIR / "temp"
    output_dir: Path = BASE_DIR / "output"


def get_settings() -> Settings:
    return Settings()
