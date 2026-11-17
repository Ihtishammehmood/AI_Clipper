"""Per-job directory layout."""
from __future__ import annotations

import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

from config import Settings


def new_job_id() -> str:
    return "project_" + uuid.uuid4().hex[:8]


@dataclass
class JobPaths:
    settings: Settings
    job_id: str

    @property
    def temp(self) -> Path: return self.settings.temp_dir / self.job_id
    @property
    def out(self) -> Path: return self.settings.output_dir / self.job_id
    @property
    def audio(self) -> Path: return self.temp / "audio.wav"
    @property
    def transcript(self) -> Path: return self.out / "transcript.json"
    @property
    def state(self) -> Path: return self.out / "state.json"
    @property
    def subtitles(self) -> Path: return self.out / "subtitles"
    @property
    def thumbnails(self) -> Path: return self.out / "thumbnails"
    @property
    def metadata(self) -> Path: return self.out / "metadata"
    @property
    def reports(self) -> Path: return self.out / "quality_reports"
    @property
    def music(self) -> Path: return self.temp / "background_music"
    @property
    def split_screen(self) -> Path: return self.temp / "split_screen"

    def ensure(self) -> None:
        for p in (self.temp, self.out, self.subtitles, self.thumbnails, self.metadata, self.reports):
            p.mkdir(parents=True, exist_ok=True)

    def delete_temp(self) -> None:
        shutil.rmtree(self.temp, ignore_errors=True)
