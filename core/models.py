"""Shared Pydantic data models."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


class VideoInfo(BaseModel):
    path: str
    duration: float
    width: int
    height: int
    fps: float
    has_audio: bool
    video_codec: str = ""


class Word(BaseModel):
    start: float
    end: float
    text: str
    probability: float = 1.0


class Segment(BaseModel):
    start: float
    end: float
    text: str
    words: list[Word] = Field(default_factory=list)


class Sentence(BaseModel):
    start: float
    end: float
    text: str


class Transcript(BaseModel):
    language: str = "unknown"
    language_probability: float = 0.0
    duration: float = 0.0
    segments: list[Segment] = Field(default_factory=list)

    def all_words(self) -> list[Word]:
        return [w for s in self.segments for w in s.words]

    def sentences(self, max_gap: float = 1.0, max_words: int = 45) -> list[Sentence]:
        """Split into sentences using punctuation, long pauses and a length cap."""
        out: list[Sentence] = []
        for seg in self.segments:
            if not seg.words:
                out.append(Sentence(start=seg.start, end=seg.end, text=seg.text.strip()))
                continue
            cur: list[Word] = []
            for w in seg.words:
                if cur and (w.start - cur[-1].end > max_gap or len(cur) >= max_words):
                    out.append(_mk_sentence(cur)); cur = []
                cur.append(w)
                if w.text.rstrip().endswith((".", "?", "!", "…")):
                    out.append(_mk_sentence(cur)); cur = []
            if cur:
                out.append(_mk_sentence(cur))
        return out


def _mk_sentence(words: list[Word]) -> Sentence:
    return Sentence(start=words[0].start, end=words[-1].end, text=" ".join(w.text for w in words))


class ClipScores(BaseModel):
    hook: float = Field(ge=0, le=10)
    value: float = Field(ge=0, le=10)
    story: float = Field(ge=0, le=10)
    emotion: float = Field(ge=0, le=10)
    clarity: float = Field(ge=0, le=10)
    retention: float = Field(ge=0, le=10)
    payoff: float = Field(ge=0, le=10)


class LLMClip(BaseModel):
    """Exactly what the LLM must return per clip."""
    start: float
    end: float
    topic: str
    description: str
    opening_hook: str
    takeaway: str
    ending_payoff: str
    rationale: str
    scores: ClipScores
    title: str
    caption: str
    hashtags: list[str] = Field(default_factory=list)


class LLMClipList(BaseModel):
    clips: list[LLMClip]


class ClipCandidate(LLMClip):
    id: str = ""
    quality_score: float = 0.0  # 0-100, computed locally from factor scores
    transcript_excerpt: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start


class JobOptions(BaseModel):
    num_clips: int = 3
    min_duration: float = 30.0
    max_duration: float = 60.0
    subtitle_style: str = "modern_creator"
    language: str | None = None
    use_ai_metadata: bool = True
    normalize_audio: bool = True
    face_tracking: bool = False
    speaker_tracking: bool = False
    jump_cuts: bool = False
    zoom: bool = False
    split_screen_path: str | None = None
    background_music_path: str | None = None
    hook_overlay: bool = True


class QAReport(BaseModel):
    passed: bool
    checks: dict[str, bool] = Field(default_factory=dict)
    issues: list[str] = Field(default_factory=list)


class ClipRecord(BaseModel):
    candidate: ClipCandidate
    status: Literal["pending", "approved", "rejected", "failed", "review"] = "pending"
    subtitle_style: str = "modern_creator"
    subtitle_text: str | None = None
    output_path: str | None = None
    qa: QAReport | None = None
    error: str | None = None


class JobState(BaseModel):
    job_id: str
    source: str
    source_path: str | None = None
    options: JobOptions = Field(default_factory=JobOptions)
    video: VideoInfo | None = None
    clips: list[ClipRecord] = Field(default_factory=list)
    stage: str = "created"
    error: str | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    speaker_tracks: list[dict[str, float | str]] = Field(default_factory=list)
