"""LLM-driven clip discovery with explainable local scoring, boundary snapping and dedupe."""
from __future__ import annotations

import math
import re

from pydantic import ValidationError

from ai.llm_client import LLM, LLMError
from ai.prompts import CLIP_PROMPT, CLIP_SYSTEM
from config import Settings
from core.models import (ClipCandidate, JobOptions, LLMClip, LLMClipList, Sentence, Transcript)
from utils.logger import get_logger

log = get_logger("discovery")


def weighted_score(scores: dict[str, float], weights: dict[str, float]) -> float:
    """Return 0-100 quality score from 0-10 factor scores using configurable weights."""
    total_w = sum(weights.values()) or 1.0
    return round(sum(weights[k] * scores.get(k, 0.0) for k in weights) / total_w * 10, 2)


def make_windows(duration: float, size: float, overlap: float) -> list[tuple[float, float]]:
    out, start = [], 0.0
    while True:
        end = min(start + size, duration)
        out.append((start, end))
        if end >= duration:
            return out
        start = end - overlap


def _fmt_sentences(sents: list[Sentence]) -> str:
    return "\n".join(f"[{s.start:.1f}-{s.end:.1f}] {s.text}" for s in sents)


def snap_and_fit(c: LLMClip, sents: list[Sentence], video_duration: float,
                 opt: JobOptions, st: Settings) -> ClipCandidate | None:
    """Snap to sentence boundaries, enforce duration limits, clamp to the video."""
    if not sents or c.end <= c.start:
        return None
    s0 = min(sents, key=lambda s: abs(s.start - c.start))
    start = s0.start if abs(s0.start - c.start) <= st.snap_tolerance_seconds else c.start
    e0 = min(sents, key=lambda s: abs(s.end - c.end))
    end = e0.end if abs(e0.end - c.end) <= st.snap_tolerance_seconds else c.end
    ends = [s.end for s in sents if start + opt.min_duration <= s.end <= start + opt.max_duration]
    if end - start > opt.max_duration:
        end = ends[-1] if ends else start + opt.max_duration
    elif end - start < opt.min_duration:
        if not ends:
            return None
        end = ends[0]
    start, end = max(0.0, start - st.pad_start), min(video_duration, end + st.pad_end)
    if end - start < min(opt.min_duration, 15.0) or start >= video_duration:
        return None
    inside = [s.text for s in sents if s.start >= start - 0.5 and s.end <= end + 0.5]
    data = c.model_dump()
    data.update(start=round(start, 2), end=round(end, 2))
    return ClipCandidate(**data, quality_score=weighted_score(c.scores.model_dump(), st.score_weights),
                         transcript_excerpt=" ".join(inside))


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", text.lower()))


def text_similarity(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    return len(ta & tb) / len(ta | tb) if ta and tb else 0.0


def select_best(cands: list[ClipCandidate], n: int, min_gap: float,
                max_similarity: float = 0.6) -> list[ClipCandidate]:
    """Rank by quality; drop overlaps, too-close clips and near-duplicate content."""
    chosen: list[ClipCandidate] = []
    for c in sorted(cands, key=lambda x: x.quality_score, reverse=True):
        ok = True
        for k in chosen:
            gap = max(c.start - k.end, k.start - c.end)  # negative => overlap
            if gap < min_gap or text_similarity(c.transcript_excerpt, k.transcript_excerpt) > max_similarity:
                ok = False
                break
        if ok:
            chosen.append(c)
        if len(chosen) >= n:
            break
    chosen.sort(key=lambda x: x.start)  # chronological ids
    for i, c in enumerate(chosen, 1):
        c.id = f"clip_{i:03d}"
    return chosen


def discover_clips(transcript: Transcript, video_duration: float, llm: LLM,
                   opt: JobOptions, st: Settings, progress=lambda f, m="": None) -> list[ClipCandidate]:
    sents = transcript.sentences()
    if not sents:
        raise LLMError("Transcript has no sentences to analyse.")
    windows = make_windows(video_duration, st.window_seconds, st.window_overlap_seconds)
    per_window = min(8, max(3, opt.num_clips + 2))
    cands: list[ClipCandidate] = []
    for i, (a, b) in enumerate(windows):
        progress(i / len(windows), f"Analysing transcript window {i + 1}/{len(windows)}")
        win = [s for s in sents if s.start >= a and s.end <= b]
        if len(win) < 3:
            continue
        prompt = CLIP_PROMPT.format(n=per_window, min_d=opt.min_duration, max_d=opt.max_duration,
                                    transcript=_fmt_sentences(win))
        try:
            result = llm.generate_model(prompt, CLIP_SYSTEM, LLMClipList)
        except (LLMError, ValidationError) as e:
            log.warning("Window %d failed: %s", i + 1, e)
            continue
        for c in result.clips:
            fitted = snap_and_fit(c, sents, video_duration, opt, st)
            if fitted:
                cands.append(fitted)
    if not cands:
        raise LLMError("No valid clip candidates were found.")
    return select_best(cands, opt.num_clips, st.min_clip_gap_seconds)
