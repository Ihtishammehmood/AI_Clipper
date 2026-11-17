"""End-to-end orchestration. Intermediate artifacts are cached so jobs resume and clips regenerate cheaply."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Callable

from ai.llm_client import GeminiClient
from config import Settings, get_settings
from core.clip_discovery import discover_clips
from core.ingestion import download_youtube, extract_audio, is_youtube_url, probe_video
from core.models import ClipRecord, JobOptions, JobState, Transcript
from core.rendering import RenderError, render_clip
from core.transcription import transcribe
from utils.file_manager import JobPaths
from utils.hardware import require_ffmpeg
from utils.job_store import import_legacy_jobs, list_jobs as list_indexed_jobs, save_job
from utils.logger import get_logger

Progress = Callable[[float, str], None]


def load_state(paths: JobPaths) -> JobState | None:
    if paths.state.exists():
        try:
            return JobState.model_validate_json(paths.state.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def save_state(paths: JobPaths, state: JobState) -> None:
    paths.out.mkdir(parents=True, exist_ok=True)
    paths.state.write_text(state.model_dump_json(indent=1), encoding="utf-8")
    import_legacy_jobs(paths.settings)
    save_job(paths.settings, state)


def list_jobs(st: Settings) -> list[JobState]:
    import_legacy_jobs(st)
    return list_indexed_jobs(st)


def make_llm(st: Settings) -> GeminiClient:
    return GeminiClient(st.gemini_api_key, st.gemini_model, st.gemini_thinking_level,
                        st.gemini_max_output_tokens, st.llm_retries)


def run_job(job_id: str, source: str, options: JobOptions, st: Settings | None = None,
            progress: Progress = lambda f, m: None, llm=None) -> JobState:
    """Run (or resume) a job: ingest -> audio -> transcribe -> discover -> render."""
    st = st or get_settings()
    log = get_logger("pipeline", job_id)
    require_ffmpeg()
    paths = JobPaths(st, job_id)
    paths.ensure()
    state = load_state(paths) or JobState(job_id=job_id, source=source, options=options)
    state.options = options
    try:
        # 1. ingest
        progress(0.02, "Ingesting video")
        src = Path(state.source_path) if state.source_path else None
        if not (src and src.exists()):
            if is_youtube_url(source):
                src = download_youtube(source, paths.temp)
            else:
                local = Path(source)
                src = paths.temp / f"source{local.suffix}"
                if local.resolve() != src.resolve():
                    shutil.copy2(local, src)
            state.source_path = str(src)
        state.video = probe_video(src)
        state.stage = "ingested"; save_state(paths, state)

        # 2. audio
        progress(0.08, "Extracting audio")
        if not state.video.has_audio:
            raise RuntimeError("This video has no audio track, so it cannot be transcribed.")
        if not paths.audio.exists():
            extract_audio(src, paths.audio)

        # 3. transcribe (cached)
        progress(0.12, "Transcribing (first run downloads the Whisper model)")
        tr = transcribe(paths.audio, st, options.language, paths.transcript,
                        progress=lambda f: progress(0.12 + 0.38 * f, "Transcribing"))
        if options.speaker_tracking:
            from core.speaker_tracking import track_speakers
            state.speaker_tracks = track_speakers(paths.audio, st.huggingface_token)
            (paths.out / "speaker_tracks.json").write_text(
                json.dumps(state.speaker_tracks, indent=1), encoding="utf-8")
        state.stage = "transcribed"; save_state(paths, state)

        # 4. discover (skipped when resuming with clips already chosen)
        if not state.clips:
            progress(0.52, "Finding the best clips with AI")
            llm = llm or make_llm(st)
            cands = discover_clips(tr, state.video.duration, llm, options, st,
                                   progress=lambda f, m="": progress(0.52 + 0.13 * f, m))
            state.clips = [ClipRecord(candidate=c, subtitle_style=options.subtitle_style) for c in cands]
            state.stage = "discovered"; save_state(paths, state)

        # 5. render
        for i, rec in enumerate(state.clips):
            if rec.output_path and Path(rec.output_path).exists():
                continue  # resume
            progress(0.66 + 0.32 * i / max(1, len(state.clips)), f"Rendering {rec.candidate.id}")
            _render_record(state, rec, tr, paths, st, options, log)
            save_state(paths, state)
        state.stage = "done"; state.error = None
        progress(1.0, "Done")
    except Exception as e:
        state.error = str(e)
        log.error("Job failed at stage %s: %s", state.stage, e)
        save_state(paths, state)
        raise
    save_state(paths, state)
    return state


def _render_record(state, rec, tr, paths, st, options, log) -> None:
    try:
        out, qa = render_clip(Path(state.source_path), state.video, tr, rec.candidate,
                              rec.subtitle_style, paths, st, options, rec.subtitle_text)
        rec.output_path, rec.qa, rec.error = str(out), qa, None
        rec.status = "pending" if qa.passed else "review"
    except RenderError as e:
        rec.status, rec.error = "failed", str(e)
        log.error("Render failed for %s: %s", rec.candidate.id, e)


def regenerate_clip(job_id: str, clip_id: str, start: float | None = None, end: float | None = None,
                    subtitle_style: str | None = None, st: Settings | None = None,
                    subtitle_text: str | None = None) -> JobState:
    """Re-render one clip from cached transcript/source (no re-transcription, no LLM call)."""
    st = st or get_settings()
    paths = JobPaths(st, job_id)
    state = load_state(paths)
    if not state or not state.video:
        raise RuntimeError("Unknown or incomplete job.")
    rec = next((c for c in state.clips if c.candidate.id == clip_id), None)
    if rec is None:
        raise RuntimeError(f"Clip {clip_id} not found.")
    if start is not None: rec.candidate.start = start
    if end is not None: rec.candidate.end = end
    if subtitle_style: rec.subtitle_style = subtitle_style
    if subtitle_text is not None:
        rec.subtitle_text = subtitle_text
    tr = Transcript.model_validate_json(paths.transcript.read_text(encoding="utf-8"))
    rec.candidate.transcript_excerpt = " ".join(
        w.text for w in tr.all_words() if rec.candidate.start <= w.start and w.end <= rec.candidate.end)
    _render_record(state, rec, tr, paths, st, state.options, get_logger("pipeline", job_id))
    save_state(paths, state)
    return state
