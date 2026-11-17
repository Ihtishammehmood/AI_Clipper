"""faster-whisper transcription with VAD, word timestamps and on-disk caching."""
from __future__ import annotations

from pathlib import Path

from config import Settings
from core.models import Segment, Transcript, Word
from utils.hardware import default_whisper_model, resolve_device
from utils.logger import get_logger

log = get_logger("transcription")


class TranscriptionError(RuntimeError):
    pass


def transcribe(audio: Path, settings: Settings, language: str | None = None,
               cache_path: Path | None = None, progress=lambda f: None) -> Transcript:
    """Transcribe ``audio``. Results are cached so regeneration never re-transcribes."""
    if cache_path and cache_path.exists():
        try:
            log.info("Using cached transcript %s", cache_path)
            return Transcript.model_validate_json(cache_path.read_text(encoding="utf-8"))
        except Exception:
            log.warning("Cached transcript invalid; re-transcribing")

    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise TranscriptionError("faster-whisper is not installed.") from e

    device = resolve_device(settings.device)
    size = settings.whisper_model or default_whisper_model(device)
    compute = "float16" if device == "cuda" else "int8"
    log.info("Loading Whisper '%s' on %s (%s)", size, device, compute)
    try:
        model = WhisperModel(size, device=device, compute_type=compute,
                             cpu_threads=settings.threads or 0)
        segments, info = model.transcribe(
            str(audio), language=language, word_timestamps=True, vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )
        out: list[Segment] = []
        for s in segments:  # generator: transcription happens here
            words = [Word(start=w.start, end=w.end, text=w.word.strip(), probability=w.probability)
                     for w in (s.words or []) if w.word.strip()]
            out.append(Segment(start=s.start, end=s.end, text=s.text.strip(), words=words))
            if info.duration:
                progress(min(1.0, s.end / info.duration))
    except Exception as e:
        raise TranscriptionError(f"Transcription failed: {e}") from e

    tr = Transcript(language=info.language, language_probability=info.language_probability,
                    duration=info.duration, segments=out)
    if not tr.all_words():
        raise TranscriptionError("No speech detected in this video.")
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(tr.model_dump_json(indent=1), encoding="utf-8")
    return tr
