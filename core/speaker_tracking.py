"""Optional pyannote-based speech diarization."""
from __future__ import annotations

from pathlib import Path


def track_speakers(audio: Path, token: str) -> list[dict[str, float | str]]:
    if not token:
        raise RuntimeError("Speaker tracking requires HUGGINGFACE_TOKEN to access the diarization model.")
    try:
        from pyannote.audio import Pipeline
    except ImportError as e:
        raise RuntimeError("Speaker tracking requires pyannote.audio; install the optional dependency.") from e
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)
    if pipeline is None:
        raise RuntimeError("Unable to load the pyannote speaker diarization model.")
    diarization = pipeline(str(audio))
    return [
        {"start": float(turn.start), "end": float(turn.end), "speaker": str(label)}
        for turn, _, label in diarization.itertracks(yield_label=True)
    ]
