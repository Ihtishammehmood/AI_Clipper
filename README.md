# AI Video Clipper

Turns long videos into vertical 1080×1920 clips with word-highlighted subtitles.
Pipeline: ingest (file / YouTube) → audio → faster-whisper (word timestamps, VAD) → Gemini clip discovery
→ local explainable scoring + sentence-boundary snapping + de-duplication → FFmpeg render → QA.

## Install
**Linux:** `sudo apt install ffmpeg python3-venv` · `python3 -m venv .venv && source .venv/bin/activate`
**Windows:** `winget install --id Gyan.FFmpeg --exact` · restart VS Code/terminal after installation ·
`py -3.11 -m venv .venv && .venv\Scripts\activate`

Then: `pip install -r requirements.txt` · copy `.env.example` to `.env` and set `GEMINI_API_KEY`.
FFmpeg must include libass (standard builds do) for subtitle burn-in.
If FFmpeg is installed but not on PATH, set `FFMPEG_PATH` to its `bin` folder (for example `C:\ffmpeg\bin`)
or directly to `ffmpeg.exe`; `ffprobe.exe` must be beside it.

## Run
- UI: `streamlit run app.py`
- CLI: `python cli.py my_video.mp4 --clips 3 --min 30 --max 60 --style kinetic`
- Optional CLI edits: `--face-tracking --jump-cuts --zoom --music track.mp3 --split-screen inset.mp4`
- Resume an interrupted job: `python cli.py <same source> --job-id project_xxxxxxxx` (cached audio/transcript/clips are reused)
- Tests: `python -m pytest -q`

## Model downloads
Whisper weights download on first use (~75 MB `base`, ~460 MB `small`) into the Hugging Face cache. Needs internet once.
Default is `base` on CPU, `small` on CUDA; override with `WHISPER_MODEL`.

## Configuration
All settings are in `config.py` and overridable via `.env` (see `.env.example`): device (auto/cpu/cuda), Whisper size,
resolution, FPS, CRF, loudness target, score weights, clip spacing, subtitle position (`SUBTITLE_Y_FRACTION`).
GPU is optional; with CUDA, Whisper uses float16 and FFmpeg uses `h264_nvenc` if present.

## LLM
`ai/llm_client.py` wraps Gemini via the Interactions API exactly as in your sample (falls back to `generate_content`
on older SDKs). Output is validated with Pydantic and retried with a correction prompt. Cost = one request per ~10 min of
transcript. Swapping to a local LLM only needs another class with `generate_model(prompt, system, schema)`.

## Output
`output/<project>/clip_001.mp4`, `subtitles/*.srt|ass`, `thumbnails/`, `metadata/`, `quality_reports/`, `transcript.json`, `state.json`.
Source video + audio stay in `temp/<project>/` so single clips can be regenerated without re-transcribing.

## Editing features
The New Project page supports multiple uploaded videos per batch, face-following vertical crops, speech-pause jump cuts,
subtle zoom, split-screen video, background music, and a separate opening-hook overlay. The Projects page provides an
editable subtitle text field; regenerating a clip applies the edited text and selected subtitle style.

Speaker-turn tracking uses pyannote's `speaker-diarization-3.1` model. Install `pyannote.audio`, accept that model's
Hugging Face terms, and set `HUGGINGFACE_TOKEN` before enabling it. Track output is saved in `speaker_tracks.json`.

The dashboard's project/clip/approval counts come from `output/dashboard.sqlite3`. Existing `state.json` projects are
imported into SQLite the first time the dashboard is opened; job folders continue to hold media and editable project
state. Face tracking uses OpenCV's bundled Haar cascade, and all render options remain opt-in except the hook overlay.
