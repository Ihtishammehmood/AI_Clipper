"""Command-line runner: python cli.py <file-or-youtube-url> --clips 3"""
import argparse

from core.models import JobOptions
from core.subtitle_engine import STYLE_PRESETS
from pipeline import run_job
from utils.file_manager import new_job_id

p = argparse.ArgumentParser()
p.add_argument("source")
p.add_argument("--clips", type=int, default=3)
p.add_argument("--min", type=float, default=30)
p.add_argument("--max", type=float, default=60)
p.add_argument("--style", default="modern_creator", choices=list(STYLE_PRESETS))
p.add_argument("--language", default=None)
p.add_argument("--job-id", default=None, help="reuse an id to resume a job")
p.add_argument("--face-tracking", action="store_true")
p.add_argument("--speaker-tracking", action="store_true")
p.add_argument("--jump-cuts", action="store_true")
p.add_argument("--zoom", action="store_true")
p.add_argument("--split-screen", default=None, help="path to an inset video")
p.add_argument("--music", default=None, help="path to background music")
p.add_argument("--no-hook-overlay", action="store_true")
a = p.parse_args()
opts = JobOptions(num_clips=a.clips, min_duration=a.min, max_duration=a.max,
                  subtitle_style=a.style, language=a.language, face_tracking=a.face_tracking,
                  speaker_tracking=a.speaker_tracking, jump_cuts=a.jump_cuts, zoom=a.zoom,
                  split_screen_path=a.split_screen, background_music_path=a.music,
                  hook_overlay=not a.no_hook_overlay)
state = run_job(a.job_id or new_job_id(), a.source, opts, progress=lambda f, m: print(f"[{f:5.0%}] {m}"))
for c in state.clips:
    print(c.candidate.id, c.status, c.output_path, c.candidate.title)
