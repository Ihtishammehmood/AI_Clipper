import os
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from ai.llm_client import parse_json
from config import Settings
from core.clip_discovery import discover_clips, make_windows, select_best, snap_and_fit, weighted_score
from core.models import ClipScores, JobOptions, JobState, LLMClip, LLMClipList, Segment, Transcript, Word
from core.jump_cuts import compress_words, speech_ranges
from core.reframing import compute_crop, face_crop_expressions, track_faces
from core.subtitle_engine import STYLE_PRESETS, ass_color, build_ass, build_srt, chunk_words, clip_words
from utils.hardware import executable_path

ST = Settings(gemini_api_key="x")


def test_ffmpeg_path_override_accepts_bin_directory(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    ffmpeg = bin_dir / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    ffprobe = bin_dir / ("ffprobe.exe" if os.name == "nt" else "ffprobe")
    ffmpeg.touch()
    ffprobe.touch()
    monkeypatch.setenv("FFMPEG_PATH", str(bin_dir))
    assert executable_path("ffmpeg") == str(ffmpeg)
    assert executable_path("ffprobe") == str(ffprobe)


def test_face_tracking_reports_incompatible_opencv(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace())
    with pytest.raises(RuntimeError, match="OpenCV build is incompatible"):
        track_faces(tmp_path / "source.mp4", 0, 1)


def make_transcript(total=300.0, sent_len=5.0):
    segs, t = [], 0.0
    while t + sent_len <= total:
        words = [Word(start=t + i * 0.5, end=t + i * 0.5 + 0.4, text=f"w{int(t)}_{i}" + ("." if i == 9 else ""))
                 for i in range(10)]
        segs.append(Segment(start=t, end=t + 5, text=" ".join(w.text for w in words), words=words))
        t += sent_len
    return Transcript(language="en", duration=total, segments=segs)


def llm_clip(start, end, score=8, **kw):
    s = ClipScores(hook=score, value=score, story=score, emotion=score, clarity=score, retention=score, payoff=score)
    return LLMClip(start=start, end=end, topic="t", description="d", opening_hook="h", takeaway="k",
                   ending_payoff="p", rationale="r", scores=s, title="T", caption="C", **kw)


def test_weights_sum_to_one():
    assert sum(ST.score_weights.values()) == pytest.approx(1.0)


def test_weighted_score_range():
    assert weighted_score({k: 10 for k in ST.score_weights}, ST.score_weights) == 100
    assert weighted_score({k: 5 for k in ST.score_weights}, ST.score_weights) == 50


def test_parse_json_handles_fences():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    with pytest.raises(ValueError):
        parse_json("nothing here")


def test_windows_cover_video():
    w = make_windows(1500, 600, 90)
    assert w[0][0] == 0 and w[-1][1] == 1500
    assert all(w[i + 1][0] < w[i][1] for i in range(len(w) - 1))


def test_snap_and_fit_respects_limits():
    tr = make_transcript()
    opt = JobOptions(min_duration=15, max_duration=30)
    c = snap_and_fit(llm_clip(10.4, 52.0), tr.sentences(), 300, opt, ST)
    assert c is not None and c.duration <= 30 + 0.5 and c.end <= 300
    assert snap_and_fit(llm_clip(10, 12), tr.sentences(), 300, JobOptions(min_duration=400, max_duration=500), ST) is None
    assert snap_and_fit(llm_clip(290, 330), tr.sentences(), 300, JobOptions(min_duration=5, max_duration=60), ST).end <= 300


def test_select_best_removes_overlap_and_duplicates():
    tr = make_transcript()
    opt = JobOptions(min_duration=15, max_duration=30)
    cs = [snap_and_fit(llm_clip(a, b, sc), tr.sentences(), 300, opt, ST)
          for a, b, sc in [(10, 35, 9), (20, 45, 8), (100, 125, 7), (200, 225, 6)]]
    got = select_best(cs, 3, 10)
    assert [c.id for c in got] == ["clip_001", "clip_002", "clip_003"]
    assert not any(a.end > b.start for a, b in zip(got, got[1:]))
    assert 20 not in [round(c.start) for c in got]


def test_discover_with_fake_llm():
    class Fake:
        def generate_model(self, prompt, system, schema):
            return LLMClipList(clips=[llm_clip(10, 40, 9), llm_clip(100, 130, 7), llm_clip(500, 530)])
    tr = make_transcript(total=300)
    out = discover_clips(tr, 300, Fake(), JobOptions(num_clips=2, min_duration=20, max_duration=40), ST)
    assert len(out) == 2 and all(0 <= c.start < c.end <= 300 for c in out)


def test_compute_crop_916():
    x, y, w, h = compute_crop(1920, 1080, 1080, 1920)
    assert (h, w) == (1080, 608) and x + w <= 1920 and abs(w / h - 9 / 16) < 0.01
    assert compute_crop(1920, 1080, 1080, 1920, cx=0.0)[0] == 0
    assert compute_crop(1920, 1080, 1080, 1920, cx=1.0)[0] + w <= 1920


def test_face_crop_expression_escapes_filter_argument_commas():
    x_expr, y_expr = face_crop_expressions(
        [(0.0, 500.0, 300.0), (1.0, 700.0, 400.0)],
        600, 800, 1920, 1080,
    )
    assert r"\,0\,1" in x_expr
    assert r"\,0\,1" in y_expr
    assert "," not in x_expr.replace(r"\,", "")
    assert "," not in y_expr.replace(r"\,", "")


def test_jump_cuts_remove_long_pauses_and_retime_words():
    words = [
        Word(start=0.2, end=0.5, text="hello"),
        Word(start=0.6, end=0.9, text="there"),
        Word(start=4.0, end=4.3, text="again"),
    ]
    ranges = speech_ranges(words, 5.0)
    assert len(ranges) == 2
    compressed = compress_words(words, ranges)
    assert [w.text for w in compressed] == ["hello", "there", "again"]
    assert compressed[-1].start < 1.5


def test_ass_color_bgr():
    assert ass_color("#FF0000") == "&H000000FF"


def test_subtitles_timing_and_highlight():
    tr = make_transcript(30)
    words = clip_words(tr, 5.0, 15.0)
    assert words and words[0].start >= 0 and words[-1].end <= 10.1
    for name, style in STYLE_PRESETS.items():
        ass = build_ass(words, style)
        events = [l for l in ass.splitlines() if l.startswith("Dialogue")]
        assert len(events) == len(words), name
        assert "\\1c" in events[0]
    srt = build_srt(words, STYLE_PRESETS["modern_creator"])
    assert "-->" in srt
    assert all(len(c) <= 6 for c in chunk_words(words, STYLE_PRESETS["modern_creator"]))


def test_hook_overlay_is_separate_and_timed():
    from core.subtitle_engine import build_hook_ass

    ass = build_hook_ass("A strong hook")
    assert "Style: Hook" in ass
    assert "A STRONG HOOK" in ass
    assert "0:00:03.00" in ass


def test_sqlite_dashboard_stats_and_job_index(tmp_path):
    from utils.job_store import dashboard_stats, list_jobs, save_job

    settings = Settings(gemini_api_key="x", output_dir=tmp_path / "output")
    state = JobState(job_id="project_test", source="input.mp4", stage="done")
    save_job(settings, state)
    assert [job.job_id for job in list_jobs(settings)] == ["project_test"]
    assert dashboard_stats(settings) == {"projects": 1, "clips": 0, "approved": 0}


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg missing")
def test_render_end_to_end(tmp_path):
    from core.ingestion import probe_video
    from core.models import ClipCandidate
    from core.rendering import render_clip
    from utils.file_manager import JobPaths

    src = tmp_path / "src.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=1280x720:r=30:d=14",
                    "-f", "lavfi", "-i", "sine=f=440:d=14", "-c:v", "libx264", "-c:a", "aac", "-shortest", str(src)],
                   check=True)
    st = Settings(gemini_api_key="x", temp_dir=tmp_path / "t", output_dir=tmp_path / "o", fonts_dir=tmp_path / "f")
    info = probe_video(src)
    assert info.has_audio and info.width == 1280
    tr = make_transcript(14)
    cand = ClipCandidate(**llm_clip(1.0, 11.0).model_dump(), id="clip_001")
    out, qa = render_clip(src, info, tr, cand, "kinetic", JobPaths(st, "job"), st, JobOptions())
    assert out.exists() and qa.passed, qa.issues
