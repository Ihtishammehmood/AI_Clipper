"""Streamlit UI. Business logic lives in pipeline.py / core/."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from config import get_settings
from core.models import JobOptions
from core.subtitle_engine import STYLE_PRESETS
from pipeline import list_jobs, load_state, regenerate_clip, run_job, save_state
from utils.file_manager import JobPaths, new_job_id
from utils.hardware import cuda_available
from utils.job_store import dashboard_stats

S = get_settings()
st.set_page_config(page_title="AI Video Clipper", page_icon="🎬", layout="wide")
st.title("🎬 AI Video Clipper")
page = st.sidebar.radio("Navigate", ["New Project", "Projects"])
st.sidebar.caption(f"GPU (CUDA): {'available' if cuda_available() else 'not detected — CPU mode'}")
if not S.gemini_api_key:
    st.sidebar.error("GEMINI_API_KEY missing — add it to .env")


def new_project() -> None:
    st.header("New project")
    tab_up, tab_url = st.tabs(["Upload file", "YouTube URL"])
    with tab_up:
        uploads = st.file_uploader("Video files (batch processing supported)", type=["mp4", "mov", "mkv", "avi", "webm"],
                                   accept_multiple_files=True)
    with tab_url:
        url = st.text_input("Public YouTube URL")
    c1, c2, c3 = st.columns(3)
    n = c1.number_input("Number of clips", 1, 15, 3)
    lo = c2.number_input("Min duration (s)", 15, 90, 30)
    hi = c3.number_input("Max duration (s)", 15, 90, 60)
    c4, c5 = st.columns(2)
    style = c4.selectbox("Subtitle style", list(STYLE_PRESETS))
    lang = c5.text_input("Language code (blank = auto-detect)", "")
    with st.expander("Advanced editing"):
        face_tracking = st.checkbox("Track faces for vertical framing")
        speaker_tracking = st.checkbox("Track speaker turns (requires pyannote and a Hugging Face token)")
        jump_cuts = st.checkbox("Remove long speech pauses")
        zoom = st.checkbox("Add a subtle zoom")
        hook_overlay = st.checkbox("Show opening hook as a separate overlay", value=True)
        split_upload = st.file_uploader("Optional split-screen video", type=["mp4", "mov", "mkv", "avi", "webm"],
                                        key="split_video")
        music_upload = st.file_uploader("Optional background music", type=["mp3", "wav", "m4a", "aac", "ogg"],
                                        key="background_music")
    if st.button("Start processing", type="primary"):
        if hi < lo:
            st.error("Max duration must be ≥ min duration."); return
        if not S.gemini_api_key:
            st.error("Set GEMINI_API_KEY in .env first."); return
        if not uploads and not url.strip():
            st.error("Upload a video or enter a YouTube URL."); return
        bar, msg = st.progress(0.0), st.empty()
        batch = uploads or [None]
        completed = []
        for index, up in enumerate(batch):
            job_id = new_job_id()
            paths = JobPaths(S, job_id)
            paths.ensure()
            if up is not None:
                src = paths.temp / f"source{Path(up.name).suffix.lower()}"
                src.write_bytes(up.getbuffer())
                source = str(src)
            else:
                source = url.strip()
            split_path = None
            music_path = None
            if split_upload is not None:
                split_path = paths.split_screen / f"video{Path(split_upload.name).suffix.lower()}"
                paths.split_screen.mkdir(parents=True, exist_ok=True)
                split_path.write_bytes(split_upload.getbuffer())
            if music_upload is not None:
                music_path = paths.music / f"music{Path(music_upload.name).suffix.lower()}"
                paths.music.mkdir(parents=True, exist_ok=True)
                music_path.write_bytes(music_upload.getbuffer())
            options = JobOptions(
                num_clips=int(n), min_duration=float(lo), max_duration=float(hi),
                subtitle_style=style, language=lang or None, face_tracking=face_tracking,
                speaker_tracking=speaker_tracking, jump_cuts=jump_cuts, zoom=zoom,
                split_screen_path=str(split_path) if split_path else None,
                background_music_path=str(music_path) if music_path else None,
                hook_overlay=hook_overlay,
            )
            try:
                run_job(job_id, source, options,
                        progress=lambda f, m: (bar.progress(min(1.0, (index + f) / len(batch))),
                                               msg.info(f"{index + 1}/{len(batch)}: {m}")))
                completed.append(job_id)
            except Exception as e:
                st.error(f"Processing failed for {up.name if up else url}: {e}")
                st.caption(f"Job `{job_id}` can be resumed from the CLI: `python cli.py <source> --job-id {job_id}`")
        if completed:
            st.session_state["job"] = completed[-1]
            st.success(f"Finished {len(completed)} of {len(batch)} project(s). Open **Projects** to review them.")


def projects() -> None:
    jobs = list_jobs(S)
    stats = dashboard_stats(S)
    c1, c2, c3 = st.columns(3)
    c1.metric("Projects", stats["projects"]); c2.metric("Clips", stats["clips"])
    c3.metric("Approved", stats["approved"])
    if not jobs:
        st.info("No projects yet."); return
    ids = [j.job_id for j in jobs]
    default = ids.index(st.session_state["job"]) if st.session_state.get("job") in ids else 0
    job_id = st.selectbox("Project", ids, index=default,
                          format_func=lambda i: f"{i} — {next(j for j in jobs if j.job_id == i).stage}")
    paths = JobPaths(S, job_id)
    state = load_state(paths)
    if state.error:
        st.warning(f"Last error: {state.error}")
    if state.speaker_tracks:
        with st.expander("Speaker turns"):
            st.dataframe(state.speaker_tracks, use_container_width=True, hide_index=True)
    for rec in state.clips:
        c = rec.candidate
        with st.container(border=True):
            left, right = st.columns([1, 2])
            if rec.output_path and Path(rec.output_path).exists():
                left.video(rec.output_path)
            else:
                left.error(rec.error or "Not rendered")
            right.subheader(f"{c.id}: {c.title}")
            right.caption(f"{c.duration:.0f}s · {c.start:.1f}–{c.end:.1f}s · topic: {c.topic} · "
                          f"score {c.quality_score:.0f}/100 · status: **{rec.status}**")
            right.write(c.caption)
            right.write(" ".join(f"#{h.lstrip('#')}" for h in c.hashtags))
            right.caption(f"Why: {c.rationale}")
            if rec.qa and not rec.qa.passed:
                right.warning("QA issues: " + "; ".join(rec.qa.issues))
            b1, b2, b3 = right.columns(3)
            if b1.button("✅ Approve", key=f"a{job_id}{c.id}"):
                rec.status = "approved"; save_state(paths, state); st.rerun()
            if b2.button("❌ Reject", key=f"r{job_id}{c.id}"):
                rec.status = "rejected"; save_state(paths, state); st.rerun()
            if rec.output_path and Path(rec.output_path).exists():
                b3.download_button("⬇ Download", Path(rec.output_path).read_bytes(), f"{c.id}.mp4",
                                   "video/mp4", key=f"d{job_id}{c.id}")
            with right.expander("Regenerate / adjust"):
                s0 = st.number_input("Start (s)", value=float(c.start), step=0.5, key=f"s{job_id}{c.id}")
                e0 = st.number_input("End (s)", value=float(c.end), step=0.5, key=f"e{job_id}{c.id}")
                subtitle_text = st.text_area(
                    "Subtitle text", value=rec.subtitle_text if rec.subtitle_text is not None else c.transcript_excerpt,
                    key=f"sub{job_id}{c.id}", help="Edit the words burned into this clip; timing is preserved when possible.",
                )
                names = list(STYLE_PRESETS)
                sty = st.selectbox("Subtitle style", names, index=names.index(rec.subtitle_style)
                                   if rec.subtitle_style in names else 0, key=f"y{job_id}{c.id}")
                if st.button("Regenerate this clip", key=f"g{job_id}{c.id}"):
                    with st.spinner("Rendering…"):
                        try:
                            regenerate_clip(job_id, c.id, s0, e0, sty, S, subtitle_text)
                        except Exception as ex:
                            st.error(str(ex))
                    st.rerun()


(new_project if page == "New Project" else projects)()
