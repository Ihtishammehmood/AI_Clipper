"""Render one clip: trim -> 9:16 crop -> burn ASS subtitles -> loudness normalise -> H.264/AAC."""
from __future__ import annotations

import subprocess
from pathlib import Path

from config import Settings
from core.jump_cuts import compress_words, speech_ranges
from core.models import ClipCandidate, JobOptions, QAReport, Transcript, VideoInfo, Word
from core.quality_assurance import check_output
from core.reframing import compute_crop, face_crop_expressions, track_faces
from core.subtitle_engine import STYLE_PRESETS, build_ass, build_hook_ass, build_srt, clip_words
from utils.file_manager import JobPaths
from utils.hardware import executable_path, resolve_device, video_encoder
from utils.logger import get_logger

log = get_logger("rendering")


class RenderError(RuntimeError):
    pass


def _ffmpeg(cmd: list[str], cwd: Path | None = None) -> None:
    cmd[0] = executable_path("ffmpeg")
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if r.returncode != 0:
        raise RenderError(f"ffmpeg failed: {r.stderr.strip()[-1500:]}")


def render_clip(source: Path, video: VideoInfo, tr: Transcript, clip: ClipCandidate,
                style_name: str, paths: JobPaths, st: Settings, opt: JobOptions,
                subtitle_text: str | None = None) -> tuple[Path, QAReport]:
    if clip.start < 0 or clip.end > video.duration + 0.01 or clip.end <= clip.start:
        raise RenderError(f"Invalid clip range {clip.start:.2f}-{clip.end:.2f} (video is {video.duration:.2f}s)")
    paths.ensure()
    style = STYLE_PRESETS.get(style_name, STYLE_PRESETS["modern_creator"])
    words = clip_words(tr, clip.start, clip.end)
    duration = clip.end - clip.start
    ranges = speech_ranges(words, duration) if opt.jump_cuts else [(0.0, duration)]
    rendered_duration = sum(end - start for start, end in ranges)
    if opt.jump_cuts:
        words = compress_words(words, ranges)
    if subtitle_text is not None:
        tokens = subtitle_text.split()
        if len(tokens) == len(words):
            words = [word.model_copy(update={"text": token}) for word, token in zip(words, tokens)]
        elif tokens:
            slot = rendered_duration / len(tokens)
            words = [
                Word(start=i * slot, end=min(rendered_duration, (i + 1) * slot),
                     text=token, probability=1.0)
                for i, token in enumerate(tokens)
            ]
        else:
            words = []

    ass_path = paths.subtitles / f"{clip.id}.ass"
    ass_path.write_text(build_ass(words, style, (st.out_width, st.out_height), st.subtitle_y_fraction),
                        encoding="utf-8")
    (paths.subtitles / f"{clip.id}.srt").write_text(build_srt(words, style), encoding="utf-8")
    hook_path = paths.subtitles / f"{clip.id}_hook.ass"
    if opt.hook_overlay and clip.opening_hook.strip():
        hook_path.write_text(build_hook_ass(clip.opening_hook, (st.out_width, st.out_height)),
                             encoding="utf-8")

    x, y, cw, ch = compute_crop(video.width, video.height, st.out_width, st.out_height)
    if opt.face_tracking:
        if opt.jump_cuts:
            faces = []
            offset = 0.0
            for start, end in ranges:
                segment_faces = track_faces(source, clip.start + start, clip.start + end)
                faces.extend((offset + t, cx, cy) for t, cx, cy in segment_faces)
                offset += end - start
        else:
            faces = track_faces(source, clip.start, clip.end)
        x_expr, y_expr = face_crop_expressions(faces, cw, ch, video.width, video.height)
    else:
        x_expr, y_expr = str(x), str(y)
    filters = [f"crop={cw}:{ch}:{x_expr}:{y_expr}",
               f"scale={st.out_width}:{st.out_height}:flags=lanczos"]
    if opt.zoom:
        zoom_w, zoom_h = int(st.out_width * 1.08) // 2 * 2, int(st.out_height * 1.08) // 2 * 2
        filters += [f"scale={zoom_w}:{zoom_h}:flags=lanczos",
                    f"crop={st.out_width}:{st.out_height}"]
    filters += [f"fps={st.output_fps}", f"subtitles={ass_path.name}"]
    if opt.hook_overlay and clip.opening_hook.strip():
        filters.append(f"subtitles={hook_path.name}")
    if st.fonts_dir.is_dir() and any(st.fonts_dir.iterdir()):
        font_arg = f":fontsdir={st.fonts_dir.as_posix()}"
        filters = [filt + font_arg if filt.startswith("subtitles=") else filt for filt in filters]

    out = paths.out / f"{clip.id}.mp4"
    dur = duration
    enc = video_encoder(resolve_device(st.device))
    venc = (["-c:v", "libx264", "-preset", st.x264_preset, "-crf", str(st.crf)] if enc == "libx264"
            else ["-c:v", enc, "-cq", str(st.crf), "-preset", "p4"])
    cmd = ["ffmpeg", "-y", "-v", "error", "-ss", f"{clip.start:.3f}", "-t", f"{dur:.3f}",
           "-i", str(source.resolve())]
    split_index = None
    music_index = None
    if opt.split_screen_path:
        split_path = Path(opt.split_screen_path)
        if not split_path.is_file():
            raise RenderError(f"Split-screen video not found: {split_path}")
        split_index = 1
        cmd += ["-stream_loop", "-1", "-i", str(split_path.resolve())]
    if opt.background_music_path:
        music_path = Path(opt.background_music_path)
        if not music_path.is_file():
            raise RenderError(f"Background music not found: {music_path}")
        music_index = 1 + int(split_index is not None)
        cmd += ["-stream_loop", "-1", "-i", str(music_path.resolve())]

    video_chain = ",".join(filters + ["format=yuv420p"])
    graph: list[str] = []
    if opt.jump_cuts:
        concat_parts = []
        graph.append(f"[0:v]split={len(ranges)}" +
                     "".join(f"[vsrc{i}]" for i in range(len(ranges))))
        if video.has_audio:
            graph.append(f"[0:a]asplit={len(ranges)}" +
                         "".join(f"[asrc{i}]" for i in range(len(ranges))))
        for i, (start, end) in enumerate(ranges):
            graph.append(f"[vsrc{i}]trim=start={start:.3f}:end={end:.3f},setpts=PTS-STARTPTS[v{i}]")
            concat_parts.append(f"[v{i}]")
            if video.has_audio:
                graph.append(f"[asrc{i}]atrim=start={start:.3f}:end={end:.3f},"
                             f"asetpts=PTS-STARTPTS[a{i}]")
                concat_parts.append(f"[a{i}]")
        graph.append(f"{''.join(concat_parts)}concat=n={len(ranges)}:v=1:a={int(video.has_audio)}"
                     "[vbase]" + ("[abase]" if video.has_audio else ""))
    else:
        graph.append("[0:v]null[vbase]")
        if video.has_audio:
            graph.append("[0:a]anull[abase]")
    graph.append(f"[vbase]{video_chain}[main]")
    if split_index is not None:
        graph += [f"[{split_index}:v]scale={st.out_width // 3}:{st.out_height // 3}:"
                  "force_original_aspect_ratio=decrease[insert]",
                  "[main][insert]overlay=W-w-36:36[vout]"]
    else:
        graph.append("[main]null[vout]")
    if video.has_audio:
        audio_filters = ["highpass=f=80"]
        if st.denoise:
            audio_filters.append("afftdn=nf=-25")
        if opt.normalize_audio:
            audio_filters.append(f"loudnorm=I={st.loudness_target_lufs}:TP={st.true_peak_db}:LRA=11")
        graph.append(f"[abase]{','.join(audio_filters)}[source_audio]")
    if music_index is not None:
        graph.append(f"[{music_index}:a]volume=0.16[music_audio]")
        if video.has_audio:
            graph += ["[source_audio]asplit=2[source_mix][sidechain]",
                      "[music_audio][sidechain]sidechaincompress=threshold=0.02:ratio=8:"
                      "attack=20:release=250[ducked_music]",
                      "[source_mix][ducked_music]amix=inputs=2:duration=first:dropout_transition=2,"
                      "alimiter=limit=0.95[aout]"]
        else:
            graph.append("[music_audio]anull[aout]")
    elif video.has_audio:
        graph.append("[source_audio]anull[aout]")
    cmd += ["-filter_complex", ";".join(graph), "-map", "[vout]"]
    if video.has_audio:
        cmd += ["-map", "[aout]", "-c:a", "aac", "-b:a", "192k", "-ar", str(st.audio_sample_rate), "-ac", "2"]
    elif music_index is not None:
        cmd += ["-map", "[aout]", "-c:a", "aac", "-b:a", "192k", "-ar", str(st.audio_sample_rate), "-ac", "2"]
    if not video.has_audio and music_index is None:
        cmd += ["-an"]
    cmd += [*venc, "-t", f"{rendered_duration:.3f}", "-movflags", "+faststart", str(out.resolve())]
    log.info("Rendering %s (%.1fs, encoder=%s)", clip.id, rendered_duration, enc)
    _ffmpeg(cmd, cwd=paths.subtitles)  # cwd => subtitle filename needs no path escaping

    _ffmpeg(["ffmpeg", "-y", "-v", "error", "-ss", f"{min(1.0, dur / 2):.2f}", "-i", str(out),
             "-frames:v", "1", "-q:v", "3", str(paths.thumbnails / f"{clip.id}.jpg")])
    qa = check_output(out, rendered_duration, video, st.out_width, st.out_height)
    (paths.reports / f"{clip.id}.json").write_text(qa.model_dump_json(indent=1), encoding="utf-8")
    (paths.metadata / f"{clip.id}.json").write_text(clip.model_dump_json(indent=1), encoding="utf-8")
    return out, qa
