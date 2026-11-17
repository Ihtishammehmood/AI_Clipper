"""ASS (word-highlighted, animated) and SRT subtitle generation."""
from __future__ import annotations

from dataclasses import dataclass

from core.models import Transcript, Word


@dataclass(frozen=True)
class SubtitleStyle:
    name: str
    font: str = "Arial"
    size: int = 84
    primary: str = "#FFFFFF"
    highlight: str = "#FFD400"
    outline: str = "#000000"
    outline_w: float = 6
    shadow: float = 2
    box: bool = False
    uppercase: bool = True
    words_per_line: int = 3
    max_lines: int = 2
    animate: bool = False
    pop_scale: int = 112


STYLE_PRESETS: dict[str, SubtitleStyle] = {
    "modern_creator": SubtitleStyle("modern_creator", size=80, words_per_line=3, animate=False),
    "kinetic": SubtitleStyle("kinetic", size=88, highlight="#00E5FF", words_per_line=2, max_lines=1, animate=True),
    "minimal_professional": SubtitleStyle("minimal_professional", size=62, highlight="#FFFFFF", outline_w=0,
                                          shadow=0, box=True, uppercase=False, words_per_line=5),
    "high_energy": SubtitleStyle("high_energy", size=104, highlight="#FF3B30", outline_w=8, words_per_line=2,
                                 max_lines=1, animate=True, pop_scale=125),
    "podcast": SubtitleStyle("podcast", size=70, highlight="#7CFC00", uppercase=False, words_per_line=4),
}


def ass_color(hex_color: str, alpha: int = 0) -> str:
    h = hex_color.lstrip("#")
    r, g, b = h[0:2], h[2:4], h[4:6]
    return f"&H{alpha:02X}{b}{g}{r}".upper()


def _ov(hex_color: str) -> str:
    return ass_color(hex_color)[4:] + "&"  # &HBBGGRR&


def clip_words(tr: Transcript, start: float, end: float) -> list[Word]:
    """Words fully inside [start, end], shifted so the clip begins at t=0."""
    out = []
    for w in tr.all_words():
        if w.start >= start - 0.05 and w.end <= end + 0.05:
            out.append(Word(start=max(0.0, w.start - start), end=max(0.0, w.end - start),
                            text=w.text, probability=w.probability))
    return out


def chunk_words(words: list[Word], st: SubtitleStyle, max_gap: float = 0.8) -> list[list[Word]]:
    size = st.words_per_line * st.max_lines
    chunks: list[list[Word]] = []
    cur: list[Word] = []
    for w in words:
        if cur and (len(cur) >= size or w.start - cur[-1].end > max_gap):
            chunks.append(cur); cur = []
        cur.append(w)
        if w.text.endswith((".", "?", "!")):
            chunks.append(cur); cur = []
    if cur:
        chunks.append(cur)
    return chunks


def _t(sec: float) -> str:
    cs = int(round(sec * 100))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def _srt_t(sec: float) -> str:
    ms = int(round(sec * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def _clean(text: str, st: SubtitleStyle) -> str:
    text = text.replace("{", "(").replace("}", ")").replace("\\", "")
    return text.upper() if st.uppercase else text


def build_ass(words: list[Word], st: SubtitleStyle, res: tuple[int, int] = (1080, 1920),
              y_fraction: float = 0.72) -> str:
    """One Dialogue event per spoken word; the active word is coloured (and optionally 'pops')."""
    w_, h_ = res
    back = ass_color("#000000", 0x70) if st.box else ass_color("#000000", 0x80)
    outline = back if st.box else ass_color(st.outline)
    header = (
        "[Script Info]\nScriptType: v4.00+\n"
        f"PlayResX: {w_}\nPlayResY: {h_}\nWrapStyle: 2\nScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, "
        "Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{st.font},{st.size},{ass_color(st.primary)},{ass_color(st.primary)},{outline},{back},"
        f"-1,0,0,0,100,100,0,0,{3 if st.box else 1},{6 if st.box else st.outline_w},{st.shadow},5,60,60,0,1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    pos = f"{{\\an5\\pos({w_ // 2},{int(h_ * y_fraction)})}}"
    lines = []
    for chunk in chunk_words(words, st):
        for j, w in enumerate(chunk):
            nxt = chunk[j + 1].start if j + 1 < len(chunk) else None
            start = w.start
            end = nxt if nxt is not None else w.end + 0.25
            if nxt is not None and nxt - w.end > 0.5:
                end = w.end + 0.25
            end = max(end, start + 0.08)
            parts = []
            for k, x in enumerate(chunk):
                txt = _clean(x.text, st)
                if k == j:
                    tag = f"\\1c{_ov(st.highlight)}"
                    if st.animate:
                        s0 = int(st.pop_scale * 0.75)
                        tag += f"\\fscx{s0}\\fscy{s0}\\t(0,90,\\fscx{st.pop_scale}\\fscy{st.pop_scale})"
                    parts.append(f"{{{tag}}}{txt}{{\\r}}")
                else:
                    parts.append(txt)
                if (k + 1) % st.words_per_line == 0 and k + 1 < len(chunk):
                    parts.append("\\N")
                elif k + 1 < len(chunk):
                    parts.append(" ")
            text = "".join(parts).replace(" \\N", "\\N")
            lines.append(f"Dialogue: 0,{_t(start)},{_t(end)},Default,0,0,0,,{pos}{text}")
    return header + "\n".join(lines) + "\n"


def build_srt(words: list[Word], st: SubtitleStyle) -> str:
    out = []
    for i, chunk in enumerate(chunk_words(words, st), 1):
        out.append(f"{i}\n{_srt_t(chunk[0].start)} --> {_srt_t(chunk[-1].end + 0.1)}\n"
                   f"{' '.join(w.text for w in chunk)}\n")
    return "\n".join(out)


def build_hook_ass(text: str, res: tuple[int, int] = (1080, 1920), duration: float = 3.0) -> str:
    """Build a separate, short title overlay for the clip's opening hook."""
    w, h = res
    clean = text.replace("{", "(").replace("}", ")").replace("\\", "").strip().upper()
    clean = clean.replace("\n", "\\N")
    return (
        "[Script Info]\nScriptType: v4.00+\n"
        f"PlayResX: {w}\nPlayResY: {h}\nWrapStyle: 2\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Hook,Arial,{max(44, int(w * 0.075))},&H00FFFFFF,&H00FFFFFF,&H00000000,&H90000000,"
        "-1,0,0,0,100,100,0,0,3,2,1,8,90,90,0,1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, MarginL, MarginR, MarginV, Effect, Text\n"
        f"Dialogue: 1,0:00:00.00,{_t(duration)},Hook,0,0,0,,{{\\an8\\pos({w // 2},{int(h * 0.12)})}}{clean}\n"
    )
