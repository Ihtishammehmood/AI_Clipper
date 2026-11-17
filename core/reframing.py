"""9:16 crop geometry and sampled face-following crop expressions."""
from __future__ import annotations

from pathlib import Path


def compute_crop(src_w: int, src_h: int, out_w: int, out_h: int,
                 cx: float = 0.5, cy: float = 0.5) -> tuple[int, int, int, int]:
    """Largest crop with the output aspect ratio, centred at fractional (cx, cy). Returns x, y, w, h."""
    target = out_w / out_h
    if src_w / src_h > target:
        ch, cw = src_h, round(src_h * target)
    else:
        cw, ch = src_w, round(src_w / target)
    cw -= cw % 2
    ch -= ch % 2
    x = min(max(round(cx * src_w - cw / 2), 0), src_w - cw)
    y = min(max(round(cy * src_h - ch / 2), 0), src_h - ch)
    return x - x % 2, y - y % 2, cw, ch


def track_faces(source: Path, start: float, end: float, sample_seconds: float = 1.0) -> list[tuple[float, float, float]]:
    """Sample the largest/nearest face, returning (clip time, center x, center y) in pixels."""
    try:
        import cv2
    except ImportError as e:
        raise RuntimeError("Face tracking requires opencv-python-headless.") from e

    if not hasattr(cv2, "CascadeClassifier") or not hasattr(cv2, "data"):
        raise RuntimeError(
            "The installed OpenCV build is incompatible with face tracking. "
            "Install opencv-python-headless>=4.9,<5 and restart the app."
        )
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if detector.empty():
        raise RuntimeError("OpenCV's frontal-face cascade could not be loaded.")
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to read video for face tracking: {source}")
    found: list[tuple[float, float, float]] = []
    previous: tuple[float, float] | None = None
    try:
        t = start
        while t < end:
            capture.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = capture.read()
            if ok:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                boxes = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
                if len(boxes):
                    centers = [(float(x + w / 2), float(y + h / 2), int(w * h)) for x, y, w, h in boxes]
                    selected = max(centers, key=lambda f: f[2]) if previous is None else min(
                        centers, key=lambda f: (f[0] - previous[0]) ** 2 + (f[1] - previous[1]) ** 2)
                    previous = selected[0], selected[1]
                    found.append((t - start, selected[0], selected[1]))
            t += sample_seconds
    finally:
        capture.release()
    return found


def face_crop_expressions(track: list[tuple[float, float, float]], cw: int, ch: int,
                          src_w: int, src_h: int) -> tuple[str, str]:
    """Create FFmpeg crop expressions that follow sampled face centers."""
    if not track:
        return f"(iw-{cw})/2", f"(ih-{ch})/2"

    def expression(axis: int, crop_size: int, src_size: int) -> str:
        coords = [max(0, min(src_size - crop_size, round(point[axis] - crop_size / 2))) for point in track]
        result = str(coords[-1])
        for i in range(len(track) - 1, 0, -1):
            t0, t1 = track[i - 1][0], track[i][0]
            fraction = f"clip((t-{t0:.3f})/{max(0.001, t1 - t0):.3f}\\,0\\,1)"
            value = f"({coords[i - 1]}+({coords[i]}-{coords[i - 1]})*{fraction})"
            result = f"if(lt(t\\,{t1:.3f})\\,{value}\\,{result})"
        return result

    return expression(1, cw, src_w), expression(2, ch, src_h)
