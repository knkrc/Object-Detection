"""Video dosyalarini kare kare isleyip yeni bir video yazar."""

from collections import Counter
from pathlib import Path
from typing import Callable

import cv2

from src.detector import Detector


def _writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    """Tarayicida oynayabilen bir mp4 yazici acmayi dener."""
    for codec in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, size)
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError("Video yazici acilamadi (codec destegi yok).")


def process_video(
    detector: Detector,
    source: Path,
    target: Path,
    conf: float = 0.35,
    keep_classes: list[str] | None = None,
    stride: int = 1,
    on_progress: Callable[[float], None] | None = None,
) -> dict:
    """Videoyu isler, kutulari cizilmis halini `target` yoluna yazar.

    `stride` > 1 ise her N karede bir tespit yapilir, aradaki karelerde
    son bulunan kutular tekrar cizilir. Uzun videolari hizlandirir.
    """
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Video acilamadi: {source}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    writer = _writer(target, fps, (width, height))
    counts: Counter[str] = Counter()
    frame_index = 0
    last_frame = None

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if frame_index % stride == 0:
                annotated, detections = detector.detect(frame, conf, keep_classes)
                counts.update(d.label for d in detections)
                last_frame = annotated
            else:
                annotated = last_frame if last_frame is not None else frame

            writer.write(annotated)
            frame_index += 1

            if on_progress and total:
                on_progress(min(frame_index / total, 1.0))
    finally:
        capture.release()
        writer.release()

    return {
        "frames": frame_index,
        "fps": fps,
        "size": (width, height),
        "counts": dict(counts.most_common()),
    }
