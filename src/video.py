"""Video dosyalarini kare kare isleyip yeni bir video yazar.

Islemin kendisi (tespit mi, takip mi) buraya ait degil: cagiran taraf bir
`on_frame(kare) -> cizilmis_kare` fonksiyonu verir. Boylece ayni dongu hem
tespit hem takip icin kullanilir.
"""

from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np


def video_info(source: Path) -> dict:
    """Videoyu acmadan once fps/boyut gibi bilgileri okur (arayuz icin)."""
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {source}")
    try:
        return {
            "fps": capture.get(cv2.CAP_PROP_FPS) or 25.0,
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0,
        }
    finally:
        capture.release()


def _writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    """Tarayicida oynayabilen bir mp4 yazici acmayi dener."""
    for codec in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, size)
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError("Could not open video writer (no codec support).")


def process_video(
    source: Path,
    target: Path,
    on_frame: Callable[[np.ndarray], np.ndarray],
    stride: int = 1,
    on_progress: Callable[[float], None] | None = None,
) -> dict:
    """Videoyu isler, `on_frame`'in dondurdugu kareleri `target`'a yazar.

    `stride` > 1 ise her N karede bir `on_frame` cagrilir, aradaki karelerde
    son cizilmis kare tekrar yazilir. Uzun videolari belirgin sekilde hizlandirir.
    """
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {source}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    writer = _writer(target, fps, (width, height))
    frame_index = 0
    last_frame: np.ndarray | None = None

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if frame_index % stride == 0:
                last_frame = on_frame(frame)

            writer.write(last_frame if last_frame is not None else frame)
            frame_index += 1

            if on_progress and total:
                on_progress(min(frame_index / total, 1.0))
    finally:
        capture.release()
        writer.release()

    return {"frames": frame_index, "fps": fps, "size": (width, height)}
