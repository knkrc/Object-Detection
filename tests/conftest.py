"""Ortak test yardimcilari.

Buradaki sahte model, ultralytics'in `model.track()` ciktisinin sadece
`TrackSession`'in kullandigi kadarini taklit eder. Amac takip mantigini
(sayim, sure, iz, cizgi gecisi) gercek bir model indirmeden test edebilmek —
o testler saniyeler yerine milisaniyeler suruyor.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np
import pytest


@dataclass
class FakeBox:
    """ultralytics Boxes'in tek bir satiri gibi davranir."""

    id: np.ndarray | None
    xyxy: np.ndarray
    cls: np.ndarray
    conf: np.ndarray


@dataclass
class FakeResult:
    boxes: list[FakeBox]
    frame: np.ndarray

    def plot(self) -> np.ndarray:
        # Gercek plot kutulari cizer; testler icin karenin kopyasi yeterli.
        return self.frame.copy()


@dataclass
class FakeModel:
    names: dict[int, str]
    # Her cagrida sirayla donulecek tespit listeleri: [(id, cls, x1,y1,x2,y2), ...]
    script: list[list[tuple]]
    predictor = None
    calls: list[dict] = field(default_factory=list)

    def track(self, source, **kwargs):
        self.calls.append(kwargs)
        index = min(len(self.calls) - 1, len(self.script) - 1)
        boxes = [
            FakeBox(
                id=None if track_id is None else np.array([track_id]),
                xyxy=np.array([[x1, y1, x2, y2]], dtype=float),
                cls=np.array([cls]),
                conf=np.array([0.9]),
            )
            for track_id, cls, x1, y1, x2, y2 in self.script[index]
        ]
        return [FakeResult(boxes=boxes, frame=source)]


@dataclass
class FakeDetector:
    model: FakeModel

    def class_ids(self, keep):
        if not keep:
            return None
        return [i for i, name in self.model.names.items() if name in keep]


def make_detector(script, names=None) -> FakeDetector:
    """Verilen tespit senaryosunu oynatan sahte bir dedektor kurar."""
    return FakeDetector(FakeModel(names or {0: "car", 1: "person"}, script))


@pytest.fixture
def blank_frame() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def synthetic_video(tmp_path):
    """Belirtilen sayida kareden olusan kucuk bir mp4 uretir."""

    def build(frames: int = 12, size: tuple[int, int] = (64, 48)) -> "Path":  # noqa: F821
        path = tmp_path / "test.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, size)
        assert writer.isOpened(), "test videosu yazilamadi"
        for i in range(frames):
            frame = np.full((size[1], size[0], 3), i * 5 % 255, np.uint8)
            writer.write(frame)
        writer.release()
        return path

    return build
