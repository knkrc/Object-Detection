"""A thin wrapper around the YOLOv8 model.

The point is to keep the Streamlit side out of ultralytics' details. The model
is loaded once, then run on individual images or frames.
"""

import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from src.config import MODELS_DIR


@dataclass
class Detection:
    """A single detection."""

    label: str
    confidence: float
    box: tuple[int, int, int, int]  # x1, y1, x2, y2


def resolve_weights(weights: str) -> str:
    """Prefers the weights under models/ when they are already there.

    Otherwise returns the name as-is, and ultralytics downloads it itself.
    """
    path = MODELS_DIR / weights
    return str(path) if path.exists() else weights


def stash_weights(weights: str) -> None:
    """Ultralytics downloads into the working directory; move the file to models/.

    Otherwise every new weights file litters the project root and gets
    downloaded again next time.
    """
    target = MODELS_DIR / weights
    downloaded = Path(weights)
    if downloaded.is_file() and not target.exists():
        shutil.move(str(downloaded), target)


class Detector:
    def __init__(self, weights: str = "yolov8n.pt"):
        self.model = YOLO(resolve_weights(weights))
        self.weights = weights
        stash_weights(weights)

    @property
    def class_names(self) -> list[str]:
        """Every class name the model knows (80 of them for COCO)."""
        return list(self.model.names.values())

    def class_ids(self, keep: list[str] | None) -> list[int] | None:
        """Converts class names into the id list ultralytics expects.

        Public because the tracker needs the same conversion.
        """
        if not keep:
            return None
        return [i for i, name in self.model.names.items() if name in keep]

    def detect(
        self,
        image: np.ndarray,
        conf: float = 0.35,
        keep_classes: list[str] | None = None,
    ) -> tuple[np.ndarray, list[Detection]]:
        """Runs the model on one BGR image.

        Returns: (image with boxes drawn, list of detections)
        """
        results = self.model.predict(
            source=image,
            conf=conf,
            classes=self.class_ids(keep_classes),
            verbose=False,
        )
        result = results[0]

        detections = []
        for box in result.boxes:
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            detections.append(
                Detection(
                    label=self.model.names[int(box.cls[0])],
                    confidence=float(box.conf[0]),
                    box=(x1, y1, x2, y2),
                )
            )

        return result.plot(), detections


def summarize(detections: list[Detection]) -> dict[str, int]:
    """Per-class counts, for summaries like "2 people, 1 car"."""
    return dict(Counter(d.label for d in detections).most_common())
