"""YOLOv8 modelinin etrafina ince bir sarmalayici.

Amac: Streamlit tarafinin ultralytics detaylariyla ugrasmamasi.
Model bir kez yuklenir, sonra tek tek kareler/resimler uzerinde calisir.
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
    """Tek bir tespit sonucu."""

    label: str
    confidence: float
    box: tuple[int, int, int, int]  # x1, y1, x2, y2


def resolve_weights(weights: str) -> str:
    """models/ altindaki agirligi varsa onu kullanir.

    Yoksa ismi oldugu gibi doner; ultralytics o zaman kendi indirir.
    """
    path = MODELS_DIR / weights
    return str(path) if path.exists() else weights


def stash_weights(weights: str) -> None:
    """Ultralytics indirmeyi calisma dizinine yapiyor; dosyayi models/ altina tasir.

    Aksi halde her yeni agirlik proje kokunu kirletiyor ve bir dahaki sefere
    yeniden indiriliyor.
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
        """Modelin tanidigi tum sinif isimleri (COCO icin 80 tane)."""
        return list(self.model.names.values())

    def class_ids(self, keep: list[str] | None) -> list[int] | None:
        """Sinif isimlerini ultralytics'in bekledigi id listesine cevirir.

        Tracker da ayni donusume ihtiyac duydugu icin bu metod disari acik.
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
        """Bir BGR resmi isler.

        Donen deger: (kutulari cizilmis resim, tespit listesi)
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
    """'2 kisi, 1 araba' seklinde ozet cikarmak icin sinif sayilari."""
    return dict(Counter(d.label for d in detections).most_common())
