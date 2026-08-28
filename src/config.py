"""Uygulama genelinde kullanilan sabitler."""

import os
from pathlib import Path

# Proje kok dizini (src/config.py -> src -> kok)
ROOT = Path(__file__).resolve().parent.parent

MODELS_DIR = ROOT / "models"
SAMPLES_DIR = ROOT / "samples"
OUTPUTS_DIR = ROOT / "outputs"
RUNS_DIR = ROOT / "runs"  # ultralytics egitim ciktilari
DATASETS_DIR = ROOT / "datasets"  # indirilen veri setleri
DOCS_DIR = ROOT / "docs"  # metrik tablosu, egitim grafikleri

for _d in (MODELS_DIR, SAMPLES_DIR, OUTPUTS_DIR, DOCS_DIR):
    _d.mkdir(exist_ok=True)

# Kullanilabilir modeller: isim -> agirlik dosyasi.
# Ilk calistirmada ultralytics bu dosyalari otomatik indirir.
AVAILABLE_MODELS = {
    "YOLOv8n (fast)": "yolov8n.pt",
    "YOLOv8s (balanced)": "yolov8s.pt",
    "YOLOv8m (slow, more accurate)": "yolov8m.pt",
}

DEFAULT_MODEL = "YOLOv8n (fast)"

# Kendi egittigimiz modeller arayuzde bu onekle listeleniyor
CUSTOM_PREFIX = "Custom: "


def is_deployed() -> bool:
    """Uygulama bir sunucuda mi calisiyor?

    Webcam sekmesi `cv2.VideoCapture(0)` ile *calisan makinenin* kamerasini
    aciyor. Yerelde bu kullanicinin kamerasi, sunucuda ise (varsa) sunucunun
    kamerasi olurdu — yani ziyaretci icin anlamsiz. Sunucuda o sekmeyi
    gizliyoruz.

    `DEPLOYED` degiskeni Dockerfile'da ayarlaniyor; `SPACE_ID` ise Hugging Face
    Spaces'in kendi ekledigi degisken.
    """
    if os.getenv("DEPLOYED", "").strip().lower() in {"1", "true", "yes"}:
        return True
    return bool(os.getenv("SPACE_ID"))


def custom_models() -> dict[str, str]:
    """models/ altindaki, hazir listede olmayan .pt dosyalari = kendi egittiklerimiz.

    Arayuz bunlari "Custom: <isim>" olarak model listesine ekler, boylece
    egitilen her model otomatik olarak tum sekmelerde kullanilabilir hale gelir.
    """
    builtin = set(AVAILABLE_MODELS.values())
    return {
        f"{CUSTOM_PREFIX}{path.stem}": path.name
        for path in sorted(MODELS_DIR.glob("*.pt"))
        if path.name not in builtin
    }


DEFAULT_CONF = 0.35

# Video islerken her N kareden birini isle (1 = her kare)
DEFAULT_FRAME_STRIDE = 1

# --- Takip (tracking) ayarlari ---
# Hareket izinde saklanan gecmis kare sayisi
DEFAULT_TRAIL_LENGTH = 32
# Cizgi gecis sayaci: cizginin karedeki varsayilan konumu (0-1 arasi oran)
DEFAULT_LINE_POSITION = 0.5

IMAGE_TYPES = ["jpg", "jpeg", "png", "bmp", "webp"]
VIDEO_TYPES = ["mp4", "mov", "avi", "mkv"]
