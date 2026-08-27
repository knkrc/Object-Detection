"""Uygulama genelinde kullanilan sabitler."""

from pathlib import Path

# Proje kok dizini (src/config.py -> src -> kok)
ROOT = Path(__file__).resolve().parent.parent

MODELS_DIR = ROOT / "models"
SAMPLES_DIR = ROOT / "samples"
OUTPUTS_DIR = ROOT / "outputs"

for _d in (MODELS_DIR, SAMPLES_DIR, OUTPUTS_DIR):
    _d.mkdir(exist_ok=True)

# Kullanilabilir modeller: isim -> agirlik dosyasi.
# Ilk calistirmada ultralytics bu dosyalari otomatik indirir.
AVAILABLE_MODELS = {
    "YOLOv8n (hizli)": "yolov8n.pt",
    "YOLOv8s (dengeli)": "yolov8s.pt",
    "YOLOv8m (yavas, daha isabetli)": "yolov8m.pt",
}

DEFAULT_MODEL = "YOLOv8n (hizli)"
DEFAULT_CONF = 0.35

# Video islerken her N kareden birini isle (1 = her kare)
DEFAULT_FRAME_STRIDE = 1

IMAGE_TYPES = ["jpg", "jpeg", "png", "bmp", "webp"]
VIDEO_TYPES = ["mp4", "mov", "avi", "mkv"]
