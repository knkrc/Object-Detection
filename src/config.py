"""Constants used across the application."""

import os
from pathlib import Path

# Project root (src/config.py -> src -> root)
ROOT = Path(__file__).resolve().parent.parent

MODELS_DIR = ROOT / "models"
SAMPLES_DIR = ROOT / "samples"
OUTPUTS_DIR = ROOT / "outputs"
RUNS_DIR = ROOT / "runs"  # ultralytics training output
DATASETS_DIR = ROOT / "datasets"  # downloaded datasets
DOCS_DIR = ROOT / "docs"  # metric tables, training plots

for _d in (MODELS_DIR, SAMPLES_DIR, OUTPUTS_DIR, DOCS_DIR):
    _d.mkdir(exist_ok=True)

# Available models: label -> weights file.
# Ultralytics downloads these automatically the first time they are used.
AVAILABLE_MODELS = {
    "YOLOv8n (fast)": "yolov8n.pt",
    "YOLOv8s (balanced)": "yolov8s.pt",
    "YOLOv8m (slow, more accurate)": "yolov8m.pt",
}

DEFAULT_MODEL = "YOLOv8n (fast)"

# Models we trained ourselves are listed in the UI under this prefix
CUSTOM_PREFIX = "Custom: "


def is_deployed() -> bool:
    """Is the app running on a server?

    The webcam tab uses `cv2.VideoCapture(0)`, which opens the camera of
    *whichever machine is running the app*. Locally that is the user's camera;
    on a server it would be the server's camera (if any) — useless to a visitor.
    So we hide that tab when deployed.

    `DEPLOYED` is set in the Dockerfile; `SPACE_ID` is added by Hugging Face
    Spaces itself.
    """
    if os.getenv("DEPLOYED", "").strip().lower() in {"1", "true", "yes"}:
        return True
    return bool(os.getenv("SPACE_ID"))


def custom_models() -> dict[str, str]:
    """The .pt files under models/ that are not built-in — i.e. ones we trained.

    The UI adds them to the model list as "Custom: <name>", so every model we
    train becomes usable in every tab without touching any code.
    """
    builtin = set(AVAILABLE_MODELS.values())
    return {
        f"{CUSTOM_PREFIX}{path.stem}": path.name
        for path in sorted(MODELS_DIR.glob("*.pt"))
        if path.name not in builtin
    }


DEFAULT_CONF = 0.35

# When processing video, handle one in every N frames (1 = every frame)
DEFAULT_FRAME_STRIDE = 1

# --- Tracking settings ---
# How many past frames a motion trail keeps
DEFAULT_TRAIL_LENGTH = 32
# Line crossing counter: the line's default position in the frame (0-1 ratio)
DEFAULT_LINE_POSITION = 0.5

IMAGE_TYPES = ["jpg", "jpeg", "png", "bmp", "webp"]
VIDEO_TYPES = ["mp4", "mov", "avi", "mkv"]
