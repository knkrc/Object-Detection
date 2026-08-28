"""Downloads the sample images into samples/.

Usage:  python scripts/download_samples.py
"""

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import SAMPLES_DIR  # noqa: E402

# Ultralytics' public demo images
SAMPLES = {
    "bus.jpg": "https://ultralytics.com/images/bus.jpg",
    "zidane.jpg": "https://ultralytics.com/images/zidane.jpg",
}


def main() -> None:
    for name, url in SAMPLES.items():
        target = SAMPLES_DIR / name
        if target.exists():
            print(f"skipped (already there): {name}")
            continue
        print(f"downloading: {name} <- {url}")
        urllib.request.urlretrieve(url, target)
        print(f"   -> {target} ({target.stat().st_size // 1024} KB)")

    print(f"\nDone. Images: {SAMPLES_DIR}")


if __name__ == "__main__":
    main()
