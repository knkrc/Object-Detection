"""Produces the UI tour GIF used in the README.

Playwright records the screen as video while walking through the app, then
ffmpeg turns that into a GIF. The tour steps live in `tour()`; update them when
the UI changes and the GIF can be refreshed with one command.

Requires:
    pip install -r requirements-dev.txt
    playwright install chromium
    ffmpeg  (brew install ffmpeg)

Usage:
    python scripts/make_demo_gif.py
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts._preview import open_tab, select_model, start_app  # noqa: E402
from src.config import DOCS_DIR  # noqa: E402

TARGET = DOCS_DIR / "demo.gif"
VIEWPORT = {"width": 1280, "height": 780}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8598)
    parser.add_argument("--width", type=int, default=720, help="GIF width in pixels")
    parser.add_argument("--fps", type=int, default=8, help="GIF frame rate")
    return parser.parse_args()


def tour(page) -> None:
    """Walks through everything in the app that is worth showing.

    The waits are deliberately long: a viewer has to notice what changed, and a
    GIF that races through its steps tells you nothing.
    """
    # 1. Detection on a sample image: 3 people + 1 bus
    open_tab(page, "Samples")
    page.wait_for_timeout(2500)

    # 2. Class filter set to person -> the bus box disappears.
    #    The option list is virtualised, so the only way to reach an entry is to
    #    type; that also shows the search working in the GIF.
    page.get_by_role("combobox").nth(1).click()
    page.wait_for_timeout(600)
    page.keyboard.type("person", delay=110)
    page.wait_for_timeout(1000)
    page.get_by_role("option", name="person", exact=True).click()
    page.wait_for_timeout(700)
    # An open dropdown marks the rest of the page aria-hidden; the other
    # widgets are unreachable until it is closed.
    page.keyboard.press("Escape")
    page.wait_for_timeout(3000)

    # 3. Clear the filter — we switch to the wildlife model next, and "person"
    #    is not one of its classes, so nothing would be found.
    page.get_by_role("combobox").nth(1).click()
    page.wait_for_timeout(400)
    page.keyboard.press("Backspace")
    page.wait_for_timeout(700)
    page.keyboard.press("Escape")
    page.wait_for_timeout(2500)

    # 4. Open the metrics tab first, then switch models.
    #    The other way round, the wildlife model re-evaluates the bus photo and
    #    produces out-of-domain hits like "elephant 0.43" — correct behaviour,
    #    but a frame that makes the model look bad to a viewer.
    open_tab(page, "Model performance", settle=2000)
    page.wait_for_timeout(1500)

    # 5. Switch to our own model -> the sidebar turns green, classes change
    select_model(page, "Custom: african-wildlife", settle=2500)
    page.wait_for_timeout(2500)

    # 6. Metrics and the before/after comparison
    for _ in range(6):
        page.mouse.wheel(0, 260)
        page.wait_for_timeout(380)
    page.wait_for_timeout(2500)


def to_gif(source: Path, target: Path, width: int, fps: int) -> None:
    """Converts to GIF with ffmpeg.

    Two passes: extract a colour palette, then encode against it. A single-pass
    conversion looks awful within the GIF format's 256-colour limit.
    """
    palette = target.with_suffix(".palette.png")
    scale = f"fps={fps},scale={width}:-1:flags=lanczos"

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(source),
            "-vf",
            f"{scale},palettegen=stats_mode=diff",
            str(palette),
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(source),
            "-i",
            str(palette),
            "-lavfi",
            f"{scale}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3",
            str(target),
        ],
        check=True,
        capture_output=True,
    )
    palette.unlink(missing_ok=True)


def warm_up(browser, url: str) -> None:
    """Loads the custom model before recording starts.

    The model is cached server-side by `@st.cache_resource`. Without warming it
    up, switching to it mid-tour leaves the screen blank for ~3 seconds, which
    looks like the app has frozen.
    """
    context = browser.new_context(viewport=VIEWPORT)
    page = context.new_page()
    page.goto(url)
    page.wait_for_timeout(5000)
    select_model(page, "Custom: african-wildlife")
    context.close()


def main() -> None:
    args = parse_args()
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg not found. On macOS: brew install ffmpeg")

    from playwright.sync_api import sync_playwright

    print("Starting Streamlit...")
    process, url = start_app(args.port)
    workdir = Path(tempfile.mkdtemp())

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()

            print("Warming up the model...")
            warm_up(browser, url)

            context = browser.new_context(
                viewport=VIEWPORT,
                record_video_dir=str(workdir),
                record_video_size=VIEWPORT,
            )
            page = context.new_page()
            page.goto(url)
            page.wait_for_timeout(3000)

            print("Recording the tour...")
            tour(page)

            context.close()  # the video is written on this line
            browser.close()

        recordings = list(workdir.glob("*.webm"))
        if not recordings:
            raise SystemExit("Playwright produced no video recording.")

        print("Converting to GIF...")
        TARGET.parent.mkdir(parents=True, exist_ok=True)
        to_gif(recordings[0], TARGET, args.width, args.fps)
        print(f"\n{TARGET.relative_to(ROOT)}  ({TARGET.stat().st_size // 1024} KB)")
    finally:
        process.terminate()
        process.wait(timeout=10)
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
