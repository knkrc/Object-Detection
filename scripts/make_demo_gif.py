"""README icin arayuz turu GIF'i uretir.

Playwright uygulamayi gezerken ekrani video olarak kaydediyor, sonra ffmpeg
onu GIF'e ceviriyor. Tur adimlari `tour()` icinde; arayuz degistikce burasi
guncellenir ve tek komutla GIF yenilenir.

Onkosullar:
    pip install -r requirements-dev.txt
    playwright install chromium
    ffmpeg  (brew install ffmpeg)

Kullanim:
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
    parser.add_argument("--width", type=int, default=720, help="GIF genisligi (piksel)")
    parser.add_argument("--fps", type=int, default=8, help="GIF kare hizi")
    return parser.parse_args()


def tour(page) -> None:
    """Uygulamada gezerken gosterilmeye deger ne varsa sirayla acar.

    Bekleme sureleri bilerek uzun: izleyen kisinin neyin degistigini fark
    etmesi gerekiyor, hizli gecen bir GIF hicbir sey anlatmiyor.
    """
    # 1. Ornek goruntude tespit: 3 insan + 1 otobus
    open_tab(page, "Samples")
    page.wait_for_timeout(2500)

    # 2. Sinif filtresi: sadece insan -> otobus kutusu kayboluyor.
    #    Liste sanallastirilmis oldugu icin secenege ancak yazarak ulasiliyor;
    #    zaten yazmak GIF'te aramanin calistigini da gosteriyor.
    page.get_by_role("combobox").nth(1).click()
    page.wait_for_timeout(600)
    page.keyboard.type("person", delay=110)
    page.wait_for_timeout(1000)
    page.get_by_role("option", name="person", exact=True).click()
    page.wait_for_timeout(700)
    # Acik dropdown sayfanin geri kalanini aria-hidden yapiyor; kapatmadan
    # diger widget'lara erisilemiyor.
    page.keyboard.press("Escape")
    page.wait_for_timeout(3000)

    # 3. Filtreyi temizle — sonraki adimda yaban hayati modeline gececegiz ve
    #    "person" o modelde olmadigi icin hicbir sey bulunamazdi.
    page.get_by_role("combobox").nth(1).click()
    page.wait_for_timeout(400)
    page.keyboard.press("Backspace")
    page.wait_for_timeout(700)
    page.keyboard.press("Escape")
    page.wait_for_timeout(2500)

    # 4. Once metrik sekmesine gec, sonra modeli degistir.
    #    Ters sirada yapinca yaban hayati modeli otobus fotografini yeniden
    #    degerlendiriyor ve "elephant 0.43" gibi alan disi tespitler cikiyor —
    #    dogru ama izleyene modelin kotu oldugunu dusundurten bir kare.
    open_tab(page, "Model performance", settle=2000)
    page.wait_for_timeout(1500)

    # 5. Kendi egittigimiz modele gec -> kenar cubugu yesile doner, siniflar degisir
    select_model(page, "Custom: african-wildlife", settle=2500)
    page.wait_for_timeout(2500)

    # 6. Metrikler ve once/sonra karsilastirmasi
    for _ in range(6):
        page.mouse.wheel(0, 260)
        page.wait_for_timeout(380)
    page.wait_for_timeout(2500)


def to_gif(source: Path, target: Path, width: int, fps: int) -> None:
    """ffmpeg ile GIF'e cevirir.

    Iki gecis: once renk paleti cikariliyor, sonra o paletle kodlaniyor.
    Tek gecisli donusum GIF'in 256 renk sinirinda berbat gorunuyor.
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
    """Ozel modeli kayit baslamadan once yukletir.

    Model `@st.cache_resource` ile sunucu tarafinda tutuluyor. Isitmazsak
    tur sirasinda modele gecerken ekran ~3 saniye bos kaliyor ve GIF'te
    uygulama donmus gibi gorunuyor.
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
        raise SystemExit("ffmpeg bulunamadi. macOS'ta: brew install ffmpeg")

    from playwright.sync_api import sync_playwright

    print("Streamlit baslatiliyor...")
    process, url = start_app(args.port)
    workdir = Path(tempfile.mkdtemp())

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()

            print("Model isitiliyor...")
            warm_up(browser, url)

            context = browser.new_context(
                viewport=VIEWPORT,
                record_video_dir=str(workdir),
                record_video_size=VIEWPORT,
            )
            page = context.new_page()
            page.goto(url)
            page.wait_for_timeout(3000)

            print("Tur cekiliyor...")
            tour(page)

            context.close()  # video bu satirda yaziliyor
            browser.close()

        recordings = list(workdir.glob("*.webm"))
        if not recordings:
            raise SystemExit("Playwright video kaydi uretmedi.")

        print("GIF'e cevriliyor...")
        TARGET.parent.mkdir(parents=True, exist_ok=True)
        to_gif(recordings[0], TARGET, args.width, args.fps)
        print(f"\n{TARGET.relative_to(ROOT)}  ({TARGET.stat().st_size // 1024} KB)")
    finally:
        process.terminate()
        process.wait(timeout=10)
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
