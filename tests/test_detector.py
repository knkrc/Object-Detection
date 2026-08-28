"""Dedektor testleri.

Buradaki `slow` isaretli testler gercek modeli indirip calistirir; CI'da
atlanir, yerelde `pytest -m slow` ile calistirilir. Isaretsiz testler saf
mantik testleri ve model gerektirmez.
"""

import cv2
import pytest

from src.config import MODELS_DIR, SAMPLES_DIR
from src.detector import Detection, summarize

# --- model gerektirmeyen testler -----------------------------------------


def make_detection(label: str, conf: float = 0.9) -> Detection:
    return Detection(label=label, confidence=conf, box=(0, 0, 10, 10))


def test_summarize_sinif_sayar():
    detections = [make_detection("person"), make_detection("car"), make_detection("person")]
    assert summarize(detections) == {"person": 2, "car": 1}


def test_summarize_cok_olandan_aza_siralar():
    detections = [make_detection("car")] + [make_detection("person")] * 3
    assert list(summarize(detections)) == ["person", "car"]


def test_summarize_bos_liste():
    assert summarize([]) == {}


def test_detection_kutu_koordinatlari():
    detection = Detection(label="car", confidence=0.5, box=(10, 20, 110, 220))
    assert detection.box == (10, 20, 110, 220)


# --- gercek model gerektiren testler --------------------------------------


@pytest.fixture(scope="module")
def detector():
    from src.detector import Detector

    return Detector("yolov8n.pt")


@pytest.fixture(scope="module")
def bus_image():
    path = SAMPLES_DIR / "bus.jpg"
    if not path.exists():
        pytest.skip("ornek gorsel yok: python scripts/download_samples.py")
    return cv2.imread(str(path))


@pytest.mark.slow
def test_coco_modeli_80_sinif_taniyor(detector):
    assert len(detector.class_names) == 80
    assert "person" in detector.class_names


@pytest.mark.slow
def test_ornek_gorselde_insan_ve_otobus_bulur(detector, bus_image):
    _, detections = detector.detect(bus_image, conf=0.35)
    bulunanlar = summarize(detections)

    assert bulunanlar.get("person", 0) >= 1
    assert bulunanlar.get("bus", 0) >= 1


@pytest.mark.slow
def test_cizilmis_gorsel_ayni_boyutta(detector, bus_image):
    annotated, _ = detector.detect(bus_image, conf=0.35)
    assert annotated.shape == bus_image.shape


@pytest.mark.slow
def test_sinif_filtresi_sadece_istenen_sinifi_dondurur(detector, bus_image):
    _, detections = detector.detect(bus_image, conf=0.35, keep_classes=["person"])
    assert detections, "filtreli aramada hic tespit yok"
    assert {d.label for d in detections} == {"person"}


@pytest.mark.slow
def test_yuksek_guven_esigi_daha_az_tespit(detector, bus_image):
    _, dusuk = detector.detect(bus_image, conf=0.25)
    _, yuksek = detector.detect(bus_image, conf=0.9)
    assert len(yuksek) <= len(dusuk)


@pytest.mark.slow
def test_class_ids_isimleri_idye_cevirir(detector):
    ids = detector.class_ids(["person"])
    assert ids == [0]  # COCO'da person 0 numarali sinif
    assert detector.class_ids([]) is None
    assert detector.class_ids(None) is None


@pytest.mark.slow
def test_egitilmis_model_kendi_siniflarini_tanir():
    """M3'te egitilen model repoda; 4 Afrika hayvanini tanimali."""
    from src.detector import Detector

    path = MODELS_DIR / "african-wildlife.pt"
    if not path.exists():
        pytest.skip("egitilmis model yok: python scripts/train.py")

    custom = Detector(path.name)
    assert sorted(custom.class_names) == ["buffalo", "elephant", "rhino", "zebra"]
