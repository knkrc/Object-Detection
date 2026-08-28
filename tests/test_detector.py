"""Detector tests.

The ones marked `slow` download and run the real model; they are skipped in CI
and run locally with `pytest -m slow`. The unmarked ones are pure logic tests
and need no model.
"""

import cv2
import pytest

import src.detector as detector_module
from src.config import MODELS_DIR, SAMPLES_DIR
from src.detector import Detection, resolve_weights, stash_weights, summarize

# --- tests that need no model --------------------------------------------


def make_detection(label: str, conf: float = 0.9) -> Detection:
    return Detection(label=label, confidence=conf, box=(0, 0, 10, 10))


def test_summarize_counts_classes():
    detections = [make_detection("person"), make_detection("car"), make_detection("person")]
    assert summarize(detections) == {"person": 2, "car": 1}


def test_summarize_sorts_most_common_first():
    detections = [make_detection("car")] + [make_detection("person")] * 3
    assert list(summarize(detections)) == ["person", "car"]


def test_summarize_handles_an_empty_list():
    assert summarize([]) == {}


def test_detection_keeps_box_coordinates():
    detection = Detection(label="car", confidence=0.5, box=(10, 20, 110, 220))
    assert detection.box == (10, 20, 110, 220)


# --- weights file handling -----------------------------------------------
# scripts/train.py once called YOLO directly, so ultralytics downloaded the
# weights into the project root. These tests keep that behaviour from returning.


def test_resolve_weights_prefers_the_file_under_models(tmp_path, monkeypatch):
    monkeypatch.setattr(detector_module, "MODELS_DIR", tmp_path)
    (tmp_path / "yolov8n.pt").touch()

    assert resolve_weights("yolov8n.pt") == str(tmp_path / "yolov8n.pt")


def test_resolve_weights_returns_the_name_when_missing(tmp_path, monkeypatch):
    """With no file present we return the name as-is and ultralytics fetches it."""
    monkeypatch.setattr(detector_module, "MODELS_DIR", tmp_path)

    assert resolve_weights("yolov8s.pt") == "yolov8s.pt"


def test_stash_weights_moves_the_download_into_models(tmp_path, monkeypatch):
    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setattr(detector_module, "MODELS_DIR", models)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "yolov8n.pt").write_bytes(b"sahte agirlik")

    stash_weights("yolov8n.pt")

    assert not (tmp_path / "yolov8n.pt").exists()
    assert (models / "yolov8n.pt").read_bytes() == b"sahte agirlik"


def test_stash_weights_does_not_overwrite_an_existing_file(tmp_path, monkeypatch):
    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setattr(detector_module, "MODELS_DIR", models)
    monkeypatch.chdir(tmp_path)
    (models / "yolov8n.pt").write_bytes(b"mevcut")
    (tmp_path / "yolov8n.pt").write_bytes(b"yeni")

    stash_weights("yolov8n.pt")

    assert (models / "yolov8n.pt").read_bytes() == b"mevcut"


def test_stash_weights_is_a_noop_when_nothing_was_downloaded(tmp_path, monkeypatch):
    monkeypatch.setattr(detector_module, "MODELS_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)

    stash_weights("yolov8n.pt")  # hata firlatmamali


# --- tests that need the real model ---------------------------------------


@pytest.fixture(scope="module")
def detector():
    from src.detector import Detector

    return Detector("yolov8n.pt")


@pytest.fixture(scope="module")
def bus_image():
    path = SAMPLES_DIR / "bus.jpg"
    if not path.exists():
        pytest.skip("no sample image: python scripts/download_samples.py")
    return cv2.imread(str(path))


@pytest.mark.slow
def test_coco_model_knows_80_classes(detector):
    assert len(detector.class_names) == 80
    assert "person" in detector.class_names


@pytest.mark.slow
def test_finds_people_and_a_bus_in_the_sample_image(detector, bus_image):
    _, detections = detector.detect(bus_image, conf=0.35)
    bulunanlar = summarize(detections)

    assert bulunanlar.get("person", 0) >= 1
    assert bulunanlar.get("bus", 0) >= 1


@pytest.mark.slow
def test_annotated_image_keeps_the_same_size(detector, bus_image):
    annotated, _ = detector.detect(bus_image, conf=0.35)
    assert annotated.shape == bus_image.shape


@pytest.mark.slow
def test_class_filter_returns_only_the_requested_class(detector, bus_image):
    _, detections = detector.detect(bus_image, conf=0.35, keep_classes=["person"])
    assert detections, "the filtered search found nothing"
    assert {d.label for d in detections} == {"person"}


@pytest.mark.slow
def test_higher_confidence_yields_fewer_detections(detector, bus_image):
    _, dusuk = detector.detect(bus_image, conf=0.25)
    _, yuksek = detector.detect(bus_image, conf=0.9)
    assert len(yuksek) <= len(dusuk)


@pytest.mark.slow
def test_class_ids_maps_names_to_ids(detector):
    ids = detector.class_ids(["person"])
    assert ids == [0]  # person is class 0 in COCO
    assert detector.class_ids([]) is None
    assert detector.class_ids(None) is None


@pytest.mark.slow
def test_trained_model_knows_its_own_classes():
    """The model trained in M3 is committed; it must know 4 African animals."""
    from src.detector import Detector

    path = MODELS_DIR / "african-wildlife.pt"
    if not path.exists():
        pytest.skip("no trained model: python scripts/train.py")

    custom = Detector(path.name)
    assert sorted(custom.class_names) == ["buffalo", "elephant", "rhino", "zebra"]
