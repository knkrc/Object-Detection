"""Yapilandirma ve model kesfi testleri.

`custom_models()` arayuzun dogrudan guvendigi bir fonksiyon: models/ altina
konan her .pt dosyasi model listesinde gorunmeli, hazir modeller ise
"Ozel:" olarak ikinci kez listelenmemeli.
"""

import src.config as config
from src.config import AVAILABLE_MODELS, DEFAULT_MODEL, IMAGE_TYPES, VIDEO_TYPES


def test_varsayilan_model_listede_var():
    assert DEFAULT_MODEL in AVAILABLE_MODELS


def test_hazir_modeller_pt_dosyasi():
    assert all(name.endswith(".pt") for name in AVAILABLE_MODELS.values())


def test_resim_ve_video_uzantilari_cakismiyor():
    assert not set(IMAGE_TYPES) & set(VIDEO_TYPES)


def test_custom_models_bos_klasorde_bos_doner(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    assert config.custom_models() == {}


def test_custom_models_egitilmis_modeli_bulur(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    (tmp_path / "african-wildlife.pt").touch()

    assert config.custom_models() == {"Ozel: african-wildlife": "african-wildlife.pt"}


def test_custom_models_hazir_modelleri_disarida_birakir(tmp_path, monkeypatch):
    """yolov8n.pt zaten listede; "Ozel:" olarak ikinci kez gorunmemeli."""
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    for name in AVAILABLE_MODELS.values():
        (tmp_path / name).touch()
    (tmp_path / "kendi-modelim.pt").touch()

    assert config.custom_models() == {"Ozel: kendi-modelim": "kendi-modelim.pt"}


def test_custom_models_pt_disi_dosyalari_yok_sayar(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    (tmp_path / "notlar.txt").touch()
    (tmp_path / "model.onnx").touch()

    assert config.custom_models() == {}


def test_custom_models_alfabetik_siralar(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    for name in ("zebra.pt", "aslan.pt", "manda.pt"):
        (tmp_path / name).touch()

    assert list(config.custom_models()) == ["Ozel: aslan", "Ozel: manda", "Ozel: zebra"]
