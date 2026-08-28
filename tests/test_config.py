"""Yapilandirma ve model kesfi testleri.

`custom_models()` arayuzun dogrudan guvendigi bir fonksiyon: models/ altina
konan her .pt dosyasi model listesinde gorunmeli, hazir modeller ise
"Ozel:" olarak ikinci kez listelenmemeli.
"""

import src.config as config
from src.config import AVAILABLE_MODELS, DEFAULT_MODEL, IMAGE_TYPES, VIDEO_TYPES, is_deployed


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


# --- is_deployed ---------------------------------------------------------
# Webcam sekmesi bu fonksiyona bakarak gizleniyor; yanlis pozitif yerelde
# ozelligi kaybettirir, yanlis negatif sunucuda kirik sekme gosterir.


def test_degisken_yoksa_yerel_sayilir(monkeypatch):
    monkeypatch.delenv("DEPLOYED", raising=False)
    monkeypatch.delenv("SPACE_ID", raising=False)
    assert is_deployed() is False


def test_deployed_1_ise_sunucu(monkeypatch):
    monkeypatch.delenv("SPACE_ID", raising=False)
    monkeypatch.setenv("DEPLOYED", "1")
    assert is_deployed() is True


def test_deployed_dogru_degerleri_kabul_eder(monkeypatch):
    monkeypatch.delenv("SPACE_ID", raising=False)
    for value in ("1", "true", "TRUE", "yes", " True "):
        monkeypatch.setenv("DEPLOYED", value)
        assert is_deployed() is True, value


def test_deployed_bos_veya_0_ise_yerel(monkeypatch):
    monkeypatch.delenv("SPACE_ID", raising=False)
    for value in ("", "0", "false", "no", "hayir"):
        monkeypatch.setenv("DEPLOYED", value)
        assert is_deployed() is False, value


def test_hugging_face_space_id_yeterli(monkeypatch):
    """HF Spaces bu degiskeni kendi ekliyor; Dockerfile'a guvenmeye gerek yok."""
    monkeypatch.delenv("DEPLOYED", raising=False)
    monkeypatch.setenv("SPACE_ID", "knkrc/object-detection")
    assert is_deployed() is True
