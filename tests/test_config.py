"""Configuration and model-discovery tests.

`custom_models()` is something the UI relies on directly: every .pt file dropped
into models/ must show up in the model list, and built-in models must not be
listed a second time under "Custom:".
"""

from pathlib import Path

import src.config as config
from src.config import AVAILABLE_MODELS, DEFAULT_MODEL, IMAGE_TYPES, VIDEO_TYPES, is_deployed


def test_default_model_is_in_the_list():
    assert DEFAULT_MODEL in AVAILABLE_MODELS


def test_builtin_models_are_pt_files():
    assert all(name.endswith(".pt") for name in AVAILABLE_MODELS.values())


def test_image_and_video_extensions_do_not_overlap():
    assert not set(IMAGE_TYPES) & set(VIDEO_TYPES)


def test_custom_models_returns_empty_for_empty_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    assert config.custom_models() == {}


def test_custom_models_finds_a_trained_model(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    (tmp_path / "african-wildlife.pt").touch()

    assert config.custom_models() == {"Custom: african-wildlife": "african-wildlife.pt"}


def test_custom_models_excludes_builtin_models(tmp_path, monkeypatch):
    """yolov8n.pt is already in the list; it must not appear again as "Custom:"."""
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    for name in AVAILABLE_MODELS.values():
        (tmp_path / name).touch()
    (tmp_path / "kendi-modelim.pt").touch()

    assert config.custom_models() == {"Custom: kendi-modelim": "kendi-modelim.pt"}


def test_custom_models_ignores_non_pt_files(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    (tmp_path / "notlar.txt").touch()
    (tmp_path / "model.onnx").touch()

    assert config.custom_models() == {}


def test_custom_models_sorts_alphabetically(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    for name in ("zebra.pt", "aslan.pt", "manda.pt"):
        (tmp_path / name).touch()

    assert list(config.custom_models()) == ["Custom: aslan", "Custom: manda", "Custom: zebra"]


# --- is_deployed ---------------------------------------------------------
# The webcam tab is hidden based on this function: a false positive loses the
# feature locally, a false negative shows a broken tab on the server.


def test_no_env_var_means_local(monkeypatch):
    monkeypatch.delenv("DEPLOYED", raising=False)
    monkeypatch.delenv("SPACE_ID", raising=False)
    assert is_deployed() is False


def test_deployed_1_means_server(monkeypatch):
    monkeypatch.delenv("SPACE_ID", raising=False)
    monkeypatch.setenv("DEPLOYED", "1")
    assert is_deployed() is True


def test_deployed_accepts_truthy_values(monkeypatch):
    monkeypatch.delenv("SPACE_ID", raising=False)
    for value in ("1", "true", "TRUE", "yes", " True "):
        monkeypatch.setenv("DEPLOYED", value)
        assert is_deployed() is True, value


def test_deployed_empty_or_zero_means_local(monkeypatch):
    monkeypatch.delenv("SPACE_ID", raising=False)
    for value in ("", "0", "false", "no", "hayir"):
        monkeypatch.setenv("DEPLOYED", value)
        assert is_deployed() is False, value


def test_hugging_face_space_id_is_enough(monkeypatch):
    """HF Spaces sets this itself, so we need not rely on the Dockerfile."""
    monkeypatch.delenv("DEPLOYED", raising=False)
    monkeypatch.setenv("SPACE_ID", "knkrc26/object-detection")
    assert is_deployed() is True


def test_streamlit_cloud_path_is_enough(monkeypatch):
    """Community Cloud sets no env var; the checkout path is the only signal."""
    monkeypatch.delenv("DEPLOYED", raising=False)
    monkeypatch.delenv("SPACE_ID", raising=False)
    monkeypatch.setattr(config, "ROOT", Path("/mount/src/object-detection"))
    assert is_deployed() is True


def test_ordinary_path_is_not_deployed(monkeypatch):
    monkeypatch.delenv("DEPLOYED", raising=False)
    monkeypatch.delenv("SPACE_ID", raising=False)
    monkeypatch.setattr(config, "ROOT", Path("/Users/kaan/Desktop/Object-Detection"))
    assert is_deployed() is False
