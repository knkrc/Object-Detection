"""Video isleme dongusu testleri — model gerektirmez.

`process_video` isin ne oldugunu bilmez, verilen `on_frame` fonksiyonunu
cagirir. Bu yuzden sahte bir `on_frame` ile dongunun kendisini test edebiliyoruz.
"""

import cv2
import numpy as np
import pytest

from src.video import process_video, video_info


def test_video_info_dogru_okuyor(synthetic_video):
    source = synthetic_video(frames=12, size=(64, 48))
    info = video_info(source)

    assert info["frames"] == 12
    assert info["width"] == 64
    assert info["height"] == 48
    assert info["fps"] == pytest.approx(10.0)


def test_video_info_olmayan_dosyada_hata(tmp_path):
    with pytest.raises(RuntimeError, match="Could not open"):
        video_info(tmp_path / "yok.mp4")


def test_process_video_olmayan_dosyada_hata(tmp_path):
    with pytest.raises(RuntimeError, match="Could not open"):
        process_video(tmp_path / "yok.mp4", tmp_path / "cikti.mp4", lambda f: f)


def test_cikti_ayni_sayida_kare_icerir(synthetic_video, tmp_path):
    source = synthetic_video(frames=12)
    target = tmp_path / "cikti.mp4"

    stats = process_video(source, target, lambda frame: frame)

    assert stats["frames"] == 12
    assert target.exists()

    capture = cv2.VideoCapture(str(target))
    try:
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 12
    finally:
        capture.release()


def test_stride_1_her_kareyi_isler(synthetic_video, tmp_path):
    source = synthetic_video(frames=10)
    cagrilar = []

    process_video(source, tmp_path / "c.mp4", lambda f: (cagrilar.append(1), f)[1], stride=1)

    assert len(cagrilar) == 10


def test_stride_3_her_ucuncu_kareyi_isler(synthetic_video, tmp_path):
    source = synthetic_video(frames=9)
    cagrilar = []

    stats = process_video(
        source, tmp_path / "c.mp4", lambda f: (cagrilar.append(1), f)[1], stride=3
    )

    # 9 kare, 0/3/6 islenir; cikti yine 9 kare olur (aradakiler tekrarlanir).
    assert len(cagrilar) == 3
    assert stats["frames"] == 9


def test_atlanan_kareler_son_cizilmis_kareyi_tekrarlar(synthetic_video, tmp_path):
    """stride > 1'de arada kalan kareler islenmis son kareyle doldurulmali."""
    source = synthetic_video(frames=4, size=(32, 32))
    target = tmp_path / "c.mp4"

    # Islenen her kareyi tek renge boyayarak hangi karenin yazildigini izliyoruz.
    renk = [50]

    def on_frame(frame):
        renk[0] += 50
        return np.full_like(frame, renk[0])

    process_video(source, target, on_frame, stride=2)

    capture = cv2.VideoCapture(str(target))
    try:
        okunan = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            okunan.append(int(frame[0, 0, 0]))
    finally:
        capture.release()

    # 4 kare / stride 2 -> iki kez islendi, her deger iki kez yazildi.
    assert len(okunan) == 4
    assert okunan[0] == okunan[1]
    assert okunan[2] == okunan[3]
    assert okunan[0] != okunan[2]


def test_ilerleme_callbacki_sona_ulasir(synthetic_video, tmp_path):
    source = synthetic_video(frames=8)
    ilerleme = []

    process_video(source, tmp_path / "c.mp4", lambda f: f, on_progress=ilerleme.append)

    assert len(ilerleme) == 8
    assert ilerleme[-1] == pytest.approx(1.0)
    assert ilerleme == sorted(ilerleme)  # monoton artmali
    assert all(0 < p <= 1.0 for p in ilerleme)


def test_ilerleme_callbacki_opsiyonel(synthetic_video, tmp_path):
    source = synthetic_video(frames=5)
    stats = process_video(source, tmp_path / "c.mp4", lambda f: f)
    assert stats["frames"] == 5


def test_stats_kaynak_bilgisini_dondurur(synthetic_video, tmp_path):
    source = synthetic_video(frames=6, size=(64, 48))
    stats = process_video(source, tmp_path / "c.mp4", lambda f: f)

    assert stats["size"] == (64, 48)
    assert stats["fps"] == pytest.approx(10.0)
