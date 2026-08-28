"""Cizgi sayaci ve takip oturumu testleri.

Cizgi yon mantigi M2'de bir kez hatali cikti (dikey cizgide saga hareket
"geri" sayiliyordu); buradaki testler o hatanin geri gelmesini engelliyor.
"""

import numpy as np
import pytest
from conftest import make_detector

from src.tracker import LineCounter, TrackSession, color_for, line_from_ratio

# --- LineCounter ---------------------------------------------------------


@pytest.fixture
def vertical_line() -> LineCounter:
    """x=100'de dikey cizgi, asagidan yukari — saga hareket pozitif taraf."""
    return LineCounter((100, 200), (100, 0), names=("right", "left"))


def test_ilk_gorulme_gecis_saymaz(vertical_line):
    assert vertical_line.update(1, (50, 100)) is None
    assert vertical_line.counts == {"right": 0, "left": 0}


def test_ayni_tarafta_kalmak_saymaz(vertical_line):
    vertical_line.update(1, (50, 100))
    for x in (60, 70, 80, 99):
        assert vertical_line.update(1, (x, 100)) is None
    assert sum(vertical_line.counts.values()) == 0


def test_soldan_saga_gecis_saga_sayilir(vertical_line):
    vertical_line.update(1, (50, 100))
    assert vertical_line.update(1, (150, 100)) == "right"
    assert vertical_line.counts == {"right": 1, "left": 0}


def test_sagdan_sola_gecis_sola_sayilir(vertical_line):
    vertical_line.update(1, (150, 100))
    assert vertical_line.update(1, (50, 100)) == "left"
    assert vertical_line.counts == {"right": 0, "left": 1}


def test_ileri_geri_gidip_gelmek_iki_kez_sayilir(vertical_line):
    vertical_line.update(1, (50, 100))
    vertical_line.update(1, (150, 100))
    vertical_line.update(1, (50, 100))
    assert vertical_line.counts == {"right": 1, "left": 1}


def test_farkli_idler_ayri_takip_edilir(vertical_line):
    vertical_line.update(1, (50, 100))
    vertical_line.update(2, (150, 100))
    # 2 numarali nesne sagdan basladi; onun ilk gorulmesi gecis degil.
    assert vertical_line.counts == {"right": 0, "left": 0}
    vertical_line.update(1, (150, 100))
    assert vertical_line.counts["right"] == 1


def test_tam_cizgi_uzerindeki_nokta_yok_sayilir(vertical_line):
    vertical_line.update(1, (50, 100))
    assert vertical_line.update(1, (100, 100)) is None
    # Taraf bilgisi bozulmamali: soldan saga gecis hala sayilmali.
    assert vertical_line.update(1, (150, 100)) == "right"


def test_gecen_idler_kaydediliyor(vertical_line):
    vertical_line.update(7, (50, 100))
    vertical_line.update(7, (150, 100))
    assert vertical_line.crossed == {7}


# --- line_from_ratio: yon kurallarinin regresyon testi --------------------


def test_dikey_cizgide_saga_hareket_saga_sayilir():
    """M2 hatasi: cizgi yukaridan asagi cizilince saga hareket 'left' oluyordu."""
    line = line_from_ratio(640, 480, "vertical", 0.5)
    line.update(1, (100, 240))
    assert line.update(1, (500, 240)) == "right"


def test_yatay_cizgide_asagi_hareket_asagi_sayilir():
    line = line_from_ratio(640, 480, "horizontal", 0.5)
    line.update(1, (320, 50))
    assert line.update(1, (320, 400)) == "down"


def test_cizgi_konumu_orana_gore_yerlesiyor():
    line = line_from_ratio(640, 480, "horizontal", 0.25)
    assert line.p1 == (0, 120)
    assert line.p2 == (640, 120)


# --- color_for -----------------------------------------------------------


def test_ayni_id_hep_ayni_renk():
    assert color_for(5) == color_for(5)


def test_farkli_idler_farkli_renk():
    renkler = {color_for(i) for i in range(10)}
    assert len(renkler) > 1


# --- TrackSession --------------------------------------------------------


def test_ayni_id_birden_cok_karede_bir_kez_sayilir(blank_frame):
    # Ayni nesne (id=1) uc kare boyunca goruluyor.
    detector = make_detector([[(1, 0, 10, 10, 50, 50)]] * 3)
    session = TrackSession(detector=detector, fps=10.0)

    for _ in range(3):
        session.step(blank_frame)

    assert session.unique_counts() == {"car": 1}
    assert session.summary()["total_objects"] == 1


def test_farkli_idler_ayri_sayilir(blank_frame):
    detector = make_detector(
        [
            [(1, 0, 10, 10, 50, 50)],
            [(1, 0, 12, 10, 52, 50), (2, 0, 100, 100, 140, 140)],
            [(2, 0, 102, 100, 142, 140), (3, 1, 200, 200, 240, 240)],
        ]
    )
    session = TrackSession(detector=detector, fps=10.0)
    for _ in range(3):
        session.step(blank_frame)

    assert session.unique_counts() == {"car": 2, "person": 1}


def test_kimliksiz_kutular_atlanir(blank_frame):
    """Tracker bazi kutulara henuz ID atamaz; bunlar sayima girmemeli."""
    detector = make_detector([[(None, 0, 10, 10, 50, 50), (1, 0, 60, 60, 90, 90)]])
    session = TrackSession(detector=detector, fps=10.0)
    _, tracks = session.step(blank_frame)

    assert len(tracks) == 1
    assert tracks[0].track_id == 1


def test_sure_fps_uzerinden_hesaplanir(blank_frame):
    detector = make_detector([[(1, 0, 10, 10, 50, 50)]] * 20)
    session = TrackSession(detector=detector, fps=10.0)
    for _ in range(20):
        session.step(blank_frame)

    row = session.durations()[0]
    assert row["frames"] == 20
    assert row["seconds"] == pytest.approx(2.0)  # 20 kare / 10 fps
    assert row["first_frame"] == 0
    assert row["last_frame"] == 19


def test_sure_tablosu_uzundan_kisaya_siralanir(blank_frame):
    detector = make_detector(
        [
            [(1, 0, 10, 10, 50, 50), (2, 0, 60, 60, 90, 90)],
            [(1, 0, 10, 10, 50, 50)],
            [(1, 0, 10, 10, 50, 50)],
        ]
    )
    session = TrackSession(detector=detector, fps=10.0)
    for _ in range(3):
        session.step(blank_frame)

    rows = session.durations()
    assert [r["id"] for r in rows] == [1, 2]


def test_iz_uzunlugu_sinirli(blank_frame):
    detector = make_detector([[(1, 0, 10, 10, 50, 50)]] * 20)
    session = TrackSession(detector=detector, fps=10.0, trail_length=5)
    for _ in range(20):
        session.step(blank_frame)

    assert len(session.trails[1]) == 5


def test_iz_merkez_noktalarini_biriktirir(blank_frame):
    detector = make_detector(
        [
            [(1, 0, 0, 0, 100, 100)],  # merkez (50, 50)
            [(1, 0, 100, 0, 200, 100)],  # merkez (150, 50)
        ]
    )
    session = TrackSession(detector=detector, fps=10.0)
    session.step(blank_frame)
    session.step(blank_frame)

    assert list(session.trails[1]) == [(50, 50), (150, 50)]


def test_sinif_filtresi_modele_iletilir(blank_frame):
    detector = make_detector([[(1, 0, 10, 10, 50, 50)]])
    session = TrackSession(detector=detector, fps=10.0, keep_classes=["person"])
    session.step(blank_frame)

    assert detector.model.calls[0]["classes"] == [1]


def test_filtre_yoksa_tum_siniflar(blank_frame):
    detector = make_detector([[(1, 0, 10, 10, 50, 50)]])
    session = TrackSession(detector=detector, fps=10.0)
    session.step(blank_frame)

    assert detector.model.calls[0]["classes"] is None


def test_cizgi_sayaci_oturuma_baglanir(blank_frame):
    detector = make_detector(
        [
            [(1, 0, 0, 0, 100, 100)],  # merkez x=50, cizginin solunda
            [(1, 0, 500, 0, 600, 100)],  # merkez x=550, saginda -> gecis
        ]
    )
    line = line_from_ratio(640, 480, "vertical", 0.5)
    session = TrackSession(detector=detector, fps=10.0, line=line)
    session.step(blank_frame)
    session.step(blank_frame)

    assert session.summary()["line"] == {"right": 1, "left": 0}


def test_cizgi_yoksa_ozette_none(blank_frame):
    detector = make_detector([[(1, 0, 10, 10, 50, 50)]])
    session = TrackSession(detector=detector, fps=10.0)
    session.step(blank_frame)

    assert session.summary()["line"] is None


def test_reset_durumu_temizler(blank_frame):
    detector = make_detector([[(1, 0, 10, 10, 50, 50)]] * 5)
    session = TrackSession(detector=detector, fps=10.0)
    for _ in range(5):
        session.step(blank_frame)

    session.reset()

    assert session.frame_index == 0
    assert session.unique_counts() == {}
    assert session.trails == {}
    assert session.durations() == []


def test_step_cizilmis_kare_dondurur(blank_frame):
    detector = make_detector([[(1, 0, 10, 10, 50, 50)]])
    session = TrackSession(detector=detector, fps=10.0)
    annotated, _ = session.step(blank_frame)

    assert isinstance(annotated, np.ndarray)
    assert annotated.shape == blank_frame.shape
