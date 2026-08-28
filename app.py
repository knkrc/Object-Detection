"""Object Detection - Streamlit arayuzu.

Calistirmak icin:  streamlit run app.py
"""

import csv
import io
import json
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from src.config import (
    AVAILABLE_MODELS,
    DEFAULT_CONF,
    DEFAULT_FRAME_STRIDE,
    DEFAULT_LINE_POSITION,
    DEFAULT_MODEL,
    DEFAULT_TRAIL_LENGTH,
    DOCS_DIR,
    IMAGE_TYPES,
    OUTPUTS_DIR,
    SAMPLES_DIR,
    VIDEO_TYPES,
    custom_models,
)
from src.detector import Detector, summarize
from src.tracker import TrackSession, line_from_ratio
from src.video import process_video, video_info

st.set_page_config(page_title="Object Detection", page_icon="🎯", layout="wide")


@st.cache_resource(show_spinner="Model yukleniyor...")
def load_detector(weights: str) -> Detector:
    """Model yuklemesi pahali; ayni agirlik icin tekrar tekrar yuklemeyelim."""
    return Detector(weights)


def to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def read_upload(uploaded) -> np.ndarray:
    """Streamlit'ten gelen dosyayi OpenCV'nin bekledigi BGR diziye cevirir."""
    buffer = np.frombuffer(uploaded.getvalue(), np.uint8)
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


def as_counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{n}× {label}" for label, n in counts.items())


def show_results(original_bgr, annotated_bgr, detections, download_name="sonuc.png"):
    """Orijinal / sonuc karsilastirmasi + tespit ozeti."""
    left, right = st.columns(2)
    with left:
        st.caption("Orijinal")
        st.image(to_rgb(original_bgr), use_container_width=True)
    with right:
        st.caption(f"Sonuc — {len(detections)} nesne")
        st.image(to_rgb(annotated_bgr), use_container_width=True)

    if not detections:
        st.info("Bu esikte hicbir nesne bulunamadi. Guven esigini dusurmeyi dene.")
        return

    st.write("**Bulunanlar:** " + as_counts(summarize(detections)))

    with st.expander("Tespit detaylari"):
        st.dataframe(
            [
                {
                    "nesne": d.label,
                    "guven": round(d.confidence, 3),
                    "x1": d.box[0],
                    "y1": d.box[1],
                    "x2": d.box[2],
                    "y2": d.box[3],
                }
                for d in detections
            ],
            use_container_width=True,
        )

    ok, encoded = cv2.imencode(".png", annotated_bgr)
    if ok:
        st.download_button(
            "Sonucu indir",
            data=encoded.tobytes(),
            file_name=download_name,
            mime="image/png",
        )


def tracking_controls(key: str) -> dict:
    """Takip modu ayarlarini cizer, secimleri sozluk olarak doner."""
    enabled = st.toggle(
        "🎯 Takip modu",
        key=f"{key}_track",
        help="Her nesneye kalici bir ID verir; ayni nesneyi iki kez saymadan "
        "'kac farkli nesne gecti' sorusunu cevaplar.",
    )
    if not enabled:
        return {"enabled": False}

    left, right = st.columns(2)
    with left:
        trails = st.checkbox("Hareket izi ciz", value=True, key=f"{key}_trails")
        trail_length = st.slider(
            "Iz uzunlugu (kare)",
            8,
            96,
            DEFAULT_TRAIL_LENGTH,
            key=f"{key}_trail_len",
            disabled=not trails,
        )
    with right:
        line_on = st.checkbox("Cizgi gecis sayimi", value=False, key=f"{key}_line")
        orientation = st.radio(
            "Cizgi yonu",
            ["yatay", "dikey"],
            horizontal=True,
            key=f"{key}_orient",
            disabled=not line_on,
        )
        position = st.slider(
            "Cizgi konumu",
            0.05,
            0.95,
            DEFAULT_LINE_POSITION,
            step=0.05,
            key=f"{key}_pos",
            disabled=not line_on,
        )

    return {
        "enabled": True,
        "trails": trails,
        "trail_length": trail_length,
        "line": {"orientation": orientation, "position": position} if line_on else None,
    }


def build_session(detector, options, conf, keep_classes, fps, size) -> TrackSession:
    """Arayuz secimlerinden bir TrackSession kurar."""
    line = None
    if options.get("line"):
        line = line_from_ratio(
            size[0], size[1], options["line"]["orientation"], options["line"]["position"]
        )
    return TrackSession(
        detector=detector,
        conf=conf,
        keep_classes=keep_classes,
        fps=fps,
        trail_length=options["trail_length"],
        draw_trails=options["trails"],
        line=line,
    )


def show_tracking_summary(session: TrackSession) -> None:
    """Benzersiz sayim + cizgi sayaci + sure tablosu."""
    summary = session.summary()

    columns = st.columns(3 if summary["cizgi"] else 2)
    columns[0].metric("Farkli nesne", summary["toplam_nesne"])
    columns[1].metric("Islenen kare", summary["kare"])
    if summary["cizgi"]:
        counts = summary["cizgi"]
        columns[2].metric(
            "Cizgiyi gecen",
            sum(counts.values()),
            help=", ".join(f"{name}: {n}" for name, n in counts.items()),
        )

    if summary["unique"]:
        st.write("**Toplam farkli nesne:** " + as_counts(summary["unique"]))
    else:
        st.info("Hicbir nesne takip edilemedi. Guven esigini dusurmeyi dene.")
        return

    rows = session.durations()
    with st.expander(f"Nesne basina sure ({len(rows)} ID)"):
        st.dataframe(rows, use_container_width=True)

        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        st.download_button(
            "Takip verisini CSV indir",
            data=buffer.getvalue(),
            file_name="takip_verisi.csv",
            mime="text/csv",
        )


# --------------------------------------------------------------------------
# Kenar cubugu: model ve tespit ayarlari
# --------------------------------------------------------------------------

st.sidebar.title("⚙️ Ayarlar")

# Kendi egittigimiz modeller models/ altinda duruyor; hazir modellerin
# yanina eklenince tum sekmeler (resim/video/webcam/takip) onlarla da calisir.
own_models = custom_models()
model_choices = {**AVAILABLE_MODELS, **own_models}

model_label = st.sidebar.selectbox(
    "Model",
    list(model_choices),
    index=list(model_choices).index(DEFAULT_MODEL),
    help="Buyuk modeller daha isabetli ama daha yavas. Ilk secimde agirlik dosyasi indirilir. "
    "'Ozel:' ile baslayanlar scripts/train.py ile egitilmis kendi modellerimiz.",
)
detector = load_detector(model_choices[model_label])
is_custom = model_label in own_models

conf = st.sidebar.slider(
    "Guven esigi",
    min_value=0.05,
    max_value=0.95,
    value=DEFAULT_CONF,
    step=0.05,
    help="Dusuk deger = daha cok tespit, daha cok yanlis alarm.",
)

keep_classes = st.sidebar.multiselect(
    "Sadece bu nesneleri ara",
    options=sorted(detector.class_names),
    default=[],
    help="Bos birakirsan modelin bildigi 80 sinifin hepsi aranir.",
)

st.sidebar.divider()
st.sidebar.caption(f"Model: `{detector.weights}` · {len(detector.class_names)} sinif")
if is_custom:
    st.sidebar.success("Kendi egittigimiz model kullaniliyor.")
    st.sidebar.caption("Siniflar: " + ", ".join(detector.class_names))
elif not own_models:
    st.sidebar.caption("Kendi modelini egitmek icin: `python scripts/train.py`")

# --------------------------------------------------------------------------
# Ana sayfa
# --------------------------------------------------------------------------

st.title("🎯 Object Detection")
st.caption("YOLOv8 ile resim, video ve canli kamera uzerinde nesne tespiti ve takibi.")

tab_image, tab_video, tab_webcam, tab_samples, tab_metrics = st.tabs(
    ["📷 Resim", "🎬 Video", "📹 Webcam", "🖼️ Ornekler", "📊 Model performansi"]
)

# --- Resim ---------------------------------------------------------------
with tab_image:
    uploaded = st.file_uploader("Bir resim yukle", type=IMAGE_TYPES, key="image_upload")
    if uploaded:
        image = read_upload(uploaded)
        if image is None:
            st.error("Resim okunamadi, baska bir dosya dene.")
        else:
            with st.spinner("Tespit ediliyor..."):
                annotated, detections = detector.detect(image, conf, keep_classes)
            show_results(image, annotated, detections, f"tespit_{uploaded.name}.png")
    else:
        st.info("JPG veya PNG bir dosya yukle. Fikrin yoksa 'Ornekler' sekmesine bak.")

# --- Video ---------------------------------------------------------------
with tab_video:
    uploaded_video = st.file_uploader("Bir video yukle", type=VIDEO_TYPES, key="video_upload")

    stride = st.slider(
        "Kare atlama",
        min_value=1,
        max_value=5,
        value=DEFAULT_FRAME_STRIDE,
        help="1 = her kareyi isle (yavas). 3 = her 3 karede bir isle (hizli). "
        "Takip modunda yuksek deger ID kararliligini bozabilir.",
    )
    track_options = tracking_controls("video")

    if uploaded_video and st.button("Videoyu isle", type="primary"):
        suffix = Path(uploaded_video.name).suffix
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(uploaded_video.getvalue())
            source = Path(tmp.name)

        target = OUTPUTS_DIR / f"tespit_{Path(uploaded_video.name).stem}.mp4"
        progress = st.progress(0.0, text="Kareler isleniyor...")

        try:
            info = video_info(source)
            session = None

            if track_options["enabled"]:
                # Kare atlanirsa sure hesabi icin efektif fps kullanilir.
                session = build_session(
                    detector,
                    track_options,
                    conf,
                    keep_classes,
                    fps=info["fps"] / stride,
                    size=(info["width"], info["height"]),
                )

                def on_frame(frame):
                    annotated, _ = session.step(frame)
                    return annotated
            else:
                counts: Counter[str] = Counter()

                def on_frame(frame):
                    annotated, detections = detector.detect(frame, conf, keep_classes)
                    counts.update(d.label for d in detections)
                    return annotated

            stats = process_video(
                source,
                target,
                on_frame,
                stride=stride,
                on_progress=lambda p: progress.progress(
                    p, text=f"Kareler isleniyor... %{p * 100:.0f}"
                ),
            )
        except RuntimeError as exc:
            progress.empty()
            st.error(str(exc))
        else:
            progress.empty()
            st.success(f"{stats['frames']} kare islendi ({stats['fps']:.0f} FPS).")

            if session is not None:
                show_tracking_summary(session)
            elif counts:
                st.write("**Toplam tespit:** " + as_counts(dict(counts.most_common())))

            st.video(str(target))
            st.download_button(
                "Videoyu indir",
                data=target.read_bytes(),
                file_name=target.name,
                mime="video/mp4",
            )
        finally:
            source.unlink(missing_ok=True)

# --- Webcam --------------------------------------------------------------
with tab_webcam:
    st.write("Bilgisayarin kamerasindan canli tespit.")
    st.caption(
        "macOS'ta ilk calistirmada kamera izni istenir. Izin verdikten sonra "
        "terminali yeniden baslatman gerekebilir."
    )

    webcam_options = tracking_controls("webcam")

    if "webcam_on" not in st.session_state:
        st.session_state.webcam_on = False

    start, stop = st.columns(2)
    if start.button("▶️ Baslat", disabled=st.session_state.webcam_on):
        st.session_state.webcam_on = True
    if stop.button("⏹️ Durdur", disabled=not st.session_state.webcam_on):
        st.session_state.webcam_on = False

    frame_slot = st.empty()
    info_slot = st.empty()

    if st.session_state.webcam_on:
        capture = cv2.VideoCapture(0)
        if not capture.isOpened():
            st.session_state.webcam_on = False
            st.error("Kamera acilamadi. Baska bir uygulama kullaniyor olabilir.")
        else:
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
            live_session = (
                build_session(
                    detector,
                    webcam_options,
                    conf,
                    keep_classes,
                    fps=capture.get(cv2.CAP_PROP_FPS) or 25.0,
                    size=(width, height),
                )
                if webcam_options["enabled"]
                else None
            )

            try:
                # Durdur'a basilinca Streamlit script'i bastan calistirir ve
                # bu dongu kendiliginden kesilir.
                while st.session_state.webcam_on:
                    ok, frame = capture.read()
                    if not ok:
                        st.warning("Kameradan goruntu alinamadi.")
                        break

                    if live_session is not None:
                        annotated, _ = live_session.step(frame)
                        text = as_counts(live_session.unique_counts())
                        text = f"Toplam farkli nesne — {text}" if text else "Nesne yok."
                    else:
                        annotated, detections = detector.detect(frame, conf, keep_classes)
                        text = as_counts(summarize(detections)) or "Goruntude nesne yok."

                    frame_slot.image(to_rgb(annotated), channels="RGB", use_container_width=True)
                    info_slot.write(text)
            finally:
                capture.release()

# --- Ornekler ------------------------------------------------------------
with tab_samples:
    sample_files = sorted(
        p for p in SAMPLES_DIR.iterdir() if p.suffix.lstrip(".").lower() in IMAGE_TYPES
    )

    if not sample_files:
        st.warning(
            "`samples/` klasoru bos. `python scripts/download_samples.py` "
            "komutuyla ornek gorselleri indirebilirsin."
        )
    else:
        choice = st.selectbox("Ornek sec", sample_files, format_func=lambda p: p.name)
        image = cv2.imread(str(choice))
        with st.spinner("Tespit ediliyor..."):
            annotated, detections = detector.detect(image, conf, keep_classes)
        show_results(image, annotated, detections, f"tespit_{choice.stem}.png")

# --- Model performansi -----------------------------------------------------
with tab_metrics:
    metrics_file = DOCS_DIR / "metrics.json"

    if not metrics_file.exists():
        st.info(
            "Henuz egitim metrigi yok. Kendi modelini egitip olcmek icin:\n\n"
            "```\n"
            "python scripts/train.py --epochs 30\n"
            "python scripts/evaluate.py\n"
            "python scripts/compare.py\n"
            "```"
        )
    else:
        metrics = json.loads(metrics_file.read_text())
        st.subheader(f"`{metrics['model']}` — {metrics['data']}")

        overall = metrics["genel"]
        cols = st.columns(4)
        cols[0].metric(
            "mAP50",
            f"{overall['mAP50']:.3f}",
            help="Kutu ortusmesi %50 esiginde ortalama isabet. Ana basari olcusu.",
        )
        cols[1].metric(
            "mAP50-95",
            f"{overall['mAP50-95']:.3f}",
            help="%50'den %95'e kadar farkli esiklerin ortalamasi. Daha zorlu olcu.",
        )
        cols[2].metric(
            "Precision", f"{overall['precision']:.3f}", help="Bulduklarinin ne kadari dogruydu."
        )
        cols[3].metric(
            "Recall", f"{overall['recall']:.3f}", help="Olmasi gerekenlerin ne kadarini buldu."
        )

        st.write("**Sinif bazinda**")
        st.dataframe(metrics["sinif_bazinda"], use_container_width=True, hide_index=True)

        comparisons = (
            sorted((DOCS_DIR / "comparison").glob("*.jpg"))
            if (DOCS_DIR / "comparison").exists()
            else []
        )
        summary_image = next((p for p in comparisons if p.stem == "ozet"), None)
        singles = [p for p in comparisons if p.stem != "ozet"]

        if singles:
            st.divider()
            st.subheader("Once / sonra")
            st.caption("Solda hazir COCO modeli, sagda kendi egittigimiz model — ayni goruntude.")
            choice = st.selectbox(
                "Gorsel sec", singles, format_func=lambda p: p.stem, key="compare_pick"
            )
            st.image(str(choice), use_container_width=True)
            if summary_image:
                with st.expander("Hepsini bir arada gor"):
                    st.image(str(summary_image), use_container_width=True)

        plots = sorted((DOCS_DIR / "plots").glob("*")) if (DOCS_DIR / "plots").exists() else []
        if plots:
            st.divider()
            st.subheader("Egitim grafikleri")
            for plot in plots:
                st.caption(plot.name)
                st.image(str(plot), use_container_width=True)
