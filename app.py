"""Object Detection - Streamlit arayuzu.

Calistirmak icin:  streamlit run app.py
"""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from src.config import (
    AVAILABLE_MODELS,
    DEFAULT_CONF,
    DEFAULT_FRAME_STRIDE,
    DEFAULT_MODEL,
    IMAGE_TYPES,
    OUTPUTS_DIR,
    SAMPLES_DIR,
    VIDEO_TYPES,
)
from src.detector import Detector, summarize
from src.video import process_video

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

    counts = summarize(detections)
    st.write("**Bulunanlar:** " + ", ".join(f"{n}× {label}" for label, n in counts.items()))

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


# --------------------------------------------------------------------------
# Kenar cubugu: model ve tespit ayarlari
# --------------------------------------------------------------------------

st.sidebar.title("⚙️ Ayarlar")

model_label = st.sidebar.selectbox(
    "Model",
    list(AVAILABLE_MODELS),
    index=list(AVAILABLE_MODELS).index(DEFAULT_MODEL),
    help="Buyuk modeller daha isabetli ama daha yavas. Ilk secimde agirlik dosyasi indirilir.",
)
detector = load_detector(AVAILABLE_MODELS[model_label])

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

# --------------------------------------------------------------------------
# Ana sayfa
# --------------------------------------------------------------------------

st.title("🎯 Object Detection")
st.caption("YOLOv8 ile resim, video ve canli kamera uzerinde nesne tespiti.")

tab_image, tab_video, tab_webcam, tab_samples = st.tabs(
    ["📷 Resim", "🎬 Video", "📹 Webcam", "🖼️ Ornekler"]
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
        help="1 = her kareyi isle (yavas). 3 = her 3 karede bir tespit yap (hizli).",
    )

    if uploaded_video and st.button("Videoyu isle", type="primary"):
        suffix = Path(uploaded_video.name).suffix
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(uploaded_video.getvalue())
            source = Path(tmp.name)

        target = OUTPUTS_DIR / f"tespit_{Path(uploaded_video.name).stem}.mp4"
        progress = st.progress(0.0, text="Kareler isleniyor...")

        try:
            stats = process_video(
                detector,
                source,
                target,
                conf=conf,
                keep_classes=keep_classes,
                stride=stride,
                on_progress=lambda p: progress.progress(p, text=f"Kareler isleniyor... %{p * 100:.0f}"),
            )
        except RuntimeError as exc:
            progress.empty()
            st.error(str(exc))
        else:
            progress.empty()
            st.success(f"{stats['frames']} kare islendi ({stats['fps']:.0f} FPS).")
            if stats["counts"]:
                st.write(
                    "**Toplam tespit:** "
                    + ", ".join(f"{n}× {label}" for label, n in stats["counts"].items())
                )
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
            try:
                # Durdur'a basilinca Streamlit script'i bastan calistirir ve
                # bu dongu kendiliginden kesilir.
                while st.session_state.webcam_on:
                    ok, frame = capture.read()
                    if not ok:
                        st.warning("Kameradan goruntu alinamadi.")
                        break
                    annotated, detections = detector.detect(frame, conf, keep_classes)
                    frame_slot.image(to_rgb(annotated), channels="RGB", use_container_width=True)
                    counts = summarize(detections)
                    info_slot.write(
                        ", ".join(f"{n}× {label}" for label, n in counts.items())
                        or "Goruntude nesne yok."
                    )
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
