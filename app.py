"""Object Detection - Streamlit interface.

To run it:  streamlit run app.py
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
    is_deployed,
)
from src.detector import Detector, summarize
from src.tracker import TrackSession, line_from_ratio
from src.video import process_video, video_info

st.set_page_config(page_title="Object Detection", page_icon="🎯", layout="wide")


@st.cache_resource(show_spinner="Loading model...")
def load_detector(weights: str) -> Detector:
    """Loading a model is expensive; do not reload the same weights each rerun."""
    return Detector(weights)


def to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def read_upload(uploaded) -> np.ndarray:
    """Turns a Streamlit upload into the BGR array OpenCV expects."""
    buffer = np.frombuffer(uploaded.getvalue(), np.uint8)
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


def as_counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{n}× {label}" for label, n in counts.items())


def show_results(original_bgr, annotated_bgr, detections, download_name="result.png"):
    """Original vs. result side by side, plus a summary of the detections."""
    left, right = st.columns(2)
    with left:
        st.caption("Original")
        st.image(to_rgb(original_bgr), use_container_width=True)
    with right:
        st.caption(f"Result — {len(detections)} objects")
        st.image(to_rgb(annotated_bgr), use_container_width=True)

    if not detections:
        st.info("Nothing found at this threshold. Try lowering the confidence.")
        return

    st.write("**Found:** " + as_counts(summarize(detections)))

    with st.expander("Detection details"):
        st.dataframe(
            [
                {
                    "object": d.label,
                    "confidence": round(d.confidence, 3),
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
            "Download result",
            data=encoded.tobytes(),
            file_name=download_name,
            mime="image/png",
        )


def tracking_controls(key: str) -> dict:
    """Draws the tracking-mode controls and returns the choices as a dict."""
    enabled = st.toggle(
        "🎯 Tracking mode",
        key=f"{key}_track",
        help="Gives every object a persistent ID, so it can answer how many "
        "distinct objects passed through without counting any of them twice.",
    )
    if not enabled:
        return {"enabled": False}

    left, right = st.columns(2)
    with left:
        trails = st.checkbox("Draw motion trails", value=True, key=f"{key}_trails")
        trail_length = st.slider(
            "Trail length (frames)",
            8,
            96,
            DEFAULT_TRAIL_LENGTH,
            key=f"{key}_trail_len",
            disabled=not trails,
        )
    with right:
        line_on = st.checkbox("Count line crossings", value=False, key=f"{key}_line")
        orientation = st.radio(
            "Line direction",
            ["horizontal", "vertical"],
            horizontal=True,
            key=f"{key}_orient",
            disabled=not line_on,
        )
        position = st.slider(
            "Line position",
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
    """Builds a TrackSession from the UI choices."""
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
    """Unique counts, line counter and the time-on-screen table."""
    summary = session.summary()

    columns = st.columns(3 if summary["line"] else 2)
    columns[0].metric("Distinct objects", summary["total_objects"])
    columns[1].metric("Frames processed", summary["frames"])
    if summary["line"]:
        counts = summary["line"]
        columns[2].metric(
            "Crossed the line",
            sum(counts.values()),
            help=", ".join(f"{name}: {n}" for name, n in counts.items()),
        )

    if summary["unique"]:
        st.write("**Distinct objects in total:** " + as_counts(summary["unique"]))
    else:
        st.info("Nothing could be tracked. Try lowering the confidence.")
        return

    rows = session.durations()
    with st.expander(f"Time on screen per object ({len(rows)} IDs)"):
        st.dataframe(rows, use_container_width=True)

        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        st.download_button(
            "Download tracking data as CSV",
            data=buffer.getvalue(),
            file_name="tracking-data.csv",
            mime="text/csv",
        )


# --------------------------------------------------------------------------
# Sidebar: model and detection settings
# --------------------------------------------------------------------------

st.sidebar.title("⚙️ Settings")

# Models we trained ourselves live under models/; adding them next to the
# built-in ones makes every tab (image/video/webcam/tracking) work with them.
own_models = custom_models()
model_choices = {**AVAILABLE_MODELS, **own_models}

model_label = st.sidebar.selectbox(
    "Model",
    list(model_choices),
    index=list(model_choices).index(DEFAULT_MODEL),
    help="Bigger models are more accurate but slower. The weights are downloaded "
    "the first time you pick one. Entries starting with 'Custom:' are models "
    "trained with scripts/train.py.",
)
detector = load_detector(model_choices[model_label])
is_custom = model_label in own_models

conf = st.sidebar.slider(
    "Confidence threshold",
    min_value=0.05,
    max_value=0.95,
    value=DEFAULT_CONF,
    step=0.05,
    help="Lower means more detections, and more false alarms.",
)

keep_classes = st.sidebar.multiselect(
    "Look for these objects only",
    options=sorted(detector.class_names),
    default=[],
    help="Leave empty to look for every class the model knows.",
)

st.sidebar.divider()
st.sidebar.caption(f"Model: `{detector.weights}` · {len(detector.class_names)} classes")
if is_custom:
    st.sidebar.success("Using a model we trained ourselves.")
    st.sidebar.caption("Classes: " + ", ".join(detector.class_names))
elif not own_models:
    st.sidebar.caption("To train your own model: `python scripts/train.py`")

# --------------------------------------------------------------------------
# Main page
# --------------------------------------------------------------------------

st.title("🎯 Object Detection")

# The webcam is meaningless on a server, so we do not show the tab there.
show_webcam = not is_deployed()
tab_labels = ["📷 Image", "🎬 Video"]
if show_webcam:
    tab_labels.append("📹 Webcam")
tab_labels += ["🖼️ Samples", "📊 Model performance"]

st.caption(
    "Object detection and tracking on images, video and a live camera, with YOLOv8."
    if show_webcam
    else "Object detection and tracking on images and video, with YOLOv8."
)

tabs = st.tabs(tab_labels)
tab_image, tab_video = tabs[0], tabs[1]
tab_webcam = tabs[2] if show_webcam else None
tab_samples, tab_metrics = tabs[-2], tabs[-1]

# --- Image ---------------------------------------------------------------
with tab_image:
    uploaded = st.file_uploader("Upload an image", type=IMAGE_TYPES, key="image_upload")
    if uploaded:
        image = read_upload(uploaded)
        if image is None:
            st.error("Could not read that image, try another file.")
        else:
            with st.spinner("Detecting..."):
                annotated, detections = detector.detect(image, conf, keep_classes)
            show_results(image, annotated, detections, f"detected_{uploaded.name}.png")
    else:
        st.info("Upload a JPG or PNG. Short on ideas? Try the 'Samples' tab.")

# --- Video ---------------------------------------------------------------
with tab_video:
    uploaded_video = st.file_uploader("Upload a video", type=VIDEO_TYPES, key="video_upload")

    stride = st.slider(
        "Frame skip",
        min_value=1,
        max_value=5,
        value=DEFAULT_FRAME_STRIDE,
        help="1 processes every frame (slow). 3 processes every third frame (fast). "
        "In tracking mode a high value can destabilise the IDs.",
    )
    track_options = tracking_controls("video")

    if uploaded_video and st.button("Process video", type="primary"):
        suffix = Path(uploaded_video.name).suffix
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(uploaded_video.getvalue())
            source = Path(tmp.name)

        target = OUTPUTS_DIR / f"detected_{Path(uploaded_video.name).stem}.mp4"
        progress = st.progress(0.0, text="Processing frames...")

        try:
            info = video_info(source)
            session = None

            if track_options["enabled"]:
                # With frame skipping, durations need the effective fps.
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
                    p, text=f"Processing frames... {p * 100:.0f}%"
                ),
            )
        except RuntimeError as exc:
            progress.empty()
            st.error(str(exc))
        else:
            progress.empty()
            st.success(f"Processed {stats['frames']} frames ({stats['fps']:.0f} FPS).")

            if session is not None:
                show_tracking_summary(session)
            elif counts:
                st.write("**Detections in total:** " + as_counts(dict(counts.most_common())))

            st.video(str(target))
            st.download_button(
                "Download video",
                data=target.read_bytes(),
                file_name=target.name,
                mime="video/mp4",
            )
        finally:
            source.unlink(missing_ok=True)

# --- Webcam --------------------------------------------------------------
# The webcam tab is never created on a server, so this block must not run either.
if show_webcam:
    with tab_webcam:
        st.write("Live detection from your computer's camera.")
        st.caption(
            "macOS asks for camera permission the first time. You may need to "
            "restart your terminal after granting it."
        )

        webcam_options = tracking_controls("webcam")

        if "webcam_on" not in st.session_state:
            st.session_state.webcam_on = False

        start, stop = st.columns(2)
        if start.button("▶️ Start", disabled=st.session_state.webcam_on):
            st.session_state.webcam_on = True
        if stop.button("⏹️ Stop", disabled=not st.session_state.webcam_on):
            st.session_state.webcam_on = False

        frame_slot = st.empty()
        info_slot = st.empty()

        if st.session_state.webcam_on:
            capture = cv2.VideoCapture(0)
            if not capture.isOpened():
                st.session_state.webcam_on = False
                st.error("Could not open the camera. Another app may be using it.")
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
                    # Pressing Stop makes Streamlit rerun the script, which
                    # breaks out of this loop on its own.
                    while st.session_state.webcam_on:
                        ok, frame = capture.read()
                        if not ok:
                            st.warning("Could not read a frame from the camera.")
                            break

                        if live_session is not None:
                            annotated, _ = live_session.step(frame)
                            text = as_counts(live_session.unique_counts())
                            text = f"Distinct objects so far — {text}" if text else "No objects."
                        else:
                            annotated, detections = detector.detect(frame, conf, keep_classes)
                            text = as_counts(summarize(detections)) or "No objects in view."

                        frame_slot.image(
                            to_rgb(annotated), channels="RGB", use_container_width=True
                        )
                        info_slot.write(text)
                finally:
                    capture.release()

# --- Samples -------------------------------------------------------------
with tab_samples:
    sample_files = sorted(
        p for p in SAMPLES_DIR.iterdir() if p.suffix.lstrip(".").lower() in IMAGE_TYPES
    )

    if not sample_files:
        st.warning(
            "The `samples/` folder is empty. Run "
            "`python scripts/download_samples.py` to fetch the sample images."
        )
    else:
        choice = st.selectbox("Pick a sample", sample_files, format_func=lambda p: p.name)
        image = cv2.imread(str(choice))
        with st.spinner("Detecting..."):
            annotated, detections = detector.detect(image, conf, keep_classes)
        show_results(image, annotated, detections, f"detected_{choice.stem}.png")

# --- Model performance -----------------------------------------------------
with tab_metrics:
    metrics_file = DOCS_DIR / "metrics.json"

    if not metrics_file.exists():
        st.info(
            "No training metrics yet. To train and measure your own model:\n\n"
            "```\n"
            "python scripts/train.py --epochs 30\n"
            "python scripts/evaluate.py\n"
            "python scripts/compare.py\n"
            "```"
        )
    else:
        metrics = json.loads(metrics_file.read_text())
        st.subheader(f"`{metrics['model']}` — {metrics['data']}")

        overall = metrics["overall"]
        cols = st.columns(4)
        cols[0].metric(
            "mAP50",
            f"{overall['mAP50']:.3f}",
            help="Mean average precision at 50% box overlap. The headline score.",
        )
        cols[1].metric(
            "mAP50-95",
            f"{overall['mAP50-95']:.3f}",
            help="Averaged over overlap thresholds from 50% to 95%. A stricter measure.",
        )
        cols[2].metric(
            "Precision", f"{overall['precision']:.3f}", help="How many detections were correct."
        )
        cols[3].metric(
            "Recall", f"{overall['recall']:.3f}", help="How many of the real objects it found."
        )

        st.write("**Per class**")
        st.dataframe(metrics["per_class"], use_container_width=True, hide_index=True)

        comparisons = (
            sorted((DOCS_DIR / "comparison").glob("*.jpg"))
            if (DOCS_DIR / "comparison").exists()
            else []
        )
        summary_image = next((p for p in comparisons if p.stem == "summary"), None)
        singles = [p for p in comparisons if p.stem != "summary"]

        if singles:
            st.divider()
            st.subheader("Before / after")
            st.caption("Left: the pretrained COCO model. Right: our own model. Same image.")
            choice = st.selectbox(
                "Pick an image", singles, format_func=lambda p: p.stem, key="compare_pick"
            )
            st.image(str(choice), use_container_width=True)
            if summary_image:
                with st.expander("See them all together"):
                    st.image(str(summary_image), use_container_width=True)

        plots = sorted((DOCS_DIR / "plots").glob("*")) if (DOCS_DIR / "plots").exists() else []
        if plots:
            st.divider()
            st.subheader("Training curves")
            for plot in plots:
                st.caption(plot.name)
                st.image(str(plot), use_container_width=True)
