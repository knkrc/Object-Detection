# CLAUDE.md

Context and development log for Claude working on this project (and for future
me). **Updated at the end of every milestone.**

---

## Project summary

An **object detection and tracking** app built for a GitHub portfolio / CV. It
uses YOLOv8's COCO-pretrained model to detect 80 classes in images, video and a
live camera; in tracking mode it gives each object a persistent ID and computes
unique counts and line crossings. A model we fine-tuned ourselves on the African
Wildlife dataset is also selectable from the UI. The interface is Streamlit.

**The goal:** a project that works, can be shown, and can be understood. No
over-engineering — reading the code should tell you what it does.

## Technology

| What | Why |
|---|---|
| Python 3.13 (`.venv`) | Isolated per-project environment; keeps the Anaconda base clean |
| ultralytics (YOLOv8) | Pretrained COCO model, inference in one line |
| ByteTrack (+ `lap`) | For tracking; fast, fine on CPU, built into ultralytics |
| OpenCV | Reading/writing images and video, frame processing |
| Streamlit | Quick, visual UI — easy to screenshot for a portfolio |
| MPS (Apple Silicon) | Where training runs; the Colab notebook is the GPU alternative |
| pytest + ruff | Tests and linting; both configured in `pyproject.toml` |
| Docker | Deployment; the same image runs locally and on Hugging Face Spaces |

## Commands

```bash
source .venv/bin/activate        # activate the environment
streamlit run app.py             # run the app
python scripts/download_samples.py   # fetch the sample images

python scripts/train.py --epochs 30  # fine-tune (produces models/<name>.pt)
python scripts/evaluate.py           # metrics -> docs/metrics.{json,md} + docs/plots/
python scripts/compare.py            # before/after images -> docs/comparison/
python scripts/screenshot.py         # README screenshots -> docs/screenshots/
python scripts/make_demo_gif.py      # README demo GIF -> docs/demo.gif (needs ffmpeg)

pytest                               # all tests
pytest -m "not slow"                 # the fast ones (what CI runs)
pytest -m slow                       # the ones that run the real model
ruff check . && ruff format .        # lint + format

docker compose up --build            # run in a container
./deploy/push_to_hf.sh <user>/<space>  # push to HF Spaces (needs HF_TOKEN)
```

## Architecture decisions

- **`src/detector.py` is the single entry point.** The UI layer (`app.py`) never
  touches ultralytics directly; it calls `Detector.detect()` and gets back
  `(annotated_image, [Detection, ...])`. Swapping the model later (YOLOv11,
  RT-DETR, one we trained) then only concerns one file.
- **Images travel as BGR numpy arrays.** That is OpenCV's default; conversion to
  RGB happens only on the way to the screen (`to_rgb`). We stick to this rule to
  avoid confusion.
- **The model is loaded once via `@st.cache_resource`.** Otherwise it reloads on
  every interaction and the app becomes unusable.
- **Weights live under `models/` and are gitignored.** They download on first
  run. `detector.resolve_weights()` / `stash_weights()` handle this, and both
  `Detector` and `scripts/train.py` use them — ultralytics downloads into the
  working directory, so without a shared place the project root gets littered.
- **Video processing sits in `src/video.py` and is *work-agnostic*.**
  `process_video` does not know what it is doing; it calls the
  `on_frame(frame) -> frame` function it was given. Detection and tracking share
  one loop instead of the code splitting in two. The "frame skip" (stride)
  setting trades accuracy for speed.
- **Tracking state lives in `TrackSession`.** IDs, trails and counters belong to
  a session; a new one is created for every video or webcam run. Because the
  model is shared through `@st.cache_resource`, `TrackSession.__post_init__`
  also resets ultralytics' tracker state — otherwise the previous video's IDs
  leak into the next one.
- **Trained models go under `models/`, and the UI finds them itself.**
  `config.custom_models()` scans for `.pt` files that are not built-in and adds
  them to the model list as "Custom: <name>". Training a new model needs no code
  change — dropping the file into `models/` is enough.
- **`models/african-wildlife.pt` is committed on purpose (5.9 MB).** There is a
  deliberate exception to the `*.pt` rule in `.gitignore`, so whoever clones the
  repo can try the "Custom:" model without waiting 31 minutes for training.
- **The demo GIF is scripted too.** `scripts/make_demo_gif.py` records the UI
  tour with playwright's video capture, and ffmpeg converts it with a two-pass
  palette (a single pass looks awful within GIF's 256-colour limit). The shared
  "start the app / walk the tabs" logic moved to `scripts/_preview.py`.
- **Screenshots are taken by a script, not by hand.** `scripts/screenshot.py`
  starts the app with playwright, walks it, and writes into `docs/screenshots/`,
  so they can be refreshed with one command whenever the UI changes. JPEG is
  used: the content is mostly photo and PNG bloats it (1.4 MB -> 576 KB).
- **Metrics and plots are committed under `docs/`.** `runs/` is gitignored;
  `evaluate.py` copies the plots worth showing into `docs/plots/`. The README
  and the Streamlit tab read the same files.
- **Tests split in two via the `slow` marker.** The fast ones use the fake model
  layer in `tests/conftest.py`: `FakeModel.track()` imitates only as much of
  ultralytics' output as `TrackSession` touches. That way the tracking logic is
  tested without touching torch, and CI takes seconds instead of minutes. The
  ones that use the real model are marked `slow`.
- **`pythonpath = ["."]` is required in the pytest config.** CI invokes `pytest`
  directly, which — unlike `python -m pytest` — does not add the working
  directory to `sys.path`, and `import src` fails.
- **When UI text changes, refresh the images.** `docs/screenshots/`,
  `docs/demo.gif` and `docs/comparison/` show the app's screen and output, so
  they go stale when the wording changes. Three commands:
  `scripts/screenshot.py`, `scripts/make_demo_gif.py`, `scripts/compare.py`.
  If metric keys change, `scripts/evaluate.py` too.
- **`is_deployed()` hides the webcam tab on a server.** `cv2.VideoCapture(0)`
  opens the camera of *whichever machine is running the app*; on a server that
  would be the server's camera, not the visitor's. The tab is not even created —
  the block sits under `if show_webcam:`, otherwise its widgets would leak onto
  the main page.
- **The Docker image is self-contained.** Model weights, samples and metrics are
  copied in; the container downloads nothing on first start. torch is installed
  from the CPU index (the PyPI build pulls CUDA packages on Linux).
- **The Space is not a copy of the repo.** `deploy/push_to_hf.sh` pushes only
  what the app needs to run; training scripts, tests, datasets, the demo GIF and
  the README screenshots stay out (the app reads `docs/metrics.json`,
  `docs/comparison` and `docs/plots`, nothing else under `docs/`). The
  Space's README is a separate file (`deploy/space-README.md`) because HF reads
  its configuration from README frontmatter, which our own README cannot carry.
- **The Space runs on the Streamlit SDK, not Docker.** HF only offers Docker
  Spaces on a paid plan, so the Dockerfile is not pushed. HF installs
  `deploy/space-requirements.txt` and `deploy/space-packages.txt` and runs
  `app.py` itself. The requirements file pins `torch==...+cpu`, because the
  PyPI wheel pulls CUDA packages on Linux and a free Space cannot afford them —
  the same problem the Dockerfile solves with an index flag. `packages.txt`
  carries the two apt packages opencv needs. `sdk_version` has to be set in the
  frontmatter too; without it HF answers CONFIG_ERROR. streamlit is deliberately
  absent from the Space requirements: HF installs the version named by
  `sdk_version`, and pinning it twice invites a conflict. The Dockerfile still
  stands for local runs and self-hosting, and CI still builds and smoke-tests it.
- **The Space push has to keep `.gitattributes`.** HF rejects binary files that
  are not in LFS ("use Xet storage"), and the LFS patterns live in that file.
  Its defaults cover `*.pt` but not images, so `push_to_hf.sh` also runs
  `git lfs track` for jpg/png/gif before staging — tracking after `git add`
  is too late and the push bounces.
- **Line crossings come from the sign of the cross product.** If the side of the
  line an object's centre is on flips between two frames, it has crossed; the
  direction of the flip separates in from out. No intersection maths needed.

## Code conventions

- **The whole codebase is in English.** Comments, docstrings, test names,
  variable names, UI strings, `metrics.json` keys, Dockerfile and CI comments —
  all of it. This file was translated too, so nothing is left in Turkish except
  `README.tr.md`.
- Docstrings should say *why* something is done that way, not *what* it does.
- **There are two READMEs:** `README.md` in English (primary, for international
  applications) and `README.tr.md` in Turkish. They link to each other. When
  something changes, update **both** — especially numbers like the test count,
  the coverage percentage and the metrics.
- A new feature gets its own module under `src/`; `app.py` stays UI only.
- When adding a dependency, update both `requirements.txt` (loose) and
  `requirements-lock.txt` (`pip freeze`).

---

## Milestone log

### ✅ M1 — Base application (2026-08-27)

**Done**
- `.venv` set up; ultralytics 8.4.131, torch 2.13.0, opencv 5.0.0, streamlit 1.62.0.
- `src/config.py` — paths, model list (n/s/m), defaults.
- `src/detector.py` — the `Detector` class, the `Detection` dataclass, `summarize()`.
- `src/video.py` — frame-by-frame processing into an mp4, progress callback.
- `app.py` — 4 tabs (Image / Video / Webcam / Samples) plus model selection,
  confidence threshold and class filter in the sidebar.
- `scripts/download_samples.py` — downloads Ultralytics' public demo images.
- README, .gitignore, requirements.
- `LICENSE` (MIT) to match what the README claims. **Replace the name in the
  copyright line with your full name.**
- `git init` run, files staged (no commit made).

**Verified**
- `samples/bus.jpg`: 4 detections (3× person, 1× bus). The class filter
  (`keep_classes`) works; weights land under `models/`.
- The Streamlit UI comes up and the "Samples" tab shows the original/result
  comparison correctly.
- Video pipeline: a 40-frame test video with stride=2 wrote 40 frames, the
  progress callback reaches 1.0, and the output mp4 can be read back.

**Known gaps / notes**
- The webcam loop leans on Streamlit's rerun mechanism; the "Stop" button breaks
  the loop by rerunning the script. Simple but fragile — if it causes trouble,
  `streamlit-webrtc` is the alternative.
- Video output is attempted with the `avc1` codec, falling back to `mp4v`.
  `mp4v` may not play in some browsers — the download button is there regardless.
- No tests yet.

---

### ✅ M2 — Object tracking (2026-08-28)

**Done**
- `src/tracker.py` — `TrackSession` (session state), `Track` (an identified
  object), `LineCounter` (line crossing counter), `color_for` (per-ID colour),
  `line_from_ratio` (UI choice -> pixel coordinates).
- ByteTrack (`bytetrack.yaml`) is used. Added the `lap>=0.5.12` dependency —
  ultralytics tries to install it itself when missing but then asks for a
  restart, so it is written into `requirements.txt` explicitly.
- `src/video.py` restructured: `process_video` now takes an `on_frame` callback
  and knows nothing about detection vs. tracking. Added `video_info()` (to read
  fps/size before processing).
- `Detector._class_ids` -> `class_ids` (the tracker needs the same conversion).
- `app.py` — a "Tracking mode" toggle on the Video and Webcam tabs, plus trail
  length and line direction/position controls; unique counts, the line counter,
  the duration table and CSV download.

**Verified**
- On a synthetic video, 5 IDs held across 60 frames without changing (no ID
  switches).
- Line direction tested in three scenarios: moving right -> `saga: 4`, moving
  down -> `asagi: 4`, moving left -> `sola: 4`. (These names were later
  translated to `right`/`down`/`left`.) A wrong-direction bug was fixed, below.
- Tracker reset: opening two sessions back to back on the same model object,
  the second still starts at ID 1 — no leakage.
- UI: the tracking controls render correctly, and the line settings stay
  disabled until "Count line crossings" is ticked.

**Fixed along the way**
- On a vertical line, left-to-right movement was counted as "backward". Cause:
  the line was drawn top to bottom, leaving the cross product's positive side on
  the left. Changed to draw the line bottom to top. The direction names also
  became direction-aware (`asagi/yukari`, `saga/sola`) instead of
  `ileri/geri`.

**Known gaps / notes**
- A high `stride` can destabilise IDs; the UI warns but does not prevent it.
- Unique counting trusts ByteTrack's IDs. An object that disappears for a while
  and comes back gets a new ID and is counted twice. BoT-SORT (re-ID) improves
  this — a sidebar option could be added if wanted.
- The trail dictionary grows over a session (a `deque` per ID). Memory could
  matter in a webcam session running for hours; negligible for now.

---

### ✅ M3 — Fine-tuning on our own dataset (2026-08-28)

**Dataset:** `african-wildlife` (a built-in ultralytics set, 100 MB, 4 classes:
buffalo, elephant, rhino, zebra). Chosen because elephant and zebra exist in
COCO while buffalo and rhino do not — which makes the "before/after" difference
both real and honest. No API key needed; `data=african-wildlife.yaml` downloads
it automatically.

**Done**
- `scripts/train.py` — the fine-tuning CLI. Picks the device automatically
  (cuda -> mps -> cpu), stops early via `patience`, and copies the best weights
  to `models/<name>.pt` when done.
- `scripts/evaluate.py` — writes validation metrics to `docs/metrics.json` and
  `docs/metrics.md`, and copies training plots into `docs/plots/`.
- `scripts/compare.py` — puts the pretrained COCO model and ours side by side on
  the same images.
- `notebooks/train_colab.ipynb` — the same training on a Colab GPU.
- `config.custom_models()` plus the `app.py` sidebar — the trained model is
  usable in every tab (image/video/webcam/tracking).
- The `app.py` "📊 Model performance" tab — metric cards, per-class table,
  before/after picker, training plots.

**Results** (YOLOv8n, 30 epochs, 640px, MPS, 31 minutes)

| Metric | Value |
|---|---|
| mAP50 | 0.957 |
| mAP50-95 | 0.791 |
| Precision | 0.954 |
| Recall | 0.895 |

Per-class mAP50: buffalo 0.970, elephant 0.927, rhino 0.972, zebra 0.958.

**Before/after evidence:** the COCO model sees a rhino as `cow 0.56` plus a
phantom `horse`; ours says `rhino 0.97`. For buffalo, COCO also says `cow`. On
elephant and zebra both are right — expected, since those are COCO classes.

**Fixed along the way**
- `compare.py` assumed the dataset used a `valid/images` layout, but this one
  uses `images/val`. Added a search that tries all four common layouts.
- Random image selection piled up on the dominant class (elephant), which made
  the comparison meaningless. Changed to read the class from the label files and
  take an even sample from each class.
- The comparison banners printed only the detection *count*; since the real
  difference is in the labels, the label list went into the banners.
- Filenames in the dataset contain spaces and parentheses (`3 (226).jpg`), so
  outputs are renamed after their content (`rhino.jpg`, `buffalo-2.jpg`).

**Known gaps / notes**
- 30 epochs is an arbitrary number; `patience=15` never triggered, so longer
  training might improve things slightly.
- Only YOLOv8n was tried. A bigger model via `--model yolov8s.pt` would probably
  lift mAP50-95.
- The Colab notebook was written but **not run on Colab** — it uses the same
  ultralytics calls as the local path, but review it on first use.
- `docs/` currently holds one model's results. Training a second model overwrites
  the files; if that becomes a thing, fold them into per-model folders.

---

### ✅ M4 — Tests + CI (2026-08-28)

**Done**
- `tests/conftest.py` — the fake model layer (`FakeModel`, `FakeResult`,
  `FakeBox`, `FakeDetector`) and a synthetic video fixture. Only as much of
  ultralytics' output as `TrackSession` uses is imitated.
- `tests/test_tracker.py` (26 tests) — line counter direction logic, not counting
  a first sighting, not counting while on one side, going back and forth, a point
  exactly on the line; `TrackSession` unique counts, durations, trail length,
  class filter, reset.
- `tests/test_video.py` (10 tests) — frame count, stride behaviour (including
  skipped frames repeating the last annotated one), progress callback, bad file.
- `tests/test_config.py` (8 tests) — `custom_models()` discovery and its
  exclusion of built-in models.
- `tests/test_detector.py` (11 tests, 7 of them `slow`) — `summarize()` logic is
  fast; class count, detection, filter and threshold behaviour with the real
  model are `slow`.
- `pyproject.toml` — pytest (marker, testpaths, pythonpath) and ruff
  (E/F/I/B/UP, 100 characters) configuration.
- `.github/workflows/ci.yml` — a ruff job plus a Python 3.11/3.12/3.13 test matrix.
- `requirements-dev.txt`, a CI badge and a tests section in the README.

**Result:** 55 tests, the fast suite runs in 0.7 s, `src/` coverage **91%**
(tracker 98%, video 95%, config 100%). `detector` sits at 54% — its
model-dependent parts are only in the `slow` tests.

**Fixed along the way**
- CI invoked `pytest` directly and `import src` failed; locally it looked fine
  because `python -m pytest` adds the working directory to `sys.path`. Added
  `pythonpath = ["."]` and verified with a bare `pytest` locally.
- Ruff found 15 issues (import order, long lines, `%` formatting); 9 were fixed
  automatically, the rest by hand. 9 files were reformatted.

**Known gaps / notes**
- Every CI job installs `torch` from the CPU index (the PyPI build pulls CUDA
  packages on Linux, ~2.5 GB). Installation still dominates the job time.
- `app.py` is not tested. Testing a Streamlit UI needs `streamlit.testing`; not
  worth it for now, the UI is verified by hand.
- The training pipeline under `scripts/` is not tested — it needs real training,
  which does not belong in CI.
- No coverage badge; the number only shows in the CI output. Codecov could be
  added if wanted.

---

### ✅ M5 — Docker + deployment pipeline (2026-08-28)

**Done**
- `Dockerfile` — python:3.12-slim, torch from the CPU index, non-root user
  (uid 1000, required by HF Spaces), healthcheck, port 8501. Model weights,
  samples and metrics are baked into the image.
- `.dockerignore` — datasets, `runs/`, tests and scripts stay out of the image.
- `docker-compose.yml` — one-command local run, with `outputs/` bind-mounted.
- `config.is_deployed()` — checks `DEPLOYED` or HF's own `SPACE_ID`; `app.py`
  skips creating the webcam tab accordingly.
- `deploy/space-README.md` — the Space's own README (frontmatter with
  `sdk: docker`, `app_port: 8501`).
- `deploy/push_to_hf.sh` — clones the Space, copies what is needed, pushes.
- A `docker` job in CI: builds the image, starts the container, health-checks it
  and verifies that server mode is actually active.
- 5 more tests in `tests/test_config.py` (`is_deployed` behaviour). 60 in total.

**Verified**
- Local mode: 5 tabs, webcam present, the subtitle mentions the live camera.
- Server mode (`DEPLOYED=1`): 4 tabs, no webcam, the subtitle changes too.
- The Docker image **could not be tested on this machine** (docker is not
  installed); the build and smoke test run in CI and pass: the image builds
  (~170 s), the container comes up, the health check responds, server mode is on.

**Fixed along the way**
- The first attempt wrapped the webcam block in
  `with tab_webcam if show_webcam else nullcontext():`. That still runs the code
  inside and the widgets fell onto the main page. Moved under `if show_webcam:`.
- The extra indentation pushed one line past 100 characters; ruff caught it.

**Known gaps / notes**
- **The live demo is not published yet** — Kaan has to create the HF account and
  the Space. The script and configuration are ready, and the README has a slot
  for the link.
- Video processing on a free-tier CPU will be slow; users will wait on long
  videos. A duration/size limit could be added to the UI if wanted.
- The image is **2.16 GB** (measured in CI), mostly torch + ultralytics. To
  shrink it, `opencv-python-headless` is an option (which also drops the need
  for `libgl1`), or a multi-stage build; not worth it for now.
- `DEPLOYED=1` is hard-coded in `docker-compose.yml`. Correct, since the webcam
  would not work in a container anyway, though on Linux `--device /dev/video0`
  could be tried.

---

### ✅ Translating the UI to English (2026-08-28)

The follow-up after M5, once the README had been translated.

**Translated**
- `app.py` — every label, help text, message, button and tab name, and the
  downloaded filenames (`detected_*.png`, `tracking-data.csv`).
- `src/config.py` — model labels (`YOLOv8n (fast)` and so on), `Ozel:` ->
  `Custom:` (now the `CUSTOM_PREFIX` constant).
- `src/tracker.py` — line direction names (`asagi/yukari` -> `down/up`,
  `saga/sola` -> `right/left`), duration table columns (`object`, `seconds`,
  `frames`, `first_frame`, `last_frame`), summary keys (`total_objects`, `line`,
  `frames`). `line_from_ratio` now takes `horizontal`/`vertical`.
- `src/video.py` — error messages.
- `scripts/evaluate.py` — `metrics.json` keys (`overall`, `per_class`, `class`)
  and the markdown headings.
- `scripts/compare.py` — the banners on the images ("Pretrained COCO model" /
  "Our own model"), summary grid renamed `ozet.jpg` -> `summary.jpg`.
- `deploy/space-README.md` — dropped the "the interface is in Turkish" note.

**Regenerated:** `docs/metrics.{json,md}`, `docs/comparison/*` (7 images),
`docs/screenshots/*` (`tespit.jpg` -> `detection.jpg`, `model-performansi.jpg`
-> `model-performance.jpg`), `docs/demo.gif`.

**Tests:** 65 tests, all updated to the new names. The line direction logic was
verified again in three scenarios (`right: 4`, `down: 4`, `left: 4`).

**Fixed along the way**
- Reordered the model switch in the demo GIF. Switching to the wildlife model
  before opening the metrics tab made it re-evaluate the bus photo and produce
  out-of-domain hits like "elephant 0.43" — correct, but a frame that makes the
  model look bad to a viewer. Now the tab is opened first and the model switched
  after.
- Changing the `metrics.json` keys forced the fields `app.py` reads to change
  too; the file was regenerated by rerunning `evaluate.py`.

---

### ✅ Translating code comments to English (2026-08-28)

A continuation of the UI translation. **No Turkish text was left in the codebase.**

**Translated**
- `src/` (4 files) and `app.py` — every docstring and inline comment
- `scripts/` (7 files) — docstrings, argparse help text, console output
- `tests/` (5 files) — comments, docstrings, local variable names and **all 65
  test function names** (`test_soldan_saga_gecis_saga_sayilir` ->
  `test_left_to_right_counts_as_right`)
- `Dockerfile`, `docker-compose.yml`, `.gitignore`, `.dockerignore`,
  `pyproject.toml`, `requirements*.txt`, `.github/workflows/ci.yml`,
  `deploy/push_to_hf.sh`
- `notebooks/train_colab.ipynb` — markdown cells and code comments

**Verified:** 65 tests pass, ruff is clean, all six scripts run with `--help`,
and `download_samples.py` was actually run. A regex sweep over 50+ Turkish words
and suffixes returns nothing.

---

### ✅ Translating this file to English (2026-08-28)

The last Turkish artefact besides `README.tr.md`.

Translated in place, without keeping a Turkish copy. Reasoning: this is a
working document updated at every milestone, and two copies would drift apart —
the two READMEs already carry an "update both" burden. `README.tr.md` stays
because it is public-facing and worth having in two languages.

Historical entries keep their original numbers (M4's "55 tests, 91%", M5's "60
tests") — this is a log, and later entries record how those figures changed.

---

## Upcoming

### 🔜 Next step — get the demo actually running
The README links to `knkrc26/object-detection`, but the Space is not up yet.
What is left:

1. **Create the Space** as public. Any SDK on the form will do — the push
   overwrites README.md with `sdk: streamlit`. Docker is paid, hence the switch.
2. **Push** — set `HF_TOKEN` and run
   `./deploy/push_to_hf.sh knkrc26/object-detection`. The first build takes a few
   minutes.
3. **Watch the build.** The `sdk: streamlit` route is the untested part: HF
   dropped Streamlit from the Space creation form, but the backend still serves
   Spaces declaring it, so setting it in frontmatter should work. If the build
   refuses the SDK, the fallbacks are Streamlit Community Cloud (free, native
   Streamlit) or writing a Gradio front end against the existing `src/` modules.

**Note on the username:** HF is `knkrc26`, GitHub is `knkrc`. The README linked
the GitHub name for a while and every visitor got a 401.

### 💡 Idea pool (unordered)
- A CLI (`python detect.py --image foo.jpg`) for batch work.
- Heatmap / density visualisation.
- Exporting detection results as JSON (the tracking CSV came in M2).
- A BoT-SORT option: remembers a long-lost object through re-ID.
- A second dataset from images you collect yourself (the M3 pipeline is ready).
- Training and comparing a bigger model (`yolov8s/m`).
- UI tests with `streamlit.testing`.
- Codecov integration and a coverage badge.
- A video size/duration limit in the demo, to protect the free-tier CPU.
- An animated demo of tracking mode — there is a UI tour GIF, but tracking is
  not in it. Every one of Ultralytics' sample videos is either their own demo
  output (with someone else's boxes burned in) or under a second long. This
  needs a royalty-free stock clip or footage Kaan shoots himself.
- A model comparison tab: n/s/m results side by side on the same image.
