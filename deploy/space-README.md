---
title: Object Detection
emoji: 🎯
colorFrom: blue
colorTo: green
sdk: streamlit
app_file: app.py
pinned: false
license: mit
---

# 🎯 Object Detection

Object detection and tracking with YOLOv8.
[Source code and details](https://github.com/knkrc/Object-Detection).

- **Image** — upload a picture, see detections drawn as boxes
- **Video** — upload an MP4, process it frame by frame, download the result
- **Tracking** — persistent IDs, unique counts, line crossings, motion trails
- **Samples** — try it without uploading anything
- **Performance** — metrics for our fine-tuned model and a before/after comparison

Two models are available from the sidebar: the pretrained COCO model (80 classes)
and one we fine-tuned on the African Wildlife dataset — buffalo, elephant, rhino
and zebra, mAP50 0.957.

> There is no webcam tab here: on a server, opening a camera would open the
> *server's* camera rather than yours. Run the repo locally and that tab appears.
