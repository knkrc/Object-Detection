Measured on arm64 Darwin 25.6.0, Python 3.13.9. Timings cover `Detector.detect()` end to end,
including drawing the boxes — the wait a user actually sees.

| Model | Device | ms / image | FPS |
|---|---|---|---|
| yolov8n.pt | mps | 9.3 | 107.2 |
| yolov8n.pt | cpu | 25.5 | 39.2 |
| yolov8s.pt | mps | 15.6 | 63.9 |
| yolov8s.pt | cpu | 50.0 | 20.0 |
| yolov8m.pt | mps | 29.9 | 33.4 |
| yolov8m.pt | cpu | 106.9 | 9.4 |
